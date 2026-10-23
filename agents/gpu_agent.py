"""GPU worker: claims queued training runs, trains a YOLO nano model, exports it for the IMX500 and uploads the results.

Run it where the NVIDIA GPU is. On Windows that means inside WSL2 (Ultralytics' IMX export only runs on Linux):

    export MT_SERVER=http://<ip of the Mac running the backend>:8000
    export MT_WORKER_TOKEN=<MT_WORKER_TOKEN from backend/.env>
    python gpu_agent.py --name gpu-3060 --device 0
"""

import argparse
import json
import logging
import math
import os
import platform
import re
import shutil
import socket
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path

from common import Api, ApiError, Heartbeat, LeaseLost, setup_logging

log = logging.getLogger("agent")


@dataclass
class TrainResult:
    best_pt: Path
    metrics: dict


def preflight(export_imx: bool, system: str | None = None, which=shutil.which) -> str | None:
    """Problems that would only show up after hours of training. Checked first so the run fails in seconds."""
    if not export_imx:
        return None
    system = system or sys.platform
    if not system.startswith("linux"):
        return ("The IMX500 export only runs on Linux, and this worker is on "
                f"{platform.system() or system}. On Windows, run the agent inside WSL2 (see agents/README.md).")
    if which("java") is None:
        return "The IMX500 export needs Java 17 or newer: sudo apt install openjdk-17-jre-headless"
    return None


def rewrite_data_path(data_yaml: Path, root: Path) -> None:
    """Point `path:` at the extracted dataset. A relative `path: .` would be resolved against Ultralytics' own
    datasets directory, not against the folder holding data.yaml, and training would not find the images."""
    text = data_yaml.read_text()
    new_line = f"path: {json.dumps(str(root.resolve()))}"  # a JSON string is a valid YAML double-quoted string
    text, n = re.subn(r"(?m)^path:.*$", new_line, text)
    data_yaml.write_text(text if n else new_line + "\n" + text)


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None  # NaN would not survive JSON


CALIBRATION_IMAGES = 150  # the IMX export quantizes against validation images; Sony suggests 300+, but every one costs time and memory


def calibration_fraction(data_yaml: Path, wanted: int = CALIBRATION_IMAGES) -> float:
    """The `fraction` of the validation split to calibrate on, so about `wanted` images are used. The export of run 5 (520 validation
    images, one at a time) ran for 13 minutes and then its process was killed (exit 137), where the export of 23 images had finished
    in 8. How much of that was memory is not known; fewer images makes both time and memory smaller."""
    val = data_yaml.parent / "images" / "val"
    n = sum(1 for p in val.iterdir() if p.is_file()) if val.is_dir() else 0
    return min(1.0, wanted / n) if n else 1.0


class UltralyticsBackend:
    """The only part that needs torch/ultralytics. Tests swap it for a fake."""

    def __init__(self, device: str):
        self.device = device

    def train(self, cfg: dict, data_yaml: Path, run_dir: Path, on_progress, should_stop) -> TrainResult:
        from ultralytics import YOLO

        model = YOLO(f"{cfg['model']}.pt")  # downloads the pretrained nano weights on first use

        def on_epoch_end(trainer):
            m = trainer.metrics
            on_progress(epoch=min(trainer.epoch + 1, trainer.epochs), epochs=trainer.epochs, message="training",  # also fires once after the last epoch
                        precision=_num(m.get("metrics/precision(B)")), recall=_num(m.get("metrics/recall(B)")),
                        map50=_num(m.get("metrics/mAP50(B)")), map50_95=_num(m.get("metrics/mAP50-95(B)")))
            if should_stop():
                trainer.stop = True  # Ultralytics checks this right after the callback, so training ends this epoch

        model.add_callback("on_fit_epoch_end", on_epoch_end)
        results = model.train(data=str(data_yaml), epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=cfg["batch"],
                              device=self.device, project=str(run_dir / "runs"), name="train", exist_ok=True, plots=False)
        best = Path(model.trainer.best)
        if not best.exists():
            best = Path(model.trainer.last)
        return TrainResult(best, final_metrics(results, model.trainer.metrics))

    def export_imx(self, best_pt: Path, data_yaml: Path, cfg: dict, run_dir: Path) -> tuple[Path, Path]:
        from ultralytics import YOLO

        # `data` supplies the calibration images for int8 quantization.
        fraction = calibration_fraction(data_yaml)
        log.info("calibrating the int8 export on %.0f%% of the validation images", fraction * 100)
        exported = YOLO(str(best_pt)).export(format="imx", data=str(data_yaml), imgsz=cfg["imgsz"], device=self.device, fraction=fraction)
        out = Path(exported)  # the <weights>_imx_model/ folder that imxconv-pt wrote packerOut.zip and labels.txt into
        if not out.is_dir():
            out = best_pt.with_name(best_pt.stem + "_imx_model")
        packer, labels = next(out.rglob("packerOut.zip"), None), next(out.rglob("labels.txt"), None)
        if packer is None or labels is None:
            found = sorted(str(p.relative_to(out)) for p in out.rglob("*")) if out.exists() else "(no output folder)"
            raise RuntimeError(f"The IMX export finished but packerOut.zip / labels.txt are missing in {out}: {found}")
        return packer, labels


def final_metrics(results, last_epoch_metrics: dict) -> dict:
    """Validation metrics of the best weights, overall and per class."""
    d = getattr(results, "results_dict", None) or last_epoch_metrics or {}
    out = {"precision": _num(d.get("metrics/precision(B)")), "recall": _num(d.get("metrics/recall(B)")),
           "map50": _num(d.get("metrics/mAP50(B)")), "map50_95": _num(d.get("metrics/mAP50-95(B)"))}
    try:
        box, names = results.box, results.names
        out["per_class"] = {names[int(i)]: {"ap50": _num(a50), "ap50_95": _num(ap)}
                            for i, a50, ap in zip(box.ap_class_index, box.ap50, box.ap)}
    except Exception:  # noqa: BLE001 - per-class numbers are a nicety; never lose a finished run over them
        log.info("per-class metrics unavailable")
    return out


def extract_dataset(zip_path: Path, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)  # extractall drops absolute paths and ".." components
    data_yaml = dest / "data.yaml"
    if not data_yaml.exists():
        raise RuntimeError("the dataset zip has no data.yaml")
    rewrite_data_path(data_yaml, dest)
    return data_yaml


def fetch_dataset(api, url: str, run_dir: Path, say=lambda message: None, tries: int = 3, pause: float = 5.0, link: str | None = None) -> Path:
    """Download and unpack the dataset. A transfer that was cut short, or an answer that is not a zip, is fetched again: the
    connection to the backend (a tunnel, for a RunPod pod) is not perfectly steady. The last error says what arrived.

    `link`, when the backend has one, is a signed link to a copy in the dataset bucket: much faster from a RunPod pod than the
    Mac's upload link. It is tried first, once; anything wrong with it and the backend's route is used instead."""
    for attempt in range(1, 3 if link else 1):  # two quick tries, not the agent's full patience: the fallback is right there
        say("downloading the dataset from the bucket")
        try:
            return _fetch_once(api, link, run_dir)
        except Exception as e:  # noqa: BLE001 - the backend route below still works
            log.warning("the bucket's copy didn't work, try %d (%s)", attempt, e)
    if link:
        log.warning("downloading the dataset through the backend instead")
    zip_path = run_dir / "dataset.zip"
    for attempt in range(1, tries + 1):
        say("downloading the dataset" if attempt == 1 else f"downloading the dataset (try {attempt} of {tries})")
        api.download(url, zip_path)
        try:
            return extract_dataset(zip_path, run_dir / "dataset")
        except zipfile.BadZipFile:
            size, head = zip_path.stat().st_size, zip_path.read_bytes()[:80]
            log.warning("the dataset download is not a valid zip (%d bytes, starts with %r)", size, head)
            if attempt == tries:
                raise RuntimeError(f"the dataset download is not a valid zip after {tries} tries (last: {size} bytes, starts with {head!r})") from None
            time.sleep(pause)


def _fetch_once(api, url: str, run_dir: Path) -> Path:
    zip_path = run_dir / "dataset.zip"
    api.call("GET", url, out=zip_path, retry=False, timeout=60)
    try:
        return extract_dataset(zip_path, run_dir / "dataset")
    except zipfile.BadZipFile:
        raise RuntimeError(f"not a valid zip ({zip_path.stat().st_size} bytes)") from None


def run_task(api: Api, task: dict, workdir: Path, backend, heartbeat_interval: float = 20.0) -> None:
    run_id, cfg = task["run_id"], task["config"]
    log.info("claimed run %s (%s): %s", run_id, task["name"], cfg)
    problem = preflight(cfg.get("export_imx", True))
    if problem:
        log.error(problem)
        api.fail(run_id, problem)
        return

    run_dir = workdir / f"run-{run_id}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    try:
        with Heartbeat(api, run_id, heartbeat_interval) as hb:
            data_yaml = fetch_dataset(api, task["dataset_url"], run_dir, lambda message: hb.update(message=message),
                                      link=task.get("dataset_link"))

            result = backend.train(cfg, data_yaml, run_dir, hb.update, hb.stopped)
            if hb.stopped():
                log.info("run %s stopped before finishing; nothing to upload", run_id)
                return
            hb.update(message="uploading the trained weights")  # a stalled upload then shows in the UI, not as a stuck "training"
            lost_upload = None
            try:
                api.upload(run_id, "best_pt", result.best_pt)
            except ApiError as e:
                if e.status != 0 or not cfg.get("export_imx", True):  # only "can't get through" is worth going on from
                    raise
                lost_upload = e  # still export: it is the part nobody has seen work yet, and the pod is paid for anyway
                log.error("could not upload best.pt (%s); exporting anyway so the run at least shows whether that works", e.detail)

            if cfg.get("export_imx", True):
                # retry=True: an outage here must not fail the run after the training is already done
                if api.heartbeat(run_id, {"message": "exporting for the IMX500 (quantizing, takes a few minutes)"}, stage="exporting", retry=True):
                    return
                hb.update(message="exporting for the IMX500 (quantizing, takes a few minutes)")
                try:
                    packer, labels = backend.export_imx(result.best_pt, data_yaml, cfg, run_dir)
                except Exception as e:  # noqa: BLE001
                    if lost_upload:
                        raise RuntimeError(f"best.pt could not be uploaded ({lost_upload.detail}) and the IMX export failed too: {e}") from e
                    raise
                if hb.stopped():
                    return
                if lost_upload:
                    api.fail(run_id, f"best.pt could not be uploaded ({lost_upload.detail}). The IMX export itself worked "
                                     f"({packer.name}, {packer.stat().st_size / 2**20:.1f} MB), so the failure is the network path, not the export.")
                    return
                hb.update(message="uploading the IMX500 package")
                api.upload(run_id, "imx_zip", packer)
                api.upload(run_id, "labels", labels)
            api.complete(run_id, result.metrics)
            log.info("run %s finished: %s", run_id, result.metrics.get("map50"))
    except LeaseLost as e:
        log.warning("run %s is no longer ours (%s); abandoning it", run_id, e.detail)
    except Exception as e:  # noqa: BLE001 - report any failure to the backend instead of dying silently
        log.error("run %s failed:\n%s", run_id, traceback.format_exc())
        api.fail(run_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
    finally:
        shutil.rmtree(run_dir / "dataset", ignore_errors=True)  # keep weights and logs, drop the copy of the images


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(argv=None, backend=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("MT_SERVER"), help="backend URL (or MT_SERVER)")
    ap.add_argument("--name", default=f"gpu-{socket.gethostname()}", help="worker name shown in the UI")
    ap.add_argument("--device", default="auto", help="CUDA device index like 0, or cpu (default: auto)")
    ap.add_argument("--workdir", default="work", help="where datasets and weights are kept while running")
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between checks for new runs")
    ap.add_argument("--heartbeat", type=float, default=20.0, help="seconds between lease renewals")
    ap.add_argument("--once", action="store_true", help="do at most one run, then exit (also exits if none is waiting)")
    ap.add_argument("--patience", type=float, default=float(os.environ.get("MT_PATIENCE") or 6 * 3600),
                    help="seconds a request keeps being retried while the backend can't be reached, before the run is given up (default 6 h, "
                         "or MT_PATIENCE; the RunPod launcher sets 15 minutes because a rented GPU bills while it waits)")
    args = ap.parse_args(argv)
    token = os.environ.get("MT_WORKER_TOKEN")  # from the environment, so it never shows up in `ps`
    if not args.server or not token:
        ap.error("set MT_SERVER (or --server) and MT_WORKER_TOKEN")

    setup_logging()
    api = Api(args.server, token, args.name, max_wait=args.patience)
    try:
        api.ping()
    except ApiError as e:
        log.error("can't use the backend: %s", e)
        return 2
    backend = backend or UltralyticsBackend(pick_device(args.device))
    log.info("worker '%s' connected to %s", args.name, args.server)

    workdir = Path(args.workdir).resolve()  # Ultralytics treats a relative project path as relative to its own runs/ folder
    while True:
        try:
            task = api.claim("gpu")
        except ApiError as e:
            log.warning("claim failed: %s", e)
            task = None
        if task:
            run_task(api, task, workdir, backend, args.heartbeat)
            if args.once:
                return 0
        elif args.once:
            return 0
        else:
            time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
