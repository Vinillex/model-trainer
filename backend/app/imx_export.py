"""The IMX500 export on this Mac, as two jobs of the job queue.

`imx_tools_setup` builds the export tools' own Python once: Python 3.11 + the same pinned packages a RunPod pod installs
(agents/runpod_start.sh), in settings.imx_export_dir/env. It never touches the backend's own packages; that is why the export
runs as a subprocess there.

`imx_export` exports one run: it takes the run's best.pt and up to CALIBRATION_IMAGES images of its dataset version (the int8
calibration set), runs agents/imx_export_mac.py in that Python, stores packerOut.zip + labels.txt as the run's artifacts and
hands the run to the Pi (awaiting_package). It is queued by itself when a pod finishes a run made with export_where "mac", and
by the Training page's "Export on this Mac" button (a retry, or a new export of an older run).
"""

import json
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import training
from .config import settings
from .datasets import load_manifest, resolve_manifest
from .models import DatasetVersion, Job, JobStatus, RunStage, TrainingRun
from .storage import get_storage
from .yolo_export import item_name, label_text

AGENT = Path(__file__).resolve().parents[2] / "agents" / "imx_export_mac.py"
# Every package of the working tools at its exact version (committed). Setup installs this list, never anything newer; without the file
# it installs TOOLS and writes the file from what it got.
LOCK = Path(__file__).resolve().parents[1] / "imx-export.lock.txt"
PYTHON_VERSION = "3.11"  # what Ultralytics declares for its IMX export environment
# The pod's list (agents/runpod_start.sh) plus the Ultralytics pin of agents/requirements-gpu.txt: change them together.
# imx_export_mac.py lifts a Linux-only check inside this exact Ultralytics version.
TOOLS = ("torch==2.9.1", "torchvision==0.24.1", "ultralytics==8.4.183", "model-compression-toolkit>=2.4.1", "edge-mdt-cl<1.1.0",
         "edge-mdt-tpc>=1.2.0", "pydantic<2.12", "imx500-converter[pt]>=3.17.3")
UV = "uv<1"
CALIBRATION_IMAGES = 150  # as the pod: Sony suggests 300+, but every image costs time and memory
EXPORT_TIMEOUT = 45 * 60  # the run 5 export took 7.5 min on this Mac
SETUP_TIMEOUT = 30 * 60
EXPORTABLE = (RunStage.awaiting_export, RunStage.awaiting_package, RunStage.done)  # best.pt exists and no worker holds the run


def base() -> Path:
    return Path(settings.imx_export_dir).resolve()  # absolute: the export runs with its work folder as the current directory


def env_python() -> Path:
    return base() / "env" / "bin" / "python"


def _marker() -> Path:
    return base() / "ready.json"


def _wanted() -> str:
    """What the environment must hold: the lock file's text, or the TOOLS list while there is no lock file."""
    try:
        return LOCK.read_text()
    except OSError:
        return "\n".join(TOOLS)


def tools_status() -> dict:
    """Whether the export tools are set up as the lock file (or TOOLS) says; a changed lock needs setting up again."""
    try:
        ready = json.loads(_marker().read_text()).get("wanted") == _wanted() and env_python().exists()
    except (OSError, ValueError):
        ready = False
    return {"ready": ready, "java": _java_version(), "path": str(base() / "env")}


def _java_version() -> int | None:
    try:
        out = subprocess.run(["java", "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    words = out.split()
    for i, w in enumerate(words[:-1]):
        if w in ("openjdk", "java") and words[i + 1].split(".")[0].isdigit():
            return int(words[i + 1].split(".")[0])
    return None


def busy(db: Session, job_type: str, run_id: int | None = None) -> Job | None:
    """A queued or running job of this type (for this run)."""
    for job in db.scalars(select(Job).where(Job.type == job_type, Job.status.in_([JobStatus.queued, JobStatus.running]))):
        if run_id is None or job.params.get("run_id") == run_id:
            return job
    return None


def queue_setup(db: Session, project_id: int) -> Job:
    if job := busy(db, "imx_tools_setup"):
        return job
    job = Job(project_id=project_id, type="imx_tools_setup", params={})
    db.add(job)
    db.flush()
    return job


def queue_export(db: Session, run: TrainingRun) -> Job:
    """Queue the export of `run` (or return the one already waiting). Raises ValueError (safe to show) when it can't be exported."""
    if run.stage not in EXPORTABLE:
        raise ValueError(f"Run {run.id} is {run.stage.value}; only a trained run that no worker holds can be exported.")
    if "best_pt" not in run.artifacts:
        raise ValueError(f"Run {run.id} has no best.pt to export.")
    if job := busy(db, "imx_export", run.id):
        return job
    job = Job(project_id=run.project_id, type="imx_export", params={"run_id": run.id})
    db.add(job)
    db.flush()
    return job


# ---------- running a command with live progress ----------

def _run(cmd: list[str], db: Session, job: Job, step: str, log: Path, timeout: float, cwd: Path | None = None) -> list[str]:
    """Run `cmd`, appending its output to `log` and showing its latest line as the job's progress. Returns the last lines.
    Raises RuntimeError when it fails, times out or the job is cancelled."""
    tail: deque[str] = deque(maxlen=40)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        out.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                stdin=subprocess.DEVNULL)
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        last_shown = 0.0
        try:
            for line in proc.stdout:
                out.write(line)
                line = line.strip()
                if line:
                    tail.append(line)
                if time.monotonic() - last_shown > 2:  # a DB write per output line would be far too many
                    last_shown = time.monotonic()
                    db.refresh(job)
                    if job.cancel_requested:
                        proc.kill()
                    job.progress = {**job.progress, "step": step, "message": line[:300]}
                    db.commit()
            proc.wait()
        finally:
            timer.cancel()
    if job.cancel_requested:
        raise RuntimeError("Cancelled.")
    if proc.returncode != 0:
        why = "it took too long" if proc.returncode < 0 and not job.cancel_requested else f"exit {proc.returncode}"
        raise RuntimeError(f"{step} failed ({why}): " + " | ".join(list(tail)[-3:])[:1500] + f" (full log: {log})")
    return list(tail)


# ---------- the jobs ----------

def handle_tools_setup(db: Session, job: Job) -> None:
    root = base()
    log = root / "setup.log"
    if (java := _java_version()) is None or java < 17:
        raise RuntimeError("Java 17 or newer is needed by Sony's converter (imxconv-pt). Install it with: brew install openjdk@17")
    _marker().unlink(missing_ok=True)
    uv_env = root / "uv"
    if not (uv_env / "bin" / "uv").exists():
        _run([sys.executable, "-m", "venv", str(uv_env)], db, job, "making a place for uv", log, SETUP_TIMEOUT)
        _run([str(uv_env / "bin" / "pip"), "install", UV], db, job, "installing uv", log, SETUP_TIMEOUT)
    uv = str(uv_env / "bin" / "uv")
    shutil.rmtree(root / "env", ignore_errors=True)  # never patch a half-built environment
    _run([uv, "venv", "--python", PYTHON_VERSION, "--seed", str(root / "env")], db, job, f"installing Python {PYTHON_VERSION}", log, SETUP_TIMEOUT)
    if LOCK.exists():  # exactly the versions that worked, nothing newer
        _run([uv, "pip", "sync", "--python", str(env_python()), str(LOCK)], db, job, "installing the export tools (about 1 GB)", log, SETUP_TIMEOUT)
    else:
        _run([uv, "pip", "install", "--python", str(env_python()), *TOOLS], db, job, "installing the export tools (about 1 GB)", log, SETUP_TIMEOUT)
    _run([str(env_python()), "-c", "import ultralytics, model_compression_toolkit, edgemdt_tpc, onnx; "
          f"assert ultralytics.__version__ == {TOOLS[2].split('==')[1]!r}, ultralytics.__version__; print('export tools import fine')"],
         db, job, "checking the export tools", log, 600)
    if not (env_python().parent / "imxconv-pt").exists():
        raise RuntimeError(f"Sony's converter imxconv-pt was not installed in {env_python().parent}. Full log: {log}")
    if not LOCK.exists():  # first install: record what worked, for every later setup (commit the file)
        frozen = subprocess.run([uv, "pip", "freeze", "--python", str(env_python())], capture_output=True, text=True, check=True).stdout
        LOCK.write_text(f"# The IMX500 export tools, exactly as installed and verified (app/imx_export.py). Python {PYTHON_VERSION}.\n" + frozen)
    _marker().write_text(json.dumps({"wanted": _wanted(), "python": PYTHON_VERSION, "at": time.time()}))
    job.progress = {"step": "done", "message": "The IMX export tools are ready."}


def _calibration_set(db: Session, run: TrainingRun, folder: Path) -> Path:
    """Up to CALIBRATION_IMAGES images of the run's dataset version (val first, train if val is short) as a small YOLO dataset."""
    version = db.get(DatasetVersion, run.dataset_id)
    if version is None:
        raise RuntimeError(f"Run {run.id}'s dataset version is gone; it supplies the calibration images.")
    manifest = resolve_manifest(db, load_manifest(version))
    rnd = random.Random(run.id)  # the same images every time this run is exported
    val = [it for it in manifest["images"] if it["split"] == "val"]
    rest = [it for it in manifest["images"] if it["split"] == "train"]
    picked = rnd.sample(val, min(len(val), CALIBRATION_IMAGES))
    picked += rnd.sample(rest, min(len(rest), CALIBRATION_IMAGES - len(picked)))
    storage = get_storage()
    for split in ("train", "val"):  # Ultralytics wants a train entry too; the same images serve
        (folder / "images" / split).mkdir(parents=True, exist_ok=True)
        (folder / "labels" / split).mkdir(parents=True, exist_ok=True)
        for it in picked:
            (folder / "images" / split / f"{item_name(it)}.{it['ext']}").write_bytes(storage.get(it["key"]))
            (folder / "labels" / split / f"{item_name(it)}.txt").write_text(label_text(it))
    names = "\n".join(f"  {c['index']}: {json.dumps(c['name'])}" for c in manifest["classes"])
    yaml = folder / "data.yaml"
    yaml.write_text(f"path: {folder}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    return yaml


def handle_export(db: Session, job: Job) -> None:
    run = db.get(TrainingRun, job.params["run_id"])
    if run is None:
        raise RuntimeError("The training run no longer exists.")
    if run.stage not in EXPORTABLE or "best_pt" not in run.artifacts:
        raise RuntimeError(f"Run {run.id} can't be exported now (it is {run.stage.value}).")
    if not tools_status()["ready"]:
        raise RuntimeError("The IMX export tools aren't set up on this Mac. Press 'Set up the IMX export tools' on the Training page.")
    work = base() / "work" / f"run-{run.id}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        run.progress = {**run.progress, "message": "exporting for the IMX500 on this Mac"}
        run.error = None
        job.progress = {"step": "preparing", "message": "copying best.pt and the calibration images"}
        db.commit()
        best = work / "best.pt"
        best.write_bytes(get_storage().get(run.artifacts["best_pt"]["key"]))
        yaml = _calibration_set(db, run, work / "calibration")
        lines = _run([str(env_python()), "-u", str(AGENT), str(best), str(yaml), str(run.config.get("imgsz", 640))],
                     db, job, "exporting (quantizing, a few minutes)", work / "export.log", EXPORT_TIMEOUT, cwd=work)
        result = next((json.loads(line[len("MT-RESULT "):]) for line in reversed(lines) if line.startswith("MT-RESULT ")), None)
        if result is None:
            raise RuntimeError("The export ended without saying where its files are: " + " | ".join(lines[-3:]))
        db.refresh(run)
        if run.stage not in EXPORTABLE:  # cancelled or deleted meanwhile: don't bring it back
            raise RuntimeError(f"Run {run.id} became {run.stage.value} during the export; its files were not stored.")
        storage, artifacts = get_storage(), dict(run.artifacts)
        for name, path in (("imx_zip", result["packer"]), ("labels", result["labels"])):
            data = Path(path).read_bytes()
            key = f"projects/{run.project_id}/training/{run.id}/{training.ARTIFACT_FILES[name]}"
            storage.put(key, data)
            artifacts[name] = {"key": key, "size": len(data)}
        if old := artifacts.pop("rpk", None):  # packaged from the previous export: the Pi packages this one again
            storage.delete(old["key"])
        run.artifacts = artifacts
        run.stage, run.finished_at = RunStage.awaiting_package, None
        run.progress = {**run.progress, "message": "exported on this Mac; waiting for the Pi to package it"}
        job.progress = {"step": "done", "message": "exported; the Pi packages it next"}
        db.commit()
    except Exception as e:
        db.rollback()
        run = db.get(TrainingRun, job.params["run_id"])
        if run is not None and run.stage in EXPORTABLE:  # best.pt stays; "Export on this Mac" tries again
            run.error = f"The IMX export on this Mac failed: {e}"[:2000]
            db.commit()
        raise
    finally:
        shutil.rmtree(work / "calibration", ignore_errors=True)  # export.log stays for a look afterwards
