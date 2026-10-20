"""Raspberry Pi worker: turns the GPU worker's packerOut.zip into the network.rpk the IMX500 loads.

`imx500-package` only runs on a Raspberry Pi. Standard library only, so there is nothing to pip install:

    sudo apt install imx500-tools
    export MT_SERVER=http://<ip of the Mac running the backend>:8000
    export MT_WORKER_TOKEN=<MT_WORKER_TOKEN from backend/.env>
    python3 pi_agent.py --name pi-camera
"""

import argparse
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import Api, ApiError, Heartbeat, LeaseLost, setup_logging

log = logging.getLogger("agent")


def package(packer_zip: Path, out_dir: Path, packager: str, timeout: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([packager, "-i", str(packer_zip), "-o", str(out_dir)], capture_output=True, text=True, timeout=timeout)
    rpk = out_dir / "network.rpk"
    if proc.returncode != 0 or not rpk.exists():
        raise RuntimeError(f"{packager} exited with {proc.returncode} and "
                           f"{'made' if rpk.exists() else 'did not make'} network.rpk:\n{(proc.stdout + proc.stderr)[-1500:]}")
    return rpk


def run_task(api: Api, task: dict, workdir: Path, packager: str, heartbeat_interval: float = 20.0, timeout: float = 1800) -> None:
    run_id = task["run_id"]
    log.info("claimed run %s (%s) for packaging", run_id, task["name"])
    run_dir = workdir / f"run-{run_id}"
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    try:
        with Heartbeat(api, run_id, heartbeat_interval) as hb:
            hb.update(message="packaging the .rpk")
            api.download(f"/workers/runs/{run_id}/artifacts/imx_zip", run_dir / "packerOut.zip")
            rpk = package(run_dir / "packerOut.zip", run_dir / "out", packager, timeout)
            if hb.stopped():
                return
            api.upload(run_id, "rpk", rpk)
            api.complete(run_id)
            log.info("run %s packaged", run_id)
    except LeaseLost as e:
        log.warning("run %s is no longer ours (%s); abandoning it", run_id, e.detail)
    except FileNotFoundError as e:
        log.error("can't run %s: %s", packager, e)
        api.fail(run_id, f"'{packager}' not found on the Pi. Install it with: sudo apt install imx500-tools")
    except Exception as e:  # noqa: BLE001 - report any failure to the backend instead of dying silently
        log.error("run %s failed:\n%s", run_id, traceback.format_exc())
        api.fail(run_id, f"{type(e).__name__}: {e}")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=os.environ.get("MT_SERVER"), help="backend URL (or MT_SERVER)")
    ap.add_argument("--name", default=f"pi-{socket.gethostname()}", help="worker name shown in the UI")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--packager", default="imx500-package", help="the packaging command")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--heartbeat", type=float, default=20.0)
    ap.add_argument("--once", action="store_true", help="do at most one run, then exit (also exits if none is waiting)")
    args = ap.parse_args(argv)
    token = os.environ.get("MT_WORKER_TOKEN")
    if not args.server or not token:
        ap.error("set MT_SERVER (or --server) and MT_WORKER_TOKEN")

    setup_logging()
    if shutil.which(args.packager) is None:
        log.error("'%s' not found. Install it with: sudo apt install imx500-tools", args.packager)
        return 2
    api = Api(args.server, token, args.name)
    try:
        api.ping()
    except ApiError as e:
        log.error("can't use the backend: %s", e)
        return 2
    log.info("worker '%s' connected to %s", args.name, args.server)

    while True:
        try:
            task = api.claim("pi")
        except ApiError as e:
            log.warning("claim failed: %s", e)
            task = None
        if task:
            run_task(api, task, Path(args.workdir), args.packager, args.heartbeat)
            if args.once:
                return 0
        elif args.once:
            return 0
        else:
            time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
