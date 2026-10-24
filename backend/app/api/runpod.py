"""Train on a rented RunPod GPU from the UI: the price check, the confirmed launch, its live log, and stopping it.

The launch itself is `python -m scripts.runpod launch --yes`, run as its own detached process. It is a long loop (tunnel, pod, watching,
then deleting the pod) with all its safety checks tested in tests/test_runpod_launch.py, so the page starts and watches it instead of
reimplementing it. Being detached, it keeps going through a backend restart; `launch.json` in `runpod_launch_dir` is how this module
finds it again.

Everything here sits behind LoopbackOnly (no other machine, nothing through a tunnel). The calls that spend money or end pods also
need the `X-MT-Spend: 1` header, which a web page open in the same browser cannot send without a CORS preflight the allow-list refuses.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import dataset_bucket
from .. import runpod_client as rp
from ..config import settings
from ..db import get_db
from ..models import RunStage, TrainingRun
from ..runpod_pod import POD_PREFIX

router = APIRouter(prefix="/runpod", tags=["runpod"])

BACKEND_DIR = Path(__file__).resolve().parents[2]
LOG_LINES = 200
_launch_lock = threading.Lock()  # two clicks must not start two launches


def spend_header(x_mt_spend: str | None = Header(None)) -> None:
    if x_mt_spend != "1":
        raise HTTPException(403, "this changes what is running or costs money, so it must come from the app (missing X-MT-Spend header)")


def launcher():
    from scripts import runpod  # imported late: the scripts package sits next to app/, and this keeps `import app` light
    return runpod


def make_api():
    return launcher().RunPodApi(settings.runpod_api_key or "")


def make_http() -> httpx.Client:
    return httpx.Client(timeout=30)


def need_setup() -> None:
    if not settings.runpod_api_key:
        raise HTTPException(409, "MT_RUNPOD_API_KEY isn't set in backend/.env.")
    if not settings.worker_token:
        raise HTTPException(409, "MT_WORKER_TOKEN isn't set in backend/.env (the pod signs in to the backend with it).")


# ---------- reading: free, nothing is started ----------

@router.get("/status")
def status():
    out = {"key_set": bool(settings.runpod_api_key), "token_set": bool(settings.worker_token),
           "cloudflared": shutil.which("cloudflared") is not None, "bucket": dataset_bucket.enabled(),
           "balance": None, "spend_per_hr": None, "pods": [], "other_pods": 0, "error": None}
    if not settings.runpod_api_key:
        return out
    try:
        api = make_api()
        acct = api.account()
        out["balance"], out["spend_per_hr"] = acct.get("balance") or 0, acct.get("spend_per_hr") or 0
        pods = api.list_pods()
        ours = [p for p in pods if str(p.get("name", "")).startswith(POD_PREFIX)]
        out["pods"] = [{"id": p["id"], "name": p.get("name"), "status": p.get("desiredStatus"), "cost_per_hr": p.get("costPerHr") or 0} for p in ours]
        out["other_pods"] = len(pods) - len(ours)
    except rp.RunPodError as e:
        out["error"] = str(e)
    return out


@router.get("/plan")
def plan(run_id: int, max_hours: float = Query(3.0, gt=0, le=12), cloud: Literal["secure", "community"] = "secure", db: Session = Depends(get_db)):
    """What launching this run would do and cost, with every check the launcher does before spending anything. Starts nothing."""
    need_setup()
    cli = launcher()
    args = SimpleNamespace(run=run_id, max_hours=max_hours, cloud=cloud, gpu=None, image=cli.IMAGE, max_cost=None)
    http = make_http()
    try:
        p = cli.make_plan(args, cli.Deps(make_api(), http, lambda: Session(db.get_bind())))
    except cli.Refusal as e:
        raise HTTPException(422, str(e))
    except rp.RunPodError as e:
        raise HTTPException(502, f"RunPod said no: {e}")
    finally:
        http.close()
    return {"run_id": p.run_id, "run_name": p.run_name, "images": p.images, "cloud": p.cloud.lower(), "max_hours": p.max_hours,
            "gpus": [{"id": g, "price": p.gpu_prices.get(g)} for g in p.gpus], "worst_price": p.worst_price, "worst_case": p.worst_case,
            "balance": p.balance, "bucket": p.bucket, "export_imx": p.export_imx}


# ---------- the launch: a detached process ----------

def state_path() -> Path:
    return Path(settings.runpod_launch_dir) / "launch.json"


def read_state() -> dict | None:
    try:
        return json.loads(state_path().read_text())
    except (OSError, ValueError):
        return None


def write_state(state: dict) -> None:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def alive(pid: int) -> bool:
    """Is that process still the launcher? The command line is checked too, so a recycled pid is never mistaken for it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    out = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    return "scripts.runpod" in out


def tail(path: str, lines: int = LOG_LINES) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 64 * 1024))
            return f.read().decode("utf-8", "replace").splitlines()[-lines:]
    except OSError:
        return []


def spawn(cmd: list[str], log) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=BACKEND_DIR, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def watch_exit(proc, pid: int) -> None:
    code = proc.wait()
    state = read_state()
    if state and state.get("pid") == pid:
        write_state({**state, "exit_code": code})


@router.get("/launch")
def launch_state():
    st = read_state()
    if not st:
        return {"state": "none", "lines": []}
    running = "exit_code" not in st and alive(st["pid"])
    return {"state": "running" if running else "finished", "run_id": st["run_id"], "started_at": st["started_at"], "max_cost": st["max_cost"],
            "exit_code": st.get("exit_code"), "stopping": bool(st.get("stop_requested")) and running,
            "lost_track": not running and "exit_code" not in st,  # the backend restarted while it ran: how it ended is in the log
            "lines": tail(st["log"])}


class LaunchIn(BaseModel):
    run_id: int
    max_hours: float = Field(3.0, gt=0, le=12)
    cloud: Literal["secure", "community"] = "secure"
    max_cost: float = Field(gt=0, le=100)  # the worst case the person saw and confirmed; the launcher refuses anything dearer


@router.post("/launch", dependencies=[Depends(spend_header)], status_code=202)
def start_launch(body: LaunchIn, request: Request, db: Session = Depends(get_db)):
    need_setup()
    if shutil.which("cloudflared") is None:
        raise HTTPException(409, "cloudflared isn't installed (the tunnel the pod uses to reach this backend). Install it with: brew install cloudflared")
    run = db.get(TrainingRun, body.run_id)
    if run is None:
        raise HTTPException(404, "training run not found")
    if run.stage != RunStage.queued:
        raise HTTPException(409, f"Run {run.id} is not waiting for a GPU (it is {run.stage.value}).")
    with _launch_lock:
        st = read_state()
        if st and "exit_code" not in st and alive(st["pid"]):
            raise HTTPException(409, "A RunPod launch is already running. Wait for it to end or stop it first.")
        started = datetime.now(timezone.utc)
        log_path = Path(settings.runpod_launch_dir) / f"launch-{started:%Y%m%d-%H%M%S}-run{run.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-u", "-m", "scripts.runpod", "launch", "--yes", "--run", str(run.id), "--max-hours", f"{body.max_hours:g}",
               "--cloud", body.cloud, "--max-cost", f"{body.max_cost:.2f}", "--backend-url", f"http://127.0.0.1:{request.url.port or 8000}"]
        with open(log_path, "ab") as log:  # the child keeps its own copy of the descriptor
            proc = spawn(cmd, log)
        write_state({"pid": proc.pid, "run_id": run.id, "started_at": started.isoformat(), "max_cost": body.max_cost, "log": str(log_path)})
        threading.Thread(target=watch_exit, args=(proc, proc.pid), daemon=True).start()
    return launch_state()


@router.post("/launch/stop", dependencies=[Depends(spend_header)])
def stop_launch():
    """SIGINT, not a kill: the launcher turns it into the Ctrl-C path that deletes the pod and closes the tunnel before it exits."""
    st = read_state()
    if not st or "exit_code" in st or not alive(st["pid"]):
        raise HTTPException(409, "No RunPod launch is running.")
    os.kill(st["pid"], signal.SIGINT)
    write_state({**st, "stop_requested": True})
    return launch_state()


@router.post("/pods/{pod_id}/delete", dependencies=[Depends(spend_header)])
def delete_pod(pod_id: str):
    """The way out when a pod is left running: only pods this project started (names start with mt-) are ever touched."""
    need_setup()
    cli = launcher()
    api = make_api()
    try:
        match = [p for p in api.list_pods() if p["id"] == pod_id and str(p.get("name", "")).startswith(POD_PREFIX)]
        if not match:
            raise HTTPException(404, f"No pod {pod_id} that this project started. Others are never touched here; use the RunPod console.")
        gone = cli.end_pod(cli.Deps(api, None, None, say=lambda *_: None), pod_id, match[0].get("name", ""))
    except rp.RunPodError as e:
        raise HTTPException(502, f"RunPod said no: {e}")
    if not gone:
        raise HTTPException(502, "RunPod did not delete the pod. Delete it in the RunPod console: https://console.runpod.io/pods")
    return {"deleted": True}
