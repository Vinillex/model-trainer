import shlex
import socket
import subprocess
import sys

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import imx_export, training
from ..config import settings
from ..db import get_db
from ..models import Job, TrainingRun
from ..schemas import JobOut, TrainingRunCreate, TrainingRunOut, WorkerOut
from ..storage import get_storage
from .projects import get_project

router = APIRouter(tags=["training"])


def get_run(db: Session, run_id: int) -> TrainingRun:
    run = db.get(TrainingRun, run_id)
    if run is None:
        raise HTTPException(404, "training run not found")
    return run


@router.post("/projects/{project_id}/training-runs", response_model=TrainingRunOut, status_code=201)
def create_training_run(project_id: int, body: TrainingRunCreate, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    try:
        run = training.create_run(db, project, body)
    except training.NotFound as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    db.commit()
    return run


@router.get("/projects/{project_id}/training-runs", response_model=list[TrainingRunOut])
def list_training_runs(project_id: int, db: Session = Depends(get_db)):
    get_project(db, project_id)
    return db.scalars(select(TrainingRun).where(TrainingRun.project_id == project_id).order_by(TrainingRun.id.desc())).all()


@router.get("/training-runs/{run_id}", response_model=TrainingRunOut)
def read_training_run(run_id: int, db: Session = Depends(get_db)):
    return get_run(db, run_id)


@router.post("/training-runs/{run_id}/cancel", response_model=TrainingRunOut)
def cancel_training_run(run_id: int, db: Session = Depends(get_db)):
    try:
        run = training.cancel_run(db, get_run(db, run_id))
    except ValueError as e:
        raise HTTPException(409, str(e))
    db.commit()
    return run


@router.delete("/training-runs/{run_id}", status_code=204)
def delete_training_run(run_id: int, db: Session = Depends(get_db)):
    try:
        training.delete_run(db, get_run(db, run_id))
    except ValueError as e:
        raise HTTPException(409, str(e))
    db.commit()


@router.get("/training-runs/{run_id}/artifacts/{name}")
def download_artifact(run_id: int, name: str, db: Session = Depends(get_db)):
    run = get_run(db, run_id)
    art = run.artifacts.get(name)
    if art is None:
        raise HTTPException(404, f"no '{name}' artifact on this run")
    filename = f"{run.name}-{training.ARTIFACT_FILES[name]}"
    return Response(get_storage().get(art["key"]), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.post("/training-runs/{run_id}/export-imx", response_model=JobOut, status_code=202)
def export_on_this_mac(run_id: int, db: Session = Depends(get_db)):
    """Export the run for the IMX500 on this Mac (a job). Also the retry after a failed export, and a fresh export of an older run."""
    try:
        job = imx_export.queue_export(db, get_run(db, run_id))
    except ValueError as e:
        raise HTTPException(409, str(e))
    db.commit()
    return job


@router.get("/imx-tools")
def imx_tools(db: Session = Depends(get_db)):
    """Whether this Mac can export for the IMX500, and how the latest setup went (running, or why it failed)."""
    job = db.scalars(select(Job).where(Job.type == "imx_tools_setup").order_by(Job.id.desc()).limit(1)).first()
    setup = {"job_id": job.id, "status": job.status.value, "message": job.progress.get("message"), "error": job.error} if job else None
    return {**imx_export.tools_status(), "setup": setup}


@router.post("/projects/{project_id}/imx-tools/setup", response_model=JobOut, status_code=202)
def set_up_imx_tools(project_id: int, db: Session = Depends(get_db)):
    """Build the export tools' own Python (a download of about 1 GB; the page asks first)."""
    job = imx_export.queue_setup(db, get_project(db, project_id).id)
    db.commit()
    return job


@router.get("/training-workers", response_model=list[WorkerOut])
def training_workers():
    """Workers that have called in recently (in memory, so it resets when the backend restarts)."""
    return training.list_workers()


def lan_hostname() -> str:
    """The name other machines on the network use for this one (<name>.local). Unlike an IP address it survives a new DHCP lease."""
    if sys.platform == "darwin":
        try:
            name = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=3).stdout.strip()
            if name:
                return f"{name}.local"
        except (OSError, subprocess.SubprocessError):
            pass
    host = socket.gethostname()
    return host if "." in host else f"{host}.local"


@router.get("/pi-setup")
def pi_setup(request: Request):
    """The one-time command that turns a Raspberry Pi into an always-on worker. It contains the worker token, so it is only
    served to this machine (the normal API is loopback-only) and the page copies it to the clipboard without showing it."""
    if not settings.worker_token:
        raise HTTPException(503, "The worker API is disabled: set MT_WORKER_TOKEN in backend/.env and restart the backend")
    server = f"http://{lan_hostname()}:{request.url.port or 8000}"
    command = (f"T={shlex.quote(settings.worker_token)} && mkdir -p ~/agent && cd ~/agent"
               f" && curl -fsS -H \"X-Worker-Token: $T\" {shlex.quote(server + '/workers/pi-agent')} | tar xz"
               f" && MT_SERVER={shlex.quote(server)} MT_WORKER_TOKEN=\"$T\" bash pi_install.sh")
    return {"server": server, "command": command}
