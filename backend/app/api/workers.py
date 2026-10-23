"""The API the remote workers use: claim a run, report progress, upload artifacts, finish.

Every route needs the worker token; the worker's name comes in the X-Worker-Name header and identifies the lease holder.
"""

import io
import re
import tarfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from .. import dataset_bucket, training
from ..db import get_db
from ..models import DatasetVersion
from ..schemas import ClaimIn, CompleteIn, FailIn, HeartbeatIn, TrainingRunOut
from ..security import require_token, worker_name
from ..storage import get_storage
from .datasets import zip_response

def served(db: Session = Depends(get_db)) -> None:
    training.mark_served(db)


# Order matters: only a request that passed the token check may count as "the server is serving workers".
router = APIRouter(prefix="/workers", tags=["workers"], dependencies=[Depends(require_token), Depends(served)])


def guarded(fn, *args):
    """Run a state-machine call and turn its errors into HTTP ones."""
    try:
        return fn(*args)
    except training.NotFound as e:
        raise HTTPException(404, str(e))
    except training.LeaseLost as e:
        raise HTTPException(409, f"lease lost: {e}")  # the worker must stop working on this run
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/ping")
def ping():
    return {"ok": True}


AGENTS_DIR = Path(__file__).resolve().parents[3] / "agents"
AGENT_FILES = ("common.py", "gpu_agent.py", "requirements-gpu.txt")


@router.get("/agent")
def agent_bundle():
    """The GPU agent's code as a tar.gz. A fresh machine (a RunPod pod) fetches it here, so it runs exactly the version on
    this Mac, including changes that were never committed."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in AGENT_FILES:
            tar.add(AGENTS_DIR / name, arcname=name)
    return Response(buf.getvalue(), media_type="application/gzip")


WORKER_LOG_MAX_BYTES = 256 * 1024


def worker_log_path(worker: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", worker).strip("._")[:100] or "worker"  # one plain file name, never a path
    return Path(training.settings.worker_log_dir) / f"{safe}.log"


@router.post("/log", status_code=204)
async def worker_log(request: Request, worker: str = Depends(worker_name)):
    """One progress line from a worker that has no agent running yet (a RunPod pod installing its tools)."""
    line = (await request.body())[:2000].decode("utf-8", "replace").replace("\r", " ").replace("\n", " ").strip()
    if line:
        path = worker_log_path(worker)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.stat().st_size < WORKER_LOG_MAX_BYTES:  # a runaway pod can't fill the disk
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    return Response(status_code=204)


@router.post("/claim", response_model=None)
def claim(body: ClaimIn, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    training.touch_worker(worker, body.kind)
    run = training.claim(db, body.kind, worker)
    if run is None:
        return Response(status_code=204)
    training.touch_worker(worker, body.kind, run.id)
    # dataset_link: a signed link to the bucket's copy, tried first; dataset_url (through this backend) stays the fallback.
    link = dataset_bucket.link_if_stored(db.get(DatasetVersion, run.dataset_id)) if body.kind == "gpu" else None
    return {"run_id": run.id, "kind": body.kind, "stage": run.stage.value, "name": run.name, "config": run.config,
            "dataset_url": f"/workers/runs/{run.id}/dataset", "dataset_link": link, "artifacts": sorted(run.artifacts),
            "lease_seconds": training.settings.worker_lease_seconds}


@router.post("/runs/{run_id}/heartbeat")
def heartbeat(run_id: int, body: HeartbeatIn, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    kind = training.kind_of(guarded(training.owned, db, run_id, worker).stage)  # read before the heartbeat can end the run
    cancel = guarded(training.heartbeat, db, run_id, worker, body.progress, body.stage)
    training.touch_worker(worker, kind, None if cancel else run_id)
    return {"cancel": cancel}


@router.get("/runs/{run_id}/dataset")
def run_dataset(run_id: int, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    run = guarded(training.owned, db, run_id, worker, training.GPU_STAGES)
    return zip_response(db, db.get(DatasetVersion, run.dataset_id))


@router.get("/runs/{run_id}/artifacts/{name}")
def run_artifact(run_id: int, name: str, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    run = guarded(training.owned, db, run_id, worker)
    art = run.artifacts.get(name)
    if art is None:
        raise HTTPException(404, f"no '{name}' artifact on this run")
    return Response(get_storage().get(art["key"]), media_type="application/octet-stream")


@router.put("/runs/{run_id}/artifacts/{name}", response_model=TrainingRunOut)
async def upload_artifact(run_id: int, name: str, request: Request, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    if name not in training.ARTIFACT_FILES:
        raise HTTPException(422, f"unknown artifact '{name}'")
    guarded(training.owned, db, run_id, worker)  # reject early, before reading a large body
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > training.MAX_ARTIFACT_BYTES:
            raise HTTPException(413, f"artifact is larger than {training.MAX_ARTIFACT_BYTES // 2**20} MB")
        chunks.append(chunk)
    return guarded(training.store_artifact, db, run_id, worker, name, b"".join(chunks))


@router.post("/runs/{run_id}/complete", response_model=TrainingRunOut)
def complete(run_id: int, body: CompleteIn, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    return guarded(training.complete, db, run_id, worker, body.metrics)


@router.post("/runs/{run_id}/fail", response_model=TrainingRunOut)
def fail(run_id: int, body: FailIn, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    return guarded(training.fail, db, run_id, worker, body.error)
