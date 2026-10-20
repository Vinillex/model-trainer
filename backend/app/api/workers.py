"""The API the remote workers use: claim a run, report progress, upload artifacts, finish.

Every route needs the worker token; the worker's name comes in the X-Worker-Name header and identifies the lease holder.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy.orm import Session

from .. import training
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


@router.post("/claim", response_model=None)
def claim(body: ClaimIn, worker: str = Depends(worker_name), db: Session = Depends(get_db)):
    training.touch_worker(worker, body.kind)
    run = training.claim(db, body.kind, worker)
    if run is None:
        return Response(status_code=204)
    training.touch_worker(worker, body.kind, run.id)
    return {"run_id": run.id, "kind": body.kind, "stage": run.stage.value, "name": run.name, "config": run.config,
            "dataset_url": f"/workers/runs/{run.id}/dataset", "artifacts": sorted(run.artifacts),
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
