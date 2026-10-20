from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import training
from ..db import get_db
from ..models import TrainingRun
from ..schemas import TrainingRunCreate, TrainingRunOut, WorkerOut
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


@router.get("/training-workers", response_model=list[WorkerOut])
def training_workers():
    """Workers that have called in recently (in memory, so it resets when the backend restarts)."""
    return training.list_workers()
