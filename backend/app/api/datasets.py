import os
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from .. import datasets
from ..db import get_db
from ..models import DatasetVersion, TrainingRun
from ..schemas import DatasetCreate, DatasetOut
from ..storage import get_storage
from ..yolo_export import MissingImages, make_zip_file
from .projects import get_project

router = APIRouter(tags=["datasets"])


def get_version(db: Session, dataset_id: int) -> DatasetVersion:
    version = db.get(DatasetVersion, dataset_id)
    if version is None:
        raise HTTPException(404, "dataset version not found")
    return version


@router.get("/projects/{project_id}/datasets/preview")
def dataset_preview(project_id: int, db: Session = Depends(get_db)):
    """What a dataset created right now would contain, and what would be left out."""
    return datasets.preview(db, get_project(db, project_id))


@router.post("/projects/{project_id}/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(project_id: int, body: DatasetCreate, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    try:
        version = datasets.create_version(db, project, body.name, body.val_ratio, body.test_ratio, body.seed, body.include_negatives)
        db.commit()
    except ValueError as e:
        db.rollback()
        raise HTTPException(422, str(e))
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "a dataset version with that name already exists")
    return version


@router.get("/projects/{project_id}/datasets", response_model=list[DatasetOut])
def list_datasets(project_id: int, db: Session = Depends(get_db)):
    get_project(db, project_id)
    return db.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id).order_by(DatasetVersion.id.desc())).all()


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def read_dataset(dataset_id: int, db: Session = Depends(get_db)):
    return get_version(db, dataset_id)


def zip_response(db: Session, version: DatasetVersion) -> FileResponse:
    """The version as a YOLO-format zip (images/, labels/, data.yaml), built from the frozen manifest."""
    storage = get_storage()
    try:
        path = make_zip_file(datasets.load_manifest(version), storage.get)
    except MissingImages as e:
        raise HTTPException(409, f"{e} Images deleted after a version was created can't be exported; create a new version.")
    filename = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{get_project(db, version.project_id).name}-{version.name}") + ".zip"
    return FileResponse(path, media_type="application/zip", filename=filename, background=BackgroundTask(os.unlink, path))


@router.get("/datasets/{dataset_id}/download")
def download_dataset(dataset_id: int, db: Session = Depends(get_db)):
    return zip_response(db, get_version(db, dataset_id))


@router.delete("/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    version = get_version(db, dataset_id)
    if db.scalar(select(TrainingRun.id).where(TrainingRun.dataset_id == dataset_id).limit(1)) is not None:
        raise HTTPException(409, "Training runs use this dataset version. Delete those runs first.")
    get_storage().delete(version.manifest_key)
    db.delete(version)
    db.commit()
