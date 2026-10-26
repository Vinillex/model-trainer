import os
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from .. import dataset_bucket, datasets
from ..db import get_db
from ..models import DatasetVersion, RunStage, TrainingRun
from ..schemas import SIMILAR_PCT_MAX, SIMILAR_PCT_MIN, CloudCopiesOut, CloudCopyOut, CloudDeleteIn, CloudDeleteOut, DatasetCreate, DatasetOut
from ..storage import get_storage
from ..yolo_export import MissingImages, make_zip_file
from .projects import get_project

router = APIRouter(tags=["datasets"])


def get_version(db: Session, dataset_id: int) -> DatasetVersion:
    version = db.get(DatasetVersion, dataset_id)
    if version is None:
        raise HTTPException(404, "dataset version not found")
    return version


def _csv(value: str | None, allowed: tuple[str, ...] | None = None) -> frozenset[str] | None:
    """None (parameter absent) = all; "" = none; "a,b" = just those."""
    if value is None:
        return None
    names = frozenset(v for v in value.split(",") if v)
    if allowed is not None and not names <= set(allowed):
        raise HTTPException(422, f"unknown source(s): {', '.join(sorted(names - set(allowed)))}")
    return names


@router.get("/projects/{project_id}/datasets/preview")
def dataset_preview(project_id: int, sources: str | None = None, imported: str | None = None,
                    skip_similar_pct: float | None = Query(None, ge=SIMILAR_PCT_MIN, le=SIMILAR_PCT_MAX), db: Session = Depends(get_db)):
    """What a dataset created right now would contain, and what would be left out.

    `sources` (comma-separated: uploaded, scraped, generated) and `imported` (comma-separated imported dataset names)
    choose the data; leave one out for "all of it", pass it empty for "none of it". `skip_similar_pct` also counts the
    repeats that "leave out similar pictures" would drop."""
    selection = datasets.Selection(_csv(sources, datasets.SELECTABLE_SOURCES), _csv(imported))
    return datasets.preview(db, get_project(db, project_id), selection, skip_similar_pct)


@router.post("/projects/{project_id}/datasets", response_model=DatasetOut, status_code=201)
def create_dataset(project_id: int, body: DatasetCreate, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    try:
        selection = datasets.Selection(frozenset(body.sources) if body.sources is not None else None,
                                       frozenset(body.imported_datasets) if body.imported_datasets is not None else None)
        version = datasets.create_version(db, project, body.name, body.val_ratio, body.test_ratio, body.seed, body.include_negatives,
                                          selection, [(m.class_ids, m.name) for m in body.merge_classes],
                                          body.similar_box_pct if body.skip_similar else None, body.group_scenes)
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


def _cloud_state(db: Session, project_id: int) -> list[CloudCopyOut]:
    """The project's zips in the bucket, each with the dataset version it is the copy of (matched the way they are named) and whether a run needs it."""
    versions = db.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id)).all()
    busy = set(db.scalars(select(TrainingRun.dataset_id).where(TrainingRun.project_id == project_id,
                                                               TrainingRun.stage.in_([RunStage.queued, RunStage.training]))))
    by_key = {}
    for v in versions:
        try:
            by_key[dataset_bucket.key_for(v)] = v
        except Exception:  # noqa: BLE001 - a version whose manifest is gone has no key to match
            pass
    try:
        copies = dataset_bucket.list_copies(project_id)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"The cloud storage did not answer: {e}")
    out = []
    for c in copies:
        v = by_key.get(c["key"])
        out.append(CloudCopyOut(**c, dataset_id=v.id if v else None, dataset_name=v.name if v else None, in_use=v is not None and v.id in busy))
    return out


@router.get("/projects/{project_id}/datasets/cloud", response_model=CloudCopiesOut)
def cloud_copies(project_id: int, db: Session = Depends(get_db)):
    """The dataset zips this project has in the cloud bucket (Cloudflare R2), for RunPod pods to download. Empty and `enabled: false` without a bucket."""
    get_project(db, project_id)
    if not dataset_bucket.enabled():
        return CloudCopiesOut(enabled=False, copies=[], total_bytes=0)
    copies = _cloud_state(db, project_id)
    return CloudCopiesOut(enabled=True, copies=copies, total_bytes=sum(c.size for c in copies))


@router.post("/projects/{project_id}/datasets/cloud/delete", response_model=CloudDeleteOut)
def delete_cloud_copies(project_id: int, body: CloudDeleteIn, db: Session = Depends(get_db)):
    """Delete these cloud copies. The dataset versions stay: the next RunPod run on one uploads its zip again (slow from this Mac). All or nothing
    before anything is deleted: a key that is not a copy of this project's, or one a run is waiting for, refuses the whole request."""
    get_project(db, project_id)
    if not dataset_bucket.enabled():
        raise HTTPException(409, "No cloud storage is set up.")
    keys = sorted(set(body.keys))
    bad = [k for k in keys if not dataset_bucket.is_copy_key(project_id, k)]
    if bad:
        raise HTTPException(422, f"not a dataset copy of this project: {bad[0]}")
    state = {c.key: c for c in _cloud_state(db, project_id)}
    busy = [state[k] for k in keys if k in state and state[k].in_use]
    if busy:
        raise HTTPException(409, f"A training run on {busy[0].dataset_name} is still waiting or training and may need the zip. Delete its cloud copy once that run has finished or been cancelled.")
    found = [k for k in keys if k in state]  # one that is already gone is just not counted
    try:
        dataset_bucket.delete_copies(found)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"The cloud storage refused or did not answer: {e}")
    return CloudDeleteOut(deleted=len(found), freed_bytes=sum(state[k].size for k in found))


@router.get("/datasets/{dataset_id}", response_model=DatasetOut)
def read_dataset(dataset_id: int, db: Session = Depends(get_db)):
    return get_version(db, dataset_id)


def zip_response(db: Session, version: DatasetVersion) -> FileResponse:
    """The version as a YOLO-format zip (images/, labels/, data.yaml), built from the frozen manifest."""
    storage = get_storage()
    try:
        path = make_zip_file(datasets.resolve_manifest(db, datasets.load_manifest(version)), storage.get)
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
    dataset_bucket.forget(version)  # before the manifest goes: the bucket's key is derived from it
    get_storage().delete(version.manifest_key)
    db.delete(version)
    db.commit()
