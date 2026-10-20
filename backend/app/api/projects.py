import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import DetClass, Image, Project, ReviewStatus
from ..schemas import (NAME_PATTERN, BulkClassCreate, BulkClassResult, ClassCreate, ClassOut, ProjectCreate,
                       ProjectOut, StatRow)

router = APIRouter(prefix="/projects", tags=["projects"])

DEFAULT_CLASSES = ["multirotor", "fixed-wing", "shahed"]
DEFAULT_NEGATIVES = ["birds", "aircraft", "helicopters", "balloons", "empty-sky"]


def add_class(db: Session, project: Project, name: str, is_negative: bool) -> DetClass:
    index = None
    if not is_negative:
        index = db.scalar(
            select(func.count()).select_from(DetClass).where(
                DetClass.project_id == project.id, DetClass.is_negative.is_(False)
            )
        )
    cls = DetClass(project_id=project.id, name=name, is_negative=is_negative, index=index)
    db.add(cls)
    db.flush()
    return cls


def normalize_class_name(raw: str) -> str:
    return re.sub(r"\s+", "-", raw.strip().lower())


def get_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")
    return project


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    project = Project(name=body.name)
    db.add(project)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "project name already exists")
    if body.seed_defaults:
        for name in DEFAULT_CLASSES:
            add_class(db, project, name, False)
        for name in DEFAULT_NEGATIVES:
            add_class(db, project, name, True)
    db.commit()
    db.refresh(project)
    return project


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.scalars(select(Project).order_by(Project.id)).all()


@router.get("/{project_id}", response_model=ProjectOut)
def read_project(project_id: int, db: Session = Depends(get_db)):
    return get_project(db, project_id)


@router.post("/{project_id}/classes", response_model=ClassOut, status_code=201)
def create_class(project_id: int, body: ClassCreate, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    try:
        cls = add_class(db, project, body.name, body.is_negative)
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "class name already exists in project")
    return cls


@router.post("/{project_id}/classes/bulk", response_model=BulkClassResult)
def create_classes_bulk(project_id: int, body: BulkClassCreate, db: Session = Depends(get_db)):
    """Add many classes at once. Names that already exist (in either category) are skipped, not changed."""
    project = get_project(db, project_id)
    existing = {c.name for c in project.classes}
    result = BulkClassResult(added=[], skipped=[], invalid=[])
    seen: set[str] = set()
    for raw in body.names:
        name = normalize_class_name(raw)
        if not name:
            continue  # blank entries from trailing commas / empty lines
        if name in seen:
            continue
        seen.add(name)
        if len(name) > 64 or not re.match(NAME_PATTERN, name):
            result.invalid.append(raw.strip())
        elif name in existing:
            result.skipped.append(name)
        else:
            result.added.append(ClassOut.model_validate(add_class(db, project, name, body.is_negative)))
    db.commit()
    return result


@router.get("/{project_id}/stats", response_model=list[StatRow])
def project_stats(project_id: int, db: Session = Depends(get_db)):
    project = get_project(db, project_id)
    rows = {
        c.id: StatRow(class_id=c.id, class_name=c.name, is_negative=c.is_negative) for c in project.classes
    }
    counts = db.execute(
        select(Image.class_id, Image.source, func.count())
        .where(Image.project_id == project_id)
        .group_by(Image.class_id, Image.source)
    )
    for class_id, source, n in counts:
        setattr(rows[class_id], source.value, n)
    by_review = db.execute(
        select(Image.class_id, Image.review_status, func.count())
        .where(Image.project_id == project_id, Image.review_status != ReviewStatus.reviewed)
        .group_by(Image.class_id, Image.review_status)
    )
    for class_id, status, n in by_review:
        setattr(rows[class_id], status.value, n)
    return list(rows.values())
