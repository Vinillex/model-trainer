from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import DetClass, Job, JobStatus
from ..schemas import JobOut, ScrapeJobCreate
from .projects import get_project

router = APIRouter(tags=["jobs"])


@router.post("/projects/{project_id}/jobs/scrape", response_model=JobOut, status_code=201)
def create_scrape_job(project_id: int, body: ScrapeJobCreate, db: Session = Depends(get_db)):
    get_project(db, project_id)
    cls = db.get(DetClass, body.class_id)
    if cls is None or cls.project_id != project_id:
        raise HTTPException(404, "class not found")
    if not settings.serpapi_key:
        # Fail now rather than queue a job that is certain to fail.
        raise HTTPException(503, "SerpAPI is not configured: set MT_SERPAPI_KEY in backend/.env and restart the backend")
    job = Job(project_id=project_id, type="scrape", params={**body.model_dump(), "query": body.query.strip()})
    db.add(job)
    db.commit()
    return job


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(project_id: int, limit: int = Query(20, le=100), db: Session = Depends(get_db)):
    get_project(db, project_id)
    return db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.id.desc()).limit(limit)).all()


@router.get("/jobs/{job_id}", response_model=JobOut)
def read_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: int, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status not in (JobStatus.queued, JobStatus.running):
        raise HTTPException(409, f"job already {job.status.value}")
    # A queued job can be cancelled outright; a running one is asked to stop at its next checkpoint.
    res = db.execute(update(Job).where(Job.id == job_id, Job.status == JobStatus.queued)
                     .values(status=JobStatus.cancelled, cancel_requested=True))
    if res.rowcount == 0:
        job.cancel_requested = True
    db.commit()
    db.refresh(job)
    return job
