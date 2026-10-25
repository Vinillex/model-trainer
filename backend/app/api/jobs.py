from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import DetClass, Job, JobStatus, SearchLog
from ..runpod_client import RunPodError, fetch_account as fetch_runpod_account
from ..schemas import CollectJobCreate, JobOut, RunPodAccount, ScrapeJobCreate, SearchHistory, SerpApiAccount, VideoJobCreate
from ..scrape import SearchError, fetch_account, normalize_query
from ..video import ffmpeg_tools
from .projects import get_project

router = APIRouter(tags=["jobs"])


@router.post("/projects/{project_id}/jobs/scrape", response_model=JobOut, status_code=201)
def create_scrape_job(project_id: int, body: ScrapeJobCreate, db: Session = Depends(get_db)):
    get_project(db, project_id)
    cls = db.get(DetClass, body.class_id)
    if cls is None or cls.project_id != project_id:
        raise HTTPException(404, "class not found")
    if cls.is_mixed:
        raise HTTPException(422, "'mixed' is filled automatically when a reviewed image shows several classes; pick another class")
    if not settings.serpapi_key:
        # Fail now rather than queue a job that is certain to fail.
        raise HTTPException(503, "SerpAPI is not configured: set MT_SERPAPI_KEY in backend/.env and restart the backend")
    job = Job(project_id=project_id, type="scrape", params={**body.model_dump(), "query": body.query.strip()})
    db.add(job)
    db.commit()
    return job


@router.post("/projects/{project_id}/jobs/collect-pages", response_model=JobOut, status_code=201)
def create_collect_job(project_id: int, body: CollectJobCreate, db: Session = Depends(get_db)):
    """Read the pictures off web pages. Needs no SerpAPI key; the pages and pictures are fetched politely (app/collect.py)."""
    get_project(db, project_id)
    cls = db.get(DetClass, body.class_id)
    if cls is None or cls.project_id != project_id:
        raise HTTPException(404, "class not found")
    if cls.is_mixed:
        raise HTTPException(422, "'mixed' is filled automatically when a reviewed image shows several classes; pick another class")
    first = urlsplit(body.urls[0]).hostname
    # `query` is what the provenance record of each picture keeps ("where did this come from"), as for a search.
    label = f"pages: {first}" + (f" (+{len(body.urls) - 1} more)" if len(body.urls) > 1 else "")
    job = Job(project_id=project_id, type="collect_pages", params={**body.model_dump(), "query": label[:300], "start_page": 0})
    db.add(job)
    db.commit()
    return job


@router.post("/projects/{project_id}/jobs/video-frames", response_model=JobOut, status_code=201)
def create_video_job(project_id: int, body: VideoJobCreate, db: Session = Depends(get_db)):
    """Cut still frames out of videos (Wikimedia Commons File: pages or direct video files). Needs ffmpeg on the server."""
    get_project(db, project_id)
    cls = db.get(DetClass, body.class_id)
    if cls is None or cls.project_id != project_id:
        raise HTTPException(404, "class not found")
    if cls.is_mixed:
        raise HTTPException(422, "'mixed' is filled automatically when a reviewed image shows several classes; pick another class")
    if ffmpeg_tools() is None:
        raise HTTPException(503, "ffmpeg is not installed on this machine: run `brew install ffmpeg` and restart the backend")
    first = urlsplit(body.urls[0]).hostname
    label = f"video: {first}" + (f" (+{len(body.urls) - 1} more)" if len(body.urls) > 1 else "")
    job = Job(project_id=project_id, type="video_frames",
              params={**body.model_dump(), "query": label[:300], "start_page": 0, "max_pages": len(body.urls), "group_by": "article"})
    db.add(job)
    db.commit()
    return job


@router.get("/projects/{project_id}/searches", response_model=SearchHistory)
def search_history(project_id: int, query: str = Query(min_length=1, max_length=300), db: Session = Depends(get_db)):
    """Which result pages of this query were already searched, so the next job can start where the last one stopped."""
    get_project(db, project_id)
    q = normalize_query(query)
    rows = db.execute(select(SearchLog.page, SearchLog.results).where(SearchLog.project_id == project_id, SearchLog.query == q)).all()
    pages = sorted({page for page, _ in rows})
    return SearchHistory(query=q, pages_searched=pages, next_page=pages[-1] + 1 if pages else 0,
                         exhausted_at=min((page for page, n in rows if n == 0), default=None))


@router.get("/serpapi/account", response_model=SerpApiAccount)
def serpapi_account():
    """Remaining SerpAPI quota. Asking is free: SerpAPI doesn't count Account API calls as searches."""
    if not settings.serpapi_key:
        raise HTTPException(503, "SerpAPI is not configured: set MT_SERPAPI_KEY in backend/.env and restart the backend")
    try:
        return fetch_account(settings.serpapi_key)
    except SearchError as e:
        raise HTTPException(502, str(e))


@router.get("/runpod/account", response_model=RunPodAccount)
def runpod_account():
    """RunPod credit balance and spend. Read-only and free."""
    if not settings.runpod_api_key:
        raise HTTPException(503, "RunPod is not configured: set MT_RUNPOD_API_KEY in backend/.env and restart the backend")
    try:
        return fetch_runpod_account(settings.runpod_api_key)
    except RunPodError as e:
        raise HTTPException(502, str(e))


@router.get("/projects/{project_id}/jobs", response_model=list[JobOut])
def list_jobs(project_id: int, limit: int = Query(20, le=100), type: str | None = Query(None, pattern=r"^[a-z_]{1,32}$"), db: Session = Depends(get_db)):
    """`type` keeps one kind of job (scrape, ai_check), so each panel shows only its own."""
    get_project(db, project_id)
    q = select(Job).where(Job.project_id == project_id)
    if type is not None:
        q = q.where(Job.type == type)
    return db.scalars(q.order_by(Job.id.desc()).limit(limit)).all()


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
