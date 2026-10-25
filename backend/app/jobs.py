"""A small DB-backed job queue. The `jobs` table is the source of truth; a worker thread claims queued jobs.

Handlers are looked up by Job.type, so new kinds of work (e.g. FLUX generation) only register a function here.
"""

import threading
from collections.abc import Callable
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from .ai.job import handle_ai_check
from .db import SessionLocal
from .models import Job, JobStatus
from .collect import handle_collect
from .scrape import handle_scrape
from .video import handle_video

HANDLERS: dict[str, Callable[[Session, Job], None]] = {
    "scrape": handle_scrape,
    "collect_pages": handle_collect,
    "video_frames": handle_video,
    "ai_check": handle_ai_check,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def recover_interrupted(session_factory: sessionmaker = SessionLocal) -> int:
    """Jobs still `running` at startup died with the previous process; fail them rather than leave them spinning."""
    with session_factory() as db:
        res = db.execute(update(Job).where(Job.status == JobStatus.running).values(
            status=JobStatus.failed, error="Interrupted by a server restart", finished_at=_now()))
        db.commit()
        return res.rowcount


def claim_next(session_factory: sessionmaker = SessionLocal) -> int | None:
    with session_factory() as db:
        job_id = db.scalar(select(Job.id).where(Job.status == JobStatus.queued).order_by(Job.id).limit(1))
        if job_id is None:
            return None
        # The conditional UPDATE makes the claim atomic if a second worker is ever added.
        res = db.execute(update(Job).where(Job.id == job_id, Job.status == JobStatus.queued)
                         .values(status=JobStatus.running, started_at=_now()))
        db.commit()
        return job_id if res.rowcount == 1 else None


def run_job(job_id: int, session_factory: sessionmaker = SessionLocal) -> None:
    with session_factory() as db:
        job = db.get(Job, job_id)
        try:
            HANDLERS[job.type](db, job)
            db.refresh(job)
            job.status = JobStatus.cancelled if job.cancel_requested else JobStatus.succeeded
        except Exception as e:  # noqa: BLE001 - a job failing must never kill the worker thread
            db.rollback()
            job = db.get(Job, job_id)
            job.status = JobStatus.failed
            job.error = f"{type(e).__name__}: {e}"[:2000]
        job.finished_at = _now()
        db.commit()


class Worker(threading.Thread):
    def __init__(self, session_factory: sessionmaker = SessionLocal, poll_seconds: float = 1.0):
        super().__init__(name="job-worker", daemon=True)
        self.session_factory = session_factory
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.is_set():
            job_id = claim_next(self.session_factory)
            if job_id is None:
                self.stop_event.wait(self.poll_seconds)
            else:
                run_job(job_id, self.session_factory)

    def stop(self) -> None:
        self.stop_event.set()
