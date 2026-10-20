from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import annotations, images, jobs as jobs_api, projects
from . import jobs
from .config import settings
from .db import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("data").mkdir(exist_ok=True)
    Base.metadata.create_all(engine)  # TODO: switch to Alembic migrations once the schema settles
    jobs.recover_interrupted()
    worker = jobs.Worker(poll_seconds=settings.worker_poll_seconds) if settings.run_worker else None
    if worker:
        worker.start()
    yield
    if worker:
        worker.stop()


app = FastAPI(title="Model Trainer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["*"], allow_headers=["*"])
app.include_router(projects.router)
app.include_router(images.router)
app.include_router(annotations.router)
app.include_router(jobs_api.router)


@app.get("/health")
def health():
    return {"ok": True}
