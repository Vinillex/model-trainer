from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import annotations, images, projects
from .db import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path("data").mkdir(exist_ok=True)
    Base.metadata.create_all(engine)  # TODO: switch to Alembic migrations once the schema settles
    yield


app = FastAPI(title="Model Trainer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["*"], allow_headers=["*"])
app.include_router(projects.router)
app.include_router(images.router)
app.include_router(annotations.router)


@app.get("/health")
def health():
    return {"ok": True}
