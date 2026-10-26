import os
import tempfile

# Tests must never touch the real database, storage or API keys from backend/.env, even if one starts the app's
# lifespan (which creates tables, fails "running" jobs and starts the background worker). Set before `app` is imported.
os.environ.update(MT_DATABASE_URL="sqlite://", MT_LOCAL_STORAGE_DIR=tempfile.mkdtemp(prefix="mt-test-storage-"), MT_RUN_WORKER="false",
                  MT_WORKER_LOG_DIR=tempfile.mkdtemp(prefix="mt-test-worker-logs-"),
                  MT_RUNPOD_LAUNCH_DIR=tempfile.mkdtemp(prefix="mt-test-runpod-"),
                  MT_PI_DISPLAY_FILE=os.path.join(tempfile.mkdtemp(prefix="mt-test-pi-display-"), "pi_display.json"),
                  MT_IMX_EXPORT_DIR=tempfile.mkdtemp(prefix="mt-test-imx-export-"),  # never the real export tools
                  MT_SERPAPI_KEY="", MT_RUNPOD_API_KEY="", MT_WORKER_TOKEN="",
                  # the real dataset bucket: a launcher test with --yes would otherwise upload its zip to it
                  MT_DATASET_BUCKET="", MT_DATASET_BUCKET_ENDPOINT="", MT_DATASET_BUCKET_ACCESS_KEY="", MT_DATASET_BUCKET_SECRET_KEY="")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app import storage
from app.main import app

from .helpers import from_ip


@pytest.fixture()
def client(tmp_path):
    # A file (not ":memory:" + StaticPool) so each thread gets its own connection, like production.
    # Tests that write from download threads would otherwise share one connection and flake.
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    app_db.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override():
        with Session() as s:
            yield s

    app.dependency_overrides[app_db.get_db] = override
    client_ = TestClient(from_ip(app, "127.0.0.1"))  # a local browser; not used as a context manager, so the app's lifespan (and its worker) never starts
    client_.session_factory = Session
    storage._storage = storage.LocalStorage(str(tmp_path))
    yield client_
    app.dependency_overrides.clear()
    engine.dispose()
    storage._storage = None
