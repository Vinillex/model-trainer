import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import db as app_db
from app import storage
from app.main import app


@pytest.fixture()
def client(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    app_db.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def override():
        with Session() as s:
            yield s

    app.dependency_overrides[app_db.get_db] = override
    client_ = TestClient(app)  # not used as a context manager, so the app's lifespan (and its worker) never starts
    client_.session_factory = Session
    storage._storage = storage.LocalStorage(str(tmp_path))
    yield client_
    app.dependency_overrides.clear()
    storage._storage = None
