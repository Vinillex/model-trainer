import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import db as app_db
from app import storage
from app.main import app


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
    client_ = TestClient(app)  # not used as a context manager, so the app's lifespan (and its worker) never starts
    client_.session_factory = Session
    storage._storage = storage.LocalStorage(str(tmp_path))
    yield client_
    app.dependency_overrides.clear()
    engine.dispose()
    storage._storage = None
