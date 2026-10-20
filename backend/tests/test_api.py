import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
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
    storage._storage = storage.LocalStorage(str(tmp_path))
    yield TestClient(app)
    app.dependency_overrides.clear()
    storage._storage = None


def png(color=(10, 20, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_project_seeds_classes(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    names = {c["name"]: c for c in p["classes"]}
    assert names["multirotor"]["index"] == 0 and names["shahed"]["index"] == 2
    assert names["birds"]["is_negative"] and names["birds"]["index"] is None
    assert client.post("/projects", json={"name": "drones"}).status_code == 409


def test_upload_dedup_reject_and_stats(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    cid = next(c["id"] for c in p["classes"] if c["name"] == "multirotor")
    files = [
        ("files", ("a.png", png(), "image/png")),
        ("files", ("dup.png", png(), "image/png")),
        ("files", ("b.png", png((1, 2, 3)), "image/png")),
        ("files", ("bad.txt", b"hello", "text/plain")),
        ("files", ("tiny.png", png(size=(8, 8)), "image/png")),
    ]
    r = client.post(f"/projects/{p['id']}/classes/{cid}/images", files=files).json()
    assert len(r["added"]) == 2
    assert r["duplicates"] == ["dup.png"]
    assert len(r["rejected"]) == 2

    imgs = client.get(f"/projects/{p['id']}/images", params={"class_id": cid, "source": "uploaded"}).json()
    assert len(imgs) == 2
    assert client.get(f"/images/{imgs[0]['id']}/file").headers["content-type"] == "image/png"

    stats = {s["class_name"]: s for s in client.get(f"/projects/{p['id']}/stats").json()}
    assert stats["multirotor"]["uploaded"] == 2 and stats["shahed"]["uploaded"] == 0

    assert client.delete(f"/images/{imgs[0]['id']}").status_code == 204
    assert client.get(f"/images/{imgs[0]['id']}/file").status_code == 404
