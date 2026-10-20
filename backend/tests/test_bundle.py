import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text

from app import storage
from app.db import Base
from app.models import Job, JobStatus, ScrapedUrl, ScrapeRecord, SearchLog, UrlOutcome
from app.storage import LocalStorage
from scripts import bundle

from .test_datasets import populate


def raw_rows(engine):
    """Every table's rows exactly as the ORM layer sees them (datetimes, enum members, JSON objects), in primary-key order."""
    with engine.connect() as conn:
        return {t.name: [tuple(r) for r in conn.execute(select(t).order_by(*t.primary_key.columns))] for t in Base.metadata.sorted_tables}


def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def source(client, tmp_path):
    """A database and storage holding a bit of everything: images, boxes, a dataset version, a training run and scrape history."""
    storage._storage = LocalStorage(str(tmp_path / "src-storage"))  # conftest's default shares tmp_path with the database file
    p = client.post("/projects", json={"name": "drones"}).json()
    ids = {c["name"]: c["id"] for c in p["classes"]}
    populate(client, p["id"], ids)
    did = client.post(f"/projects/{p['id']}/datasets", json={}).json()["id"]
    client.post(f"/projects/{p['id']}/training-runs", json={"dataset_id": did, "epochs": 3})
    with client.session_factory() as db:
        job = Job(project_id=p["id"], type="scrape", params={"query": "dji", "max_images": 5}, status=JobStatus.succeeded,
                  progress={"added": 1})
        db.add(job)
        db.flush()
        image_id = client.get(f"/projects/{p['id']}/images", params={"limit": 1}).json()[0]["id"]
        db.add_all([ScrapeRecord(image_id=image_id, job_id=job.id, query="dji", original_url="https://a.test/x.jpg", title="ü ✓"),
                    ScrapedUrl(project_id=p["id"], url_sha="a" * 64, url="https://a.test/x.jpg", outcome=UrlOutcome.added, job_id=job.id),
                    SearchLog(project_id=p["id"], query="dji", page=0, results=100, job_id=job.id)])
        db.commit()
    return client.session_factory.kw["bind"], tmp_path / "src-storage"


def test_round_trip_restores_every_row_and_file_exactly(source, tmp_path):
    engine, root = source
    out = tmp_path / "bundle"
    exported = bundle.export_bundle(engine, root, out)
    assert exported["files"] > 20 and exported["tables"]["images"] > 20 and exported["tables"]["dataset_versions"] == 1
    assert all(exported["tables"][t] >= 1 for t in ("projects", "classes", "annotations", "image_hashes", "jobs", "scrape_records",
                                                     "scraped_urls", "search_log", "training_runs"))

    target = create_engine(f"sqlite:///{tmp_path / 'restored.db'}")
    restored_storage = LocalStorage(str(tmp_path / "dst-storage"))
    imported = bundle.import_bundle(target, restored_storage, out)
    assert imported["tables"] == exported["tables"]

    assert raw_rows(target) == raw_rows(engine)                    # same rows, same ids, datetimes, enums and JSON
    assert tree(tmp_path / "dst-storage") == tree(root)            # same files, byte for byte
    assert restored_storage.get(next(iter(tree(root)))) == tree(root)[next(iter(tree(root)))]


def test_a_restored_database_works_like_the_original(source, tmp_path):
    from app.main import app
    from app.db import get_db
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import sessionmaker
    from .helpers import from_ip

    engine, root = source
    bundle.export_bundle(engine, root, tmp_path / "bundle")
    target = create_engine(f"sqlite:///{tmp_path / 'restored.db'}", connect_args={"check_same_thread": False})
    restored_storage = LocalStorage(str(tmp_path / "dst-storage"))
    bundle.import_bundle(target, restored_storage, tmp_path / "bundle")

    Session = sessionmaker(bind=target, autoflush=False, expire_on_commit=False)

    def override():
        with Session() as s:
            yield s

    app.dependency_overrides[get_db] = override
    storage._storage = restored_storage
    try:
        c = TestClient(from_ip(app, "127.0.0.1"))
        pid = c.get("/projects").json()[0]["id"]
        assert c.get(f"/projects/{pid}/datasets").json()[0]["name"] == "v1"
        assert len(c.get(f"/projects/{pid}/training-runs").json()) == 1
        image = c.get(f"/projects/{pid}/images", params={"limit": 1}).json()[0]
        assert c.get(f"/images/{image['id']}/file").content                        # an image file really serves
        zipped = c.get(f"/datasets/{c.get(f'/projects/{pid}/datasets').json()[0]['id']}/download")
        assert zipped.status_code == 200 and zipped.content[:2] == b"PK"           # the frozen manifest still resolves to its files
        new_project = c.post("/projects", json={"name": "next"}).json()["id"]      # new ids continue after the imported ones
        assert new_project > pid
    finally:
        app.dependency_overrides.clear()


def test_export_is_repeatable_and_replaces_its_own_output(source, tmp_path):
    engine, root = source
    out = tmp_path / "bundle"
    bundle.export_bundle(engine, root, out)
    first = tree(out)
    bundle.export_bundle(engine, root, out)
    assert tree(out) == first                                       # nothing volatile (no timestamps) in the output
    victim = next(p for p in root.rglob("*") if p.is_file())
    victim.unlink()
    bundle.export_bundle(engine, root, out)
    assert len(tree(out)) == len(first) - 1                         # a file removed from storage disappears from the bundle


def test_export_refuses_to_write_into_a_folder_that_is_not_a_bundle(source, tmp_path):
    engine, root = source
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "notes.txt").write_text("keep me")
    with pytest.raises(bundle.BundleError, match="refusing"):
        bundle.export_bundle(engine, root, precious)
    assert (precious / "notes.txt").read_text() == "keep me"


def test_import_refuses_a_database_that_already_has_data(source, tmp_path):
    engine, root = source
    bundle.export_bundle(engine, root, tmp_path / "bundle")
    with pytest.raises(bundle.BundleError, match="isn't empty"):
        bundle.import_bundle(engine, LocalStorage(str(tmp_path / "dst")), tmp_path / "bundle")
    assert not (tmp_path / "dst").exists()                          # and it restored no files either


def test_import_detects_a_corrupted_or_missing_file_before_changing_anything(source, tmp_path):
    engine, root = source
    out = tmp_path / "bundle"
    bundle.export_bundle(engine, root, out)
    victim = next(p for p in (out / "storage").rglob("*") if p.is_file())
    original = victim.read_bytes()
    victim.write_bytes(original.replace(b"\n", b"\r\n") if b"\n" in original else original + b"\r\n")   # what a bad checkout does
    target = create_engine(f"sqlite:///{tmp_path / 'r.db'}")
    with pytest.raises(bundle.BundleError, match="corrupted"):
        bundle.import_bundle(target, LocalStorage(str(tmp_path / "dst")), out)
    victim.unlink()
    with pytest.raises(bundle.BundleError, match="missing"):
        bundle.import_bundle(target, LocalStorage(str(tmp_path / "dst")), out)
    assert not (tmp_path / "dst").exists()
    with target.connect() as conn:                                  # no tables were even created
        assert conn.execute(text("select count(*) from sqlite_master where type='table'")).scalar() == 0


def test_import_tolerates_a_bundle_from_a_different_schema(source, tmp_path):
    engine, root = source
    out = tmp_path / "bundle"
    bundle.export_bundle(engine, root, out)
    db = json.loads((out / "db.json").read_text())
    db["tables"]["old_feature"] = [{"id": 1}]                       # a table the current schema doesn't have
    for row in db["tables"]["images"]:
        row["removed_column"] = "x"                                 # a column it no longer has
    (out / "db.json").write_text(json.dumps(db))
    target = create_engine(f"sqlite:///{tmp_path / 'r.db'}")
    result = bundle.import_bundle(target, LocalStorage(str(tmp_path / "dst")), out)
    assert result["skipped_tables"] == ["old_feature"] and result["dropped_columns"] == len(db["tables"]["images"])
    assert result["tables"]["images"] == len(db["tables"]["images"])


def test_import_rejects_anything_that_is_not_a_bundle(tmp_path):
    target = create_engine(f"sqlite:///{tmp_path / 'r.db'}")
    (tmp_path / "empty").mkdir()
    with pytest.raises(bundle.BundleError, match="not a format"):
        bundle.import_bundle(target, LocalStorage(str(tmp_path / "dst")), tmp_path / "empty")


def test_import_only_supports_sqlite(tmp_path):
    class Fake:
        class dialect:
            name = "postgresql"
    with pytest.raises(bundle.BundleError, match="SQLite only"):
        bundle.import_bundle(Fake, LocalStorage(str(tmp_path)), tmp_path)


def test_synthetic_seed_makes_a_project_a_run_can_start_on(client):
    from scripts.seed_synthetic import seed

    other = client.post("/projects", json={"name": "real"}).json()["id"]
    out = seed(client, per_class=8, start_run=True, epochs=2)
    assert out["dataset"]["stats"]["images"]["val"] > 0 and out["dataset"]["stats"]["boxes"]["train"] > 0
    assert out["run"]["stage"] == "queued" and out["run"]["config"]["epochs"] == 2
    rows = client.get(f"/projects/{other}/stats").json()
    assert rows and all(r["uploaded"] + r["scraped"] + r["generated"] == 0 for r in rows)  # the other project was left untouched


def test_export_leaves_out_finder_and_explorer_junk(source, tmp_path):
    engine, root = source
    for junk in (root / ".DS_Store", root / "projects" / ".DS_Store", root / "projects" / "._shadow.jpg", root / "Thumbs.db"):
        junk.parent.mkdir(parents=True, exist_ok=True)
        junk.write_bytes(b"not part of the project")
    exported = bundle.export_bundle(engine, root, tmp_path / "bundle")
    keys = json.loads((tmp_path / "bundle" / "files.json").read_text())
    assert not [k for k in keys if k.rsplit("/", 1)[-1] in {".DS_Store", "Thumbs.db"} or k.rsplit("/", 1)[-1].startswith("._")]
    assert exported["files"] == len(keys) > 20                       # real files still exported
