"""Uploading a folder: the images go to <class>/uploaded/<folder>, keep that folder when a review moves them, and can be listed by it."""

import itertools

import pytest

from app import storage
from app.ai.job import select_images
from app.api.images import clean_folder
from app.models import Image, UploadRecord

from .helpers import photo

_seed = itertools.count(5000)


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def jpg(n=1):
    return [("files", (f"img{next(_seed)}.jpg", photo(next(_seed)), "image/jpeg")) for _ in range(n)]


def up(client, pid, cid, files, folder=None, paths=None):
    params = {"folder": folder} if folder is not None else {}
    return client.post(f"/projects/{pid}/classes/{cid}/images", params=params, files=files, data={"relative_paths": paths} if paths else None)


def key_of(client, image_id):
    with client.session_factory() as db:
        return db.get(Image, image_id).storage_key


def on_disk(key):
    return (storage._storage.root / key).exists()


def listed(client, pid, **params):
    return sorted(i["id"] for i in client.get(f"/projects/{pid}/images", params=params).json())


def review(client, image_id, class_id):
    body = {"boxes": [{"class_id": class_id, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.1}], "review_status": "reviewed"}
    assert client.put(f"/images/{image_id}/annotations", json=body).status_code == 200


# ---------- the folder name ----------

@pytest.mark.parametrize("raw, want", [
    ("flight-1", "flight-1"),
    ("Flight 1 (day2)", "Flight-1-day2"),
    ("../../etc", "etc"),
    ("a/b\\c", "a-b-c"),
    (".hidden", "hidden"),
    ("a..b", "a.b"),
    ("  spaced  ", "spaced"),
    ("..", None),
    ("///", None),
    ("日本語", None),
    ("", None),
])
def test_folder_names_are_made_safe_and_unusable_ones_are_refused(raw, want):
    assert clean_folder(raw) == want


def test_a_very_long_folder_name_is_cut_to_what_the_column_holds():
    cleaned = clean_folder("x" * 200)
    assert cleaned is not None and len(cleaned) == 80


# ---------- uploading ----------

def test_a_folder_upload_goes_to_class_uploaded_folder(client, project):
    pid, ids = project
    r = up(client, pid, ids["multirotor"], jpg(2), folder="flight-1", paths=["flight-1/a.jpg", "flight-1/day2/b.jpg"]).json()
    assert r["folder"] == "flight-1" and len(r["added"]) == 2 and all(i["folder"] == "flight-1" for i in r["added"])
    for img in r["added"]:
        key = key_of(client, img["id"])
        assert key.startswith(f"projects/{pid}/raw/multirotor/uploaded/flight-1/") and on_disk(key)
    with client.session_factory() as db:
        paths = sorted(u.original_path for u in db.query(UploadRecord))
    assert paths == ["flight-1/a.jpg", "flight-1/day2/b.jpg"]            # where each sat is kept for the record


def test_loose_files_still_go_straight_into_uploaded(client, project):
    pid, ids = project
    r = up(client, pid, ids["multirotor"], jpg(1)).json()
    img = r["added"][0]
    assert r["folder"] is None and img["folder"] is None
    key = key_of(client, img["id"])
    assert key.startswith(f"projects/{pid}/raw/multirotor/uploaded/") and key.count("/") == 5   # no folder level
    with client.session_factory() as db:
        assert db.query(UploadRecord).count() == 0


def test_an_unusable_folder_name_stores_nothing(client, project):
    pid, ids = project
    r = up(client, pid, ids["multirotor"], jpg(2), folder="日本語")
    assert r.status_code == 422 and "folder name" in r.json()["detail"]
    assert listed(client, pid) == []


def test_relative_paths_must_match_the_files(client, project):
    pid, ids = project
    r = up(client, pid, ids["multirotor"], jpg(2), folder="f", paths=["only-one.jpg"])
    assert r.status_code == 422 and listed(client, pid) == []


def test_a_folder_name_cannot_walk_out_of_the_class_folder(client, project):
    pid, ids = project
    r = up(client, pid, ids["multirotor"], jpg(1), folder="../../../outside", paths=["../../x\x00y.jpg"]).json()
    key = key_of(client, r["added"][0]["id"])
    assert r["folder"] == "outside" and ".." not in key and key.startswith(f"projects/{pid}/raw/multirotor/uploaded/outside/")
    assert (storage._storage.root / key).resolve().is_relative_to(storage._storage.root.resolve())
    with client.session_factory() as db:
        assert "\x00" not in db.query(UploadRecord).one().original_path   # the record is data only, cleaned of control characters


def test_uploading_into_the_same_folder_again_adds_to_it_and_skips_copies(client, project):
    pid, ids = project
    files = jpg(2)
    first = up(client, pid, ids["multirotor"], files, folder="f1").json()
    again = up(client, pid, ids["multirotor"], [files[0]] + jpg(1), folder="f1").json()
    assert len(first["added"]) == 2 and len(again["added"]) == 1 and len(again["duplicates"]) == 1
    assert len(listed(client, pid, folder="f1")) == 3
    elsewhere = up(client, pid, ids["fixed-wing"], [files[1]], folder="f2").json()
    assert elsewhere["added"] == [] and len(elsewhere["duplicates"]) == 1  # the same picture is not filed twice, even in another folder


# ---------- listing ----------

def test_the_folder_filter_lists_only_that_folder(client, project):
    pid, ids = project
    a = [i["id"] for i in up(client, pid, ids["multirotor"], jpg(2), folder="a").json()["added"]]
    b = [i["id"] for i in up(client, pid, ids["multirotor"], jpg(1), folder="b").json()["added"]]
    loose = [i["id"] for i in up(client, pid, ids["multirotor"], jpg(1)).json()["added"]]
    assert listed(client, pid, folder="a") == sorted(a) and listed(client, pid, folder="b") == b
    assert listed(client, pid) == sorted(a + b + loose)
    assert client.get(f"/projects/{pid}/images", params={"folder": "../x"}).status_code == 422


def test_the_folder_tree_counts_each_place_by_review_state(client, project):
    pid, ids = project
    a = up(client, pid, ids["multirotor"], jpg(3), folder="a").json()["added"]
    up(client, pid, ids["multirotor"], jpg(1))
    up(client, pid, ids["fixed-wing"], jpg(1), folder="a")
    review(client, a[0]["id"], ids["multirotor"])
    client.put(f"/images/{a[1]['id']}/annotations", json={"boxes": [], "review_status": "rejected"})
    rows = {(r["class_id"], r["source"], r["folder"]): r for r in client.get(f"/projects/{pid}/folders").json()}
    mine = rows[(ids["multirotor"], "uploaded", "a")]
    assert (mine["total"], mine["reviewed"], mine["pending"], mine["rejected"]) == (3, 1, 1, 1)
    assert rows[(ids["multirotor"], "uploaded", None)]["total"] == 1       # loose files are their own row
    assert rows[(ids["fixed-wing"], "uploaded", "a")]["total"] == 1        # the same folder name under another class is another row
    assert client.get("/projects/9999/folders").status_code == 404


# ---------- keeping the folder ----------

def test_a_reviewed_image_keeps_its_folder_when_it_moves_to_another_class(client, project):
    pid, ids = project
    img = up(client, pid, ids["multirotor"], jpg(1), folder="flight-1").json()["added"][0]
    old = key_of(client, img["id"])
    review(client, img["id"], ids["fixed-wing"])                           # it shows a fixed-wing drone
    new = key_of(client, img["id"])
    assert new != old and f"/raw/fixed-wing/uploaded/flight-1/" in new and on_disk(new) and not on_disk(old)
    assert listed(client, pid, folder="flight-1") == [img["id"]]
    assert client.get(f"/images/{img['id']}").json()["folder"] == "flight-1"


def test_refiling_old_reviews_also_keeps_the_folder(client, project):
    pid, ids = project
    img = up(client, pid, ids["multirotor"], jpg(1), folder="flight-1").json()["added"][0]
    with client.session_factory() as db:                                    # reviewed before the folder rule existed: boxes saved, image not moved
        from app.models import Annotation, AnnotationStatus, ReviewStatus
        row = db.get(Image, img["id"])
        db.add(Annotation(image_id=row.id, class_id=ids["fixed-wing"], cx=0.5, cy=0.5, w=0.2, h=0.1, status=AnnotationStatus.reviewed))
        row.review_status = ReviewStatus.reviewed
        db.commit()
    assert client.post(f"/projects/{pid}/images/refile", params={"dry_run": False}).json()["moved"] == 1
    assert "/raw/fixed-wing/uploaded/flight-1/" in key_of(client, img["id"])


def test_deleting_an_image_removes_its_folder_record(client, project):
    pid, ids = project
    img = up(client, pid, ids["multirotor"], jpg(1), folder="gone").json()["added"][0]
    assert client.delete(f"/images/{img['id']}").status_code == 204
    assert [r for r in client.get(f"/projects/{pid}/folders").json() if r["folder"] == "gone"] == []
    with client.session_factory() as db:
        assert db.query(UploadRecord).count() == 0


# ---------- the AI batch ----------

def test_an_ai_batch_limited_to_a_folder_does_not_take_other_images(client, project):
    pid, ids = project
    a = [i["id"] for i in up(client, pid, ids["multirotor"], jpg(2), folder="a").json()["added"]]
    up(client, pid, ids["multirotor"], jpg(2), folder="b")
    up(client, pid, ids["multirotor"], jpg(1))
    with client.session_factory() as db:
        assert sorted(select_images(db, pid, {"folder": "a"})) == sorted(a)
        assert len(select_images(db, pid, {})) == 5                         # no folder = everything, as before


def test_the_ai_job_form_accepts_a_folder_and_refuses_a_path(client, project, monkeypatch):
    pid, _ = project
    monkeypatch.setattr("app.api.ai.service.missing_extras", lambda: None)
    ok = client.post(f"/projects/{pid}/jobs/ai-check", json={"folder": "flight-1"})
    assert ok.status_code == 201 and ok.json()["params"]["folder"] == "flight-1"
    assert client.post(f"/projects/{pid}/jobs/ai-check", json={"folder": "../x"}).status_code == 422
