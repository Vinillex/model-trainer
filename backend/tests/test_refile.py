import io
import zipfile

import pytest

from app import refile as refile_mod
from app import storage
from app.models import Image

from .helpers import photo
from .test_datasets import add, box, populate


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def key_of(client, image_id):
    with client.session_factory() as db:
        return db.get(Image, image_id).storage_key


def class_of(client, image_id):
    return client.get(f"/images/{image_id}").json()["class_id"]


def on_disk(key):
    return (storage._storage.root / key).exists()


def put(client, image_id, boxes, status="reviewed"):
    body = {"boxes": [{"class_id": c, "cx": cx, "cy": cy, "w": w, "h": h} for c, cx, cy, w, h in boxes], "review_status": status}
    r = client.put(f"/images/{image_id}/annotations", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def mixed_id(client, pid):
    ids = [c["id"] for c in client.get(f"/projects/{pid}").json()["classes"] if c["name"] == "mixed"]
    return ids[0] if ids else None


# ---------- moving ----------

def test_a_reviewed_image_moves_to_the_folder_of_the_class_its_box_shows(client, project):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], status="pending")                  # filed under multirotor at upload
    old = key_of(client, img["id"])
    assert "/raw/multirotor/uploaded/" in old and on_disk(old)

    out = put(client, img["id"], [box(ids["fixed-wing"])])                       # ...but it shows a fixed-wing drone
    new = key_of(client, img["id"])
    assert out["class_id"] == ids["fixed-wing"] and class_of(client, img["id"]) == ids["fixed-wing"]
    assert "/raw/fixed-wing/uploaded/" in new and on_disk(new) and not on_disk(old)       # moved, not copied
    served = client.get(f"/images/{img['id']}/file")
    assert served.status_code == 200 and len(served.content) > 0                 # image URLs don't depend on the path


def test_an_image_showing_two_classes_goes_to_the_mixed_folder(client, project):
    pid, ids = project
    first = add(client, pid, ids["multirotor"], status="pending")
    second = add(client, pid, ids["fixed-wing"], status="pending")
    assert mixed_id(client, pid) is None                                         # not created until needed
    put(client, first["id"], [box(ids["multirotor"], 0.2), box(ids["fixed-wing"], 0.7)])
    put(client, second["id"], [box(ids["shahed"], 0.3), box(ids["multirotor"], 0.8)])
    m = mixed_id(client, pid)
    assert m is not None and class_of(client, first["id"]) == m and class_of(client, second["id"]) == m
    assert "/raw/mixed/uploaded/" in key_of(client, first["id"]) and on_disk(key_of(client, second["id"]))
    assert [c["name"] for c in client.get(f"/projects/{pid}").json()["classes"]].count("mixed") == 1   # created once, reused


@pytest.mark.parametrize("case", ["same class", "no boxes", "still pending", "rejected"])
def test_nothing_moves_unless_a_reviewed_image_has_boxes_of_another_class(client, project, case):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], status="pending")
    old = key_of(client, img["id"])
    if case == "same class":
        put(client, img["id"], [box(ids["multirotor"])])
    elif case == "no boxes":
        put(client, img["id"], [])
    elif case == "still pending":
        put(client, img["id"], [box(ids["fixed-wing"])], status="pending")
    else:
        put(client, img["id"], [box(ids["fixed-wing"])], status="rejected")
    assert class_of(client, img["id"]) == ids["multirotor"] and key_of(client, img["id"]) == old and on_disk(old)


def test_editing_an_already_reviewed_image_moves_it_too(client, project):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], [box(ids["multirotor"])])          # reviewed and correctly filed
    assert class_of(client, img["id"]) == ids["multirotor"]
    r = client.put(f"/images/{img['id']}/annotations", json={"boxes": [{"class_id": ids["fixed-wing"], "cx": .5, "cy": .5, "w": .2, "h": .2}]})
    assert r.status_code == 200                                                  # no review_status in the body: it stays reviewed
    assert class_of(client, img["id"]) == ids["fixed-wing"]


def test_an_image_leaves_mixed_when_its_boxes_come_down_to_one_class(client, project):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], [box(ids["multirotor"], 0.2), box(ids["fixed-wing"], 0.7)])
    assert class_of(client, img["id"]) == mixed_id(client, pid)
    mixed_key = key_of(client, img["id"])
    put(client, img["id"], [box(ids["fixed-wing"], 0.7)])
    assert class_of(client, img["id"]) == ids["fixed-wing"] and "/raw/fixed-wing/" in key_of(client, img["id"])
    assert not on_disk(mixed_key)


# ---------- `mixed` is a folder, never a label ----------

def test_mixed_is_reserved_and_never_a_class_or_a_label(client, project):
    pid, ids = project
    assert client.post(f"/projects/{pid}/classes", json={"name": "mixed"}).status_code == 422
    assert "mixed" in client.post(f"/projects/{pid}/classes/bulk", json={"names": ["mixed", "vtol"]}).json()["invalid"]
    img = add(client, pid, ids["multirotor"], [box(ids["multirotor"], 0.2), box(ids["fixed-wing"], 0.7)])
    m = mixed_id(client, pid)
    assert m is not None
    r = client.put(f"/images/{img['id']}/annotations", json={"boxes": [{"class_id": m, "cx": .5, "cy": .5, "w": .2, "h": .2}]})
    assert r.status_code == 422 and "folder" in r.json()["detail"]               # no box can carry the label "mixed"
    up = client.post(f"/projects/{pid}/classes/{m}/images", files=[("files", ("x.jpg", photo(77), "image/jpeg"))])
    assert up.status_code == 422                                                 # nothing is uploaded into it by hand...
    assert client.post(f"/projects/{pid}/jobs/scrape", json={"class_id": m, "query": "x"}).status_code == 422   # ...or scraped
    row = next(r for r in client.get(f"/projects/{pid}/stats").json() if r["class_name"] == "mixed")
    assert row["is_mixed"] is True and row["uploaded"] == 1


def test_mixed_images_train_with_their_real_classes_but_mixed_is_not_a_class(client, project):
    pid, ids = project
    populate(client, pid, ids)
    add(client, pid, ids["multirotor"], [box(ids["multirotor"], 0.2), box(ids["fixed-wing"], 0.7)])     # lands in mixed
    assert "mixed" not in client.get(f"/projects/{pid}/datasets/preview").json()["boxes_per_class"]
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    names = [c["name"] for c in v["config"]["classes"]]
    assert "mixed" not in names and [c["index"] for c in v["config"]["classes"]] == list(range(len(names)))
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/datasets/{v['id']}/download").content))
    assert "mixed" not in z.read("data.yaml").decode()
    assert sum(len(z.read(n).splitlines()) for n in z.namelist() if n.startswith("labels/")) == v["stats"]["boxes"]["train"] + v["stats"]["boxes"]["val"] + v["stats"]["boxes"]["test"]


# ---------- frozen dataset versions ----------

def test_a_version_frozen_before_a_move_still_downloads_the_same_images_and_labels(client, project):
    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    before = zipfile.ZipFile(io.BytesIO(client.get(f"/datasets/{v['id']}/download").content))
    victim = next(i for i in client.get(f"/projects/{pid}/images", params={"class_id": ids["multirotor"], "limit": 50}).json()
                  if i["review_status"] == "reviewed")
    old_key = key_of(client, victim["id"])
    put(client, victim["id"], [box(ids["fixed-wing"])])                          # re-boxed after the version was frozen: moves
    assert key_of(client, victim["id"]) != old_key and not on_disk(old_key)

    r = client.get(f"/datasets/{v['id']}/download")
    assert r.status_code == 200                                                  # the old path is gone, the version still works
    after = zipfile.ZipFile(io.BytesIO(r.content))
    assert sorted(after.namelist()) == sorted(before.namelist())
    assert all(after.read(n) == before.read(n) for n in before.namelist())      # same image bytes, same frozen labels


def test_the_worker_download_of_a_frozen_version_also_follows_moved_images(client, project, monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.main import app
    from .helpers import from_ip

    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    run = client.post(f"/projects/{pid}/training-runs", json={"dataset_id": v["id"], "epochs": 1})
    assert run.status_code == 201
    victim = next(i for i in client.get(f"/projects/{pid}/images", params={"class_id": ids["multirotor"], "limit": 50}).json()
                  if i["review_status"] == "reviewed")
    put(client, victim["id"], [box(ids["fixed-wing"])])
    monkeypatch.setattr(settings, "worker_token", "t")
    lan = TestClient(from_ip(app, "192.168.1.50"))
    h = {"X-Worker-Token": "t", "X-Worker-Name": "gpu-1"}
    task = lan.post("/workers/claim", json={"kind": "gpu"}, headers=h).json()
    assert lan.get(task["dataset_url"], headers=h).status_code == 200


def test_a_deleted_images_reused_id_is_never_swapped_into_a_frozen_version(client, project):
    pid, ids = project
    imgs = [add(client, pid, ids["multirotor"], [box(ids["multirotor"])]) for _ in range(12)]
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    last = max(imgs, key=lambda i: i["id"])
    assert client.delete(f"/images/{last['id']}").status_code == 204
    newcomer = client.post(f"/projects/{pid}/classes/{ids['multirotor']}/images",
                           files=[("files", ("n.jpg", photo(4242), "image/jpeg"))]).json()["added"][0]
    assert newcomer["id"] == last["id"], "SQLite should reuse the highest id; the scenario needs it"
    r = client.get(f"/datasets/{v['id']}/download")
    assert r.status_code == 409 and "no longer exist" in r.json()["detail"]      # missing, not silently a different picture


# ---------- failures and older data ----------

def test_a_failed_move_never_costs_the_boxes(client, project, monkeypatch):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], status="pending")
    old = key_of(client, img["id"])
    with monkeypatch.context() as m:
        m.setattr(storage._storage, "put", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        out = put(client, img["id"], [box(ids["fixed-wing"])])
    assert out["review_status"] == "reviewed" and len(out["annotations"]) == 1   # the review is saved...
    assert class_of(client, img["id"]) == ids["multirotor"] and key_of(client, img["id"]) == old and on_disk(old)   # ...the image stays
    assert not (storage._storage.root / "projects" / str(pid) / "raw" / "fixed-wing").exists()
    put(client, img["id"], [box(ids["fixed-wing"])])                             # saving again retries the move
    assert class_of(client, img["id"]) == ids["fixed-wing"]


def test_the_refile_endpoint_applies_the_rule_to_images_reviewed_before_it_existed(client, project, monkeypatch):
    pid, ids = project
    with monkeypatch.context() as m:
        m.setattr(refile_mod, "target_class", lambda db, image: None)           # reviews saved while the rule was off
        a = add(client, pid, ids["multirotor"], [box(ids["fixed-wing"])])
        b = add(client, pid, ids["multirotor"], [box(ids["multirotor"], 0.2), box(ids["shahed"], 0.7)])
        c = add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    assert [class_of(client, i["id"]) for i in (a, b, c)] == [ids["multirotor"]] * 3

    dry = client.post(f"/projects/{pid}/images/refile").json()                    # dry run is the default
    assert dry["dry_run"] is True and dry["would_move"] == 2 and dry["unchanged"] == 1
    assert {m["to"] for m in dry["moves"]} == {"fixed-wing", "mixed"}
    assert [class_of(client, i["id"]) for i in (a, b, c)] == [ids["multirotor"]] * 3          # nothing changed

    done = client.post(f"/projects/{pid}/images/refile", params={"dry_run": "false"}).json()
    assert done["moved"] == 2 and "failed" not in done
    assert class_of(client, a["id"]) == ids["fixed-wing"] and class_of(client, b["id"]) == mixed_id(client, pid)
    assert class_of(client, c["id"]) == ids["multirotor"]
    again = client.post(f"/projects/{pid}/images/refile", params={"dry_run": "false"}).json()
    assert again["moved"] == 0 and again["unchanged"] == 3
