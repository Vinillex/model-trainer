"""The trash: rejected images are kept out of the working lists, can be restored, and can be deleted for good (with a warning when an existing
dataset version contains them)."""

import pytest

from app import storage
from app.models import AiCheck, Annotation, Image, UploadRecord

from .test_ai_check import FakeModels, flagged_image, listed, save_boxes
from .test_datasets import add, box
from .test_upload_folders import jpg, up
from app.ai import service


@pytest.fixture(autouse=True)
def no_models_left_behind():
    yield
    service.set_models(None)


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def ids_of(client, pid, **params):
    return sorted(i["id"] for i in client.get(f"/projects/{pid}/images", params={"limit": 500, **params}).json())


def trash_it(client, pid, *image_ids):
    assert client.post(f"/projects/{pid}/images/reject", json={"image_ids": list(image_ids)}).status_code == 200


def restore(client, pid, ids):
    return client.post(f"/projects/{pid}/images/restore", json={"image_ids": ids})


def purge(client, pid, ids, **extra):
    return client.post(f"/projects/{pid}/images/delete", json={"image_ids": ids, **extra})


def key_of(client, image_id):
    with client.session_factory() as db:
        return db.get(Image, image_id).storage_key


def on_disk(key):
    return (storage._storage.root / key).exists()


def exists_row(client, image_id):
    with client.session_factory() as db:
        return db.get(Image, image_id) is not None


# ---------- listing ----------

def test_the_trash_can_be_left_out_of_a_list_but_an_explicit_status_still_finds_it(client, project):
    pid, ids = project
    m = ids["multirotor"]
    keep, gone, pending = add(client, pid, m, [box(m)])["id"], add(client, pid, m, [box(m)])["id"], add(client, pid, m, status="pending")["id"]
    trash_it(client, pid, gone)
    assert ids_of(client, pid) == sorted([keep, gone, pending])                                  # as before: no filter, everything
    assert ids_of(client, pid, exclude_rejected=True) == sorted([keep, pending])
    assert ids_of(client, pid, exclude_rejected=True, review_status="rejected") == [gone]       # asking for the rejected ones by name wins
    assert ids_of(client, pid, review_status="rejected") == [gone]
    assert ids_of(client, pid, exclude_rejected=True, review_status="reviewed") == [keep]


# ---------- restoring ----------

def test_restore_puts_rejected_images_back_as_not_reviewed_with_their_boxes(client, project):
    pid, ids = project
    m = ids["multirotor"]
    a, b, other = (add(client, pid, m, [box(m)])["id"] for _ in range(3))
    trash_it(client, pid, a, b)
    assert restore(client, pid, [a]).json() == {"restored": 1}
    assert client.get(f"/images/{a}").json()["review_status"] == "pending" and client.get(f"/images/{b}").json()["review_status"] == "rejected"
    assert len(client.get(f"/images/{a}").json()["annotations"]) == 1                              # the boxes are as they were
    assert a in ids_of(client, pid, exclude_rejected=True) and b not in ids_of(client, pid, exclude_rejected=True)
    assert client.get(f"/images/{other}").json()["review_status"] == "reviewed"


def test_restore_and_delete_only_take_rejected_images_of_this_project_all_or_nothing(client, project):
    pid, ids = project
    m = ids["multirotor"]
    rejected, live = add(client, pid, m, [box(m)])["id"], add(client, pid, m, [box(m)])["id"]
    trash_it(client, pid, rejected)
    other = client.post("/projects", json={"name": "other"}).json()
    foreign = add(client, other["id"], {c["name"]: c["id"] for c in other["classes"]}["multirotor"], [])["id"]
    trash_it(client, other["id"], foreign)
    for call in (restore, purge):
        for bad in ([rejected, live], [rejected, foreign], [rejected, 99999]):
            r = call(client, pid, bad)
            assert r.status_code == 422 and "not rejected images of this project" in r.json()["detail"]
        assert call(client, pid, []).status_code == 422
        assert call(client, 9999, [rejected]).status_code == 404
    assert client.get(f"/images/{rejected}").json()["review_status"] == "rejected" and exists_row(client, rejected) and exists_row(client, foreign)
    assert client.get(f"/images/{live}").json()["review_status"] == "reviewed"


def test_a_restored_image_gets_the_ais_doubts_back(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)                                                            # "needs a look"
    save_boxes(client, img, m, status="rejected", dismiss=True)                                    # the review screen's Reject
    assert listed(client, pid, "flagged")[0] == [] and listed(client, pid, "dismissed")[0] == [img["id"]]
    restore(client, pid, [img["id"]])
    assert listed(client, pid, "flagged")[0] == [img["id"]] and listed(client, pid, "dismissed")[0] == []
    with client.session_factory() as db:
        assert db.get(AiCheck, img["id"]).verdict == "suspect"                                     # the check itself is untouched


# ---------- deleting for good ----------

def test_delete_forever_removes_the_files_the_rows_and_everything_attached_and_nothing_else(client, project):
    pid, ids = project
    m = ids["multirotor"]
    in_folder = up(client, pid, m, jpg(1), folder="f1").json()["added"][0]["id"]
    client.put(f"/images/{in_folder}/annotations", json={"boxes": [{"class_id": m, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]})
    service.set_models(FakeModels([]))
    client.post(f"/images/{in_folder}/ai-check")
    plain = add(client, pid, m, [box(m)])["id"]
    survivor = add(client, pid, m, [box(m)])["id"]
    trash_it(client, pid, in_folder, plain)
    keys = {i: key_of(client, i) for i in (in_folder, plain, survivor)}
    assert all(on_disk(k) for k in keys.values())
    assert purge(client, pid, [in_folder, plain]).json() == {"deleted": 2}
    assert not exists_row(client, in_folder) and not exists_row(client, plain)
    assert not on_disk(keys[in_folder]) and not on_disk(keys[plain])                                # the files are gone too
    with client.session_factory() as db:
        assert db.query(Annotation).filter(Annotation.image_id.in_([in_folder, plain])).count() == 0
        assert db.query(UploadRecord).count() == 0 and db.query(AiCheck).count() == 0
    assert client.get(f"/images/{in_folder}/file").status_code == 404
    assert exists_row(client, survivor) and on_disk(keys[survivor])
    assert [r for r in client.get(f"/projects/{pid}/folders").json() if r["folder"] == "f1"] == []


def frozen_version_with(client, pid, ids, count=8):
    m = ids["multirotor"]
    images = [add(client, pid, m, [box(m)])["id"] for _ in range(count)]
    assert client.post(f"/projects/{pid}/datasets", json={}).status_code == 201
    return images


def test_deleting_images_a_dataset_version_contains_asks_first_and_then_works_when_told(client, project):
    pid, ids = project
    images = frozen_version_with(client, pid, ids)
    inside, free = images[0], add(client, pid, ids["multirotor"], [box(ids["multirotor"])])["id"]   # `free` came after v1 was frozen
    trash_it(client, pid, inside, free)
    inside_key = key_of(client, inside)
    r = purge(client, pid, [inside, free])
    assert r.status_code == 409 and "dataset version v1" in r.json()["detail"] and "1 of these images is part of dataset version v1. Deleting it means" in r.json()["detail"]
    assert exists_row(client, inside) and exists_row(client, free) and on_disk(inside_key)        # all or nothing
    assert purge(client, pid, [free]).json() == {"deleted": 1}                                    # not in any version: no question
    assert purge(client, pid, [inside], allow_breaking_versions=True).json() == {"deleted": 1}
    assert not exists_row(client, inside) and not on_disk(inside_key)


def test_a_reused_image_id_is_not_mistaken_for_the_frozen_one(client, project):
    pid, ids = project
    m = ids["multirotor"]
    images = frozen_version_with(client, pid, ids)
    last = max(images)
    assert client.delete(f"/images/{last}").status_code == 204                                    # the single Delete is unchanged: allowed
    reused = add(client, pid, m, [box(m)])["id"]
    assert reused == last                                                                           # SQLite hands the id out again
    trash_it(client, pid, reused)
    assert purge(client, pid, [reused]).json() == {"deleted": 1}                                  # a different picture: the version does not contain it


def test_the_single_delete_button_still_deletes_an_image_a_version_contains(client, project):
    pid, ids = project
    images = frozen_version_with(client, pid, ids)
    assert client.delete(f"/images/{images[0]}").status_code == 204
    assert not exists_row(client, images[0])
