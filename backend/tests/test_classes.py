import io
import zipfile

import pytest

from app.models import Job, JobStatus

from .test_datasets import add, box, populate


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def class_names(client, pid):
    return [c["name"] for c in client.get(f"/projects/{pid}").json()["classes"]]


def indexes(client, pid):
    return {c["name"]: c["index"] for c in client.get(f"/projects/{pid}").json()["classes"] if not c["is_negative"]}


def test_an_unused_class_can_be_removed_and_its_name_reused(client, project):
    pid, ids = project
    assert client.delete(f"/projects/{pid}/classes/{ids['delta-wing']}").status_code == 204
    assert "delta-wing" not in class_names(client, pid)
    assert "delta-wing" not in [r["class_name"] for r in client.get(f"/projects/{pid}/stats").json()]
    assert client.delete(f"/projects/{pid}/classes/{ids['delta-wing']}").status_code == 404            # already gone
    assert client.post(f"/projects/{pid}/classes", json={"name": "delta-wing"}).status_code == 201     # the name is free again


def test_hard_negative_classes_can_be_removed_too(client, project):
    pid, ids = project
    assert client.delete(f"/projects/{pid}/classes/{ids['birds']}").status_code == 204
    assert "birds" not in class_names(client, pid)


def test_a_class_of_another_project_is_not_found(client, project):
    pid, ids = project
    other = client.post("/projects", json={"name": "other"}).json()["id"]
    assert client.delete(f"/projects/{other}/classes/{ids['multirotor']}").status_code == 404
    assert "multirotor" in class_names(client, pid)                                               # untouched
    assert client.delete(f"/projects/9999/classes/{ids['multirotor']}").status_code == 404


def test_a_class_with_images_is_refused(client, project):
    pid, ids = project
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    r = client.delete(f"/projects/{pid}/classes/{ids['multirotor']}")
    assert r.status_code == 409 and "1 image filed under it" in r.json()["detail"]
    assert "multirotor" in class_names(client, pid)


def test_a_class_used_only_by_boxes_on_other_images_is_refused(client, project):
    pid, ids = project
    # Two classes on one image: it moves to the `mixed` folder, so fixed-wing and delta-wing have boxes but no images of their own.
    add(client, pid, ids["multirotor"], [box(ids["fixed-wing"], 0.3), box(ids["delta-wing"], 0.7)])
    stats = {r["class_name"]: r for r in client.get(f"/projects/{pid}/stats").json()}
    assert stats["fixed-wing"]["boxes"] == 1 and stats["fixed-wing"]["uploaded"] == 0 and stats["mixed"]["uploaded"] == 1
    r = client.delete(f"/projects/{pid}/classes/{ids['fixed-wing']}")
    assert r.status_code == 409 and "1 box on 1 image" in r.json()["detail"]


def test_a_class_with_a_queued_or_running_scrape_job_is_refused_until_it_ends(client, project):
    pid, ids = project
    with client.session_factory() as db:
        job = Job(project_id=pid, type="scrape", params={"class_id": ids["delta-wing"], "query": "x"}, status=JobStatus.running)
        db.add(job)
        db.commit()
        job_id = job.id
    r = client.delete(f"/projects/{pid}/classes/{ids['delta-wing']}")
    assert r.status_code == 409 and "scrape job" in r.json()["detail"]
    assert client.delete(f"/projects/{pid}/classes/{ids['birds']}").status_code == 204            # other classes are unaffected
    with client.session_factory() as db:
        db.get(Job, job_id).status = JobStatus.succeeded
        db.commit()
    assert client.delete(f"/projects/{pid}/classes/{ids['delta-wing']}").status_code == 204


def test_a_new_class_never_gets_a_live_classs_index_after_a_removal(client, project):
    pid, ids = project
    assert indexes(client, pid) == {"multirotor": 0, "fixed-wing": 1, "delta-wing": 2}
    client.delete(f"/projects/{pid}/classes/{ids['fixed-wing']}")                                  # leaves a gap at 1
    client.post(f"/projects/{pid}/classes", json={"name": "vtol"})
    bulk = client.post(f"/projects/{pid}/classes/bulk", json={"names": ["quad-copter", "flying-wing"]}).json()
    assert [c["index"] for c in bulk["added"]] == [4, 5]
    idx = indexes(client, pid)
    assert idx == {"multirotor": 0, "delta-wing": 2, "vtol": 3, "quad-copter": 4, "flying-wing": 5}  # all unique, none reused
    assert len(set(idx.values())) == len(idx)


def test_existing_dataset_versions_survive_removing_a_class(client, project):
    pid, ids = project
    spare = client.post(f"/projects/{pid}/classes", json={"name": "vtol"}).json()["id"]           # unused, but part of the version
    populate(client, pid, ids)
    v1 = client.post(f"/projects/{pid}/datasets", json={}).json()
    assert "vtol" in [c["name"] for c in v1["config"]["classes"]]

    assert client.delete(f"/projects/{pid}/classes/{spare}").status_code == 204

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/datasets/{v1['id']}/download").content))          # the old snapshot is untouched
    assert "vtol" in z.read("data.yaml").decode()
    v2 = client.post(f"/projects/{pid}/datasets", json={}).json()
    names = [c["name"] for c in v2["config"]["classes"]]
    assert "vtol" not in names and [c["index"] for c in v2["config"]["classes"]] == list(range(len(names)))  # new one is contiguous


# ---------- renaming a class ----------

def keys_of(client, *image_ids):
    from app.models import Image
    with client.session_factory() as db:
        return [db.get(Image, i).storage_key for i in image_ids]


def rename(client, pid, class_id, name):
    return client.patch(f"/projects/{pid}/classes/{class_id}", json={"name": name})


def test_renaming_a_class_moves_its_images_and_keeps_their_boxes_and_bytes(client, project):
    from app import storage
    pid, ids = project
    d = ids["delta-wing"]
    a = add(client, pid, d, [box(d)])["id"]
    b = add(client, pid, d, [box(d, 0.3, 0.6)])["id"]
    other = add(client, pid, ids["multirotor"], [box(ids["multirotor"])])["id"]
    before = keys_of(client, a, b, other)
    data = {k: storage.get_storage().get(k) for k in before}
    assert all("/raw/delta-wing/" in k for k in before[:2])

    r = rename(client, pid, d, "flying-wing")
    assert r.status_code == 200 and r.json()["name"] == "flying-wing" and r.json()["moved"] == 2 and r.json()["id"] == d
    after = keys_of(client, a, b, other)
    assert all("/raw/flying-wing/" in k for k in after[:2]) and after[2] == before[2]               # only this class's images moved
    for old, new in zip(before[:2], after[:2]):
        assert storage.get_storage().get(new) == data[old]                                          # the same bytes
        assert not storage._storage._path(old).exists()                                             # the old copy is gone
    assert "flying-wing" in class_names(client, pid) and "delta-wing" not in class_names(client, pid)
    assert indexes(client, pid)["flying-wing"] == 2                                                 # its place in the label list stays
    assert len(client.get(f"/images/{a}").json()["annotations"]) == 1 and client.get(f"/images/{a}").json()["class_id"] == d
    assert rename(client, pid, d, "flying-wing").json()["moved"] == 0                               # same name again: nothing to do


def test_a_hard_negative_class_can_be_renamed_too(client, project):
    pid, ids = project
    bird = add(client, pid, ids["birds"])["id"]
    assert rename(client, pid, ids["birds"], "birds-in-flight").json()["moved"] == 1
    assert "/raw/birds-in-flight/" in keys_of(client, bird)[0]
    renamed = next(c for c in client.get(f"/projects/{pid}").json()["classes"] if c["id"] == ids["birds"])
    assert renamed["name"] == "birds-in-flight" and renamed["is_negative"] is True


def test_renaming_never_breaks_a_frozen_dataset_version(client, project):
    pid, ids = project
    populate(client, pid, ids)
    vid = client.post(f"/projects/{pid}/datasets", json={"name": "v1"}).json()["id"]
    before = zipfile.ZipFile(io.BytesIO(client.get(f"/datasets/{vid}/download").content))
    assert rename(client, pid, ids["delta-wing"], "flying-wing").status_code == 200
    assert rename(client, pid, ids["multirotor"], "quadcopter").json()["moved"] > 0
    after = zipfile.ZipFile(io.BytesIO(client.get(f"/datasets/{vid}/download").content))
    assert sorted(after.namelist()) == sorted(before.namelist())                                      # every picture is still found
    assert after.read("data.yaml") == before.read("data.yaml") and b'"multirotor"' in after.read("data.yaml")  # the version keeps its own, old names


def test_a_rename_that_cannot_happen_is_refused_and_changes_nothing(client, project):
    pid, ids = project
    d = ids["delta-wing"]
    img = add(client, pid, d)["id"]
    before = keys_of(client, img)
    assert rename(client, pid, d, "multirotor").status_code == 409                                    # name taken
    assert rename(client, pid, d, "birds").status_code == 409                                         # also taken, as a hard negative
    assert rename(client, pid, d, "mixed").status_code == 422                                         # reserved
    assert rename(client, pid, d, "Has Spaces").status_code == 422 and rename(client, pid, d, "").status_code == 422
    assert rename(client, pid, 99999, "x").status_code == 404
    other = client.post("/projects", json={"name": "other"}).json()["id"]
    assert rename(client, other, d, "elsewhere").status_code == 404                                   # a class of another project
    with client.session_factory() as db:
        job = Job(project_id=pid, type="ai_check", params={}, status=JobStatus.running)
        db.add(job)
        db.commit()
    r = rename(client, pid, d, "flying-wing")
    assert r.status_code == 409 and "job" in r.json()["detail"]                                       # files must not move under a running job
    assert class_names(client, pid).count("delta-wing") == 1 and keys_of(client, img) == before


def test_the_mixed_folder_cannot_be_renamed(client, project):
    pid, ids = project
    add(client, pid, ids["multirotor"], [box(ids["fixed-wing"], 0.3), box(ids["delta-wing"], 0.7)])  # makes the mixed folder
    mixed = next(c["id"] for c in client.get(f"/projects/{pid}").json()["classes"] if c["name"] == "mixed")
    assert rename(client, pid, mixed, "whatever").status_code == 422


def test_a_failure_halfway_leaves_every_file_and_row_as_it_was(client, project, monkeypatch):
    from app import storage
    pid, ids = project
    d = ids["delta-wing"]
    imgs = [add(client, pid, d, [box(d)])["id"] for _ in range(3)]
    before = keys_of(client, *imgs)
    real = storage.get_storage()
    calls = {"n": 0}
    original_put = real.put

    def flaky_put(key, data):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        original_put(key, data)
    monkeypatch.setattr(real, "put", flaky_put)
    r = rename(client, pid, d, "flying-wing")
    assert r.status_code == 500 and "Nothing was changed" in r.json()["detail"]
    monkeypatch.undo()
    assert keys_of(client, *imgs) == before and "delta-wing" in class_names(client, pid)
    assert all(real.get(k) for k in before)                                                            # the originals are all still there
    leftovers = list((real.root / "projects" / str(pid) / "raw").glob("flying-wing/**/*.*"))
    assert leftovers == []                                                                             # and the two copies made before the failure were removed
