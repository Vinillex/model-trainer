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
    assert client.delete(f"/projects/{pid}/classes/{ids['shahed']}").status_code == 204
    assert "shahed" not in class_names(client, pid)
    assert "shahed" not in [r["class_name"] for r in client.get(f"/projects/{pid}/stats").json()]
    assert client.delete(f"/projects/{pid}/classes/{ids['shahed']}").status_code == 404            # already gone
    assert client.post(f"/projects/{pid}/classes", json={"name": "shahed"}).status_code == 201     # the name is free again


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
    assert r.status_code == 409 and "1 images" in r.json()["detail"]
    assert "multirotor" in class_names(client, pid)


def test_a_class_used_only_by_boxes_on_other_images_is_refused(client, project):
    pid, ids = project
    # Two classes on one image: it moves to the `mixed` folder, so fixed-wing and shahed have boxes but no images of their own.
    add(client, pid, ids["multirotor"], [box(ids["fixed-wing"], 0.3), box(ids["shahed"], 0.7)])
    stats = {r["class_name"]: r for r in client.get(f"/projects/{pid}/stats").json()}
    assert stats["fixed-wing"]["boxes"] == 1 and stats["fixed-wing"]["uploaded"] == 0 and stats["mixed"]["uploaded"] == 1
    r = client.delete(f"/projects/{pid}/classes/{ids['fixed-wing']}")
    assert r.status_code == 409 and "1 boxes" in r.json()["detail"]


def test_a_class_with_a_queued_or_running_scrape_job_is_refused_until_it_ends(client, project):
    pid, ids = project
    with client.session_factory() as db:
        job = Job(project_id=pid, type="scrape", params={"class_id": ids["shahed"], "query": "x"}, status=JobStatus.running)
        db.add(job)
        db.commit()
        job_id = job.id
    r = client.delete(f"/projects/{pid}/classes/{ids['shahed']}")
    assert r.status_code == 409 and "scrape job" in r.json()["detail"]
    assert client.delete(f"/projects/{pid}/classes/{ids['birds']}").status_code == 204            # other classes are unaffected
    with client.session_factory() as db:
        db.get(Job, job_id).status = JobStatus.succeeded
        db.commit()
    assert client.delete(f"/projects/{pid}/classes/{ids['shahed']}").status_code == 204


def test_a_new_class_never_gets_a_live_classs_index_after_a_removal(client, project):
    pid, ids = project
    assert indexes(client, pid) == {"multirotor": 0, "fixed-wing": 1, "shahed": 2}
    client.delete(f"/projects/{pid}/classes/{ids['fixed-wing']}")                                  # leaves a gap at 1
    client.post(f"/projects/{pid}/classes", json={"name": "vtol"})
    bulk = client.post(f"/projects/{pid}/classes/bulk", json={"names": ["quad-copter", "flying-wing"]}).json()
    assert [c["index"] for c in bulk["added"]] == [4, 5]
    idx = indexes(client, pid)
    assert idx == {"multirotor": 0, "shahed": 2, "vtol": 3, "quad-copter": 4, "flying-wing": 5}  # all unique, none reused
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
