"""Phase 4: rejecting several images at once, running the AI on a selection, and the per-place AI counts the overview and the drawer badge show."""

import pytest

from app.ai import service
from app.ai.check import Detection
from app.ai.job import select_images
from app.config import settings
from app.models import AiDismissal, Annotation, AnnotationStatus, Image

from .test_ai_autolabel import FakeModels, det, start, unlabelled
from .test_ai_check import flagged_image, listed, run_one_job, save_boxes
from .test_datasets import add, box
from .test_upload_folders import jpg, up


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ai_dir", str(tmp_path / "ai"))
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    yield
    service.set_models(None)


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def reject(client, pid, ids):
    return client.post(f"/projects/{pid}/images/reject", json={"image_ids": ids})


def state(client, image_id):
    """Everything a reject is supposed to settle the same way, however it was done."""
    d = client.get(f"/images/{image_id}").json()
    with client.session_factory() as db:
        image = db.get(Image, image_id)
        dismissal = db.get(AiDismissal, image_id)
        boxes = sorted((a.class_id, round(a.cx, 4), round(a.cy, 4), round(a.w, 4), round(a.h, 4), a.status.value) for a in image.annotations)
        stale = image.ai_check.stale if image.ai_check else None
    return {"status": d["review_status"], "ai_status": d["ai_status"], "ai_labelled": d["ai_labelled"], "stale": stale,
            "dismissed_boxes": dismissal is not None, "boxes": boxes}


# ---------- bulk reject ----------

def test_bulk_reject_rejects_the_images_and_keeps_their_boxes(client, project):
    pid, ids = project
    m = ids["multirotor"]
    a, b, c = (add(client, pid, m, [box(m)])["id"] for _ in range(3))
    before = state(client, a)["boxes"]
    assert reject(client, pid, [a, b]).json() == {"rejected": 2}
    assert state(client, a)["status"] == "rejected" and state(client, b)["status"] == "rejected" and state(client, c)["status"] == "reviewed"
    assert [x[:5] for x in state(client, a)["boxes"]] == [x[:5] for x in before]


def test_a_bulk_reject_leaves_the_ai_lists_exactly_as_a_single_reject_does(client, project):
    pid, ids = project
    m = ids["multirotor"]
    single, bulk = flagged_image(client, pid, m), flagged_image(client, pid, m)           # both flagged "needs a look"
    save_boxes(client, single, m, status="rejected", dismiss=True)                         # what the review screen's Reject sends
    reject(client, pid, [bulk["id"]])
    assert state(client, bulk["id"])["boxes"] == state(client, single["id"])["boxes"]
    keys = ("status", "ai_status", "ai_labelled", "dismissed_boxes")
    assert {k: state(client, bulk["id"])[k] for k in keys} == {k: state(client, single["id"])[k] for k in keys}
    assert state(client, bulk["id"])["ai_status"] == "dismissed"
    assert listed(client, pid, "flagged")[0] == [] and sorted(listed(client, pid, "dismissed")[0]) == sorted([single["id"], bulk["id"]])


def test_a_bulk_reject_ends_ai_drew_these_boxes_like_a_single_reject(client, project):
    pid, ids = project
    one, two = unlabelled(client, pid, ids), unlabelled(client, pid, ids)
    service.set_models(FakeModels([det((0.4, 0.4, 0.6, 0.6))]))
    start(client, pid)
    assert client.get(f"/images/{one}").json()["ai_labelled"] and client.get(f"/images/{two}").json()["ai_labelled"]
    boxes = [{"class_id": ids["multirotor"], "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    client.put(f"/images/{one}/annotations", json={"boxes": boxes, "review_status": "rejected", "ai_dismiss": True})
    reject(client, pid, [two])
    assert state(client, one)["ai_labelled"] is False and state(client, two)["ai_labelled"] is False
    assert state(client, one)["status"] == state(client, two)["status"] == "rejected"
    assert client.get(f"/projects/{pid}/images", params={"ai": "autolabelled"}).json() == []            # neither lingers in the AI-drew-the-boxes list


def test_a_bulk_reject_marks_a_stale_check_the_same_way(client, project):
    pid, ids = project
    m = ids["multirotor"]
    a = flagged_image(client, pid, m)
    save_boxes(client, a, m, cx=0.3)                                                      # boxes changed after the check: stale
    b = flagged_image(client, pid, m)
    reject(client, pid, [a["id"], b["id"]])
    assert state(client, a["id"])["stale"] is True and state(client, a["id"])["ai_status"] == "stale"      # still out of date; rejecting did not make it current
    assert state(client, b["id"])["stale"] is False and state(client, b["id"])["ai_status"] == "dismissed"  # boxes unchanged: the doubt is settled


def test_bulk_reject_is_all_or_nothing_and_stays_inside_the_project(client, project):
    pid, ids = project
    m = ids["multirotor"]
    mine = add(client, pid, m, [box(m)])["id"]
    other = client.post("/projects", json={"name": "other"}).json()
    foreign = add(client, other["id"], {c["name"]: c["id"] for c in other["classes"]}["multirotor"], [])["id"]
    for bad in ([mine, foreign], [mine, 99999]):
        r = reject(client, pid, bad)
        assert r.status_code == 422 and "not images of this project" in r.json()["detail"]
    assert state(client, mine)["status"] == "reviewed" and client.get(f"/images/{foreign}").json()["review_status"] == "reviewed"
    assert reject(client, pid, []).status_code == 422                                      # an empty list is not "everything"
    big = reject(client, pid, list(range(1, 502)))
    assert big.status_code == 422 and "at most 500" in str(big.json())                      # capped (not just "those ids don't exist")
    assert reject(client, 9999, [mine]).status_code == 404
    assert reject(client, pid, [mine, mine]).json() == {"rejected": 1}                     # duplicates count once


# ---------- running the AI on a selection ----------

def test_an_ai_job_can_be_limited_to_chosen_images(client, project):
    pid, ids = project
    a, b, c = (unlabelled(client, pid, ids) for _ in range(3))
    with client.session_factory() as db:
        assert sorted(select_images(db, pid, {"image_ids": [a, c]})) == [a, c]
        assert select_images(db, pid, {"image_ids": []}) == []                            # empty means none, never everything
        assert len(select_images(db, pid, {})) == 3
    service.set_models(FakeModels([det((0.4, 0.4, 0.6, 0.6))]))
    _, progress, _ = start(client, pid, image_ids=[b])
    assert progress["total"] == 1 and [x["ai_labelled"] for x in (client.get(f"/images/{i}").json() for i in (a, b, c))] == [False, True, False]


def test_chosen_images_still_obey_the_rules_of_the_mode_and_the_project(client, project):
    pid, ids = project
    m = ids["multirotor"]
    open_ = unlabelled(client, pid, ids)
    done = add(client, pid, m, [box(m)])["id"]
    bird = add(client, pid, ids["birds"])["id"]
    other = client.post("/projects", json={"name": "other"}).json()
    foreign = add(client, other["id"], {c["name"]: c["id"] for c in other["classes"]}["multirotor"], status="pending")["id"]
    with client.session_factory() as db:
        everything = [open_, done, bird, foreign]
        assert select_images(db, pid, {"image_ids": everything, "mode": "autolabel"}) == [open_]     # pending + no boxes only
        assert sorted(select_images(db, pid, {"image_ids": everything})) == sorted([open_, done])    # never negatives, never another project's


def test_an_ai_job_with_an_empty_or_oversize_selection_is_refused(client, project):
    pid, _ = project
    for bad in ([], list(range(1, 502))):
        for path in (f"/projects/{pid}/jobs/ai-check", f"/projects/{pid}/jobs/ai-check/preview"):
            r = client.post(path, json={"image_ids": bad})
            assert r.status_code == 422 and ("at least 1" in str(r.json()) or "at most 500" in str(r.json()))
    assert client.get(f"/projects/{pid}/jobs").json() == []


# ---------- the AI numbers per place ----------

def folder_rows(client, pid):
    return {(r["source"], r["folder"], r["class_id"]): r for r in client.get(f"/projects/{pid}/folders").json()}


def in_folder(client, pid, cid, folder):
    return up(client, pid, cid, jpg(1), folder=folder).json()["added"][0]["id"]


def test_the_counts_per_place_are_the_gallery_filters_counted(client, project):
    pid, ids = project
    m = ids["multirotor"]
    # place "f1": one of each state; place "f2": loose images of another class
    ok, flagged, dismissed, stale, unchecked = (in_folder(client, pid, m, "f1") for _ in range(5))
    for i in (ok, flagged, dismissed, stale):
        client.put(f"/images/{i}/annotations", json={"boxes": [{"class_id": m, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]})
    service.set_models(FakeModels([Detection((0.4, 0.4, 0.6, 0.6), 0.9)]))
    client.post(f"/images/{ok}/ai-check")                                                  # the detector agrees: ok
    service.set_models(FakeModels([]))
    for i in (flagged, dismissed, stale):
        client.post(f"/images/{i}/ai-check")                                               # nothing found at the box: suspect
    client.put(f"/images/{dismissed}/annotations", json={"boxes": [{"class_id": m, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}], "ai_dismiss": True})
    client.put(f"/images/{stale}/annotations", json={"boxes": [{"class_id": m, "cx": 0.3, "cy": 0.3, "w": 0.2, "h": 0.2}]})
    other = [up(client, pid, ids["fixed-wing"], jpg(1)).json()["added"][0]["id"] for _ in range(2)]
    service.set_models(FakeModels([]))
    client.post(f"/images/{other[0]}/ai-check")                                            # no boxes, nothing found: "none" counts as ok

    rows = folder_rows(client, pid)
    f1 = rows[("uploaded", "f1", m)]
    assert (f1["total"], f1["ai_ok"], f1["ai_flagged"], f1["ai_unchecked"]) == (5, 1, 1, 1)           # dismissed and stale are neither ok nor flagged
    loose = rows[("uploaded", None, ids["fixed-wing"])]
    assert (loose["total"], loose["ai_ok"], loose["ai_flagged"], loose["ai_unchecked"]) == (2, 1, 0, 1)
    # and they are what the gallery lists when it filters by the same place and state
    for state_, field in (("ok", "ai_ok"), ("flagged", "ai_flagged"), ("unchecked", "ai_unchecked")):
        got = len(client.get(f"/projects/{pid}/images", params={"ai": state_, "folder": "f1", "limit": 500}).json())
        assert got == f1[field], state_
    assert [i["id"] for i in client.get(f"/projects/{pid}/images", params={"ai": "flagged", "folder": "f1"}).json()] == [flagged]


def test_the_counts_do_not_multiply_with_the_number_of_boxes(client, project):
    pid, ids = project
    m = ids["multirotor"]
    image = in_folder(client, pid, m, "f1")
    boxes = [{"class_id": m, "cx": 0.1 + 0.15 * k, "cy": 0.5, "w": 0.1, "h": 0.1} for k in range(5)]
    client.put(f"/images/{image}/annotations", json={"boxes": boxes})
    service.set_models(FakeModels([]))
    client.post(f"/images/{image}/ai-check")
    row = folder_rows(client, pid)[("uploaded", "f1", m)]
    assert row["total"] == 1 and row["ai_flagged"] == 1 and row["ai_ok"] == 0
