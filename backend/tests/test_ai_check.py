"""The AI check: box maths, the verdict rules, the API and the batch job. Always with fake models: the real ones (SAM, YOLO) are heavy and optional."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app import jobs
from app.ai import service
from app.ai.check import POLICY_VERSION, Detection, LabelBox, Policy, run_check
from app.ai.device import Device
from app.ai.geometry import crop_window, iou, mask_to_box, side_on_input
from app.ai.sam import Refined
from app.ai.verdict import fingerprint, worst
from app.models import AiCheck, AiDismissal, Annotation, Image, Job, JobStatus

from .test_datasets import add, box

IMG = np.zeros((240, 320, 3), np.uint8)
LABEL = (0.4, 0.4, 0.6, 0.6)  # a 64 x 48 px box on the 320 x 240 test image


class FakeModels:
    """A detector that returns what it is told, and a SAM that shrinks the prompt's sides to `shrink` times (so a SAM box is recognisable)."""

    label = "fake"

    def __init__(self, detections=(), sam=True, edge=False, shrink=0.9):
        self.detections, self.sam, self.edge, self.shrink, self.calls = list(detections), sam, edge, shrink, 0

    def detect(self, image):
        self.calls += 1
        return list(self.detections)

    def refine(self, image, prompt=None, point=None):
        if not self.sam:
            return None
        if prompt is None:
            prompt = (point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5)
        w, h = prompt[2] - prompt[0], prompt[3] - prompt[1]
        m = (1 - self.shrink) / 2
        return Refined((prompt[0] + w * m, prompt[1] + h * m, prompt[2] - w * m, prompt[3] - h * m), 0.9, 100, self.edge)


@pytest.fixture(autouse=True)
def no_models_left_behind():
    yield
    service.set_models(None)


# ---------- geometry ----------

def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    assert iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0
    assert iou((0, 0, 2, 1), (1, 0, 3, 1)) == pytest.approx(1 / 3)


def test_crop_window_is_square_has_context_and_stays_inside_the_image():
    x1, y1, x2, y2 = crop_window((100, 100, 110, 110), 640, 480, context=4, min_side=96)
    assert (x2 - x1, y2 - y1) == (96, 96)  # a tiny box still gets the minimum window
    assert x1 <= 100 and x2 >= 110 and y1 <= 100 and y2 >= 110
    x1, y1, x2, y2 = crop_window((0, 0, 20, 20), 640, 480, context=4, min_side=96)
    assert (x1, y1) == (0, 0)  # at the corner the window moves in rather than leaving the image
    x1, y1, x2, y2 = crop_window((600, 440, 640, 480), 640, 480, context=4, min_side=96)
    assert (x2, y2) == (640, 480)
    assert crop_window((10, 10, 300, 300), 320, 240, context=4, min_side=96) == (35, 0, 275, 240)  # never bigger than the image: 240 square, centred on the box


def test_mask_to_box_and_empty_mask():
    m = np.zeros((10, 10), bool)
    assert mask_to_box(m) is None
    m[2:5, 3:8] = True
    assert mask_to_box(m) == (3, 2, 8, 5)


def test_side_on_input_scales_to_the_640_network_input():
    assert side_on_input((0, 0, 0.1, 0.1), 640, 640) == pytest.approx(64)
    assert side_on_input((0, 0, 0.1, 0.1), 320, 240) == pytest.approx(0.1 * 320 * 2 * (240 / 320) ** 0.5 * 1.0, rel=0.2)  # 320 wide -> x2


def test_fingerprint_ignores_order_and_notices_a_changed_box():
    a, b = (0.1, 0.1, 0.2, 0.2), (0.5, 0.5, 0.7, 0.7)
    assert fingerprint([a, b]) == fingerprint([b, a])
    assert fingerprint([a, b]) != fingerprint([a, (0.5, 0.5, 0.7, 0.71)])
    assert fingerprint([]) == fingerprint([])


def test_worst_orders_suspect_first_and_defaults_to_none():
    assert worst(["ok", "adjust", "too_small"]) == "adjust"
    assert worst(["ok", "suspect", "missing"]) == "suspect"
    assert worst([]) == "none"


# ---------- the verdict rules ----------

def check(labels, detections, **kw):
    return run_check(IMG, [LabelBox(i, b) for i, b in enumerate(labels)], FakeModels(detections, **kw))


def test_a_label_the_detector_confirms_is_ok_and_gets_no_suggestion():
    r = check([LABEL], [Detection(LABEL, 0.9)])
    assert r["verdict"] == "ok" and r["labels"][0]["verdict"] == "ok" and r["labels"][0]["suggested"] is None
    assert r["labels"][0]["detector_iou"] == 1 and r["missing"] == []


def test_a_label_that_only_partly_covers_the_drone_gets_the_tight_box_suggested():
    shifted = (0.5, 0.5, 0.7, 0.7)  # IoU with LABEL = 0.14 -> not the same drone ... use a milder shift
    shifted = (0.46, 0.46, 0.66, 0.66)  # IoU with LABEL about 0.47: related but sloppy
    r = check([shifted], [Detection(LABEL, 0.9)])
    item = r["labels"][0]
    assert item["verdict"] == "adjust" and r["verdict"] == "adjust"
    assert iou(tuple(item["suggested"]), LABEL) > 0.8  # SAM's polish of the detector's box, close to the true drone


def test_a_loose_label_around_a_detected_drone_is_adjust_with_the_tight_box_not_missing_or_suspect():
    drone = (0.45, 0.45, 0.55, 0.55)
    loose = (0.4, 0.4, 0.6, 0.6)  # 2x the drone: IoU only 0.25, but the detection sits wholly inside it
    r = check([loose], [Detection(drone, 0.9)])
    item = r["labels"][0]
    assert item["verdict"] == "adjust" and "loose around it" in item["reason"]
    assert iou(tuple(item["suggested"]), drone) > 0.8
    assert r["missing"] == []  # the drone inside the loose label is that label's drone, not an unlabelled one


def test_a_label_the_detector_cannot_confirm_says_whether_sam_sees_an_object_there():
    assert check([LABEL], [])["labels"][0]["sam_object"] is True  # the fake SAM finds an object in any box
    item = check([LABEL], [], sam=False)["labels"][0]
    assert item["verdict"] == "suspect" and item["sam_object"] is False and "SAM sees no object" in item["reason"]
    assert check([LABEL], [], edge=True)["labels"][0]["sam_object"] is False  # a mask running off the crop is not an object


def test_the_size_floor_is_16_px_on_the_640_input():
    barely_small = (0.5, 0.5, 0.52, 0.53)  # about 14 px on the 640 input
    assert check([barely_small], [])["labels"][0]["verdict"] == "too_small"
    big_enough = (0.5, 0.5, 0.55, 0.56)  # about 31 px
    assert check([big_enough], [])["labels"][0]["verdict"] == "suspect"


def test_a_mildly_loose_label_the_detector_agrees_with_is_ok_but_a_clearly_loose_one_is_adjust():
    detected = (0.42, 0.42, 0.58, 0.58)  # IoU 0.64 with LABEL: the detector agrees
    # SAM's tight box has IoU ~0.52 with the label: the ordinary looseness of a hand-drawn box, so no flag
    mild = check([LABEL], [Detection(detected, 0.9)], shrink=0.9)["labels"][0]
    assert mild["verdict"] == "ok" and mild["suggested"] is None
    clear = check([LABEL], [Detection(detected, 0.9)], shrink=0.8)["labels"][0]  # IoU ~0.41: clearly looser than the object
    assert clear["verdict"] == "adjust" and clear["suggested"] is not None


def test_a_label_the_detector_sees_elsewhere_is_suspect_and_the_detection_is_not_also_missing():
    far = (0.05, 0.05, 0.15, 0.15)
    r = check([far], [Detection(LABEL, 0.9)])
    assert r["labels"][0]["verdict"] == "suspect" and "elsewhere" in r["labels"][0]["reason"]
    assert iou(tuple(r["labels"][0]["suggested"]), LABEL) > 0.8
    assert r["missing"] == []  # explained by that label, not reported twice


def test_a_big_label_with_nothing_detected_is_suspect():
    r = check([LABEL], [])
    assert r["labels"][0]["verdict"] == "suspect" and r["labels"][0]["suggested"] is None


def test_a_tiny_label_with_nothing_detected_is_too_small_not_wrong():
    tiny = (0.5, 0.5, 0.51, 0.52)  # about 6 px on the 640 input
    r = check([tiny], [])
    assert r["labels"][0]["verdict"] == "too_small" and r["verdict"] == "too_small"


def test_a_confident_detection_with_no_label_is_a_missing_label_and_a_weak_one_is_ignored():
    r = check([LABEL], [Detection(LABEL, 0.9), Detection((0.05, 0.05, 0.15, 0.15), 0.8), Detection((0.8, 0.8, 0.9, 0.9), 0.3)])
    assert [m["score"] for m in r["missing"]] == [0.8]  # 0.3 is below the confident threshold
    assert r["verdict"] == "missing"


def test_an_image_without_boxes_gets_the_detections_as_proposals():
    r = check([], [Detection(LABEL, 0.9)])
    assert r["labels"] == [] and len(r["missing"]) == 1 and r["verdict"] == "missing"
    assert check([], [])["verdict"] == "none"


def test_detections_below_the_floor_score_are_ignored_altogether():
    r = check([LABEL], [Detection(LABEL, 0.1)])
    assert r["labels"][0]["verdict"] == "suspect"  # a 0.1 detection does not confirm anything


def test_sam_boxes_that_run_off_the_crop_or_are_missing_fall_back_to_the_detector_box():
    shifted = (0.46, 0.46, 0.66, 0.66)
    for models in ({"edge": True}, {"sam": False}):
        item = check([shifted], [Detection(LABEL, 0.9)], **models)["labels"][0]
        assert item["suggested"] == list(LABEL) and "detector" in item["reason"]


def test_the_policy_is_what_moves_the_thresholds():
    shifted = (0.46, 0.46, 0.66, 0.66)
    strict = run_check(IMG, [LabelBox(1, shifted)], FakeModels([Detection(LABEL, 0.9)]), Policy(related_iou=0.9))
    assert strict["labels"][0]["verdict"] == "suspect"  # no longer "the same drone"


# ---------- API ----------

@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def snapshot(client, image_id):
    d = client.get(f"/images/{image_id}").json()
    return d["review_status"], d["class_id"], [(a["id"], a["cx"], a["cy"], a["w"], a["h"], a["status"]) for a in d["annotations"]]


def test_status_says_why_the_ai_check_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(service, "missing_extras", lambda: "install the extras")
    s = client.get("/ai/status").json()
    assert s["available"] is False and s["reason"] == "install the extras"


def test_checking_an_image_without_the_extras_is_a_clear_503_and_stores_nothing(client, project, monkeypatch):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    monkeypatch.setattr(service, "missing_extras", lambda: "install the extras")
    r = client.post(f"/images/{img['id']}/ai-check")
    assert r.status_code == 503 and "install the extras" in r.json()["detail"]
    assert client.get(f"/images/{img['id']}/ai-check").status_code == 404


def test_a_check_stores_its_result_and_never_changes_boxes_or_review_status(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = add(client, pid, m, [box(m, 0.5, 0.5, 0.2, 0.2)])
    before = snapshot(client, img["id"])
    service.set_models(FakeModels([Detection((0.35, 0.35, 0.55, 0.55), 0.9), Detection((0.8, 0.1, 0.9, 0.2), 0.9)]))
    r = client.post(f"/images/{img['id']}/ai-check")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == f"fake · rules v{POLICY_VERSION}" and body["stale"] is False and body["verdict"] == "missing"  # an unlabelled drone outranks a loose box
    assert body["result"]["summary"] == {"ok": 0, "adjust": 1, "suspect": 0, "too_small": 0, "missing": 1, "labels": 1}
    assert snapshot(client, img["id"]) == before  # the boxes and the status are exactly as the person left them
    assert client.get(f"/images/{img['id']}/ai-check").json() == body
    assert client.get(f"/images/{img['id']}").json()["ai_status"] == "missing"
    with client.session_factory() as db:
        assert all(a.status.value == "reviewed" for a in db.query(Annotation))  # suggestions never became annotations


def test_checking_again_replaces_the_row(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = add(client, pid, m, [box(m, 0.5, 0.5, 0.2, 0.2)])
    service.set_models(FakeModels([Detection((0.4, 0.4, 0.6, 0.6), 0.9)]))
    client.post(f"/images/{img['id']}/ai-check")
    service.set_models(FakeModels([]))
    assert client.post(f"/images/{img['id']}/ai-check").json()["verdict"] == "suspect"
    with client.session_factory() as db:
        assert db.query(AiCheck).count() == 1


def test_editing_the_boxes_makes_the_check_stale_and_saving_the_same_boxes_does_not(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = add(client, pid, m, [box(m, 0.5, 0.5, 0.2, 0.2)])
    service.set_models(FakeModels([Detection((0.4, 0.4, 0.6, 0.6), 0.9)]))
    client.post(f"/images/{img['id']}/ai-check")
    same = {"boxes": [{"class_id": m, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}], "review_status": "reviewed"}
    client.put(f"/images/{img['id']}/annotations", json=same)
    assert client.get(f"/images/{img['id']}").json()["ai_status"] == "ok"
    moved = {"boxes": [{"class_id": m, "cx": 0.55, "cy": 0.5, "w": 0.2, "h": 0.2}], "review_status": "reviewed"}
    client.put(f"/images/{img['id']}/annotations", json=moved)
    assert client.get(f"/images/{img['id']}").json()["ai_status"] == "stale"
    assert client.get(f"/images/{img['id']}/ai-check").json()["stale"] is True
    client.put(f"/images/{img['id']}/annotations", json=same)  # put the boxes back: the old check is true again
    assert client.get(f"/images/{img['id']}").json()["ai_status"] == "ok"


def test_checking_again_after_an_edit_makes_the_check_current(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = add(client, pid, m, [box(m, 0.5, 0.5, 0.2, 0.2)])
    service.set_models(FakeModels([Detection((0.4, 0.4, 0.6, 0.6), 0.9)]))
    client.post(f"/images/{img['id']}/ai-check")
    client.put(f"/images/{img['id']}/annotations", json={"boxes": [{"class_id": m, "cx": 0.55, "cy": 0.5, "w": 0.2, "h": 0.2}], "review_status": "reviewed"})
    assert client.get(f"/images/{img['id']}").json()["ai_status"] == "stale"
    again = client.post(f"/images/{img['id']}/ai-check").json()
    assert again["stale"] is False and client.get(f"/images/{img['id']}").json()["ai_status"] != "stale"


def test_a_check_made_under_older_rules_is_redone_by_the_batch_job(client, project, monkeypatch):
    pid, ids = project
    m = ids["multirotor"]
    img = add(client, pid, m, [box(m)])
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    fake = FakeModels([Detection((0.4, 0.45, 0.6, 0.55), 0.9)])
    service.set_models(fake)
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    run_one_job(client)
    with client.session_factory() as db:
        db.get(AiCheck, img["id"]).model = "fake · rules v1"  # as if made before the rules changed
        db.commit()
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    status, progress, _ = run_one_job(client)
    assert progress["skipped"] == 0 and fake.calls == 2
    assert client.get(f"/images/{img['id']}/ai-check").json()["model"].endswith(f"rules v{POLICY_VERSION}")


def test_the_image_list_filters_on_the_ai_check(client, project):
    pid, ids = project
    m = ids["multirotor"]
    good = add(client, pid, m, [box(m)])
    bad = add(client, pid, m, [box(m)])
    never = add(client, pid, m, [box(m)])
    edited = add(client, pid, m, [box(m)])
    for img, dets in ((good, [Detection((0.4, 0.45, 0.6, 0.55), 0.9)]), (bad, []), (edited, [Detection((0.4, 0.45, 0.6, 0.55), 0.9)])):
        service.set_models(FakeModels(dets))
        client.post(f"/images/{img['id']}/ai-check")
    client.put(f"/images/{edited['id']}/annotations", json={"boxes": [{"class_id": m, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}], "review_status": "reviewed"})

    def ids_for(ai):
        return sorted(r["id"] for r in client.get(f"/projects/{pid}/images", params={"ai": ai}).json())

    assert ids_for("ok") == [good["id"]] and ids_for("flagged") == [bad["id"]]
    assert ids_for("unchecked") == [never["id"]] and ids_for("stale") == [edited["id"]]
    assert client.get(f"/projects/{pid}/images", params={"ai": "nonsense"}).status_code == 422


# ---------- dismissing the AI's doubts ----------

def flagged_image(client, pid, m):
    """An image whose current AI check is flagged: nothing is detected at its (big) box, so the check is suspect."""
    img = add(client, pid, m, [box(m)])
    service.set_models(FakeModels([]))
    assert client.post(f"/images/{img['id']}/ai-check").json()["verdict"] == "suspect"
    return img


def save_boxes(client, img, m, cx=0.5, dismiss=False, status="reviewed"):
    body = {"boxes": [{"class_id": m, "cx": cx, "cy": 0.5, "w": 0.2, "h": 0.1}], "review_status": status}
    if dismiss:
        body["ai_dismiss"] = True
    r = client.put(f"/images/{img['id']}/annotations", json=body)
    assert r.status_code == 200
    return r.json()


def listed(client, pid, ai):
    rows = client.get(f"/projects/{pid}/images", params={"ai": ai, "limit": 500}).json()
    return sorted(r["id"] for r in rows), {r["ai_status"] for r in rows}


def test_accepting_an_image_the_ai_flagged_takes_it_off_the_needs_a_look_list(client, project):
    pid, ids = project
    m = ids["multirotor"]
    a, b = flagged_image(client, pid, m), flagged_image(client, pid, m)
    assert save_boxes(client, a, m, dismiss=True)["ai_status"] == "dismissed"
    assert listed(client, pid, "flagged")[0] == [b["id"]] and listed(client, pid, "dismissed")[0] == [a["id"]]
    check = client.get(f"/images/{a['id']}/ai-check").json()
    assert check["dismissed"] is True and check["verdict"] == "suspect" and check["stale"] is False  # the AI's own opinion is untouched
    assert client.get(f"/images/{b['id']}/ai-check").json()["dismissed"] is False
    # the list filter (SQL) and the status on each row (Python) are one rule written twice: they must agree
    assert listed(client, pid, "flagged")[1] <= {"suspect", "missing", "adjust"}
    assert listed(client, pid, "dismissed")[1] == {"dismissed"}


def test_saving_without_the_flag_leaves_the_image_flagged(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m)  # plain Save / stepping past an image is not a decision about the AI's doubts
    assert listed(client, pid, "flagged")[0] == [img["id"]] and listed(client, pid, "dismissed")[0] == []


def test_rejecting_an_image_the_ai_flagged_takes_it_off_the_list_too(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m, dismiss=True, status="rejected")
    assert listed(client, pid, "flagged")[0] == [] and listed(client, pid, "dismissed")[0] == [img["id"]]


def test_nothing_is_recorded_for_an_image_the_ai_did_not_flag(client, project):
    pid, ids = project
    m = ids["multirotor"]
    never = add(client, pid, m, [box(m)])
    fine = add(client, pid, m, [box(m)])
    service.set_models(FakeModels([Detection((0.4, 0.45, 0.6, 0.55), 0.9)]))
    assert client.post(f"/images/{fine['id']}/ai-check").json()["verdict"] == "ok"
    save_boxes(client, never, m, dismiss=True)
    save_boxes(client, fine, m, dismiss=True)
    with client.session_factory() as db:
        assert db.query(AiDismissal).count() == 0
    # so a later check can still raise doubts the person never saw
    service.set_models(FakeModels([]))
    client.post(f"/images/{fine['id']}/ai-check")
    assert listed(client, pid, "flagged")[0] == [fine["id"]]


def test_checking_the_same_boxes_again_does_not_bring_a_dismissed_image_back(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m, dismiss=True)
    service.set_models(FakeModels([]))  # a forced re-run with the same result
    client.post(f"/images/{img['id']}/ai-check")
    assert listed(client, pid, "flagged")[0] == [] and listed(client, pid, "dismissed")[0] == [img["id"]]
    # and one that now finds something else about the same boxes (an unlabelled drone: "missing")
    service.set_models(FakeModels([Detection((0.4, 0.45, 0.6, 0.55), 0.9), Detection((0.8, 0.1, 0.9, 0.2), 0.9)]))
    assert client.post(f"/images/{img['id']}/ai-check").json()["verdict"] == "missing"
    assert listed(client, pid, "flagged")[0] == [] and client.get(f"/images/{img['id']}").json()["ai_status"] == "dismissed"


def test_boxes_edited_while_dismissing_stay_off_the_list_when_they_are_checked_again(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    assert save_boxes(client, img, m, cx=0.55, dismiss=True)["ai_status"] == "stale"  # accepted a suggestion, then Accept & next
    assert listed(client, pid, "flagged")[0] == [] and listed(client, pid, "dismissed")[0] == []  # stale: neither list
    service.set_models(FakeModels([]))
    client.post(f"/images/{img['id']}/ai-check")  # the new boxes are checked and flagged again, but the person already reviewed them
    assert listed(client, pid, "flagged")[0] == [] and listed(client, pid, "dismissed")[0] == [img["id"]]


def test_dismissing_again_after_editing_the_boxes_moves_the_record_to_the_new_boxes(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m, dismiss=True)
    save_boxes(client, img, m, cx=0.6, dismiss=True)  # reviewed again after an edit: the decision now covers the new boxes
    service.set_models(FakeModels([]))
    client.post(f"/images/{img['id']}/ai-check")
    assert listed(client, pid, "flagged")[0] == [] and listed(client, pid, "dismissed")[0] == [img["id"]]
    with client.session_factory() as db:
        assert db.query(AiDismissal).count() == 1  # replaced, not added to


def test_a_dismissed_image_whose_boxes_the_ai_later_approves_shows_as_ok_not_dismissed(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m, dismiss=True)
    service.set_models(FakeModels([Detection((0.4, 0.45, 0.6, 0.55), 0.9)]))  # a better detector now finds the drone at the box
    check = client.post(f"/images/{img['id']}/ai-check").json()
    assert check["verdict"] == "ok" and check["dismissed"] is False
    assert client.get(f"/images/{img['id']}").json()["ai_status"] == "ok"
    assert listed(client, pid, "ok")[0] == [img["id"]] and listed(client, pid, "dismissed")[0] == []


def test_a_dismissed_image_is_listed_again_once_its_boxes_change_and_the_new_boxes_are_flagged(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m, dismiss=True)
    save_boxes(client, img, m, cx=0.6)  # edited later, without a new decision
    service.set_models(FakeModels([]))
    client.post(f"/images/{img['id']}/ai-check")
    assert listed(client, pid, "flagged")[0] == [img["id"]] and client.get(f"/images/{img['id']}").json()["ai_status"] == "suspect"


def test_undoing_a_dismissal_puts_the_image_back_and_is_harmless_to_repeat(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m, dismiss=True)
    out = client.delete(f"/images/{img['id']}/ai-dismiss")
    assert out.status_code == 200 and out.json()["dismissed"] is False
    assert listed(client, pid, "flagged")[0] == [img["id"]]
    assert client.delete(f"/images/{img['id']}/ai-dismiss").status_code == 200
    unchecked = add(client, pid, m, [box(m)])
    assert client.delete(f"/images/{unchecked['id']}/ai-dismiss").status_code == 404
    assert client.delete("/images/99999/ai-dismiss").status_code == 404


def test_deleting_an_image_removes_its_dismissal(client, project):
    pid, ids = project
    m = ids["multirotor"]
    img = flagged_image(client, pid, m)
    save_boxes(client, img, m, dismiss=True)
    assert client.delete(f"/images/{img['id']}").status_code == 204
    with client.session_factory() as db:
        assert db.query(AiDismissal).count() == 0


def test_deleting_an_image_removes_its_check(client, project):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    service.set_models(FakeModels([]))
    client.post(f"/images/{img['id']}/ai-check")
    assert client.delete(f"/images/{img['id']}").status_code == 204
    with client.session_factory() as db:
        assert db.query(AiCheck).count() == 0


def test_click_to_box_returns_sams_box_and_stores_nothing(client, project):
    pid, ids = project
    img = add(client, pid, ids["multirotor"], [])
    service.set_models(FakeModels())
    r = client.post(f"/images/{img['id']}/ai-box", json={"x": 0.5, "y": 0.5}).json()
    assert r["found"] is True and 0 <= r["box"][0] < r["box"][2] <= 1
    assert client.post(f"/images/{img['id']}/ai-box", json={"box": [0.4, 0.4, 0.6, 0.6]}).json()["found"] is True
    service.set_models(FakeModels(edge=True))
    assert client.post(f"/images/{img['id']}/ai-box", json={"x": 0.5, "y": 0.5}).json() == {"found": False, "box": None, "score": None}
    assert client.post(f"/images/{img['id']}/ai-box", json={}).status_code == 422
    assert client.post(f"/images/{img['id']}/ai-box", json={"x": 0.5}).status_code == 422
    with client.session_factory() as db:
        assert db.query(AiCheck).count() == 0


def test_the_ai_endpoints_are_closed_to_the_network(client, project):
    from app.main import app
    from fastapi.testclient import TestClient
    from .helpers import from_ip

    lan = TestClient(from_ip(app, "10.0.0.7"))
    assert lan.get("/ai/status").status_code == 403
    assert lan.post("/images/1/ai-check").status_code == 403


# ---------- the batch job ----------

def run_one_job(client):
    with client.session_factory() as db:
        job_id = db.query(Job).order_by(Job.id.desc()).first().id
    jobs.run_job(job_id, client.session_factory)
    with client.session_factory() as db:
        job = db.get(Job, job_id)
        return job.status, dict(job.progress), job.error


def test_the_batch_job_checks_the_selected_images_skips_done_ones_and_leaves_negatives_alone(client, project, monkeypatch):
    pid, ids = project
    m = ids["multirotor"]
    add(client, pid, m, [box(m)])
    add(client, pid, m, [box(m)])
    add(client, pid, ids["birds"])  # a hard-negative image: nothing to check
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    fake = FakeModels([Detection((0.4, 0.45, 0.6, 0.55), 0.9)])
    service.set_models(fake)
    assert client.post(f"/projects/{pid}/jobs/ai-check", json={}).status_code == 201
    status, progress, error = run_one_job(client)
    assert status == JobStatus.succeeded and error is None
    assert progress["total"] == 2 and progress["done"] == 2 and progress["skipped"] == 0 and fake.calls == 2
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    status, progress, _ = run_one_job(client)
    assert progress["skipped"] == 2 and fake.calls == 2  # nothing changed since the last check: no model calls
    client.post(f"/projects/{pid}/jobs/ai-check", json={"force": True})
    assert run_one_job(client)[1]["skipped"] == 0 and fake.calls == 4


def test_the_job_list_can_be_limited_to_one_type_so_panels_show_only_their_own(client, project, monkeypatch):
    pid, ids = project
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    with client.session_factory() as db:
        db.add(Job(project_id=pid, type="scrape", params={"query": "x"}))
        db.commit()
    kinds = lambda **q: sorted(j["type"] for j in client.get(f"/projects/{pid}/jobs", params=q).json())
    assert kinds() == ["ai_check", "scrape"]
    assert kinds(type="ai_check") == ["ai_check"] and kinds(type="scrape") == ["scrape"]
    assert client.get(f"/projects/{pid}/jobs", params={"type": "Bad Type!"}).status_code == 422


def test_the_batch_job_rechecks_an_image_whose_boxes_were_edited(client, project, monkeypatch):
    pid, ids = project
    m = ids["multirotor"]
    img = add(client, pid, m, [box(m)])
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    fake = FakeModels([Detection((0.4, 0.45, 0.6, 0.55), 0.9)])
    service.set_models(fake)
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    run_one_job(client)
    client.put(f"/images/{img['id']}/annotations", json={"boxes": [{"class_id": m, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}], "review_status": "reviewed"})
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    assert run_one_job(client)[1]["skipped"] == 0 and fake.calls == 2


def test_one_unreadable_image_does_not_stop_the_batch(client, project, monkeypatch):
    pid, ids = project
    m = ids["multirotor"]
    first = add(client, pid, m, [box(m)])
    add(client, pid, m, [box(m)])
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    service.set_models(FakeModels([Detection((0.4, 0.45, 0.6, 0.55), 0.9)]))
    real = service.load_rgb

    def flaky(image):
        if image.id == first["id"]:
            raise OSError("cannot read this one")
        return real(image)

    monkeypatch.setattr(service, "load_rgb", flaky)
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    status, progress, _ = run_one_job(client)
    assert status == JobStatus.succeeded and progress["failed"] == 1 and progress["done"] == 2 and "cannot read" in progress["last_error"]


def test_the_batch_job_stops_when_cancelled(client, project, monkeypatch):
    pid, ids = project
    m = ids["multirotor"]
    for _ in range(3):
        add(client, pid, m, [box(m)])
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    service.set_models(FakeModels([]))
    job_id = client.post(f"/projects/{pid}/jobs/ai-check", json={}).json()["id"]
    with client.session_factory() as db:
        db.get(Job, job_id).cancel_requested = True
        db.commit()
    status, progress, _ = run_one_job(client)
    assert status == JobStatus.cancelled and progress["done"] == 0


def test_creating_a_batch_job_without_the_extras_is_refused_up_front(client, project, monkeypatch):
    pid, _ = project
    monkeypatch.setattr(service, "missing_extras", lambda: "install the extras")
    r = client.post(f"/projects/{pid}/jobs/ai-check", json={})
    assert r.status_code == 503
    assert client.get(f"/projects/{pid}/jobs").json() == []


def test_the_batch_job_fails_clearly_when_there_is_no_detector_yet(client, project, monkeypatch):
    pid, ids = project
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    monkeypatch.setattr(service, "detector_weights", lambda db: None)
    service.set_models(None)
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    status, _, error = run_one_job(client)
    assert status == JobStatus.failed and "No detector yet" in error


# ---------- the device ----------

def test_a_call_the_gpu_refuses_runs_on_the_cpu_once_without_giving_up_the_gpu():
    d = Device("mps")
    calls = []

    def fn(dev):
        calls.append(dev)
        if dev == "mps" and len(calls) == 1:
            raise RuntimeError("unsupported op")
        return f"done on {dev}"

    assert d.call(fn) == "done on cpu" and d.fallbacks == 1 and "unsupported op" in d.last_error
    assert d.name == "mps" and d.call(fn) == "done on mps"  # the next call tries the GPU again


def test_five_gpu_failures_in_a_row_move_everything_to_the_cpu_and_a_success_resets_the_count():
    d = Device("mps")
    bad = lambda dev: (_ for _ in ()).throw(RuntimeError("no")) if dev == "mps" else "cpu result"
    for _ in range(4):
        assert d.call(bad) == "cpu result"
    assert d.name == "mps"
    assert d.call(lambda dev: dev) == "mps"  # a success in between starts the count again
    for _ in range(4):
        d.call(bad)
    assert d.name == "mps"  # four more failures since the success: still not five in a row
    d.call(bad)
    assert d.name == "cpu" and d.fallbacks == 9
    assert d.summary()["device"] == "cpu"


def test_a_cpu_device_never_counts_a_failure_as_a_fallback():
    d = Device("cpu")
    with pytest.raises(RuntimeError):
        d.call(lambda dev: (_ for _ in ()).throw(RuntimeError("real bug")))
    assert d.fallbacks == 0


def test_status_and_job_progress_report_the_device(client, project, monkeypatch):
    pid, ids = project
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    monkeypatch.setattr(service, "detector_weights", lambda db: (Path("x.pt"), "yolo run 1"))
    service.set_models(FakeModels([]))
    s = client.get("/ai/status").json()
    assert s["available"] is True and s["fallbacks"] == 0 and s["last_fallback"] is None
    client.post(f"/projects/{pid}/jobs/ai-check", json={})
    assert "device" in run_one_job(client)[1]  # None for the fakes; the real models report their device and any fallbacks


# ---------- the backend must not need the AI extras ----------

def test_the_backend_imports_without_loading_torch_or_ultralytics():
    code = "import sys; import app.main, app.jobs, app.ai.service; assert 'torch' not in sys.modules and 'ultralytics' not in sys.modules"
    done = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
