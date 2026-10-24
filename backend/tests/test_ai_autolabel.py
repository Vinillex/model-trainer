"""Phase 3 of the AI section: auto-labelling unlabelled images, choosing the detector and SAM size per run, and storing uploaded weights.
Always with fake models. The hard rules under test: auto boxes never replace a person's boxes, never reach a training set before a person
accepts them, and an uploaded .pt is stored but never opened."""

import io

import pytest

from app import storage
from app.ai import job as ai_job
from app.ai import registry, service
from app.ai.check import Detection
from app.config import settings
from app.models import AiCheck, Annotation, AnnotationStatus, Image, JobStatus, RunStage, TrainingRun

from .test_ai_check import FakeModels, run_one_job
from .test_datasets import add, box

ZIP = b"PK\x03\x04" + b"\x00" * 64  # what a PyTorch .pt starts with


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ai_dir", str(tmp_path / "ai"))  # run weights and uploads must never land in the real backend/data/ai
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    service._real.clear()
    yield
    service.set_models(None)
    service._real.clear()


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def det(box_, score=0.9, name="multirotor"):
    return Detection(box_, score, name)


BOX = (0.4, 0.4, 0.6, 0.6)


def unlabelled(client, pid, ids, cls="multirotor"):
    return add(client, pid, ids[cls], status="pending")["id"]


def start(client, pid, **body):
    r = client.post(f"/projects/{pid}/jobs/ai-check", json={"mode": "autolabel", **body})
    assert r.status_code == 201, r.text
    return run_one_job(client)


def boxes_of(client, image_id):
    with client.session_factory() as db:
        return [(a.class_id, a.status, a.confidence, round(a.cx, 3), round(a.cy, 3), round(a.w, 3), round(a.h, 3)) for a in db.get(Image, image_id).annotations]


def detail(client, image_id):
    return client.get(f"/images/{image_id}").json()


# ---------- auto-labelling ----------

def test_autolabel_draws_auto_boxes_with_the_class_found_by_name_and_leaves_the_image_pending(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX, 0.9, "fixed-wing")], shrink=1.0))
    status, progress, error = start(client, pid)
    assert status == JobStatus.succeeded and error is None
    assert boxes_of(client, image) == [(ids["fixed-wing"], AnnotationStatus.auto, 0.9, 0.5, 0.5, 0.2, 0.2)]   # filed by the detector's class NAME
    assert detail(client, image)["review_status"] == "pending"                                                  # a person still has to accept it
    assert (progress["mode"], progress["labelled"], progress["boxes"], progress["nothing"], progress["failed"]) == ("autolabel", 1, 1, 0, 0)


def test_autolabel_boxes_are_tightened_by_sam_or_fall_back_to_the_detectors_box(client, project):
    pid, ids = project
    a, b, c = (unlabelled(client, pid, ids) for _ in range(3))
    service.set_models(FakeModels([det(BOX)], shrink=0.8))                       # a believable SAM box (it overlaps the detector's by 64%): used
    start(client, pid, limit=1)
    assert boxes_of(client, a)[0][5:] == (0.16, 0.16)
    service.set_models(FakeModels([det(BOX)], shrink=0.5))                       # SAM's box overlaps the detector's by only 25%: not believable, not used
    start(client, pid, limit=1)
    assert boxes_of(client, b)[0][5:] == (0.2, 0.2)
    service.set_models(FakeModels([det(BOX)], sam=False))                        # SAM finds nothing: the detector's own box is kept
    start(client, pid)
    assert boxes_of(client, c)[0][5:] == (0.2, 0.2)


def test_autolabel_only_touches_pending_images_without_boxes(client, project):
    pid, ids = project
    m = ids["multirotor"]
    open_ = unlabelled(client, pid, ids)
    mine = add(client, pid, m, status="pending")["id"]
    assert client.put(f"/images/{mine}/annotations", json={"boxes": [{"class_id": m, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1}]}).status_code == 200  # pending, but a person drew a box
    background = add(client, pid, m, [], status="reviewed")["id"]                 # reviewed with no boxes: a deliberate background picture
    rejected = add(client, pid, m, [], status="rejected")["id"]
    bird = add(client, pid, ids["birds"])["id"]                                   # a hard-negative image
    before = {i: boxes_of(client, i) for i in (mine, background, rejected, bird)}
    service.set_models(FakeModels([det(BOX)]))
    _, progress, _ = start(client, pid)
    assert progress["total"] == 1 and boxes_of(client, open_) != []
    assert {i: boxes_of(client, i) for i in before} == before
    assert detail(client, background)["review_status"] == "reviewed" and detail(client, rejected)["review_status"] == "rejected"


def test_a_second_run_does_not_label_again_and_images_with_nothing_found_stay_eligible(client, project):
    pid, ids = project
    found, empty = unlabelled(client, pid, ids), unlabelled(client, pid, ids)
    fake = FakeModels([det(BOX)])
    service.set_models(fake)
    start(client, pid, limit=1)                                                   # the lower id gets the box
    service.set_models(FakeModels([]))
    _, progress, _ = start(client, pid)                                           # only the image without boxes is left to try
    assert progress["total"] == 1 and progress["nothing"] == 1 and progress["labelled"] == 0
    assert len(boxes_of(client, found)) == 1 and boxes_of(client, empty) == []
    service.set_models(FakeModels([det(BOX)]))
    _, progress, _ = start(client, pid)
    assert progress["labelled"] == 1 and len(boxes_of(client, empty)) == 1


def test_detections_below_the_score_or_of_an_unknown_class_are_left_out_and_counted(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX, 0.9, "multirotor"), det((0.1, 0.1, 0.2, 0.2), 0.3, "multirotor"), det((0.7, 0.7, 0.9, 0.9), 0.95, "balloon")]))
    _, progress, _ = start(client, pid)                                           # 0.3 is under the default 0.5; "balloon" is not a class here
    assert [b[0] for b in boxes_of(client, image)] == [ids["multirotor"]] and progress["unmapped"] == 1
    other = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det((0.1, 0.1, 0.2, 0.2), 0.3)]))
    start(client, pid, min_score=0.25)                                            # lowered by the person
    assert len(boxes_of(client, other)) == 1


def test_a_hard_negative_class_name_is_not_a_label_either(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX, 0.9, "birds")]))                      # "birds" is a project class, but a negative one: it has no boxes
    _, progress, _ = start(client, pid)
    assert boxes_of(client, image) == [] and progress["unmapped"] == 1


def test_the_same_drone_seen_as_two_classes_gets_one_box_the_surer_one(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX, 0.7, "multirotor"), det((0.41, 0.41, 0.6, 0.6), 0.9, "fixed-wing"), det((0.8, 0.1, 0.9, 0.2), 0.8, "multirotor")]))
    start(client, pid)
    got = boxes_of(client, image)
    assert len(got) == 2 and sorted(b[0] for b in got) == sorted([ids["fixed-wing"], ids["multirotor"]]) and {b[2] for b in got} == {0.9, 0.8}


def test_boxes_at_the_image_edge_are_clipped_and_slivers_are_dropped(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det((0.9, 0.9, 1.2, 1.3), 0.9), det((0.1, 0.1, 0.1005, 0.5), 0.9)], shrink=1.0))
    start(client, pid)
    got = boxes_of(client, image)
    assert len(got) == 1 and got[0][3] + got[0][5] / 2 <= 1.0 + 1e-9 and got[0][4] + got[0][6] / 2 <= 1.0 + 1e-9


def test_writing_boxes_makes_an_earlier_ai_check_out_of_date_but_finding_nothing_does_not(client, project):
    pid, ids = project
    image, empty = unlabelled(client, pid, ids), unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX)]))
    for i in (image, empty):
        client.post(f"/images/{i}/ai-check")                                       # both were checked while they had no boxes
    service.set_models(FakeModels([det(BOX)]))
    start(client, pid, limit=1)
    with client.session_factory() as db:
        assert db.get(AiCheck, image).stale is True and db.get(AiCheck, empty).stale is False
    service.set_models(FakeModels([]))
    _, progress, _ = start(client, pid)                                            # a run that finds nothing on the other image
    assert progress["nothing"] == 1
    with client.session_factory() as db:
        assert db.get(AiCheck, empty).stale is False                               # no boxes written: the check still describes the image


def test_an_image_a_person_labels_while_the_models_run_is_left_alone(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)

    class Busy(FakeModels):
        def detect(self, img):                                                     # the person saves a box in the review screen mid-run
            r = client.put(f"/images/{image}/annotations", json={"boxes": [{"class_id": ids["shahed"], "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1}]})
            assert r.status_code == 200
            return super().detect(img)

    service.set_models(Busy([det(BOX)]))
    _, progress, _ = start(client, pid)
    assert [b[0] for b in boxes_of(client, image)] == [ids["shahed"]] and boxes_of(client, image)[0][1] == AnnotationStatus.reviewed
    assert progress["skipped"] == 1 and progress["labelled"] == 0


@pytest.mark.parametrize("decision", ["rejected", "reviewed"])
def test_an_image_a_person_rejects_or_accepts_as_background_while_the_models_run_is_left_alone(client, project, decision):
    pid, ids = project
    image = unlabelled(client, pid, ids)

    class Busy(FakeModels):
        def detect(self, img):                                                     # no boxes drawn, but the person decided about the image
            assert client.put(f"/images/{image}/annotations", json={"boxes": [], "review_status": decision}).status_code == 200
            return super().detect(img)

    service.set_models(Busy([det(BOX)]))
    _, progress, _ = start(client, pid)
    assert boxes_of(client, image) == [] and detail(client, image)["review_status"] == decision and progress["skipped"] == 1


def test_a_model_trained_on_other_classes_fails_the_job_before_any_image_is_touched(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)

    class Drone(FakeModels):
        def class_names(self):
            return ["drone"]

    service.set_models(Drone([det(BOX, 0.9, "drone")]))
    status, _, error = start(client, pid)
    assert status == JobStatus.failed and "None of this model's classes (drone)" in error and "multirotor" in error
    assert boxes_of(client, image) == []


def test_the_cancel_button_stops_an_autolabel_run(client, project):
    pid, ids = project
    unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX)]))
    client.post(f"/projects/{pid}/jobs/ai-check", json={"mode": "autolabel"})
    with client.session_factory() as db:
        from app.models import Job
        job = db.query(Job).one()
        job.cancel_requested = True
        db.commit()
    _, progress, _ = run_one_job(client)
    assert progress["done"] == 0


# ---------- unconfirmed means unconfirmed ----------

def test_auto_boxes_are_flagged_in_the_gallery_until_a_person_saves_the_image(client, project):
    pid, ids = project
    image, other = unlabelled(client, pid, ids), unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX)]))
    start(client, pid, limit=1)
    rows = {r["id"]: r for r in client.get(f"/projects/{pid}/images").json()}
    assert rows[image]["ai_labelled"] is True and rows[other]["ai_labelled"] is False
    assert [r["id"] for r in client.get(f"/projects/{pid}/images", params={"ai": "autolabelled"}).json()] == [image]
    assert detail(client, image)["ai_labelled"] is True
    boxes = [{"class_id": ids["multirotor"], "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    client.put(f"/images/{image}/annotations", json={"boxes": boxes, "review_status": "reviewed"})   # Accept
    assert detail(client, image)["ai_labelled"] is False and client.get(f"/projects/{pid}/images", params={"ai": "autolabelled"}).json() == []
    assert boxes_of(client, image)[0][1] == AnnotationStatus.reviewed


def test_a_plain_save_after_an_edit_also_clears_the_marker_because_the_person_has_now_touched_the_boxes(client, project):
    pid, ids = project
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX)]))
    start(client, pid)
    client.put(f"/images/{image}/annotations", json={"boxes": [{"class_id": ids["multirotor"], "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3}]})
    assert detail(client, image)["ai_labelled"] is False and detail(client, image)["review_status"] == "pending"


def test_auto_boxes_never_reach_a_dataset_version(client, project):
    pid, ids = project
    m = ids["multirotor"]
    for _ in range(6):
        add(client, pid, m, [box(m)])                                              # reviewed images with a person's boxes
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX)]))
    start(client, pid)
    assert len(boxes_of(client, image)) == 1
    preview = client.get(f"/projects/{pid}/datasets/preview").json()
    assert preview["boxes_per_class"]["multirotor"] == 6 and preview["pending"] == 1      # the auto box is not counted, its image is pending
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    assert sum(v["stats"]["boxes"].values()) == 6 and v["stats"]["excluded"]["pending"] == 1


def test_even_if_an_image_were_marked_reviewed_its_auto_boxes_would_still_be_left_out(client, project):
    pid, ids = project
    m = ids["multirotor"]
    for _ in range(6):
        add(client, pid, m, [box(m)])
    image = unlabelled(client, pid, ids)
    service.set_models(FakeModels([det(BOX)]))
    start(client, pid)
    with client.session_factory() as db:                                           # a status flip that bypasses the save endpoint
        db.get(Image, image).review_status = "reviewed"
        db.commit()
    assert client.get(f"/projects/{pid}/datasets/preview").json()["boxes_per_class"]["multirotor"] == 6


def test_preview_counts_what_the_job_would_take(client, project):
    pid, ids = project
    unlabelled(client, pid, ids), unlabelled(client, pid, ids)
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    assert client.post(f"/projects/{pid}/jobs/ai-check/preview", json={"mode": "autolabel"}).json() == {"count": 2}
    assert client.post(f"/projects/{pid}/jobs/ai-check/preview", json={}).json() == {"count": 3}            # check mode: every non-negative image
    assert client.post(f"/projects/{pid}/jobs/ai-check/preview", json={"mode": "autolabel", "limit": 1}).json() == {"count": 1}
    assert client.get(f"/projects/{pid}/jobs").json() == []                                                  # previews start nothing


# ---------- choosing the models ----------

def finished_run(client, pid, name="r", best=True, stage=RunStage.done, map50=0.6):
    key = f"projects/{pid}/training/{name}/best.pt"
    storage.get_storage().put(key, ZIP)
    with client.session_factory() as db:
        run = TrainingRun(project_id=pid, dataset_id=1, name=name, config={}, stage=stage, progress={}, metrics={"map50": map50},
                          artifacts={"best_pt": {"key": key, "size": len(ZIP)}} if best else {}, cancel_requested=False, attempts=0)
        db.add(run)
        db.commit()
        return run.id


def test_the_model_list_has_finished_runs_uploaded_files_and_the_sam_sizes(client, project, tmp_path):
    pid, _ = project
    old, new = finished_run(client, pid, "old", map50=0.5), finished_run(client, pid, "new", map50=0.65)
    finished_run(client, pid, "nobest", best=False)
    finished_run(client, pid, "failed", stage=RunStage.failed)
    client.post("/ai/models/upload", files={"file": ("mine.pt", ZIP)}, headers={"X-MT-Upload": "1"})
    (tmp_path / "ai" / "models" / "sam2.1_b.pt").write_bytes(b"x")
    out = client.get("/ai/models").json()
    assert [d["id"] for d in out["detectors"]] == [f"run:{new}", f"run:{old}", "file:mine.pt", "yoloe:26s"]  # newest run first; unfinished or empty runs left out
    assert out["detectors"][0]["map50"] == 0.65 and out["detectors"][2]["kind"] == "file"
    assert out["detectors"][3]["kind"] == "prompt" and out["detectors"][3]["downloaded"] is False         # the text-prompt detector is always offered
    assert out["default_detector"] == f"yolo run {new} (new)"
    assert {s["size"]: s["downloaded"] for s in out["sams"]} == {"t": False, "s": False, "b": True, "l": False} and out["default_sam"] == settings.ai_sam_weights


def test_the_model_cache_reuses_a_set_keeps_two_and_lets_fakes_win(client, project, monkeypatch):
    pid, _ = project
    a, b = finished_run(client, pid, "a"), finished_run(client, pid, "b")
    built = []

    class Spy:
        def __init__(self, path, label, sam_path):
            built.append((path.name, sam_path.name, label))

    monkeypatch.setattr(service, "RealModels", Spy)
    with client.session_factory() as db:
        first = service.get_models(db, f"run:{a}", "t")
        assert service.get_models(db, f"run:{a}", "t") is first and len(built) == 1                       # same choice: same loaded set
        service.get_models(db, f"run:{a}", "b")                                                            # another SAM size: another set
        service.get_models(db, f"run:{b}", "t")                                                            # a third set drops the least recently used
        assert len(service._real) == 2 and len(built) == 3
        assert service.get_models(db, f"run:{a}", "t") is not first and len(built) == 4                    # the evicted one is rebuilt
        assert built[0][:2] == (f"detector-run{a}.pt", "sam2.1_t.pt")
        default = service.get_models(db)                                                                   # no choice = newest run + the setting
        assert built[-1][0] == f"detector-run{b}.pt" and built[-1][1] == settings.ai_sam_weights and default is not first
        fake = FakeModels()
        service.set_models(fake)
        count = len(built)
        assert service.get_models(db, f"run:{a}", "l") is fake and len(built) == count                    # tests' fakes win over any choice


def test_a_set_the_job_holds_stays_usable_after_the_cache_drops_it(client, project, monkeypatch):
    pid, _ = project
    runs = [finished_run(client, pid, n) for n in "abc"]

    class Spy:
        def __init__(self, *a):
            self.alive = True

    monkeypatch.setattr(service, "RealModels", Spy)
    with client.session_factory() as db:
        held = service.get_models(db, f"run:{runs[0]}")
        for r in runs[1:]:
            service.get_models(db, f"run:{r}")
        assert held not in service._real.values() and held.alive                                           # dropped from the cache, still the job's own


def test_the_job_uses_the_detector_and_sam_size_it_was_started_with(client, project, monkeypatch):
    pid, ids = project
    unlabelled(client, pid, ids)
    run = finished_run(client, pid)
    asked = []

    def fake_get_models(db, detector=None, sam=None):
        asked.append((detector, sam))
        return FakeModels([det(BOX)])

    monkeypatch.setattr(service, "get_models", fake_get_models)
    start(client, pid, detector=f"run:{run}", sam="b")
    assert asked == [(f"run:{run}", "b")]


def test_a_job_with_a_detector_that_cannot_work_is_refused_up_front(client, project):
    pid, _ = project
    finished_run(client, pid, "nobest", best=False)
    for bad in ("run:999", "file:ghost.pt", "run:1"):
        r = client.post(f"/projects/{pid}/jobs/ai-check", json={"detector": bad})
        assert r.status_code == 422, bad
    for worse in ("../../etc/passwd", "file:../x.pt", "file:x.txt", "run:abc", "anything"):
        assert client.post(f"/projects/{pid}/jobs/ai-check", json={"detector": worse}).status_code == 422, worse
    assert client.post(f"/projects/{pid}/jobs/ai-check", json={"sam": "xl"}).status_code == 422
    assert client.get(f"/projects/{pid}/jobs").json() == []


# ---------- uploaded weights ----------

def upload(client, name="mine.pt", data=ZIP, headers=None):
    return client.post("/ai/models/upload", files={"file": (name, io.BytesIO(data))}, headers={"X-MT-Upload": "1"} if headers is None else headers)


def stored(tmp_path):
    folder = tmp_path / "ai" / "models" / "custom"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


class Booby:
    """Stands in for anything that would open weights: touching it fails the test."""

    def __init__(self, *a, **k):
        raise AssertionError("an uploaded model was opened")


def test_an_upload_without_the_app_header_is_refused_and_writes_nothing(client, tmp_path):
    for headers in ({}, {"X-MT-Upload": "0"}):
        assert upload(client, headers=headers).status_code == 403
    assert stored(tmp_path) == []


def test_an_upload_is_stored_and_listed_but_never_opened(client, project, tmp_path, monkeypatch):
    monkeypatch.setattr(service, "RealModels", Booby)
    monkeypatch.setattr("app.ai.detector.Detector", Booby)
    r = upload(client)
    assert r.status_code == 201 and r.json() == {"id": "file:mine.pt", "label": "mine.pt (uploaded)", "kind": "file", "map50": None, "downloaded": None}
    assert stored(tmp_path) == ["mine.pt"]
    assert [d["id"] for d in client.get("/ai/models").json()["detectors"]] == ["file:mine.pt", "yoloe:26s"]
    pid, _ = project
    assert client.post(f"/projects/{pid}/jobs/ai-check", json={"detector": "file:mine.pt"}).status_code == 201   # picking it queues a run; the load waits for the job


def test_an_uploaded_file_name_is_cleaned_and_cannot_leave_the_models_folder(client, tmp_path):
    assert upload(client, "../../evil model.pt").json()["id"] == "file:evil-model.pt"
    assert upload(client, "..\\..\\x.pt").json()["id"] == "file:x.pt"
    assert stored(tmp_path) == ["evil-model.pt", "x.pt"] and not (tmp_path / "evil-model.pt").exists()


@pytest.mark.parametrize("name, data, code", [
    ("model.txt", ZIP, 422), ("noext", ZIP, 422), (".pt", ZIP, 422),
    ("fake.pt", b"#!/bin/sh\nrm -rf /\n", 422),      # not a zip archive
    ("empty.pt", b"", 422),
])
def test_files_that_are_not_pytorch_zips_are_refused_and_leave_nothing_behind(client, tmp_path, name, data, code):
    assert upload(client, name, data).status_code == code
    assert stored(tmp_path) == []


def test_an_oversize_upload_is_refused_without_leaving_a_partial_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "MAX_UPLOAD", 1000)
    assert upload(client, "big.pt", ZIP + b"\x00" * 2000).status_code == 413
    assert stored(tmp_path) == []


def test_uploading_a_name_that_exists_does_not_overwrite_it(client, tmp_path):
    assert upload(client, "a.pt", ZIP + b"1").status_code == 201
    assert upload(client, "a.pt", ZIP + b"2").status_code == 409
    assert (tmp_path / "ai" / "models" / "custom" / "a.pt").read_bytes() == ZIP + b"1"


def test_a_removed_upload_fails_the_job_with_a_clear_message(client, project, tmp_path):
    pid, ids = project
    unlabelled(client, pid, ids)
    upload(client)
    assert client.post(f"/projects/{pid}/jobs/ai-check", json={"mode": "autolabel", "detector": "file:mine.pt"}).status_code == 201
    (tmp_path / "ai" / "models" / "custom" / "mine.pt").unlink()
    status, _, error = run_one_job(client)
    assert status == JobStatus.failed and "mine.pt is not there any more" in error


def test_the_model_endpoints_are_closed_to_the_network(client):
    from fastapi.testclient import TestClient

    from app.main import app

    from .helpers import from_ip

    lan = TestClient(from_ip(app, "10.0.0.7"))
    assert lan.get("/ai/models").status_code == 403
    assert lan.post("/ai/models/upload", files={"file": ("a.pt", ZIP)}, headers={"X-MT-Upload": "1"}).status_code == 403


def test_the_ai_is_available_without_a_trained_detector_because_the_text_prompt_one_needs_none(client, project):
    status = client.get("/ai/status").json()
    assert status["available"] is True and status["reason"] is None and status["detector"] is None    # no run, no upload: no default detector, but YOLOE can run
    upload(client)
    status = client.get("/ai/status").json()
    assert status["available"] is True and status["reason"] is None and status["detector"] is None   # no default, but a run can pick the upload
