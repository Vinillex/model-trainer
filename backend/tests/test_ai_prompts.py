"""The text-prompt detector (YOLOE) and what rides on it: AI labelling with words, the negatives check, the gallery's "keep as is"."""

import threading
import time
import types
from pathlib import Path

import pytest

from app.ai import service
from app.ai.check import Detection
from app.ai.prompts import DRONE_LABEL, default_object_prompt, object_label
from app.config import settings
from app.models import AnnotationStatus, DetClass, Image, JobStatus, ReviewStatus

from .test_ai_autolabel import boxes_of, detail
from .test_ai_check import FakeModels, run_one_job
from .test_datasets import add, box

BOX = (0.4, 0.4, 0.6, 0.6)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ai_dir", str(tmp_path / "ai"))  # nothing may land in the real backend/data/ai
    monkeypatch.setattr(service, "missing_extras", lambda: None)
    service._real.clear()
    yield
    service.set_models(None)
    service._real.clear()


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def drone(box_=BOX, score=0.6):
    return Detection(box_, score, DRONE_LABEL)


def thing(class_id, box_, score=0.8):
    return Detection(box_, score, object_label(class_id))


def start(client, pid, expect=201, **body):
    r = client.post(f"/projects/{pid}/jobs/ai-check", json=body)
    assert r.status_code == expect, r.text
    return run_one_job(client) if expect == 201 else r


class PerImage(FakeModels):
    """Detections by call order, so each image of a run gets its own."""

    def __init__(self, per_image):
        super().__init__()
        self.per_image = list(per_image)

    def detect(self, image):
        self.calls += 1
        return list(self.per_image.pop(0)) if self.per_image else []


# ---------- the detector class ----------

def fake_ultralytics(monkeypatch, log):
    """Stand-ins for the YOLOE model and the MobileCLIP encoder, so the loading code runs for real without the 285 MB of files."""

    class Inner:
        clip_model = "unset"

        def parameters(self):
            return iter([types.SimpleNamespace(device="cpu")])

        def get_text_pe(self, texts, cache_clip_model=False):
            log.append(("encode", list(texts), self.clip_model))
            return "embeddings"

    class FakeYoloe:
        def __init__(self, path):
            log.append(("weights", Path(path).name))
            self.model, self.result = Inner(), None

        def set_classes(self, names, embeddings):
            log.append(("set_classes", list(names), embeddings, self.model.clip_model))

        def predict(self, image, imgsz, conf, device, verbose):
            log.append(("predict", device, conf))
            r = types.SimpleNamespace(boxes=types.SimpleNamespace(xyxyn=[[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.9, 0.9]], conf=[0.7, 0.2], cls=[1, 0]))
            return [r]

    class FakeEncoder:
        def __init__(self, device, weight):
            log.append(("encoder", Path(weight).name))

    monkeypatch.setattr("ultralytics.YOLOE", FakeYoloe)
    monkeypatch.setattr("ultralytics.nn.text_model.MobileCLIPTS", FakeEncoder)


def test_the_text_prompt_detector_loads_from_its_folder_never_downloads_and_names_boxes_by_label(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    import numpy as np

    from app.ai.yoloe import ENCODER_FILE, WEIGHTS_FILE, YoloeDetector

    folder = tmp_path / "yoloe"
    folder.mkdir()
    (folder / WEIGHTS_FILE).write_bytes(b"w")
    (folder / ENCODER_FILE).write_bytes(b"e")

    def no_download(*a, **k):
        raise AssertionError("a present file must never be downloaded (it would land in the working directory)")

    monkeypatch.setattr("ultralytics.utils.downloads.safe_download", no_download)
    log = []
    fake_ultralytics(monkeypatch, log)
    detector = YoloeDetector(folder, [("drone", "drone aircraft"), ("class:4", "bird")], "cpu")
    assert detector.class_names() == ["drone", "class:4"] and log == []                          # nothing loads until the first image

    found = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
    assert [(d.class_name, d.score) for d in found] == [("class:4", 0.7), ("drone", 0.2)]       # the class index is read back as the label
    assert log[:4] == [("weights", WEIGHTS_FILE), ("encoder", ENCODER_FILE), ("encode", ["drone aircraft", "bird"], log[2][2]),
                       ("set_classes", ["drone aircraft", "bird"], "embeddings", None)]
    assert log[2][2].__class__.__name__ == "FakeEncoder" and log[-1][:2] == ("predict", "cpu")  # the encoder was used from the folder, then dropped (None)
    detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
    assert [e[0] for e in log].count("weights") == 1                                              # loaded once

    with pytest.raises(ValueError):
        YoloeDetector(folder, [], "cpu")


def test_the_text_prompt_detector_downloads_a_missing_file_to_its_own_folder(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    from app.ai.yoloe import ENCODER_FILE, WEIGHTS_FILE, YoloeDetector

    fetched = []

    def fake_download(url, file=None, **k):
        fetched.append((url.rsplit("/", 1)[-1], Path(file)))
        Path(file).write_bytes(b"x" * 200_000)

    monkeypatch.setattr("ultralytics.utils.downloads.safe_download", fake_download)
    fake_ultralytics(monkeypatch, [])
    folder = tmp_path / "models" / "yoloe"
    YoloeDetector(folder, [("drone", "drone aircraft")], "cpu")._load()
    assert fetched == [(WEIGHTS_FILE, folder / WEIGHTS_FILE), (ENCODER_FILE, folder / ENCODER_FILE)]


# ---------- registry, prompts and the job rules ----------

def test_the_text_prompt_detector_is_always_listed_and_says_whether_it_is_downloaded(client, project, tmp_path):
    listed = client.get("/ai/models").json()["detectors"]
    assert listed == [{"id": "yoloe:26s", "label": "YOLOE-26s (text prompts)", "kind": "prompt", "map50": None, "downloaded": False}]
    folder = tmp_path / "ai" / "models" / "yoloe"
    folder.mkdir(parents=True)
    (folder / "yoloe-26s-seg.pt").write_bytes(b"w")
    assert client.get("/ai/models").json()["detectors"][0]["downloaded"] is False                 # both files are needed
    (folder / "mobileclip2_b.ts").write_bytes(b"e")
    assert client.get("/ai/models").json()["detectors"][0]["downloaded"] is True


def test_default_words_for_what_a_negative_class_holds():
    assert [default_object_prompt(n) for n in ("birds", "Balloons", "aircraft", "helicopters", "empty-sky", "smoke")] == \
           ["bird", "balloon", "airplane", "helicopter", "", ""]


def test_the_prompts_endpoint_gives_the_drone_phrase_and_each_negative_class_with_its_words(client, project):
    pid, ids = project
    out = client.get(f"/projects/{pid}/ai/prompts").json()
    assert out["drone_prompt"] == "drone aircraft"
    assert {c["name"]: c["prompt"] for c in out["negative_classes"]} == {"birds": "bird", "aircraft": "airplane", "helicopters": "helicopter",
                                                                         "balloons": "balloon", "empty-sky": ""}
    assert {c["class_id"] for c in out["negative_classes"]} == {ids[n] for n in ("birds", "aircraft", "helicopters", "balloons", "empty-sky")}
    assert client.get("/projects/999/ai/prompts").status_code == 404


def test_the_api_refuses_runs_that_cannot_work_with_what_the_text_prompt_detector_is(client, project):
    pid, ids = project
    url = f"/projects/{pid}/jobs/ai-check"
    cases = [
        ({"mode": "check", "detector": "yoloe:26s"}, "box check"),                         # the check's thresholds were tuned on the trained detector
        ({"mode": "negatives", "detector": "run:1"}, "text-prompt detector"),               # negatives need words
        ({"mode": "autolabel", "min_score": 0.15}, "25%"),                                  # a trained detector has its own floor
        ({"mode": "negatives", "object_prompts": {str(ids["multirotor"]): "drone"}}, "not a hard-negative class"),
        ({"mode": "negatives", "object_prompts": {str(ids["birds"]): "Drone Aircraft"}}, "same as the words that find a drone"),  # indistinguishable
        ({"mode": "negatives", "object_prompts": {"birds": "bird"}}, None),                 # keys are class ids
        ({"mode": "negatives", "prompt": ""}, None),
        ({"mode": "negatives", "min_size": 0.9}, None),
    ]
    for body, text in cases:
        r = client.post(url, json=body)
        assert r.status_code == 422, body
        if text:
            assert text in r.text, (body, r.text)
    ok = client.post(url, json={"mode": "autolabel", "detector": "yoloe:26s", "min_score": 0.1})
    assert ok.status_code == 201 and ok.json()["params"]["min_score"] == 0.1
    neg = client.post(url, json={"mode": "negatives"})                                       # no detector named: the text-prompt one is the only choice
    assert neg.status_code == 201 and neg.json()["params"]["detector"] == "yoloe:26s"


def test_the_job_selects_negative_images_only_in_the_negatives_mode(client, project):
    pid, ids = project
    bird, sky = add(client, pid, ids["birds"])["id"], add(client, pid, ids["empty-sky"])["id"]
    drone_image = add(client, pid, ids["multirotor"], [box(ids["multirotor"])])["id"]
    count = lambda **b: client.post(f"/projects/{pid}/jobs/ai-check/preview", json=b).json()["count"]
    assert count(mode="negatives") == 2 and count(mode="check") == 1 and count(mode="negatives", class_id=ids["birds"]) == 1
    assert count(mode="negatives", image_ids=[drone_image]) == 0 and count(mode="negatives", image_ids=[bird, sky]) == 2


def test_keeping_a_negative_picture_in_the_review_window_marks_it_reviewed_and_sets_the_doubt_aside(client, project):
    pid, ids = project
    bird = add(client, pid, ids["birds"])["id"]
    negatives_run(client, pid, FakeModels([]))                                    # no bird found: flagged
    with client.session_factory() as db:                                          # a picture nobody has reviewed yet
        db.get(Image, bird).review_status = ReviewStatus.pending
        db.commit()
    # what the review window's Keep as is sends: an Accept with no boxes
    r = client.put(f"/images/{bird}/annotations", json={"boxes": [], "review_status": "reviewed", "ai_dismiss": True})
    assert r.status_code == 200
    d = detail(client, bird)
    assert (d["review_status"], d["class_id"], d["ai_status"], boxes_of(client, bird)) == ("reviewed", ids["birds"], "dismissed", [])


def test_the_negatives_mode_leaves_the_trash_alone_unless_asked_for_it(client, project):
    pid, ids = project
    bird, trashed = add(client, pid, ids["birds"])["id"], add(client, pid, ids["birds"])["id"]
    assert client.post(f"/projects/{pid}/images/reject", json={"image_ids": [trashed]}).status_code == 200
    count = lambda **b: client.post(f"/projects/{pid}/jobs/ai-check/preview", json=b).json()["count"]
    assert count(mode="negatives") == 1 and count(mode="negatives", image_ids=[bird, trashed]) == 1
    assert count(mode="negatives", review_status="rejected") == 1   # asking for the trash by name still works


# ---------- the negatives check ----------

def negatives_run(client, pid, models, **body):
    service.set_models(models)
    return start(client, pid, mode="negatives", **body)


def test_the_negatives_check_flags_drone_like_things_small_birds_and_missing_birds_and_leaves_the_rest_ok(client, project):
    pid, ids = project
    b = ids["birds"]
    big, small, nothing, hit, sky, sky_hit = (add(client, pid, ids[n])["id"] for n in ("birds",) * 4 + ("empty-sky",) * 2)
    # images are 320x240 (short side 240): a 0.5 x 0.4 box is 160 x 96 px = 40% of the short side, 0.1 x 0.1 is 32 x 24 px = 10%: use 0.05 for 5%
    models = PerImage([
        [thing(b, (0.2, 0.2, 0.7, 0.6))],                                                              # a big bird: fine
        [thing(b, (0.2, 0.2, 0.7, 0.6)), thing(b, (0.8, 0.8, 0.85, 0.85)), drone((0.1, 0.1, 0.2, 0.2), 0.1)],  # also a tiny one (below the floor and the drone score)
        [],                                                                                             # no bird at all
        [thing(b, (0.2, 0.2, 0.7, 0.6)), drone(BOX, 0.7)],                                              # a bird and something the model reads as a drone
        [],                                                                                             # empty sky: nothing to look for, nothing found
        [drone(BOX, 0.4)],                                                                              # empty sky with a drone-like hit
    ])
    status, progress, error = negatives_run(client, pid, models)
    assert status == JobStatus.succeeded and error is None
    assert (progress["mode"], progress["total"], progress["done"], progress["flagged"], progress["failed"]) == ("negatives", 6, 6, 4, 0)

    def verdict(i):
        c = client.get(f"/images/{i}/ai-check").json()
        return c["verdict"], c

    assert verdict(big)[0] == "ok" and verdict(sky)[0] == "ok"
    v, c = verdict(small)
    assert v == "suspect" and c["result"]["summary"]["suspect"] == 1 and c["result"]["missing"] == []      # the 0.1-sure drone is below the 0.25 default
    assert "5%" in c["result"]["labels"][0]["reason"] and "10% floor" in c["result"]["labels"][0]["reason"]
    assert c["result"]["negatives"] == {"object": "bird", "found": 2, "smallest": 0.05, "min_size": 0.1, "drone_score": 0.25}
    v, c = verdict(nothing)
    assert v == "suspect" and c["result"]["labels"][0]["box"] == [0.0, 0.0, 1.0, 1.0] and "no bird" in c["result"]["labels"][0]["reason"]
    v, c = verdict(hit)
    assert v == "missing" and c["result"]["missing"][0]["class_name"] == DRONE_LABEL and "reads this as a drone" in c["result"]["missing"][0]["reason"]
    assert verdict(sky_hit)[0] == "missing"
    # the pictures stay as they were: no boxes, still reviewed, and "AI: needs a look" lists exactly the four
    assert all(boxes_of(client, i) == [] and detail(client, i)["review_status"] == "reviewed" for i in (big, small, nothing, hit, sky, sky_hit))
    flagged = client.get(f"/projects/{pid}/images", params={"ai": "flagged"}).json()
    assert sorted(i["id"] for i in flagged) == sorted([small, nothing, hit, sky_hit])


def test_a_flock_of_hits_is_cut_to_the_surest_few(client, project):
    pid, ids = project
    image = add(client, pid, ids["empty-sky"])["id"]
    flock = [drone((0.01 * i, 0.1, 0.01 * i + 0.05, 0.2), 0.3 + i / 100) for i in range(20)]
    negatives_run(client, pid, FakeModels(flock))
    scores = [m["score"] for m in client.get(f"/images/{image}/ai-check").json()["result"]["missing"]]
    assert len(scores) == 8 and scores == sorted(scores, reverse=True) and scores[0] == 0.49


def test_a_negatives_run_does_not_repeat_finished_images_but_new_words_or_floor_make_them_stale(client, project):
    pid, ids = project
    image = add(client, pid, ids["birds"])["id"]
    seen = [thing(ids["birds"], (0.2, 0.2, 0.7, 0.6))]
    first = FakeModels(seen)
    assert negatives_run(client, pid, first)[1]["skipped"] == 0 and first.calls == 1
    again = FakeModels(seen)
    assert negatives_run(client, pid, again)[1]["skipped"] == 1 and again.calls == 0                 # same settings: skipped
    floor = FakeModels(seen)
    assert negatives_run(client, pid, floor, min_size=0.5)[1]["skipped"] == 0 and floor.calls == 1   # another floor: redone
    words = FakeModels(seen)
    assert negatives_run(client, pid, words, object_prompts={str(ids["birds"]): "gull"}, min_size=0.5)[1]["skipped"] == 0 and words.calls == 1
    forced = FakeModels(seen)
    assert negatives_run(client, pid, forced, object_prompts={str(ids["birds"]): "gull"}, min_size=0.5, force=True)[1]["skipped"] == 0 and forced.calls == 1
    assert client.get(f"/images/{image}/ai-check").json()["result"]["negatives"]["object"] == "gull"


def test_a_blank_object_prompt_turns_the_size_check_off_for_that_class(client, project):
    pid, ids = project
    image = add(client, pid, ids["birds"])["id"]
    _, progress, _ = negatives_run(client, pid, FakeModels([]), object_prompts={str(ids["birds"]): ""})
    assert progress["flagged"] == 0 and client.get(f"/images/{image}/ai-check").json()["verdict"] == "ok"   # no bird found, but nobody asked for one


# ---------- AI labelling with words ----------

def test_the_text_prompt_detector_files_its_boxes_under_the_class_each_picture_is_filed_under(client, project):
    pid, ids = project
    m, f, mixed = (add(client, pid, ids[n], status="pending")["id"] for n in ("multirotor", "fixed-wing", "multirotor"))
    with client.session_factory() as db:  # the "mixed" folder is created when a reviewed picture shows several classes: make one and file a picture there
        folder = DetClass(project_id=pid, name="mixed", is_negative=False)
        db.add(folder)
        db.flush()
        db.get(Image, mixed).class_id = folder.id
        db.commit()
    service.set_models(FakeModels([drone(BOX, 0.3)], shrink=1.0))
    status, progress, _ = start(client, pid, mode="autolabel", detector="yoloe:26s", min_score=0.25)
    assert status == JobStatus.succeeded
    assert boxes_of(client, m)[0][:3] == (ids["multirotor"], AnnotationStatus.auto, 0.3)
    assert boxes_of(client, f)[0][:2] == (ids["fixed-wing"], AnnotationStatus.auto)                   # the detector never names a class: the folder does
    assert boxes_of(client, mixed) == []                                                              # "mixed" is not a class a box can have
    assert (progress["labelled"], progress["unmapped"]) == (2, 1)
    assert detail(client, m)["review_status"] == "pending"


def test_the_text_prompt_detector_respects_the_score_threshold(client, project):
    pid, ids = project
    image = add(client, pid, ids["multirotor"], status="pending")["id"]
    service.set_models(FakeModels([drone(BOX, 0.2)], shrink=1.0))
    assert start(client, pid, mode="autolabel", detector="yoloe:26s", min_score=0.25)[1]["nothing"] == 1 and boxes_of(client, image) == []
    assert start(client, pid, mode="autolabel", detector="yoloe:26s", min_score=0.15)[1]["boxes"] == 1


def test_each_negative_class_gets_its_own_detector_with_only_the_drone_phrase_and_its_own_words(client, project, monkeypatch):
    pid, ids = project
    add(client, pid, ids["birds"])
    add(client, pid, ids["empty-sky"])
    asked = []

    def fake_get_models(db, detector=None, sam=None, prompts=None):
        asked.append((detector, prompts))
        return FakeModels([])

    monkeypatch.setattr(service, "get_models", fake_get_models)
    service.set_models(None)
    start(client, pid, mode="negatives", object_prompts={str(ids["birds"]): "bird", str(ids["aircraft"]): "airplane"})
    by_class = {tuple(p[1] for p in prompts) for _, prompts in asked}
    # a detector hands each object the one best-scoring phrase: with "airplane" loaded beside "bird", a gull came back as an airplane
    assert ("drone aircraft", "bird") in by_class and ("drone aircraft", "airplane") in by_class and ("drone aircraft",) in by_class
    assert all(len(p) <= 2 and d == "yoloe:26s" for d, p in asked)


def test_one_model_set_serves_one_prompt_set(client, project, monkeypatch):
    built = []

    class Spy:
        def __init__(self, path, label, sam_path, prompts=None):
            built.append((path.name, label, prompts))

    monkeypatch.setattr(service, "RealModels", Spy)
    p1, p2 = [("drone", "drone aircraft")], [("drone", "small drone flying in the sky")]
    with client.session_factory() as db:
        a = service.get_models(db, "yoloe:26s", None, p1)
        assert service.get_models(db, "yoloe:26s", None, list(p1)) is a and len(built) == 1
        assert service.get_models(db, "yoloe:26s", None, p2) is not a and len(built) == 2             # other words: another set (set_classes changes the model)
        assert built[0] == ("yoloe", "YOLOE-26s (text prompts)", p1)
        with pytest.raises(service.AiUnavailable):
            service.get_models(db, "yoloe:26s")                                                        # it cannot run without words
        with pytest.raises(service.AiUnavailable):
            service.get_models(db, None, None, p1)                                                     # and a trained detector does not take them


def test_the_text_prompt_detector_and_sam_never_use_the_gpu_at_the_same_time():
    pytest.importorskip("ultralytics")
    import numpy as np

    from app.ai.sam import SamRefiner
    from app.ai.yoloe import YoloeDetector

    active, peak, lock = [0], [0], threading.Lock()

    def work(*a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        with lock:
            active[0] -= 1
        return [types.SimpleNamespace(masks=None, boxes=types.SimpleNamespace(xyxyn=[], conf=[], cls=[]))]

    det = YoloeDetector(Path("."), [("drone", "drone aircraft")], "cpu")
    det._model = types.SimpleNamespace(predict=work)
    sam = SamRefiner(Path("x.pt"), "cpu")
    sam._model = work
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    threads = [threading.Thread(target=lambda: (det.detect(image), sam.refine(image, prompt=(10, 10, 50, 50)))) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 1


# ---------- keeping flagged images ----------

def test_keep_as_is_sets_the_ai_doubts_aside_for_flagged_images_only(client, project):
    pid, ids = project
    flagged, fine, unchecked = (add(client, pid, ids["birds"])["id"] for _ in range(3))
    service.set_models(PerImage([[], [thing(ids["birds"], (0.2, 0.2, 0.7, 0.6))]]))
    start(client, pid, mode="negatives", image_ids=[flagged, fine])
    assert client.get(f"/images/{flagged}/ai-check").json()["verdict"] == "suspect" and client.get(f"/images/{fine}/ai-check").json()["verdict"] == "ok"
    out = client.post(f"/projects/{pid}/images/ai-dismiss", json={"image_ids": [flagged, fine, unchecked]})
    assert out.status_code == 200 and out.json() == {"dismissed": 1}                                   # the ok and the unchecked image had nothing to set aside
    assert client.get(f"/images/{flagged}/ai-check").json()["dismissed"] is True
    assert client.get(f"/projects/{pid}/images", params={"ai": "flagged"}).json() == []
    assert [i["id"] for i in client.get(f"/projects/{pid}/images", params={"ai": "dismissed"}).json()] == [flagged]
    assert client.post(f"/projects/{pid}/images/ai-dismiss", json={"image_ids": [flagged]}).json() == {"dismissed": 1}   # harmless to repeat
    assert client.delete(f"/images/{flagged}/ai-dismiss").status_code == 200                           # and undoable like any dismissal
    assert [i["id"] for i in client.get(f"/projects/{pid}/images", params={"ai": "flagged"}).json()] == [flagged]


def test_keep_as_is_refuses_an_unknown_image_and_changes_nothing(client, project):
    pid, ids = project
    flagged = add(client, pid, ids["birds"])["id"]
    service.set_models(FakeModels([]))
    start(client, pid, mode="negatives")
    r = client.post(f"/projects/{pid}/images/ai-dismiss", json={"image_ids": [flagged, 99999]})
    assert r.status_code == 422 and client.get(f"/images/{flagged}/ai-check").json()["dismissed"] is False
