"""Glue between the AI check and the app: loads the real models on first use, reads an image and its boxes, stores the result.

A check only ever writes its own row in `ai_checks`. The auto-label job is the one thing here that writes boxes, and only as `auto` boxes on a
pending image that has none (never `reviewed`, so nothing it writes can reach a training set before a person accepts it)."""

import hashlib
import io
import importlib.util
import json
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import storage
from ..config import settings
from ..models import AiCheck, Annotation, AnnotationStatus, DetClass, Image, ReviewStatus, RunStage, TrainingRun
from . import registry
from .check import POLICY_VERSION, LabelBox, Models, Policy, run_check, suggest
from .geometry import iou, side_on_input, to_normalized, to_pixels
from .prompts import DRONE_LABEL, OBJECT_SCORE
from .registry import SAM_FILES, YOLOE_ID, AiUnavailable, models_dir, resolve_detector, run_weights  # noqa: F401 - AiUnavailable is part of this module's API
from .verdict import fingerprint, worst

KEEP_LOADED = 2  # model sets kept in memory at once (the Mac has 16 GB); the least recently used one is dropped

_override: Models | None = None  # tests put fakes here; they win over any choice
_real: "OrderedDict[tuple[str, str, str], RealModels]" = OrderedDict()
_build_lock = threading.Lock()


def set_models(models: Models | None) -> None:
    """Tests (and nothing else) put fake models here."""
    global _override
    _override = models


def ai_dir() -> Path:
    return Path(settings.ai_dir)


def detector_weights(db: Session) -> tuple[Path, str] | None:
    """The default detector: the configured file, else best.pt of the newest finished training run (copied out of storage once)."""
    if settings.ai_detector_weights:
        path = Path(settings.ai_detector_weights)
        return (path, path.name) if path.exists() else None
    for run in db.scalars(select(TrainingRun).where(TrainingRun.stage == RunStage.done).order_by(TrainingRun.id.desc())):
        path = run_weights(run)
        if path is not None:
            return path, f"yolo run {run.id} ({run.name})"
    return None


def missing_extras() -> str | None:
    if importlib.util.find_spec("ultralytics") is None:
        return "The AI extras are not installed. Run: cd backend && .venv/bin/pip install -r requirements-ai.txt, then restart the backend."
    return None


class RealModels:
    """The trained YOLO (or, with `prompts`, the text-prompt detector, whose path is its folder) as detector and one SAM size as refiner.
    Nothing heavy loads until the first image."""

    def __init__(self, detector_path: Path, detector_label: str, sam_path: Path, prompts: list[tuple[str, str]] | None = None):
        from .detector import Detector  # the heavy imports
        from .sam import SamRefiner

        if prompts:
            from .yoloe import YoloeDetector

            self.detector = YoloeDetector(detector_path, prompts, settings.ai_device)
        else:
            self.detector = Detector(detector_path, settings.ai_device)
        self.sam = SamRefiner(sam_path, settings.ai_device)
        self.label = f"{sam_path.name} + {detector_label}"

    def detect(self, image_rgb):
        return self.detector.detect(image_rgb)

    def refine(self, image_rgb, prompt=None, point=None):
        return self.sam.refine(image_rgb, prompt, point)

    def class_names(self) -> list[str]:
        return self.detector.class_names()

    def device_summary(self):
        parts = [self.sam.device.summary(), self.detector.device.summary()]
        return {"device": parts[0]["device"], "fallbacks": sum(p["fallbacks"] for p in parts),
                "last_error": next((p["last_error"] for p in parts if p["last_error"]), None)}


def get_models(db: Session, detector: str | None = None, sam: str | None = None, prompts: list[tuple[str, str]] | None = None) -> Models:
    """The models for a detector choice ("run:5", "file:name.pt", "yoloe:26s", None = the default) and a SAM size ("t" "s" "b" "l", None = the setting).
    The text-prompt detector also needs `prompts`, (label, words) pairs: one loaded set serves one prompt set, so the set is part of the cache key.
    Built on first use and kept (the last KEEP_LOADED sets), so switching back is free. Weights load on the first image, so a first check is slow.
    A caller keeps the object it got: being dropped from the cache never unloads it from under a running job."""
    if _override is not None:
        return _override
    if (reason := missing_extras()) is not None:
        raise AiUnavailable(reason)
    if (detector == YOLOE_ID) != bool(prompts):
        raise AiUnavailable("the text-prompt detector needs prompts, and only it takes them")
    with _build_lock:
        det_key, det_path, det_label = resolve_detector(db, detector, detector_weights)
        sam_file = SAM_FILES[sam] if sam else settings.ai_sam_weights
        key = (det_key, sam_file, hashlib.sha1(json.dumps(sorted(prompts or [])).encode()).hexdigest()[:12])
        models = _real.get(key)
        if models is None:
            models = _real[key] = RealModels(det_path, det_label, models_dir() / sam_file, *([prompts] if prompts else []))
            while len(_real) > KEEP_LOADED:
                _real.popitem(last=False)
        else:
            _real.move_to_end(key)
        return models


def model_label(models: Models) -> str:
    """Stored with every check. Includes the rules version: a check made under older rules no longer counts as up to date."""
    return f"{getattr(models, 'label', 'fake models')} · rules v{POLICY_VERSION}"


def device_summary() -> dict | None:
    """What the loaded models run on, how often the GPU failed a call and why (None until real models have loaded)."""
    models = _override or (next(reversed(_real.values())) if _real else None)
    return models.device_summary() if hasattr(models, "device_summary") else None


def load_rgb(image: Image) -> np.ndarray:
    with PILImage.open(io.BytesIO(storage.get_storage().get(image.storage_key))) as im:
        return np.array(im.convert("RGB"))


def label_boxes(image: Image) -> list[LabelBox]:
    return [LabelBox(a.id, (a.cx - a.w / 2, a.cy - a.h / 2, a.cx + a.w / 2, a.cy + a.h / 2)) for a in image.annotations]


def _store_check(db: Session, image: Image, model: str, result: dict) -> AiCheck:
    row = image.ai_check or AiCheck(image_id=image.id)
    row.checked_at = datetime.now(timezone.utc)
    row.model = model
    row.fingerprint = result.pop("fingerprint")
    row.verdict = result.pop("verdict")
    row.stale = False
    row.result = result
    image.ai_check = row
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def check_image(db: Session, image: Image, models: Models | None = None, policy: Policy = Policy()) -> AiCheck:
    """Run the AI check on one image and store (replace) its row. Never touches the image's boxes or review status."""
    models = models or get_models(db)
    return _store_check(db, image, model_label(models), run_check(load_rgb(image), label_boxes(image), models, policy))


# ---------- checking negative images ----------

MAX_DRONE_HITS = 8  # a flock of tiny birds gives dozens of hits: the surest few are enough to look at


class NegativeRules(NamedTuple):
    drone_score: float  # a drone-phrase hit this sure is reported
    min_size: float  # the smallest the class's own thing may be, as a share of the image's short side (the importer's floor)
    drone_words: str
    object_label: str | None  # the detector's label for what the class holds ("class:4"); None = no object check
    object_words: str  # "bird"


def negatives_label(models: Models, rules: NegativeRules) -> str:
    """Like model_label, plus a digest of the prompts and thresholds: another floor or other words make older results out of date."""
    digest = hashlib.sha1(json.dumps(list(rules)).encode()).hexdigest()[:8]
    return f"{model_label(models)} · negatives {digest}"


def _item(box, side_px: float, reason: str) -> dict:
    return {"annotation_id": None, "box": list(box), "side_px": round(side_px, 1), "verdict": "suspect", "reason": reason,
            "suggested": None, "detector_iou": None, "detector_score": None, "sam_object": None}


def check_negative(db: Session, image: Image, models: Models, rules: NegativeRules) -> AiCheck:
    """What a hard-negative picture must hold: nothing that reads as a drone, and (when the class has words for its thing) that thing at a
    useful size. Same row and verdicts as the box check, so the gallery's "AI: needs a look" lists the pictures to look at; boxes and review
    status are never touched. A drone hit is "missing" (a drone with no label), a too-small or absent object is "suspect"."""
    rgb = load_rgb(image)
    height, width = rgb.shape[:2]
    dets = models.detect(rgb)
    missing = [{"box": list(d.box), "detector_box": list(d.box), "score": round(d.score, 3), "class_name": DRONE_LABEL,
                "side_px": round(side_on_input(d.box, width, height), 1),
                "reason": f"the model reads this as a drone ({d.score:.0%} sure): a real drone, or something that looks like one"}
               for d in sorted(dets, key=lambda d: -d.score) if d.class_name == DRONE_LABEL and d.score >= rules.drone_score][:MAX_DRONE_HITS]
    items, found, smallest = [], 0, None
    if rules.object_label:
        things = [d for d in dets if d.class_name == rules.object_label and d.score >= OBJECT_SCORE]
        found = len(things)
        short = min(width, height)
        sizes = [min((d.box[2] - d.box[0]) * width, (d.box[3] - d.box[1]) * height) / short for d in things]
        if things:
            smallest = min(sizes)
            for d, size in zip(things, sizes):
                if size < rules.min_size:
                    items.append(_item(d.box, side_on_input(d.box, width, height),
                                       f"this {rules.object_words} is {size:.0%} of the picture's short side: smaller than the {rules.min_size:.0%} floor, "
                                       "so against the sky it could pass for a drone"))
        else:
            items.append(_item((0.0, 0.0, 1.0, 1.0), side_on_input((0.0, 0.0, 1.0, 1.0), width, height),
                               f"the model finds no {rules.object_words} in this picture (not what its folder says?)"))
    verdicts = [it["verdict"] for it in items] + ["missing"] * len(missing)
    result = {"labels": items, "missing": missing, "fingerprint": fingerprint([]), "verdict": worst(verdicts) if verdicts else "ok",
              "summary": {"ok": 0, "adjust": 0, "suspect": len(items), "too_small": 0, "missing": len(missing), "labels": len(items)},
              "negatives": {"object": rules.object_words or None, "found": found, "smallest": None if smallest is None else round(smallest, 4),
                            "min_size": rules.min_size, "drone_score": rules.drone_score}}
    return _store_check(db, image, negatives_label(models, rules), result)


def suggest_box(db: Session, image: Image, x: float | None, y: float | None, box: list[float] | None) -> dict:
    """Click-to-box: SAM's tight box (normalized) for a click and/or a rough box, for the person to accept or redraw."""
    models = get_models(db)
    rgb = load_rgb(image)
    h, w = rgb.shape[:2]
    point = (x * w, y * h) if x is not None and y is not None else None
    prompt = to_pixels(tuple(box), w, h) if box else None
    if point is None and prompt is None:
        return {"found": False}
    ref = models.refine(rgb, prompt=prompt, point=point)
    if ref is None or ref.touches_edge:
        return {"found": False}
    return {"found": True, "box": list(to_normalized(ref.box, w, h)), "score": round(ref.score, 3)}


# ---------- auto-labelling ----------

class Autolabel(NamedTuple):
    skipped: bool  # the image was no longer an unlabelled, pending one: nothing was written
    boxes: int
    unmapped: int  # detections whose class name is not one of the project's classes (left out, never guessed)


def class_map(db: Session, project_id: int) -> dict[str, int]:
    """The project's classes a box can have, by name. Names, never indexes: dataset versions renumber the classes."""
    return {c.name: c.id for c in db.scalars(select(DetClass).where(DetClass.project_id == project_id)) if c.is_trainable}


def require_matching_classes(models: Models, classes: dict[str, int]) -> None:
    """Fail before the first image when the model was trained on other classes than this project's (models that can't say are let through)."""
    names = getattr(models, "class_names", None)
    if names is None:
        return
    got = names()
    if not {n.lower() for n in got} & set(classes):
        raise AiUnavailable(f"None of this model's classes ({', '.join(got)}) match this project's classes ({', '.join(classes)}). "
                            "Pick a detector trained on this project's classes.")


def _open_for_autolabel(image: Image) -> bool:
    """Only pending images with no boxes: a reviewed image without boxes is a deliberate background picture, and nobody's boxes are replaced."""
    return image.review_status == ReviewStatus.pending and not image.annotations


def autolabel_image(db: Session, image: Image, models: Models, classes: dict[str, int], min_score: float = 0.5, policy: Policy = Policy()) -> Autolabel:
    """Draw boxes on an unlabelled pending image: the detector finds the drones, SAM tightens each box. The boxes are `auto` (model-proposed,
    awaiting review) and the image stays pending, so they stay out of every dataset version until a person accepts the image."""
    if not _open_for_autolabel(image):
        return Autolabel(True, 0, 0)
    rgb = load_rgb(image)
    height, width = rgb.shape[:2]
    kept: list[tuple[tuple[float, float, float, float], float, int]] = []  # (tight box, score, class id)
    unmapped = 0
    for det in sorted((d for d in models.detect(rgb) if d.score >= min_score), key=lambda d: -d.score):
        class_id = classes.get((det.class_name or "").lower())
        if class_id is None:
            unmapped += 1
            continue
        if any(iou(det.box, k[0]) >= 0.6 for k in kept):  # the same drone seen as two classes: the surer one stays
            continue
        box, _ = suggest(models, rgb, det, width, height, policy)
        x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in box)
        if (x2 - x1) * width < 2 or (y2 - y1) * height < 2:
            continue
        kept.append(((x1, y1, x2, y2), det.score, class_id))
    db.refresh(image)  # the person may have opened and edited it while the models ran: look again right before writing
    if not _open_for_autolabel(image):
        return Autolabel(True, 0, unmapped)
    for (x1, y1, x2, y2), score, class_id in kept:
        image.annotations.append(Annotation(class_id=class_id, cx=(x1 + x2) / 2, cy=(y1 + y2) / 2, w=x2 - x1, h=y2 - y1,
                                            status=AnnotationStatus.auto, confidence=round(score, 3)))
    if kept and image.ai_check is not None:  # its verdict was about an image without boxes
        image.ai_check.stale = True
    db.commit()
    return Autolabel(False, len(kept), unmapped)
