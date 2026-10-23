"""Glue between the AI check and the app: loads the real models on first use, reads an image and its boxes, stores the result.

A check only ever writes its own row in `ai_checks`. The auto-label job is the one thing here that writes boxes, and only as `auto` boxes on a
pending image that has none (never `reviewed`, so nothing it writes can reach a training set before a person accepts it)."""

import io
import importlib.util
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
from .geometry import iou, to_normalized, to_pixels
from .registry import SAM_FILES, AiUnavailable, models_dir, resolve_detector, run_weights  # noqa: F401 - AiUnavailable is part of this module's API

KEEP_LOADED = 2  # model sets kept in memory at once (the Mac has 16 GB); the least recently used one is dropped

_override: Models | None = None  # tests put fakes here; they win over any choice
_real: "OrderedDict[tuple[str, str], RealModels]" = OrderedDict()
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
    """The trained YOLO as detector and one SAM size as refiner. Nothing heavy loads until the first image."""

    def __init__(self, detector_path: Path, detector_label: str, sam_path: Path):
        from .detector import Detector  # the heavy imports
        from .sam import SamRefiner

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


def get_models(db: Session, detector: str | None = None, sam: str | None = None) -> Models:
    """The models for a detector choice ("run:5", "file:name.pt", None = the default) and a SAM size ("t" "s" "b" "l", None = the setting).
    Built on first use and kept (the last KEEP_LOADED sets), so switching back is free. Weights load on the first image, so a first check is slow.
    A caller keeps the object it got: being dropped from the cache never unloads it from under a running job."""
    if _override is not None:
        return _override
    if (reason := missing_extras()) is not None:
        raise AiUnavailable(reason)
    with _build_lock:
        det_key, det_path, det_label = resolve_detector(db, detector, detector_weights)
        sam_file = SAM_FILES[sam] if sam else settings.ai_sam_weights
        key = (det_key, sam_file)
        models = _real.get(key)
        if models is None:
            models = _real[key] = RealModels(det_path, det_label, models_dir() / sam_file)
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


def check_image(db: Session, image: Image, models: Models | None = None, policy: Policy = Policy()) -> AiCheck:
    """Run the AI check on one image and store (replace) its row. Never touches the image's boxes or review status."""
    models = models or get_models(db)
    result = run_check(load_rgb(image), label_boxes(image), models, policy)
    row = image.ai_check or AiCheck(image_id=image.id)
    row.checked_at = datetime.now(timezone.utc)
    row.model = model_label(models)
    row.fingerprint = result.pop("fingerprint")
    row.verdict = result.pop("verdict")
    row.stale = False
    row.result = result
    image.ai_check = row
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


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
