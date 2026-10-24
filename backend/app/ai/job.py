"""Batch AI check as a background job: the images the gallery filters select, one at a time, resumable (checked images are skipped)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import DetClass, Image, ImportRecord, Job, ReviewStatus, UploadRecord
from . import service
from .prompts import DRONE_LABEL, DRONE_PROMPT, default_object_prompt, object_label
from .registry import YOLOE_ID
from .verdict import FLAGGED, fingerprint


def select_images(db: Session, project_id: int, p: dict) -> list[int]:
    # Hard-negative images carry no boxes to check or draw on, so check and autolabel skip them; the negatives mode looks at nothing else.
    q = (select(Image.id).join(DetClass, DetClass.id == Image.class_id)
         .where(Image.project_id == project_id, DetClass.is_negative.is_(p.get("mode") == "negatives")))
    if p.get("class_id") is not None:
        q = q.where(Image.class_id == p["class_id"])
    if p.get("source"):
        q = q.where(Image.source == p["source"])
    if p.get("dataset"):
        q = q.join(ImportRecord, ImportRecord.image_id == Image.id).where(ImportRecord.dataset == p["dataset"])
    if p.get("image_ids") is not None:  # `is not None`: an empty list must select nothing, not everything
        q = q.where(Image.id.in_(p["image_ids"]))
    if p.get("folder"):
        q = q.join(UploadRecord, UploadRecord.image_id == Image.id).where(UploadRecord.folder == p["folder"])
    if p.get("review_status"):
        q = q.where(Image.review_status == p["review_status"])
    elif p.get("mode") == "negatives":  # the trash was thrown out by eye: looking inside it only makes flags nobody will see
        q = q.where(Image.review_status != ReviewStatus.rejected)
    if p.get("mode") == "autolabel":  # unlabelled images nobody has looked at yet: a reviewed image without boxes is a background picture
        q = q.where(Image.review_status == ReviewStatus.pending, ~Image.annotations.any())
    return list(db.scalars(q.order_by(Image.id).limit(int(p.get("limit", 500)))))


def build_prompts(db: Session, project_id: int, p: dict) -> dict[int | None, list[tuple[str, str]]]:
    """The words the text-prompt detector gets for a run, as {key: [(label, words)]}. Key None: the drone phrase alone (drawing boxes). In the
    negatives mode one entry per hard-negative class: the drone phrase plus the words for what that class holds ("bird"; a blank or missing entry
    means no size check, the phrase alone). One class at a time on purpose: the detector gives each object the single best-scoring phrase, so
    with "airplane" and "balloon" loaded as well a gull is called an airplane and never found as a bird (seen on real pictures).
    Raises AiUnavailable with something the person can act on."""
    drone = (p.get("prompt") or DRONE_PROMPT).strip()
    if p.get("mode") != "negatives":
        return {None: [(DRONE_LABEL, drone)]}
    negatives = {c.id: c for c in db.scalars(select(DetClass).where(DetClass.project_id == project_id, DetClass.is_negative.is_(True)))}
    given = p.get("object_prompts")
    if given is not None and (unknown := sorted(k for k in given if int(k) not in negatives)):
        raise service.AiUnavailable(f"class {unknown[0]} is not a hard-negative class of this project")
    out = {}
    for cid, cls in negatives.items():
        words = (given.get(str(cid), "") if given is not None else default_object_prompt(cls.name)).strip()
        out[cid] = [(DRONE_LABEL, drone)] + ([(object_label(cid), words)] if words else [])
        if len({w.lower() for _, w in out[cid]}) != len(out[cid]):
            raise service.AiUnavailable(f"the words for {cls.name} are the same as the words that find a drone; the detector could not tell them apart")
    return out


def handle_ai_check(db: Session, job: Job) -> None:
    """mode "check" (the default; older jobs have no mode): store the AI's opinion of each image's boxes. mode "autolabel": draw boxes on
    unlabelled pending images. mode "negatives": look inside the hard-negative classes (a drone-like object, a too-small or missing bird) with
    the text-prompt detector. Either way the job holds the models it was started with, whatever else is loaded meanwhile."""
    mode = job.params.get("mode", "check")
    ids = select_images(db, job.project_id, job.params)
    progress = {"mode": mode, "total": len(ids), "done": 0, "skipped": 0, "flagged": 0, "labelled": 0, "boxes": 0, "nothing": 0, "unmapped": 0,
                "failed": 0, "last_error": None}
    detector = YOLOE_ID if mode == "negatives" else job.params.get("detector")
    prompt_sets = build_prompts(db, job.project_id, job.params) if detector == YOLOE_ID else {}
    if mode == "negatives" and not prompt_sets:
        raise service.AiUnavailable("this project has no hard-negative class to look inside")
    prompts = prompt_sets.get(None)  # drawing boxes: one set for everything
    # raises AiUnavailable (the job fails with the instruction) before any image is touched
    models_for = {key: service.get_models(db, detector, job.params.get("sam"), *([ps] if ps else [])) for key, ps in (prompt_sets or {None: None}).items()}
    models = models_for[None] if None in models_for else None
    label = service.model_label(models) if models else None
    classes = None
    if mode == "autolabel":
        classes = service.class_map(db, job.project_id)
        if prompts is None:
            service.require_matching_classes(models, classes)
        elif not classes:
            raise service.AiUnavailable("this project has no class a box can be filed under")
    min_score = job.params.get("min_score")  # None: the mode's own default (the text-prompt detector scores lower than a trained one)
    rules: dict[int, service.NegativeRules] = {}
    for cid, ps in prompt_sets.items():
        if cid is not None:
            rules[cid] = service.NegativeRules(float(min_score or 0.25), float(job.params.get("min_size", 0.10)), ps[0][1],
                                               ps[1][0] if len(ps) > 1 else None, ps[1][1] if len(ps) > 1 else "")
    for image_id in ids:
        db.refresh(job)
        if job.cancel_requested:
            break
        image = db.get(Image, image_id)
        if image is None:
            continue
        try:
            if mode == "negatives":
                rule, class_models = rules[image.class_id], models_for[image.class_id]
                done_label = service.negatives_label(class_models, rule)
                if not job.params.get("force") and image.ai_check is not None and not image.ai_check.stale and image.ai_check.model == done_label:
                    progress["skipped"] += 1
                else:
                    progress["flagged"] += service.check_negative(db, image, class_models, rule).verdict in FLAGGED
            elif mode == "autolabel":
                if prompts is None:
                    filed = classes
                else:  # the text-prompt detector cannot name our classes (measured): its boxes go under the class the picture is filed under
                    filed = {DRONE_LABEL: image.class_id} if image.class_id in classes.values() else {}
                r = service.autolabel_image(db, image, models, filed, float(min_score or (0.25 if prompts else 0.5)))
                progress["skipped"] += r.skipped
                progress["labelled"] += r.boxes > 0
                progress["boxes"] += r.boxes
                progress["nothing"] += not r.skipped and r.boxes == 0
                progress["unmapped"] += r.unmapped
            elif not job.params.get("force") and image.ai_check is not None and not image.ai_check.stale and image.ai_check.model == label \
                    and image.ai_check.fingerprint == fingerprint(lb.box for lb in service.label_boxes(image)):
                progress["skipped"] += 1
            else:
                row = service.check_image(db, image, models)
                progress["flagged"] += row.verdict in ("suspect", "missing", "adjust")
        except Exception as e:  # noqa: BLE001 - one unreadable image must not stop a batch of thousands
            db.rollback()
            progress["failed"] += 1
            progress["last_error"] = f"image {image_id}: {type(e).__name__}: {e}"[:300]
        progress["done"] += 1
        progress["device"] = service.device_summary()
        job.progress = dict(progress)
        db.commit()
    job.progress = dict(progress)
    db.commit()
