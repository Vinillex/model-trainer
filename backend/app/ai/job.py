"""Batch AI check as a background job: the images the gallery filters select, one at a time, resumable (checked images are skipped)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models import DetClass, Image, ImportRecord, Job, ReviewStatus, UploadRecord
from . import service
from .verdict import fingerprint


def select_images(db: Session, project_id: int, p: dict) -> list[int]:
    q = (select(Image.id).join(DetClass, DetClass.id == Image.class_id)
         .where(Image.project_id == project_id, DetClass.is_negative.is_(False)))  # hard-negative images carry no boxes to check
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
    if p.get("mode") == "autolabel":  # unlabelled images nobody has looked at yet: a reviewed image without boxes is a background picture
        q = q.where(Image.review_status == ReviewStatus.pending, ~Image.annotations.any())
    return list(db.scalars(q.order_by(Image.id).limit(int(p.get("limit", 500)))))


def handle_ai_check(db: Session, job: Job) -> None:
    """mode "check" (the default; older jobs have no mode): store the AI's opinion of each image's boxes. mode "autolabel": draw boxes on
    unlabelled pending images. Either way the job holds the models it was started with, whatever else is loaded meanwhile."""
    mode = job.params.get("mode", "check")
    ids = select_images(db, job.project_id, job.params)
    progress = {"mode": mode, "total": len(ids), "done": 0, "skipped": 0, "flagged": 0, "labelled": 0, "boxes": 0, "nothing": 0, "unmapped": 0,
                "failed": 0, "last_error": None}
    # raises AiUnavailable (the job fails with the instruction) before any image is touched
    models = service.get_models(db, job.params.get("detector"), job.params.get("sam"))
    label = service.model_label(models)
    classes = None
    if mode == "autolabel":
        classes = service.class_map(db, job.project_id)
        service.require_matching_classes(models, classes)
    for image_id in ids:
        db.refresh(job)
        if job.cancel_requested:
            break
        image = db.get(Image, image_id)
        if image is None:
            continue
        try:
            if mode == "autolabel":
                r = service.autolabel_image(db, image, models, classes, float(job.params.get("min_score", 0.5)))
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
