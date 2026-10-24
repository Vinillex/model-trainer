from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .. import refile
from ..ai.verdict import FLAGGED, fingerprint
from ..db import get_db
from ..models import AiDismissal, Annotation, AnnotationStatus, DetClass, Image, ReviewStatus
from ..schemas import AnnotationsPut, DismissResult, ImageDetail, ImageIdsIn, RejectResult
from .projects import get_project

router = APIRouter(tags=["annotations"])

EPS = 1e-6  # tolerance for float rounding at the image edge


def apply_annotations(db: Session, image: Image, boxes: list[dict], review_status: ReviewStatus | None = None, ai_dismiss: bool = False) -> None:
    """What saving an image does, without the commit: replace its boxes (every saved box is `reviewed`, which also ends "AI drew these"), set the
    review status, mark an AI check out of date when the boxes changed and record the person's decision on its doubts. The editor's save and the
    bulk Reject both come through here, so a rejected image leaves the AI lists the same way however it was rejected."""
    image.annotations.clear()
    db.flush()
    for box in boxes:
        image.annotations.append(Annotation(**box, status=AnnotationStatus.reviewed))
    if review_status is not None:
        image.review_status = review_status
    if image.ai_check is not None:  # an AI check describes the boxes it saw: once they change it is out of date
        saved = fingerprint((b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2) for b in boxes)
        image.ai_check.stale = saved != image.ai_check.fingerprint
        if ai_dismiss and image.ai_check.verdict in FLAGGED:  # the person saw the AI's doubts and decided: these boxes stand
            if image.ai_dismissal is None:
                image.ai_dismissal = AiDismissal(image_id=image.id, fingerprint=saved)
            else:
                image.ai_dismissal.fingerprint, image.ai_dismissal.dismissed_at = saved, datetime.now(timezone.utc)


def set_doubts_aside(image: Image) -> bool:
    """Record that the AI's doubts about the image's boxes as they are now have been seen and set aside (it leaves "AI: needs a look"). Does
    nothing, and says so, when there is no current flagged check."""
    check = image.ai_check
    if check is None or check.stale or check.verdict not in FLAGGED:
        return False
    if image.ai_dismissal is None:
        image.ai_dismissal = AiDismissal(image_id=image.id, fingerprint=check.fingerprint)
    else:
        image.ai_dismissal.fingerprint, image.ai_dismissal.dismissed_at = check.fingerprint, datetime.now(timezone.utc)
    return True


@router.put("/images/{image_id}/annotations", response_model=ImageDetail)
def put_annotations(image_id: int, body: AnnotationsPut, db: Session = Depends(get_db)):
    """Replace all boxes on an image (the editor always saves the full set) and optionally set its review status."""
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")

    classes = {c.id: c for c in db.query(DetClass).filter(DetClass.project_id == image.project_id)}
    for i, box in enumerate(body.boxes):
        cls = classes.get(box.class_id)
        if cls is None:
            raise HTTPException(422, f"box {i}: class {box.class_id} is not in this project")
        if cls.is_negative:
            raise HTTPException(422, f"box {i}: '{cls.name}' is a negative category and cannot have boxes")
        if cls.is_mixed:
            raise HTTPException(422, f"box {i}: '{cls.name}' is a folder, not a class; give the box its real class")
        if box.cx - box.w / 2 < -EPS or box.cx + box.w / 2 > 1 + EPS or box.cy - box.h / 2 < -EPS or box.cy + box.h / 2 > 1 + EPS:
            raise HTTPException(422, f"box {i}: extends outside the image")

    apply_annotations(db, image, [box.model_dump() for box in body.boxes], body.review_status, body.ai_dismiss)
    db.commit()
    db.refresh(image)
    refile.refile(db, image)  # a reviewed image moves to the folder its boxes show (never raises; the boxes are saved)
    db.refresh(image)
    return image


@router.post("/projects/{project_id}/images/reject", response_model=RejectResult)
def reject_images(project_id: int, body: ImageIdsIn, db: Session = Depends(get_db)):
    """Reject several images at once, exactly as the review screen's Reject does for one (their boxes are kept as they are). All or nothing: an id
    that is not an image of this project refuses the whole request. There is no bulk accept on purpose: accepting means a person looked."""
    get_project(db, project_id)
    ids = sorted(set(body.image_ids))
    images = db.scalars(select(Image).where(Image.project_id == project_id, Image.id.in_(ids))
                        .options(selectinload(Image.annotations), selectinload(Image.ai_check), selectinload(Image.ai_dismissal))).all()
    unknown = set(ids) - {i.id for i in images}
    if unknown:
        raise HTTPException(422, f"not images of this project: {sorted(unknown)[:10]}")
    for image in images:
        boxes = [{"class_id": a.class_id, "cx": a.cx, "cy": a.cy, "w": a.w, "h": a.h} for a in image.annotations]
        apply_annotations(db, image, boxes, ReviewStatus.rejected, ai_dismiss=True)
    db.commit()
    return RejectResult(rejected=len(images))


@router.post("/projects/{project_id}/images/ai-dismiss", response_model=DismissResult)
def dismiss_ai_doubts(project_id: int, body: ImageIdsIn, db: Session = Depends(get_db)):
    """Keep several images as they are although the AI flagged them: the same record a review Save makes (the AI's doubts about exactly these
    boxes are set aside, so the images leave "AI: needs a look"), without saving anything. This is how a flagged hard-negative picture, which the
    review screen only shows, is kept. A later check that flags other boxes brings an image back. All or nothing, like Reject."""
    get_project(db, project_id)
    ids = sorted(set(body.image_ids))
    images = db.scalars(select(Image).where(Image.project_id == project_id, Image.id.in_(ids))
                        .options(selectinload(Image.ai_check), selectinload(Image.ai_dismissal))).all()
    unknown = set(ids) - {i.id for i in images}
    if unknown:
        raise HTTPException(422, f"not images of this project: {sorted(unknown)[:10]}")
    count = sum(set_doubts_aside(image) for image in images)
    db.commit()
    return DismissResult(dismissed=count)
