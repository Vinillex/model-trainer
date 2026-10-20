from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import refile
from ..db import get_db
from ..models import Annotation, AnnotationStatus, DetClass, Image
from ..schemas import AnnotationsPut, ImageDetail

router = APIRouter(tags=["annotations"])

EPS = 1e-6  # tolerance for float rounding at the image edge


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

    image.annotations.clear()
    db.flush()
    for box in body.boxes:
        image.annotations.append(Annotation(**box.model_dump(), status=AnnotationStatus.reviewed))
    if body.review_status is not None:
        image.review_status = body.review_status
    db.commit()
    db.refresh(image)
    refile.refile(db, image)  # a reviewed image moves to the folder its boxes show (never raises; the boxes are saved)
    db.refresh(image)
    return image
