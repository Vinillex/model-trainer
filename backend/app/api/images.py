import hashlib
import io

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import DetClass, Image, ReviewStatus, Source
from ..schemas import ImageDetail, ImageOut, UploadResult
from ..storage import get_storage
from .projects import get_project

router = APIRouter(tags=["images"])

EXT_BY_FORMAT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def ingest_image(db: Session, project_id: int, cls: DetClass, source: Source, data: bytes,
                 group_key: str | None = None) -> Image | None:
    """Validate and store one image. Returns None if it duplicates an existing one.

    Raises ValueError for undecodable, unsupported or too-small images.
    """
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            im.verify()
        with PILImage.open(io.BytesIO(data)) as im:
            fmt, (width, height) = im.format, im.size
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("not a decodable image") from e
    if fmt not in EXT_BY_FORMAT:
        raise ValueError(f"unsupported format {fmt}")
    if min(width, height) < settings.min_image_side:
        raise ValueError(f"image smaller than {settings.min_image_side}px")

    sha = hashlib.sha256(data).hexdigest()
    if db.scalar(select(Image.id).where(Image.project_id == project_id, Image.sha256 == sha)):
        return None

    key = f"projects/{project_id}/raw/{cls.name}/{source.value}/{sha}.{EXT_BY_FORMAT[fmt]}"
    get_storage().put(key, data)
    image = Image(project_id=project_id, class_id=cls.id, source=source, storage_key=key, sha256=sha,
                  width=width, height=height, group_key=group_key,
                  review_status=ReviewStatus.reviewed if cls.is_negative else ReviewStatus.pending)
    db.add(image)
    db.flush()
    return image


@router.post("/projects/{project_id}/classes/{class_id}/images", response_model=UploadResult)
def upload_images(project_id: int, class_id: int, files: list[UploadFile] = File(...),
                  group_key: str | None = None, db: Session = Depends(get_db)):
    get_project(db, project_id)
    cls = db.get(DetClass, class_id)
    if cls is None or cls.project_id != project_id:
        raise HTTPException(404, "class not found")

    result = UploadResult(added=[], duplicates=[], rejected=[])
    for f in files:
        name = f.filename or "unnamed"
        try:
            image = ingest_image(db, project_id, cls, Source.uploaded, f.file.read(), group_key)
        except ValueError as e:
            result.rejected.append(f"{name}: {e}")
            continue
        if image is None:
            result.duplicates.append(name)
        else:
            result.added.append(ImageOut.model_validate(image))
    db.commit()
    return result


@router.get("/projects/{project_id}/images", response_model=list[ImageOut])
def list_images(project_id: int, class_id: int | None = None, source: Source | None = None,
                review_status: ReviewStatus | None = None, limit: int = Query(50, le=500), offset: int = 0, db: Session = Depends(get_db)):
    get_project(db, project_id)
    q = select(Image).where(Image.project_id == project_id)
    if class_id is not None:
        q = q.where(Image.class_id == class_id)
    if source is not None:
        q = q.where(Image.source == source)
    if review_status is not None:
        q = q.where(Image.review_status == review_status)
    return db.scalars(q.order_by(Image.id.desc()).limit(limit).offset(offset)).all()


@router.get("/images/{image_id}", response_model=ImageDetail)
def read_image(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return image


@router.get("/images/{image_id}/file")
def image_file(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    ext = image.storage_key.rsplit(".", 1)[-1]
    media = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[ext]
    return Response(get_storage().get(image.storage_key), media_type=media)


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    get_storage().delete(image.storage_key)
    db.delete(image)
    db.commit()
