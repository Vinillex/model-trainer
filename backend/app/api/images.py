import hashlib
import io
import logging
import re
from collections.abc import Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from sqlalchemy import and_, case, exists, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from ..ai.filters import AI_FILTER_VALUES, ai_condition
from ..config import settings
from ..db import get_db
from .. import datasets, imagehash, refile as refile_mod
from ..models import AiDismissal, Annotation, DetClass, Image, ImageHash, ImportRecord, ManualCheck, ReviewStatus, Source, UploadRecord
from ..refile import storage_key_for
from ..schemas import FOLDER_PATTERN, FolderOut, ImageCount, ImageDetail, ImageIdsIn, ImageOut, ImportedDatasetOut, PurgeResult, RestoreResult, TrashDeleteIn, UploadResult
from ..storage import get_storage
from .annotations import set_doubts_aside
from .projects import get_project

router = APIRouter(tags=["images"])
log = logging.getLogger(__name__)

EXT_BY_FORMAT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def clean_folder(name: str) -> str | None:
    """A folder name that is safe as a storage path part and in a URL: letters, digits, dot, dash and underscore.
    Anything else becomes a dash; a name with nothing usable in it gives None (the caller rejects it, never guesses)."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    cleaned = re.sub(r"\.{2,}", ".", cleaned).strip(".-_")[:80].strip(".-_")  # no "..", no leading dot or dash
    return cleaned if re.fullmatch(FOLDER_PATTERN, cleaned) else None


class NearDuplicate(ValueError):
    """The image is a re-encoded/resized copy of one the caller already has."""


def ingest_image(db: Session, project_id: int, cls: DetClass, source: Source, data: bytes,
                 group_key: str | None = None, min_side: int | None = None,
                 is_near_duplicate: Callable[[int], bool] | None = None, subfolder: str | None = None) -> Image | None:
    """Validate and store one image. Returns None if its bytes duplicate an existing image.

    Raises ValueError for undecodable, unsupported or too-small images, and NearDuplicate if
    `is_near_duplicate(dhash)` says so. Only the caller knows whether near-duplicates matter
    (scraping: yes; a deliberate upload: no).
    """
    min_side = settings.min_image_side if min_side is None else min_side
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            im.verify()
        with PILImage.open(io.BytesIO(data)) as im:
            fmt, (width, height) = im.format, im.size
        dhash = imagehash.dhash(data)
    except (UnidentifiedImageError, OSError, SyntaxError) as e:  # Pillow raises SyntaxError for a PNG with a bad checksum
        raise ValueError("not a decodable image") from e
    if fmt not in EXT_BY_FORMAT:
        raise ValueError(f"unsupported format {fmt}")
    if min(width, height) < min_side:
        raise ValueError(f"image smaller than {min_side}px")

    sha = hashlib.sha256(data).hexdigest()
    if db.scalar(select(Image.id).where(Image.project_id == project_id, Image.sha256 == sha)):
        return None
    if is_near_duplicate is not None and is_near_duplicate(dhash):
        raise NearDuplicate("near-duplicate of an existing image")

    key = storage_key_for(project_id, cls.name, source.value, sha, EXT_BY_FORMAT[fmt], subfolder)
    get_storage().put(key, data)
    image = Image(project_id=project_id, class_id=cls.id, source=source, storage_key=key, sha256=sha,
                  width=width, height=height, group_key=group_key,
                  review_status=ReviewStatus.reviewed if cls.is_negative else ReviewStatus.pending,
                  hash=ImageHash(dhash=imagehash.to_hex(dhash)))
    db.add(image)
    db.flush()
    return image


@router.post("/projects/{project_id}/classes/{class_id}/images", response_model=UploadResult)
def upload_images(project_id: int, class_id: int, files: list[UploadFile] = File(...), group_key: str | None = None,
                  folder: str | None = None, relative_paths: list[str] = Form([]), db: Session = Depends(get_db)):
    """`folder` files the images under <class>/uploaded/<folder>. `relative_paths` (one per file, same order) says where each sat
    inside the picked folder; it is kept for the record and never used as a path."""
    get_project(db, project_id)
    if relative_paths and len(relative_paths) != len(files):
        raise HTTPException(422, "relative_paths must have one entry per file")
    folder_name = None
    if folder is not None:
        folder_name = clean_folder(folder)
        if folder_name is None:
            raise HTTPException(422, f"folder name {folder!r} has no letters or digits to use; rename the folder (A-Z, a-z, 0-9, dot, dash, underscore)")
    cls = db.get(DetClass, class_id)
    if cls is None or cls.project_id != project_id:
        raise HTTPException(404, "class not found")
    if cls.is_mixed:
        raise HTTPException(422, "'mixed' is filled automatically when a reviewed image shows several classes; pick another class")

    result = UploadResult(added=[], duplicates=[], rejected=[], folder=folder_name)
    for i, f in enumerate(files):
        name = f.filename or "unnamed"
        try:
            image = ingest_image(db, project_id, cls, Source.uploaded, f.file.read(), group_key, subfolder=folder_name)
        except ValueError as e:
            result.rejected.append(f"{name}: {e}")
            continue
        if image is None:
            result.duplicates.append(name)
        else:
            if folder_name:
                path = re.sub(r"[\x00-\x1f]", "", relative_paths[i] if relative_paths else name)[:500]
                image.upload_record = UploadRecord(folder=folder_name, original_path=path or name)
            result.added.append(ImageOut.model_validate(image))
    db.commit()
    return result


@router.post("/projects/{project_id}/images/refile")
def refile_images(project_id: int, dry_run: bool = True, db: Session = Depends(get_db)):
    """Apply the folder rule to reviewed images (those reviewed before it existed). Dry run by default: it only reports."""
    get_project(db, project_id)
    return refile_mod.refile_project(db, project_id, dry_run)


def _filtered(q, class_id, box_class_id, source, dataset, folder, manually_checked, ai, review_status, exclude_rejected):
    """The filters the Images page offers, applied to any select over `Image`: the list and its count share them, so they can't disagree."""
    if class_id is not None:
        q = q.where(Image.class_id == class_id)
    if box_class_id is not None:
        q = q.where(exists().where(Annotation.image_id == Image.id, Annotation.class_id == box_class_id))
    if source is not None:
        q = q.where(Image.source == source)
    if dataset is not None:
        q = q.join(ImportRecord, ImportRecord.image_id == Image.id).where(ImportRecord.dataset == dataset)
    if folder is not None:
        q = q.where(Image.upload_record.has(UploadRecord.folder == folder))
    if manually_checked is True:
        q = q.where(Image.manual_check.has())
    elif manually_checked is False:
        q = q.where(Image.source == Source.imported, ~Image.manual_check.has())
    if ai is not None:
        q = q.where(ai_condition(ai))
    if review_status is not None:
        q = q.where(Image.review_status == review_status)
    elif exclude_rejected:
        q = q.where(Image.review_status != ReviewStatus.rejected)
    return q


@router.get("/projects/{project_id}/images", response_model=list[ImageOut])
def list_images(project_id: int, class_id: int | None = None, box_class_id: int | None = None, source: Source | None = None,
                dataset: str | None = Query(None, pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$"), folder: str | None = Query(None, pattern=FOLDER_PATTERN),
                manually_checked: bool | None = None, exclude_rejected: bool = False,
                ai: str | None = Query(None, pattern="^(" + "|".join(AI_FILTER_VALUES) + ")$"),
                review_status: ReviewStatus | None = None, limit: int = Query(50, le=500), offset: int = 0, db: Session = Depends(get_db)):
    """`dataset` keeps only the images imported from that public dataset, so each one can be reviewed on its own.
    `class_id` is the folder an image is filed under; `box_class_id` keeps the images that have at least one box of that class, wherever they are
    filed (an image in the `mixed` folder has boxes of several classes), which is how to find what still uses a class.
    `folder` keeps only the images uploaded in that folder.
    `exclude_rejected` leaves out the trash (rejected images): the Images page and the review queue ask for it unless a review status is chosen.
    `manually_checked` is a tag only imported images have: true = opened in the review window, false = imported but not yet.
    `ai` filters on the AI check: unchecked, stale (boxes edited since), ok, flagged (suspect/missing/adjust that nobody has dismissed),
    dismissed (flagged, but a person reviewed those boxes anyway), too_small or autolabelled (boxes drawn by the auto-label job, not yet confirmed)."""
    get_project(db, project_id)
    q = select(Image).where(Image.project_id == project_id).options(
        selectinload(Image.import_record), selectinload(Image.upload_record), selectinload(Image.manual_check), selectinload(Image.ai_check), selectinload(Image.ai_dismissal),
        selectinload(Image.annotations))
    q = _filtered(q, class_id, box_class_id, source, dataset, folder, manually_checked, ai, review_status, exclude_rejected)
    return db.scalars(q.order_by(Image.id.desc()).limit(limit).offset(offset)).all()


@router.get("/projects/{project_id}/images/count", response_model=ImageCount)
def count_images(project_id: int, class_id: int | None = None, box_class_id: int | None = None, source: Source | None = None,
                 dataset: str | None = Query(None, pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$"), folder: str | None = Query(None, pattern=FOLDER_PATTERN),
                 manually_checked: bool | None = None, exclude_rejected: bool = False,
                 ai: str | None = Query(None, pattern="^(" + "|".join(AI_FILTER_VALUES) + ")$"),
                 review_status: ReviewStatus | None = None, db: Session = Depends(get_db)):
    """How many images `GET .../images` would list with the same filters (it takes no `limit` or `offset`)."""
    get_project(db, project_id)
    q = select(func.count(Image.id)).select_from(Image).where(Image.project_id == project_id)
    q = _filtered(q, class_id, box_class_id, source, dataset, folder, manually_checked, ai, review_status, exclude_rejected)
    return ImageCount(count=db.scalar(q) or 0)


@router.get("/projects/{project_id}/folders", response_model=list[FolderOut])
def folders(project_id: int, db: Session = Depends(get_db)):
    """Where the images live: one row per class, source and (imported dataset or upload folder), with how many are in each review state.
    `ai_flagged` leaves out the trash, like the Images page does: a rejected image the AI doubted is not waiting for anybody to look."""
    get_project(db, project_id)
    where = func.coalesce(ImportRecord.dataset, UploadRecord.folder)
    count = lambda cond: func.sum(case((cond, 1), else_=0))  # noqa: E731
    # Only one-row-per-image tables are joined (import and upload records) and the AI conditions are EXISTS subqueries: nothing multiplies the rows.
    rows = db.execute(
        select(Image.class_id, Image.source, where, func.count(), count(Image.review_status == ReviewStatus.reviewed),
               count(Image.review_status == ReviewStatus.pending), count(Image.review_status == ReviewStatus.rejected),
               count(ai_condition("ok")), count(and_(ai_condition("flagged"), Image.review_status != ReviewStatus.rejected)), count(ai_condition("unchecked")))
        .outerjoin(ImportRecord, ImportRecord.image_id == Image.id).outerjoin(UploadRecord, UploadRecord.image_id == Image.id)
        .where(Image.project_id == project_id).group_by(Image.class_id, Image.source, where).order_by(Image.class_id, Image.source, where))
    return [FolderOut(class_id=c, source=s, folder=f, total=t, reviewed=r, pending=p, rejected=x, ai_ok=ok, ai_flagged=fl, ai_unchecked=un)
            for c, s, f, t, r, p, x, ok, fl, un in rows]


@router.get("/projects/{project_id}/imported-datasets", response_model=list[ImportedDatasetOut])
def imported_datasets(project_id: int, db: Session = Depends(get_db)):
    """Every imported dataset in the project with how many of its images are in each review state."""
    get_project(db, project_id)
    rows = db.execute(
        select(ImportRecord.dataset, Image.review_status, func.count())
        .join(Image, Image.id == ImportRecord.image_id)
        .where(Image.project_id == project_id)
        .group_by(ImportRecord.dataset, Image.review_status)
    ).all()
    counts: dict[str, dict[str, int]] = {}
    for name, status, n in rows:
        counts.setdefault(name, {})[status.value] = n
    checked = dict(db.execute(
        select(ImportRecord.dataset, func.count())
        .join(ManualCheck, ManualCheck.image_id == ImportRecord.image_id)
        .join(Image, Image.id == ImportRecord.image_id)
        .where(Image.project_id == project_id)
        .group_by(ImportRecord.dataset)
    ).all())
    return [ImportedDatasetOut(name=name, total=sum(c.values()), reviewed=c.get("reviewed", 0),
                               pending=c.get("pending", 0), rejected=c.get("rejected", 0), checked=checked.get(name, 0))
            for name, c in sorted(counts.items())]


@router.get("/images/{image_id}", response_model=ImageDetail)
def read_image(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return image


@router.post("/images/{image_id}/manual-check", response_model=ImageOut)
def mark_manually_checked(image_id: int, db: Session = Depends(get_db)):
    """The review window calls this when it shows an imported image: opening it there is what counts as checking it by hand.
    Safe to repeat."""
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    if image.source != Source.imported:
        raise HTTPException(409, "only imported images have a manual check")
    if image.manual_check is None:
        image.manual_check = ManualCheck()
        try:
            db.commit()
        except IntegrityError:  # two requests marked it at once: it is checked either way
            db.rollback()
    return image


@router.get("/images/{image_id}/file")
def image_file(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    ext = image.storage_key.rsplit(".", 1)[-1]
    media = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}[ext]
    return Response(get_storage().get(image.storage_key), media_type=media)


def purge_images(db: Session, project_id: int, images: list[Image], *, protect_versions: bool = False) -> None:
    """Delete images for good: the rows (with their boxes, checks and records), then the files. The rows go first: a file left behind is harmless,
    a row whose file is gone is not. With `protect_versions`, images that an existing dataset version contains refuse the whole request (409):
    deleting them stays possible (a version then reports its missing files), but only when asked for explicitly."""
    using = datasets.versions_using(db, project_id, {i.id for i in images}) if protect_versions else {}
    if using:
        names = ", ".join(sorted(using))
        count = len({i for ids in using.values() for i in ids})
        raise HTTPException(409, f"{count} of these images {'is' if count == 1 else 'are'} part of dataset version {names}. Deleting "
                                 f"{'it' if count == 1 else 'them'} means that version can no longer be downloaded. Delete anyway only if you do not need it.")
    keys = [i.storage_key for i in images]
    for image in images:
        db.delete(image)
    db.commit()
    for key in keys:
        try:
            get_storage().delete(key)
        except Exception:  # noqa: BLE001 - the rows are gone; a leftover file is harmless
            log.warning("could not delete the file %s", key)


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    purge_images(db, image.project_id, [image])


def _in_trash(db: Session, project_id: int, image_ids: list[int]) -> list[Image]:
    """The rejected images with these ids. All or nothing: an id that is not a rejected image of this project refuses the whole request."""
    ids = sorted(set(image_ids))
    found = db.scalars(select(Image).where(Image.project_id == project_id, Image.id.in_(ids), Image.review_status == ReviewStatus.rejected)
                       .options(selectinload(Image.ai_dismissal), selectinload(Image.ai_check))).all()
    other = set(ids) - {i.id for i in found}
    if other:
        raise HTTPException(422, f"not rejected images of this project: {sorted(other)[:10]}")
    return list(found)


@router.post("/projects/{project_id}/images/restore", response_model=RestoreResult)
def restore_images(project_id: int, body: ImageIdsIn, db: Session = Depends(get_db)):
    """Take rejected images out of the trash: they stay in the class they were filed under, boxes as they were, and become "not reviewed" again,
    except a hard-negative picture (no boxes, nothing to review), which is reviewed as it was. Restoring says nothing about the AI: a doubt it
    had about the picture is set aside, so a restored image never lands on "AI: needs a look"."""
    get_project(db, project_id)
    images = _in_trash(db, project_id, body.image_ids)
    negatives = set(db.scalars(select(DetClass.id).where(DetClass.project_id == project_id, DetClass.is_negative.is_(True))))
    for image in images:
        image.review_status = ReviewStatus.reviewed if image.class_id in negatives else ReviewStatus.pending
        set_doubts_aside(image)
    db.commit()
    return RestoreResult(restored=len(images))


@router.post("/projects/{project_id}/images/delete", response_model=PurgeResult)
def delete_trashed_images(project_id: int, body: TrashDeleteIn, db: Session = Depends(get_db)):
    """Delete rejected images for good (files included). Only the trash can be deleted in bulk. Images that an existing dataset version contains
    are refused (409, naming the versions) unless `allow_breaking_versions` says the person knows."""
    get_project(db, project_id)
    images = _in_trash(db, project_id, body.image_ids)
    purge_images(db, project_id, images, protect_versions=not body.allow_breaking_versions)
    return PurgeResult(deleted=len(images))
