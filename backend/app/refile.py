"""Keeping each image in the folder of what its boxes show.

The class chosen at scrape or upload time is only a guess (one search returns many kinds of drone). Once an image is
reviewed its boxes are the truth, so the image (its file and its class) moves to that class's folder, or to `mixed` when
its boxes show two or more classes.

The move is the last step and never raises: the review is already committed, and a failed move must not cost anyone their
boxes. Order: copy the file, switch the database row, and only then delete the old copy.
"""

import logging

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from .models import MIXED, DetClass, Image, ReviewStatus
from .storage import get_storage

log = logging.getLogger(__name__)


def storage_key_for(project_id: int, folder: str, source: str, sha: str, ext: str, subfolder: str | None = None) -> str:
    """raw/<class>/<source>/[<subfolder>/]<sha>.<ext>. Imported images get a subfolder named after their dataset."""
    middle = f"{source}/{subfolder}" if subfolder else source
    return f"projects/{project_id}/raw/{folder}/{middle}/{sha}.{ext}"


def ensure_mixed(db: Session, project_id: int) -> DetClass:
    """The project's `mixed` folder class, created on first use. It has no index: it is never a label."""
    find = select(DetClass).where(DetClass.project_id == project_id, DetClass.name == MIXED)
    cls = db.scalar(find)
    if cls is None:
        cls = DetClass(project_id=project_id, name=MIXED, is_negative=False, index=None)
        db.add(cls)
        try:
            db.flush()
        except IntegrityError:  # another request created it first
            db.rollback()
            cls = db.scalar(find)
    return cls


def target_class(db: Session, image: Image) -> DetClass | None:
    """Where this image belongs, or None when there's no reason to move it (not reviewed, or no boxes to go by)."""
    if image.review_status != ReviewStatus.reviewed or not image.annotations:
        return None
    classes = {a.class_id for a in image.annotations}
    return db.get(DetClass, next(iter(classes))) if len(classes) == 1 else ensure_mixed(db, image.project_id)


def refile(db: Session, image: Image) -> bool:
    """Move a reviewed image to the folder its boxes show. True if it moved. Never raises."""
    storage = get_storage()
    new_key = None
    try:
        target = target_class(db, image)
        if target is None or target.id == image.class_id:
            return False
        old_key = image.storage_key
        # an imported image stays in its dataset's folder, an uploaded one in the folder it was uploaded in
        subfolder = image.import_record.dataset if image.import_record else image.upload_record.folder if image.upload_record else None
        new_key = storage_key_for(image.project_id, target.name, image.source.value, image.sha256, old_key.rsplit(".", 1)[-1], subfolder)
        if new_key == old_key:
            return False
        if db.scalar(select(Image.id).where(Image.storage_key == new_key)):
            log.warning("image %s stays put: another image already lives at %s", image.id, new_key)
            return False
        storage.put(new_key, storage.get(old_key))
        image.class_id, image.storage_key = target.id, new_key
        try:
            db.commit()
        except Exception:
            db.rollback()
            storage.delete(new_key)  # safe: nothing else uses that path (checked above)
            raise
        try:
            storage.delete(old_key)
        except Exception:  # noqa: BLE001 - the row is already switched; a leftover file is harmless
            log.warning("could not delete the old copy %s", old_key)
        return True
    except Exception:  # noqa: BLE001 - the boxes are already saved; the image just stays where it is
        log.exception("could not move image %s to its class folder", image.id)
        db.rollback()
        return False


def rename_class(db: Session, cls: DetClass, new_name: str) -> int:
    """Give a class a new name and move the files of every image filed under it to the folder of that name. Returns how many moved.

    All or nothing: the new copies are made first, then the class name and every row switch in ONE commit; if anything fails the copies are
    removed and nothing changed. The old files go last (a leftover is harmless). Frozen dataset versions keep working, they find an image by id + SHA-256
    (datasets.resolve_manifest), and keep the old name in their own labels on purpose."""
    storage = get_storage()
    images = db.scalars(select(Image).where(Image.class_id == cls.id).options(selectinload(Image.import_record), selectinload(Image.upload_record))).all()
    copied: list[str] = []
    moves: list[tuple[Image, str, str]] = []
    try:
        for image in images:
            old_key = image.storage_key
            subfolder = image.import_record.dataset if image.import_record else image.upload_record.folder if image.upload_record else None
            new_key = storage_key_for(image.project_id, new_name, image.source.value, image.sha256, old_key.rsplit(".", 1)[-1], subfolder)
            if new_key == old_key:
                continue
            if db.scalar(select(Image.id).where(Image.storage_key == new_key)):
                raise ValueError(f"another image already lives at {new_key}")
            storage.put(new_key, storage.get(old_key))
            copied.append(new_key)
            moves.append((image, old_key, new_key))
        cls.name = new_name
        for image, _, new_key in moves:
            image.storage_key = new_key
        db.commit()
    except Exception:
        db.rollback()
        for key in copied:
            try:
                storage.delete(key)  # safe: no row points at these copies (checked above, and the commit did not happen)
            except Exception:  # noqa: BLE001
                log.warning("could not remove the copy %s", key)
        raise
    for _, old_key, _ in moves:
        try:
            storage.delete(old_key)
        except Exception:  # noqa: BLE001 - the rows are switched; a leftover file is harmless
            log.warning("could not delete the old copy %s", old_key)
    return len(moves)


def refile_project(db: Session, project_id: int, dry_run: bool) -> dict:
    """Apply the rule to every reviewed image of a project (for data reviewed before the rule existed)."""
    images = db.scalars(select(Image).where(Image.project_id == project_id, Image.review_status == ReviewStatus.reviewed)
                        .order_by(Image.id).options(selectinload(Image.annotations), selectinload(Image.import_record), selectinload(Image.upload_record))).all()
    names = {c.id: c.name for c in db.scalars(select(DetClass).where(DetClass.project_id == project_id))}
    moves, unchanged, failed = [], 0, 0
    for img in images:
        classes = {a.class_id for a in img.annotations}
        if not classes:
            unchanged += 1
            continue
        want = names.get(next(iter(classes))) if len(classes) == 1 else MIXED
        if want is None or want == names.get(img.class_id):
            unchanged += 1
            continue
        move = {"image_id": img.id, "from": names.get(img.class_id), "to": want}
        if dry_run or refile(db, img):
            moves.append(move)
        else:
            failed += 1
    out = {"dry_run": dry_run, "would_move" if dry_run else "moved": len(moves), "unchanged": unchanged, "moves": moves[:200]}
    if failed:
        out["failed"] = failed  # left where they were; see the server log
    return out
