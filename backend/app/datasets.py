"""Dataset versions: freeze the reviewed images + boxes, split them train/val/test, and record what was left out."""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from .models import AnnotationStatus, DatasetVersion, DetClass, Image, Project, ReviewStatus, ScrapeRecord, Source
from .storage import get_storage
from .yolo_export import SPLITS

NEGATIVE = "(negative)"
MIN_BOXES_WARNING = 20
KEPT_BOX_STATUSES = (AnnotationStatus.reviewed, AnnotationStatus.auto_exact)  # never unreviewed or rejected boxes


SELECTABLE_SOURCES = ("uploaded", "scraped", "generated")  # imported images are chosen per imported dataset instead


@dataclass(frozen=True)
class Selection:
    """Which reviewed images a dataset version may use. None means "all of them"; an empty set means "none of them".

    `sources` covers uploaded / scraped / generated. Imported images are chosen only through `imported`, by dataset name
    (their folder under raw/<class>/imported/), so one imported dataset can be in a version while another is not.
    """

    sources: frozenset[str] | None = None
    imported: frozenset[str] | None = None

    def wants(self, img: Image) -> bool:
        if img.source == Source.imported:
            name = img.import_record.dataset if img.import_record else None
            return self.imported is None or name in self.imported
        return self.sources is None or img.source.value in self.sources


def source_label(img: Image) -> str:
    """"uploaded", "scraped", "generated", or "imported:<dataset>"."""
    if img.source == Source.imported:
        return f"imported:{img.import_record.dataset if img.import_record else '?'}"
    return img.source.value


def _round(x: float) -> int:
    return int(x + 0.5)  # round half up (Python's round() is banker's rounding)


def split_quota(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    """How many of n groups go to (val, test). Small strata still get a validation group, and train is never empty."""
    n_val, n_test = _round(n * val_ratio), _round(n * test_ratio)
    if val_ratio > 0 and n >= 2:
        n_val = max(1, n_val)
    if test_ratio > 0 and n >= 3:
        n_test = max(1, n_test)
    while n_val + n_test >= n and (n_val or n_test):
        if n_test:
            n_test -= 1
        else:
            n_val -= 1
    return n_val, n_test


def assign_splits(items: list[dict], val_ratio: float, test_ratio: float, seed: int) -> None:
    """Set item["split"] in place.

    Images sharing a group (same source page / video) always land in the same split, so near-copies can't leak
    between train and val. Groups are split per class (stratified) so every class tries to appear in val and test.
    Generated (synthetic) images only go to train: the test set stays real-only.
    """
    groups: dict[str, dict] = {}
    for it in items:
        g = groups.setdefault(it["group"], {"items": [], "strata": Counter(), "generated": False})
        g["items"].append(it)
        g["strata"][it["stratum"]] += 1
        g["generated"] |= it["source"] == "generated"

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for name, g in groups.items():
        by_stratum[sorted(g["strata"].items(), key=lambda kv: (-kv[1], kv[0]))[0][0]].append(name)

    order = lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()  # seeded, stable under new data
    split_of: dict[str, str] = {}
    for names in by_stratum.values():
        real = sorted((n for n in names if not groups[n]["generated"]), key=order)
        split_of.update({n: "train" for n in names if groups[n]["generated"]})
        n_val, n_test = split_quota(len(real), val_ratio, test_ratio)
        for i, n in enumerate(real):
            split_of[n] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    for name, g in groups.items():
        for it in g["items"]:
            it["split"] = split_of[name]


def preview(db: Session, project: Project, selection: Selection | None = None) -> dict:
    """What a new version would contain right now, for the data chosen in `selection` (all of it if None)."""
    by_status = dict(db.execute(select(Image.review_status, func.count()).where(Image.project_id == project.id)
                                .group_by(Image.review_status)).all())
    reviewed_all = db.scalars(select(Image).where(Image.project_id == project.id, Image.review_status == ReviewStatus.reviewed)
                              .options(selectinload(Image.annotations), selectinload(Image.import_record))).all()
    available = Counter(source_label(i) for i in reviewed_all)
    reviewed = [i for i in reviewed_all if selection is None or selection.wants(i)]
    negatives = {c.id for c in project.classes if c.is_negative}
    names = {c.id: c.name for c in project.classes}
    boxes: Counter = Counter()
    with_boxes = empty = neg = 0
    for img in reviewed:
        kept = [a for a in img.annotations if a.status in KEPT_BOX_STATUSES]
        if img.class_id in negatives:
            neg += 1
        elif kept:
            with_boxes += 1
        else:
            empty += 1
        boxes.update(names[a.class_id] for a in kept)
    return {"reviewed_images": len(reviewed), "reviewed_with_boxes": with_boxes, "reviewed_empty": empty,
            "negative_images": neg, "pending": by_status.get(ReviewStatus.pending, 0),
            "rejected": by_status.get(ReviewStatus.rejected, 0),
            "reviewed_total": len(reviewed_all), "not_selected": len(reviewed_all) - len(reviewed),
            # everything that could be chosen, whatever is selected now: what the picker offers
            "available": {"sources": {src: available.get(src, 0) for src in SELECTABLE_SOURCES},
                          "imported_datasets": [{"name": k.split(":", 1)[1], "reviewed": n} for k, n in sorted(available.items())
                                                if k.startswith("imported:")]},
            "boxes_per_class": {c.name: boxes.get(c.name, 0) for c in project.classes if c.is_trainable}}


def create_version(db: Session, project: Project, name: str | None, val_ratio: float, test_ratio: float, seed: int,
                   include_negatives: bool, selection: Selection | None = None) -> DatasetVersion:
    """Snapshot the project (only the data in `selection`, all of it if None).
    Raises ValueError (message safe to show) if there is nothing usable to train on."""
    classes = sorted((c for c in project.classes if c.is_trainable), key=lambda c: c.index)
    class_idx = {c.id: i for i, c in enumerate(classes)}  # contiguous 0..N-1, as YOLO requires
    by_id = {c.id: c for c in project.classes}

    images = db.scalars(select(Image).where(Image.project_id == project.id, Image.review_status == ReviewStatus.reviewed)
                        .order_by(Image.id).options(selectinload(Image.annotations), selectinload(Image.scrape_record),
                                                    selectinload(Image.import_record))).all()
    items = []
    not_selected = 0
    for img in images:
        if selection is not None and not selection.wants(img):
            not_selected += 1
            continue
        cls = by_id[img.class_id]
        if cls.is_negative and not include_negatives:
            continue
        rec: ScrapeRecord | None = img.scrape_record
        group = (rec.page_url if rec and rec.page_url else None) or img.group_key or f"image-{img.id}"
        boxes = [[class_idx[a.class_id], a.cx, a.cy, a.w, a.h] for a in img.annotations
                 if a.status in KEPT_BOX_STATUSES and a.class_id in class_idx]
        # Balance the split by what the boxes say the image shows, not by the class it happened to be filed under: a mixed
        # search filed under one class would otherwise leave the other classes' images to chance in val and test.
        if boxes:
            top = sorted(Counter(b[0] for b in boxes).items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            stratum = classes[top].name
        else:
            stratum = NEGATIVE if cls.is_negative else cls.name
        items.append({
            "image_id": img.id, "key": img.storage_key, "sha256": img.sha256, "ext": img.storage_key.rsplit(".", 1)[-1],
            "width": img.width, "height": img.height, "source": img.source.value, "folder_class": cls.name,
            "dataset": img.import_record.dataset if img.import_record else None,
            "group": f"{img.source.value}:{group}", "stratum": stratum, "boxes": boxes,
        })
    if not items and not_selected:
        raise ValueError("None of the data you selected has reviewed images. Tick other data, or review more images.")
    if not items:
        raise ValueError("No reviewed images yet. Review some images first (pending and rejected images are never exported).")
    if not any(it["boxes"] for it in items):
        raise ValueError("None of the reviewed images has a box. Draw boxes in the review screen first.")

    assign_splits(items, val_ratio, test_ratio, seed)
    for it in items:
        del it["stratum"]
    stats = _stats(items, classes, db, project, not_selected)
    if stats["images"]["train"] == 0:
        raise ValueError("The train split would be empty; review more images.")

    count = db.scalar(select(func.count()).select_from(DatasetVersion).where(DatasetVersion.project_id == project.id)) or 0
    version = DatasetVersion(project_id=project.id, name=name or f"v{count + 1}", stats=stats, manifest_key="",
                             config={"val_ratio": val_ratio, "test_ratio": test_ratio, "seed": seed, "include_negatives": include_negatives,
                                     "sources": sorted(selection.sources) if selection and selection.sources is not None else None,
                                     "imported_datasets": sorted(selection.imported) if selection and selection.imported is not None else None,
                                     "classes": [{"index": i, "name": c.name} for i, c in enumerate(classes)]})
    db.add(version)
    db.flush()  # assigns the id used in the manifest path
    version.manifest_key = f"projects/{project.id}/datasets/{version.id}/manifest.json"
    manifest = {"version_id": version.id, "name": version.name, "project_id": project.id,
                "created_at": datetime.now(timezone.utc).isoformat(), "config": version.config,
                "classes": version.config["classes"], "images": items}
    get_storage().put(version.manifest_key, json.dumps(manifest).encode())
    return version


def resolve_manifest(db: Session, manifest: dict) -> dict:
    """Point each manifest item at its image's *current* storage key.

    A reviewed image can move to another class folder after a version was frozen, but the version must keep working. An item
    is re-pointed only if the image row still has the same SHA-256: SQLite reuses the id of a deleted image, and an id alone
    could swap a different picture into a frozen version. Anything else keeps its recorded key (and, if that file is gone,
    shows up as a missing image, as before).
    """
    items = manifest["images"]
    current: dict[int, tuple[str, str]] = {}
    for i in range(0, len(items), 500):
        ids = [it["image_id"] for it in items[i:i + 500]]
        current.update({iid: (sha, key) for iid, sha, key in
                        db.execute(select(Image.id, Image.sha256, Image.storage_key).where(Image.id.in_(ids)))})
    for it in items:
        row = current.get(it["image_id"])
        if row and row[0] == it["sha256"]:
            it["key"] = row[1]
    return manifest


def versions_using(db: Session, project_id: int, image_ids: set[int]) -> dict[str, list[int]]:
    """{version name: the given image ids it contains}. A frozen version reads its image files whenever it is downloaded, so deleting one of
    these images for good would make that version impossible to download. Matched on SHA-256 too: SQLite reuses the ids of deleted images."""
    shas = dict(db.execute(select(Image.id, Image.sha256).where(Image.id.in_(image_ids))).all())
    found: dict[str, list[int]] = {}
    for version in db.scalars(select(DatasetVersion).where(DatasetVersion.project_id == project_id).order_by(DatasetVersion.id)):
        try:
            items = load_manifest(version)["images"]
        except Exception:  # noqa: BLE001 - a version whose manifest is gone cannot be broken any further
            continue
        hit = sorted(it["image_id"] for it in items if shas.get(it["image_id"]) == it["sha256"])
        if hit:
            found[version.name] = hit
    return found


def load_manifest(version: DatasetVersion) -> dict:
    return json.loads(get_storage().get(version.manifest_key))


def _stats(items: list[dict], classes: list[DetClass], db: Session, project: Project, not_selected: int = 0) -> dict:
    names = [c.name for c in classes]
    images = {s: 0 for s in SPLITS}
    boxes = {s: 0 for s in SPLITS}
    per_class = {n: {s: {"images": 0, "boxes": 0} for s in SPLITS} for n in names}
    background = Counter()
    for it in items:
        s = it["split"]
        images[s] += 1
        boxes[s] += len(it["boxes"])
        if not it["boxes"]:
            background[s] += 1
        for ci in {b[0] for b in it["boxes"]}:
            per_class[names[ci]][s]["images"] += 1
        for b in it["boxes"]:
            per_class[names[b[0]]][s]["boxes"] += 1
    by_status = dict(db.execute(select(Image.review_status, func.count()).where(Image.project_id == project.id)
                                .group_by(Image.review_status)).all())

    warnings = []
    for s in ("val", "test"):
        if images[s] == 0:
            warnings.append(f"The {s} split is empty (too few images), so there is nothing to measure {'accuracy' if s == 'val' else 'the final model'} on.")
    for n in names:
        total = sum(per_class[n][s]["boxes"] for s in SPLITS)
        if total == 0:
            warnings.append(f"Class '{n}' has no boxes at all; a model can't learn it.")
        elif per_class[n]["val"]["boxes"] == 0 and images["val"] > 0:
            warnings.append(f"Class '{n}' has no boxes in val, so its accuracy can't be measured.")
        elif total < MIN_BOXES_WARNING:
            warnings.append(f"Class '{n}' has only {total} boxes (aim for {MIN_BOXES_WARNING}+ to start with, far more for good accuracy).")
    return {"images": images, "boxes": boxes, "background_images": dict(background), "per_class": per_class,
            "excluded": {"pending": by_status.get(ReviewStatus.pending, 0), "rejected": by_status.get(ReviewStatus.rejected, 0),
                         "not_selected": not_selected},
            "by_source": dict(sorted(Counter(f"imported:{it['dataset']}" if it["source"] == "imported" else it["source"]
                                             for it in items).items())),
            "warnings": warnings}
