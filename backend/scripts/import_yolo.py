"""Import a public YOLO-format dataset (such as a Roboflow export) as reviewed images with boxes.

    python -m scripts.import_yolo import <folder> --dataset anti-uav --map UAV=multirotor --author "Yogith" --dry-run
    python -m scripts.import_yolo import <folder> --dataset anti-uav --map UAV=multirotor --author "Yogith"
    python -m scripts.import_yolo import <folder> --dataset birds-v1 --negative bird=birds --author "Somebody" --dry-run
    python -m scripts.import_yolo rollback --dataset anti-uav

Negative mode (`--negative SOURCE=target`): frames whose boxes are ALL of that source class become hard-negative images filed under
the project's negative class `target` (birds, balloons ...). Their boxes are used only to judge the frame and are not stored: a negative
image has none. A frame that also holds a mapped (`--map`) or unmapped class, such as a drone next to the bird, is dropped, and so is a frame
where any boxed object is smaller than --negative-min-size of the image's short side: a tiny bird against the sky looks like a drone,
and teaching "not a drone" from it would blur the very thing the detector must find.

What it does, in order (all of it deterministic, so a re-run imports the same images):
  1. One image per source frame. Roboflow writes augmented copies as `<frame>_jpg.rf.<hash>.jpg`; the extra copies are dropped.
  2. Drops frames with no boxes, with a class that isn't mapped, or with any box smaller than --min-box-px (a tiny drone left
     unlabelled in the image would teach the model to ignore it). Negative frames follow the rules above instead.
  3. Groups frames into *scenes* by a colour-thumbnail descriptor (robust to flips and brightness; a plain grayscale hash chains
     unrelated sky scenes into one giant group) and keeps at most --max-per-scene frames per scene, spread evenly, clean
     (un-augmented) ones first. The scene becomes the image's group_key, so dataset versions never split a scene across
     train/val/test. The dataset's own train/valid/test split is ignored: video frames were split at random, so it leaks.
  4. Stores the images as source "imported" under raw/<class>/imported/<dataset>/, as reviewed, with an ImportRecord for attribution.
     Positives get their boxes; negatives get none. Review a negative dataset's images in the review window (it counts as the manual check)
     and reject any that show a drone.
`rollback` removes everything imported under one --dataset tag, files included.
"""

import argparse
import collections
import io
import random
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.images import ingest_image
from app.config import settings
from app.db import Base
from app import models  # noqa: F401 - registers every table on Base.metadata
from app.models import Annotation, AnnotationStatus, DetClass, Image, ImportRecord, Project, ReviewStatus, Source
from app.storage import LocalStorage, get_storage

SPLITS = ("test", "valid", "val", "train")  # clean copies first when choosing which frames to keep
RF_NAME = re.compile(r"^(?P<stem>.+)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+\.(?:jpg|jpeg|png)$", re.I)
DEFAULT_SIM, DEFAULT_CAP, DEFAULT_MIN_BOX_PX, BATCH = 0.97, 20, 5.0, 50
DEFAULT_NEG_MIN_SIZE = 0.10  # a negative frame's smallest box, as a fraction of the image's short side. 68% of our drone boxes are smaller than this
DATASET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")  # it becomes a folder name and part of every scene's group key


class ImportProblem(Exception):
    """Something the person running the import needs to fix (message is safe to show)."""


@dataclass
class Frame:
    split: str
    stem: str
    path: Path
    boxes: list  # [(class_name, cx, cy, w, h)]
    width: int
    height: int
    augmented: bool  # Roboflow made several variants of this source frame
    scene: str = ""
    negative: bool = False  # a hard-negative image: boxes[0][0] is the negative class it is filed under, and its boxes are not stored

    @property
    def original_name(self) -> str:
        return self.path.name


def _natural(stem: str):
    return (0, int(stem), "") if stem.isdigit() else (1, 0, stem)


def read_data_yaml(root: Path) -> dict:
    """The few fields we need from data.yaml, parsed by hand (it is flat apart from the roboflow block)."""
    path = root / "data.yaml"
    if not path.exists():
        raise ImportProblem(f"{root} has no data.yaml")
    text = path.read_text(encoding="utf-8")
    names = re.search(r"^names:\s*\[(.*)\]\s*$", text, re.M)
    if names:
        class_names = [n.strip().strip("'\"") for n in names.group(1).split(",") if n.strip()]
    else:  # block style: names:\n  0: UAV
        class_names = [m.group(2).strip().strip("'\"") for m in re.finditer(r"^\s+(\d+):\s*(.+)$", text.split("names:", 1)[-1], re.M)]
    meta = {k: (re.search(rf"^\s+{k}:\s*(.+)$", text, re.M) or [None, ""])[1].strip() for k in ("license", "url", "workspace", "project")}
    return {"names": class_names, **meta}


def load_frames(root: Path, class_map: dict[str, str], min_box_px: float, stats: collections.Counter,
                negative_map: dict[str, str] | None = None, negative_min_size: float = DEFAULT_NEG_MIN_SIZE) -> list[Frame]:
    negative_map = negative_map or {}
    both = sorted(set(class_map) & set(negative_map))
    if both:
        raise ImportProblem(f"class '{both[0]}' is in both --map and --negative: a source class is either a drone class or a negative")
    names = read_data_yaml(root)["names"]
    frames: list[Frame] = []
    for split in SPLITS:
        img_dir, lab_dir = root / split / "images", root / split / "labels"
        if not img_dir.is_dir():
            continue
        variants: dict[str, list[Path]] = collections.defaultdict(list)
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = RF_NAME.match(p.name)
            variants[m["stem"] if m else p.stem].append(p)
        for stem, paths in variants.items():
            stats["source_frames"] += 1
            stats["variants_dropped"] += len(paths) - 1
            path = paths[0]  # any variant will do; the first by name keeps it deterministic
            label = lab_dir / (path.stem + ".txt")
            lines = [ln.split() for ln in label.read_text().splitlines() if ln.strip()] if label.exists() else []
            if not lines:
                stats["dropped_no_boxes"] += 1
                continue
            with PILImage.open(path) as im:
                width, height = im.size
            boxes, ok, kinds = [], True, set()
            for tok in lines:
                try:
                    cid, cx, cy, w, h = int(tok[0]), *map(float, tok[1:5])
                    negative = names[cid] in negative_map
                    cname = negative_map[names[cid]] if negative else class_map[names[cid]]
                except (ValueError, IndexError, KeyError):
                    stats["dropped_unmapped_or_malformed"] += 1
                    ok = False
                    break
                x0, y0, x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(1.0, cx + w / 2), min(1.0, cy + h / 2)
                w, h = x1 - x0, y1 - y0  # clamp to the image: rounding in other tools can leave a box a hair outside
                smallest = negative_min_size * min(width, height) if negative else min_box_px
                if w <= 0 or h <= 0 or min(w * width, h * height) < smallest:
                    stats["dropped_negative_too_small" if negative else "dropped_tiny_box"] += 1
                    ok = False
                    break
                box = (cname, (x0 + x1) / 2, (y0 + y1) / 2, w, h)
                if any(_iou(box, b) > 0.9 for b in boxes):
                    stats["duplicate_boxes_removed"] += 1
                    continue
                boxes.append(box)
                kinds.add(negative)
            if ok and kinds == {True, False}:
                stats["dropped_negative_with_drone_class"] += 1  # a bird in a frame that also has a drone class: not a clean negative
                ok = False
            if ok and boxes:
                frames.append(Frame(split, stem, path, boxes, width, height, augmented=len(paths) > 1, negative=kinds == {True}))
    frames.sort(key=lambda f: (f.augmented, SPLITS.index(f.split), _natural(f.stem)))
    return frames


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def descriptors(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray]:
    """12x12 colour thumbnails, brightness-centred and unit length; plus the mirrored image (augmented copies may be flipped)."""
    thumbs = []
    for f in frames:
        with PILImage.open(f.path) as im:
            thumbs.append(np.asarray(im.convert("RGB").resize((12, 12), PILImage.Resampling.BOX), dtype=np.float32))
    a = np.stack(thumbs)

    def prep(x):
        v = x.reshape(len(x), -1)
        v = v - v.mean(1, keepdims=True)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6)

    return prep(a), prep(a[:, :, ::-1, :])


def assign_scenes(frames: list[Frame], sim: float) -> None:
    """Single-linkage on descriptor similarity. Scenes are numbered in order of their first frame, so numbering is stable."""
    if not frames:
        return
    v, vm = descriptors(frames)
    n = len(frames)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i0 in range(0, n, 500):
        s = np.maximum(v[i0:i0 + 500] @ v.T, v[i0:i0 + 500] @ vm.T)
        ii, jj = np.where(s >= sim)
        for a, b in zip((ii + i0).tolist(), jj.tolist()):
            if a < b:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)  # the root is always the earliest frame
    numbering: dict[int, str] = {}
    for i, f in enumerate(frames):
        root = find(i)
        f.scene = numbering.setdefault(root, f"scene-{len(numbering) + 1:04d}")


def pick_spread(members: list, cap: int) -> list:
    """At most `cap` members, evenly spaced. Clean (un-augmented) frames are used first: the spread is over them alone when
    there are enough, and augmented ones only fill whatever is left."""
    clean = [m for m in members if not m.augmented]
    rest = [m for m in members if m.augmented]
    if len(members) <= cap:
        return members
    if len(clean) >= cap:
        return _spaced(clean, cap)
    return clean + _spaced(rest, cap - len(clean))


def _spaced(items: list, k: int) -> list:
    if k <= 0:
        return []
    if k == 1:
        return items[:1]
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def select_frames(frames: list[Frame], cap: int) -> list[Frame]:
    by_scene: dict[str, list[Frame]] = collections.defaultdict(list)
    for f in frames:
        by_scene[f.scene].append(f)
    chosen = [f for members in by_scene.values() for f in pick_spread(members, cap)]
    chosen.sort(key=lambda f: (f.scene, f.augmented, SPLITS.index(f.split), _natural(f.stem)))
    return chosen


def plan(root: Path, class_map: dict[str, str], sim=DEFAULT_SIM, cap=DEFAULT_CAP, min_box_px=DEFAULT_MIN_BOX_PX,
         negative_map: dict[str, str] | None = None, negative_min_size=DEFAULT_NEG_MIN_SIZE):
    stats: collections.Counter = collections.Counter()
    frames = load_frames(root, class_map, min_box_px, stats, negative_map, negative_min_size)
    stats["kept_after_filters"] = len(frames)
    assign_scenes(frames, sim)
    chosen = select_frames(frames, cap)
    sizes = collections.Counter(f.scene for f in frames)
    stats["scenes"] = len(sizes)
    stats["singleton_scenes"] = sum(1 for n in sizes.values() if n == 1)
    stats["largest_scene"] = max(sizes.values(), default=0)
    stats["selected"] = len(chosen)
    stats["selected_clean"] = sum(1 for f in chosen if not f.augmented)
    stats["selected_negatives"] = sum(1 for f in chosen if f.negative)
    return chosen, stats


def summarize(chosen: list[Frame], stats: collections.Counter) -> dict:
    px = sorted(min(w * f.width, h * f.height) for f in chosen for _, _, _, w, h in f.boxes)
    q = (lambda p: round(px[int(p * (len(px) - 1))], 1)) if px else (lambda p: None)
    return {**dict(stats), "selected_by_class": dict(collections.Counter(f.boxes[0][0] for f in chosen)),
            "box_min_side_px": {"p10": q(.1), "p50": q(.5), "p90": q(.9)},
            "selected_by_original_split": dict(collections.Counter(f.split for f in chosen)),
            "images_on_disk_mb": round(sum(f.path.stat().st_size for f in chosen) / 1e6, 1)}


def write_sample_sheet(chosen: list[Frame], path: Path, n: int = 48, cols: int = 8, cell: int = 160, seed: int = 1) -> None:
    pick = random.Random(seed).sample(chosen, min(n, len(chosen)))
    rows = (len(pick) + cols - 1) // cols
    sheet = PILImage.new("RGB", (cols * cell, rows * cell), (20, 20, 20))
    for k, f in enumerate(pick):
        with PILImage.open(f.path) as im:
            im = im.convert("RGB").resize((cell, cell))
        d = ImageDraw.Draw(im)
        for _, cx, cy, w, h in f.boxes:
            d.rectangle([(cx - w / 2) * cell - 2, (cy - h / 2) * cell - 2, (cx + w / 2) * cell + 2, (cy + h / 2) * cell + 2],
                        outline=(255, 90, 90) if f.negative else (255, 255, 0))  # red: a negative's box, only shown here, never stored
        sheet.paste(im, ((k % cols) * cell, (k // cols) * cell))
    sheet.save(path, quality=85)


def backup_database(engine, dest_dir: Path, tag: str) -> Path | None:
    """A consistent copy of a live SQLite database (the online backup API, not cp). Returns None for other databases."""
    if engine.dialect.name != "sqlite" or not engine.url.database or engine.url.database == ":memory:":
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"pre-import-{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
    src, dst = sqlite3.connect(engine.url.database), sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def run_import(db: Session, project_id: int, chosen: list[Frame], dataset: str, meta: dict) -> dict:
    """Store the chosen frames. Commits every BATCH images, so a live backend's own writes never wait long."""
    Base.metadata.create_all(db.get_bind())  # the import_records table must exist even if the backend hasn't restarted
    classes = {c.name: c for c in db.scalars(select(DetClass).where(DetClass.project_id == project_id))}
    for f in chosen:  # every target is checked before anything is written
        for cname in {b[0] for b in f.boxes}:
            cls = classes.get(cname)
            if f.negative and (cls is None or not cls.is_negative):
                raise ImportProblem(f"--negative needs one of the project's hard-negative classes (birds, balloons ...); '{cname}' isn't one")
            if not f.negative and (cls is None or not cls.is_trainable):
                raise ImportProblem(f"class '{cname}' doesn't exist (or isn't a trainable class) in this project")
    added = duplicates = rejected = 0
    for i, f in enumerate(chosen, 1):
        cls = classes[f.boxes[0][0]]
        try:
            image = ingest_image(db, project_id, cls, Source.imported, f.path.read_bytes(), group_key=f"{dataset}:{f.scene}",
                                 subfolder=dataset)
        except ValueError:
            rejected += 1
            continue
        if image is None:
            duplicates += 1  # already imported (same bytes): this is what makes a re-run harmless
        else:
            image.review_status = ReviewStatus.reviewed
            for cname, cx, cy, w, h in ([] if f.negative else f.boxes):  # a negative image has no boxes
                image.annotations.append(Annotation(class_id=classes[cname].id, cx=cx, cy=cy, w=w, h=h, status=AnnotationStatus.reviewed))
            image.import_record = ImportRecord(dataset=dataset, original_name=f.original_name[:300], original_split=f.split,
                                               dataset_url=meta["url"][:500], license=meta["license"][:80], author=meta["author"][:200])
            added += 1
        if i % BATCH == 0:
            db.commit()
    db.commit()
    return {"added": added, "already_imported": duplicates, "rejected": rejected}


def sweep_orphans(db: Session, project_id: int | None = None) -> int:
    """Delete files under raw/*/imported/ that no image row uses (what a crashed import can leave behind), then any folders
    left empty. Local storage only."""
    storage = get_storage()
    if not isinstance(storage, LocalStorage):
        return 0
    used = set(db.scalars(select(Image.storage_key).where(Image.source == Source.imported)))
    removed = 0
    for path in storage.root.glob("projects/*/raw/*/imported/**/*"):
        key = path.relative_to(storage.root).as_posix()
        if path.is_file() and key not in used:
            path.unlink()
            removed += 1
    for folder in sorted(storage.root.glob("projects/*/raw/*/imported/**/*"), key=lambda p: len(p.parts), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    return removed


def rollback(db: Session, dataset: str) -> dict:
    storage = get_storage()
    ids = list(db.scalars(select(ImportRecord.image_id).where(ImportRecord.dataset == dataset)))
    for i in range(0, len(ids), BATCH):
        for image in db.scalars(select(Image).where(Image.id.in_(ids[i:i + BATCH]))):
            storage.delete(image.storage_key)
            db.delete(image)  # cascades to its boxes, hash and import record
        db.commit()
    return {"images_removed": len(ids), "orphan_files_removed": sweep_orphans(db)}


def _parse_map(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ImportProblem(f"--map and --negative expect SOURCE=target, got '{p}'")
        a, b = p.split("=", 1)
        out[a.strip()] = b.strip()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="import a YOLO-format dataset folder")
    imp.add_argument("folder", type=Path)
    imp.add_argument("--dataset", required=True, help="the dataset's name, e.g. anti-uav: its folder under raw/<class>/imported/ and the tag dataset versions select")
    imp.add_argument("--project", type=int, default=1)
    imp.add_argument("--map", action="append", default=[], metavar="SOURCE=target", help="dataset class -> our class, e.g. UAV=multirotor")
    imp.add_argument("--negative", action="append", default=[], metavar="SOURCE=target",
                     help="dataset class -> one of our hard-negative classes, e.g. bird=birds. Frames with only these classes are imported as "
                          "background images without boxes")
    imp.add_argument("--negative-min-size", type=float, default=DEFAULT_NEG_MIN_SIZE, metavar="FRACTION",
                     help="drop a negative frame when any boxed object is smaller than this fraction of the image's short side "
                          "(default 0.10: tiny birds look like drones)")
    imp.add_argument("--author", required=True, help="who to credit (for the attribution record)")
    imp.add_argument("--license", help="defaults to the license in data.yaml")
    imp.add_argument("--url", help="defaults to the url in data.yaml")
    imp.add_argument("--sim", type=float, default=DEFAULT_SIM, help="descriptor similarity that links two frames into one scene")
    imp.add_argument("--max-per-scene", type=int, default=DEFAULT_CAP)
    imp.add_argument("--min-box-px", type=float, default=DEFAULT_MIN_BOX_PX, help="drop frames with a box whose short side is smaller")
    imp.add_argument("--dry-run", action="store_true", help="report what would be imported; change nothing")
    imp.add_argument("--sample-sheet", type=Path, help="write a contact sheet of random selected frames (with their boxes)")
    imp.add_argument("--no-backup", action="store_true")
    rb = sub.add_parser("rollback", help="remove everything imported under a --dataset tag")
    rb.add_argument("--dataset", required=True)
    args = ap.parse_args(argv)

    from app.db import SessionLocal, engine
    try:
        if args.cmd == "rollback":
            Base.metadata.create_all(engine)
            with SessionLocal() as db:
                print("rolled back:", rollback(db, args.dataset))
            return 0

        if not DATASET_NAME.match(args.dataset):
            raise ImportProblem("--dataset must be lowercase letters, digits, '-' or '_' (it becomes a folder name), e.g. anti-uav")
        class_map, negative_map = _parse_map(args.map), _parse_map(args.negative)
        meta = read_data_yaml(args.folder)
        if not class_map and not negative_map:
            raise ImportProblem("give at least one --map, e.g. --map UAV=multirotor, or --negative, e.g. --negative bird=birds")
        if not 0 <= args.negative_min_size < 1:
            raise ImportProblem("--negative-min-size is a fraction of the image's short side, between 0 and 1")
        meta = {"url": args.url or meta["url"], "license": args.license or meta["license"], "author": args.author}
        chosen, stats = plan(args.folder, class_map, args.sim, args.max_per_scene, args.min_box_px, negative_map, args.negative_min_size)
        report = summarize(chosen, stats)
        print(("DRY RUN: " if args.dry_run else "") + "plan:")
        for k, v in report.items():
            print(f"  {k}: {v}")
        if args.sample_sheet:
            write_sample_sheet(chosen, args.sample_sheet)
            print("  sample sheet:", args.sample_sheet)
        if args.dry_run:
            return 0
        if not args.no_backup:
            backup = backup_database(engine, Path("data/backups"), args.dataset)
            print("  database backed up to:", backup)
        with SessionLocal() as db:
            if db.get(Project, args.project) is None:
                raise ImportProblem(f"project {args.project} not found")
            print("imported:", run_import(db, args.project, chosen, args.dataset, meta))
        return 0
    except ImportProblem as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
