import collections
import random
import sqlite3
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import ImageDraw, ImageEnhance, ImageOps
from sqlalchemy import func, select

from app import datasets, storage
from app.models import Annotation, DatasetVersion, Image, ImageHash, ImportRecord, ReviewStatus, Source
from app.storage import LocalStorage
from scripts import bundle, import_yolo
from scripts.import_yolo import ImportProblem, plan, rollback, run_import

from .helpers import png

SIZE = 96
CLASS_MAP = {"UAV": "multirotor"}
META = {"url": "https://example.test/ds/1", "license": "CC BY 4.0", "author": "Somebody"}
N_SCENES, PER_SCENE = 12, 6  # per scene: 2 train frames (3 augmented variants each), 2 valid, 2 test


def _scene_pattern(seed: int) -> PILImage.Image:
    rnd = random.Random(seed)
    tiny = PILImage.new("RGB", (12, 12))
    tiny.putdata([(rnd.randrange(256), rnd.randrange(256), rnd.randrange(256)) for _ in range(144)])
    return tiny.resize((SIZE, SIZE), PILImage.Resampling.NEAREST)


def _write(root: Path, split: str, name: str, im: PILImage.Image, boxes: list[str]) -> None:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    im.save(root / split / "images" / f"{name}.jpg", quality=92)
    (root / split / "labels" / f"{name}.txt").write_text("".join(b + "\n" for b in boxes))


def build_dataset(root: Path) -> dict:
    """A Roboflow-style export: stems restart per split, train frames come as 3 augmented variants, valid/test are clean."""
    rnd = random.Random(0)
    (root).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\ntest: ../test/images\n\nnc: 2\n"
                                    "names: ['UAV', 'bird']\n\nroboflow:\n  workspace: someone\n  project: demo\n  version: 1\n"
                                    "  license: CC BY 4.0\n  url: https://example.test/ds/1\n")
    counters = collections.Counter()
    truth: dict[tuple[str, str], int] = {}  # (split, stem) -> the scene it really belongs to

    def stem(split):
        counters[split] += 1
        return f"{counters[split]:05d}"

    def drone_frame(scene, box_px=8):
        im = _scene_pattern(100 + scene).copy()
        x, y = rnd.randrange(10, SIZE - 10), rnd.randrange(10, SIZE - 10)
        ImageDraw.Draw(im).rectangle([x - box_px // 2, y - box_px // 2, x + box_px // 2, y + box_px // 2], fill=(10, 10, 10))
        return im, (x / SIZE, y / SIZE, box_px / SIZE, box_px / SIZE)

    for scene in range(N_SCENES):
        for k in range(PER_SCENE):
            split = ("train", "valid", "test")[k % 3]
            im, (cx, cy, w, h) = drone_frame(scene)
            s = stem(split)
            truth[(split, s)] = scene
            if split == "train":  # Roboflow's 3 augmented outputs: original, mirrored, dimmer
                for v, (img, bx) in enumerate([(im, cx), (ImageOps.mirror(im), 1 - cx), (ImageEnhance.Brightness(im).enhance(0.88), cx)]):
                    _write(root, "train", f"{s}_jpg.rf.{v}abc{scene:02d}{k:02d}", img, [f"0 {bx:.6f} {cy:.6f} {w:.6f} {h:.6f}"])
            else:
                _write(root, split, f"{s}_jpg.rf.f{scene:02d}{k:02d}ab", im, [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"])

    # frames that must be dropped or cleaned (single variants, all train, all of scene 0)
    im, (cx, cy, w, h) = drone_frame(0)
    s = stem("train"); _write(root, "train", f"{s}_jpg.rf.e1", im, []); truth[("train", s)] = 0                                    # no boxes
    s = stem("train"); _write(root, "train", f"{s}_jpg.rf.e2", im, [f"0 {cx:.6f} {cy:.6f} {2/SIZE:.6f} {2/SIZE:.6f}"]); truth[("train", s)] = 0   # 2px box
    s = stem("train"); _write(root, "train", f"{s}_jpg.rf.e3", im, [f"1 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]); truth[("train", s)] = 0          # 'bird': unmapped
    s = stem("train"); _write(root, "train", f"{s}_jpg.rf.e4", im, [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"] * 2); truth[("train", s)] = 0       # duplicate box
    return {"truth": truth, "train_sources": N_SCENES * 2, "clean": N_SCENES * 4}


@pytest.fixture(autouse=True)
def own_storage(client, tmp_path):
    """conftest's storage shares tmp_path with the database file and with the dataset built here; the bundle would see them all."""
    storage._storage = LocalStorage(str(tmp_path / "store"))


@pytest.fixture()
def dataset(tmp_path):
    root = tmp_path / "ds"
    info = build_dataset(root)
    return root, info


def scenes_of(frames, truth):
    """frame -> the scene it truly belongs to, to compare partitions."""
    return {f.scene: truth[(f.split, f.stem)] for f in frames}


# ---------- planning ----------

def test_plan_collapses_variants_filters_bad_frames_and_finds_the_scenes(dataset):
    root, info = dataset
    chosen, stats = plan(root, CLASS_MAP, sim=0.97, cap=100, min_box_px=5)
    assert stats["variants_dropped"] == 2 * info["train_sources"]                 # 3 augmented copies -> 1 per source frame
    assert stats["dropped_no_boxes"] == 1 and stats["dropped_tiny_box"] == 1 and stats["dropped_unmapped_or_malformed"] == 1
    assert stats["duplicate_boxes_removed"] == 1
    assert stats["kept_after_filters"] == info["train_sources"] + info["clean"] + 1   # + the duplicate-box frame, kept with one box
    assert len(chosen) == stats["kept_after_filters"]
    assert collections.Counter(len(f.boxes) for f in chosen)[2] == 0
    # scenes: every true scene found, none merged with another, none split
    assert stats["scenes"] == N_SCENES
    by_scene = collections.defaultdict(set)
    for f in chosen:
        by_scene[f.scene].add(info["truth"][(f.split, f.stem)])
    assert all(len(v) == 1 for v in by_scene.values()) and len({next(iter(v)) for v in by_scene.values()}) == N_SCENES


def test_the_per_scene_cap_keeps_clean_frames_first(dataset):
    root, info = dataset
    chosen, stats = plan(root, CLASS_MAP, cap=2)
    per = collections.Counter(f.scene for f in chosen)
    assert set(per.values()) == {2} and len(chosen) == 2 * N_SCENES
    assert not any(f.augmented for f in chosen)                                   # each scene has 4 clean frames, so no augmented one is needed
    chosen5, stats5 = plan(root, CLASS_MAP, cap=5)
    # scene 0 also holds the single-variant duplicate-box frame, which is clean: 5 clean there, 4 clean + 1 augmented elsewhere
    assert stats5["selected_clean"] == 4 * N_SCENES + 1 and len(chosen5) - stats5["selected_clean"] == N_SCENES - 1


def test_selection_is_deterministic(dataset):
    root, _ = dataset
    a = [(f.scene, f.split, f.stem) for f in plan(root, CLASS_MAP, cap=3)[0]]
    b = [(f.scene, f.split, f.stem) for f in plan(root, CLASS_MAP, cap=3)[0]]
    assert a == b


def test_box_size_threshold_is_in_pixels_of_the_actual_image(dataset):
    root, _ = dataset
    assert plan(root, CLASS_MAP, min_box_px=5)[1]["dropped_tiny_box"] == 1
    assert plan(root, CLASS_MAP, min_box_px=9)[1]["dropped_tiny_box"] > 1         # the 8px boxes now fail too


# ---------- importing ----------

def import_it(client, root, tag="demo-v1", cap=100):
    pid = client.post("/projects", json={"name": "p"}).json()["id"]
    chosen, _ = plan(root, CLASS_MAP, cap=cap)
    with client.session_factory() as db:
        result = run_import(db, pid, chosen, tag, META)
    return pid, chosen, result


def test_import_stores_reviewed_images_with_boxes_and_attribution(client, dataset):
    root, _ = dataset
    pid, chosen, result = import_it(client, root)
    assert result == {"added": len(chosen), "already_imported": 0, "rejected": 0}
    with client.session_factory() as db:
        images = db.scalars(select(Image)).all()
        assert len(images) == len(chosen) and all(i.source == Source.imported and i.review_status == ReviewStatus.reviewed for i in images)
        assert all("/raw/multirotor/imported/" in i.storage_key and (storage._storage.root / i.storage_key).exists() for i in images)
        assert all(i.group_key.startswith("demo-v1:scene-") for i in images)
        assert db.scalar(select(func.count()).select_from(Annotation)) == len(chosen)
        rec = db.scalars(select(ImportRecord)).first()
        assert (rec.dataset, rec.license, rec.author, rec.dataset_url) == ("demo-v1", "CC BY 4.0", "Somebody", "https://example.test/ds/1")
        assert rec.original_split in ("train", "valid", "test") and rec.original_name.endswith(".jpg")
    stats = {r["class_name"]: r for r in client.get(f"/projects/{pid}/stats").json()}
    assert stats["multirotor"]["imported"] == len(chosen) and stats["multirotor"]["pending"] == 0 and stats["multirotor"]["boxes"] == len(chosen)
    assert len(client.get(f"/projects/{pid}/images", params={"source": "imported", "limit": 500}).json()) == len(chosen)


def test_running_the_import_again_adds_nothing(client, dataset):
    root, _ = dataset
    pid, chosen, first = import_it(client, root)
    with client.session_factory() as db:
        second = run_import(db, pid, chosen, "demo-v1", META)
        assert db.scalar(select(func.count()).select_from(Image)) == first["added"]
    assert second == {"added": 0, "already_imported": len(chosen), "rejected": 0}


def test_an_unknown_target_class_is_a_clear_error_and_changes_nothing(client, dataset):
    root, _ = dataset
    pid = client.post("/projects", json={"name": "p"}).json()["id"]
    chosen, _ = plan(root, {"UAV": "no-such-class"})
    with client.session_factory() as db, pytest.raises(ImportProblem, match="no-such-class"):
        run_import(db, pid, chosen, "demo-v1", META)
    with client.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Image)) == 0


def test_a_dataset_version_never_splits_a_scene_across_train_val_and_test(client, dataset):
    root, _ = dataset
    pid, chosen, _ = import_it(client, root)
    v = client.post(f"/projects/{pid}/datasets", json={"val_ratio": 0.2, "test_ratio": 0.2}).json()
    with client.session_factory() as db:
        manifest = datasets.load_manifest(db.get(DatasetVersion, v["id"]))
    splits_per_group = collections.defaultdict(set)
    for it in manifest["images"]:
        splits_per_group[it["group"]].add(it["split"])
    assert len(splits_per_group) == N_SCENES and all(len(s) == 1 for s in splits_per_group.values())
    assert len({it["split"] for it in manifest["images"]}) == 3                   # and all three splits are used


def test_deleting_an_imported_image_removes_its_record_and_file(client, dataset):
    root, _ = dataset
    pid, chosen, _ = import_it(client, root)
    victim = client.get(f"/projects/{pid}/images", params={"source": "imported", "limit": 1}).json()[0]
    with client.session_factory() as db:
        key = db.get(Image, victim["id"]).storage_key
    assert client.delete(f"/images/{victim['id']}").status_code == 204
    with client.session_factory() as db:
        assert db.get(ImportRecord, victim["id"]) is None and db.get(Image, victim["id"]) is None
        assert db.scalar(select(func.count()).select_from(ImportRecord)) == len(chosen) - 1
    assert not (storage._storage.root / key).exists()


# ---------- rollback, backup, bundle ----------

def test_rollback_removes_imported_images_files_and_orphans_but_nothing_else(client, dataset):
    root, _ = dataset
    pid, chosen, _ = import_it(client, root)
    mine = client.post(f"/projects/{pid}/classes/{client.get(f'/projects/{pid}').json()['classes'][0]['id']}/images",
                       files=[("files", ("x.png", png(), "image/png"))]).json()["added"][0]
    orphan = storage._storage.root / "projects" / str(pid) / "raw" / "multirotor" / "imported" / "left-by-a-crash.jpg"
    orphan.write_bytes(b"x")
    with client.session_factory() as db:
        result = rollback(db, "demo-v1")
    assert result == {"images_removed": len(chosen), "orphan_files_removed": 1}
    with client.session_factory() as db:
        assert [db.scalar(select(func.count()).select_from(t)) for t in (ImportRecord, Annotation, ImageHash)] == [0, 0, 1]
        assert db.scalar(select(func.count()).select_from(Image)) == 1 and db.get(Image, mine["id"]) is not None
    assert not orphan.exists() and not list((storage._storage.root / "projects").glob("*/raw/*/imported/*"))


def test_backup_is_a_readable_copy_of_the_live_database(client, tmp_path):
    client.post("/projects", json={"name": "keep-me"})
    engine = client.session_factory.kw["bind"]
    dest = import_yolo.backup_database(engine, tmp_path / "backups", "demo")
    assert dest.exists() and dest.parent == tmp_path / "backups"
    assert sqlite3.connect(dest).execute("select name from projects").fetchall() == [("keep-me",)]


def test_the_bundle_leaves_imported_images_out_unless_asked(client, dataset, tmp_path):
    root, _ = dataset
    pid, chosen, _ = import_it(client, root)
    cls = client.get(f"/projects/{pid}").json()["classes"][0]["id"]
    client.post(f"/projects/{pid}/classes/{cls}/images", files=[("files", ("x.png", png(), "image/png"))])   # an ordinary upload
    engine = client.session_factory.kw["bind"]
    default = bundle.export_bundle(engine, storage._storage.root, tmp_path / "b1")
    assert default["tables"]["images"] == 1 and default["tables"]["annotations"] == 0 and default["tables"]["import_records"] == 0
    assert default["left_out_imported"] == len(chosen) and default["files"] == 1
    everything = bundle.export_bundle(engine, storage._storage.root, tmp_path / "b2", include_imported=True)
    assert everything["tables"]["images"] == len(chosen) + 1 and everything["tables"]["import_records"] == len(chosen)
    assert everything["files"] == len(chosen) + 1


def test_the_bundle_also_leaves_out_files_a_crashed_import_left_behind(client, dataset, tmp_path):
    root, _ = dataset
    pid, chosen, _ = import_it(client, root)
    stray = storage._storage.root / "projects" / str(pid) / "raw" / "multirotor" / "imported" / "stray.jpg"
    stray.write_bytes(b"x")
    engine = client.session_factory.kw["bind"]
    out = bundle.export_bundle(engine, storage._storage.root, tmp_path / "b")
    assert out["files"] == 0 and not (tmp_path / "b" / "storage" / "projects" / str(pid) / "raw" / "multirotor" / "imported" / "stray.jpg").exists()


# ---------- command line ----------

def test_dry_run_reports_and_writes_a_sample_sheet_but_changes_nothing(dataset, tmp_path, capsys):
    root, _ = dataset
    sheet = tmp_path / "sheet.jpg"
    code = import_yolo.main(["import", str(root), "--dataset", "demo-v1", "--map", "UAV=multirotor", "--author", "Somebody",
                             "--dry-run", "--sample-sheet", str(sheet)])
    out = capsys.readouterr().out
    assert code == 0 and "DRY RUN" in out and "selected:" in out and sheet.exists() and "backed up" not in out


def test_import_without_a_class_map_is_refused(dataset, capsys):
    root, _ = dataset
    assert import_yolo.main(["import", str(root), "--dataset", "d", "--author", "x", "--dry-run"]) == 1
    assert "--map" in capsys.readouterr().err


def test_a_mirrored_copy_of_a_scene_is_grouped_with_it(tmp_path):
    """Roboflow mirrors half of its augmented frames; a flipped frame must still land in the same scene."""
    pattern = _scene_pattern(7)
    other = _scene_pattern(8)
    paths = {}
    for name, im in (("a", pattern), ("a_flipped", ImageOps.mirror(pattern)), ("b", other)):
        paths[name] = tmp_path / f"{name}.jpg"
        im.save(paths[name], quality=95)
    frames = [import_yolo.Frame("valid", n, p, [("multirotor", .5, .5, .1, .1)], SIZE, SIZE, augmented=False) for n, p in paths.items()]
    import_yolo.assign_scenes(frames, 0.97)
    scene = {f.stem: f.scene for f in frames}
    assert scene["a"] == scene["a_flipped"] != scene["b"]
