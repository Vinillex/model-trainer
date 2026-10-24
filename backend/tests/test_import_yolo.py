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
        assert all("/raw/multirotor/imported/demo-v1/" in i.storage_key and (storage._storage.root / i.storage_key).exists() for i in images)
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


def test_dataset_names_must_be_safe_folder_names(dataset, capsys):
    root, _ = dataset
    for bad in ("Bad Name", "../escape", "UPPER", "", "-leading-dash", "a" * 65):
        assert import_yolo.main(["import", str(root), f"--dataset={bad}", "--map", "UAV=multirotor", "--author", "x", "--dry-run"]) == 1, bad
    assert "folder name" in capsys.readouterr().err
    assert import_yolo.main(["import", str(root), "--dataset", "anti-uav_2", "--map", "UAV=multirotor", "--author", "x", "--dry-run"]) == 0


def test_a_rolled_back_dataset_leaves_no_empty_folder_behind(client, dataset):
    root, _ = dataset
    pid, chosen, _ = import_it(client, root)
    folder = storage._storage.root / "projects" / str(pid) / "raw" / "multirotor" / "imported" / "demo-v1"
    assert folder.is_dir() and any(folder.iterdir())
    with client.session_factory() as db:
        rollback(db, "demo-v1")
    assert not folder.exists()


def test_an_imported_image_boxed_as_another_class_stays_inside_its_dataset_folder(client, dataset):
    root, _ = dataset
    pid, chosen, _ = import_it(client, root)
    ids = {c["name"]: c["id"] for c in client.get(f"/projects/{pid}").json()["classes"]}
    victim = client.get(f"/projects/{pid}/images", params={"source": "imported", "limit": 1}).json()[0]
    r = client.put(f"/images/{victim['id']}/annotations", json={"boxes": [{"class_id": ids["fixed-wing"], "cx": .5, "cy": .5, "w": .1, "h": .1}]})
    assert r.status_code == 200 and r.json()["class_id"] == ids["fixed-wing"]            # re-filed by what its box now shows
    with client.session_factory() as db:
        key = db.get(Image, victim["id"]).storage_key
    assert "/raw/fixed-wing/imported/demo-v1/" in key and (storage._storage.root / key).exists()


# ---------- negative mode ----------

NEG = {"bird": "birds"}


def build_bird_dataset(root: Path) -> None:
    """Frames for negative mode (96 px images): clean big birds in two scenes, and every kind of frame that must be dropped."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("nc: 2\nnames: ['UAV', 'bird']\n\nroboflow:\n  license: CC BY 4.0\n  url: https://example.test/ds/2\n")
    n = collections.Counter()

    def add(scene, boxes, split="valid"):
        n[split] += 1
        im = _scene_pattern(200 + scene).copy()
        ImageDraw.Draw(im).rectangle([n[split] % 40, 5, n[split] % 40 + 3, 8], fill=(10, 10, 10))  # frames of a scene differ a little
        _write(root, split, f"{n[split]:05d}_jpg.rf.b{scene}{n[split]:03d}", im, boxes)

    big = "1 0.5 0.5 0.4 0.4"          # 38 px of 96: far above the 10% floor
    for scene in (0, 1):
        for _ in range(3):
            add(scene, [big])
    add(0, [big, "1 0.1 0.1 0.04 0.04"])                  # a flock with one tiny bird: dropped
    add(2, ["1 0.5 0.5 0.05 0.05"])                       # a single tiny bird: dropped
    add(0, [big, "0 0.2 0.2 0.3 0.3"])                    # a bird and a drone together: never a clean negative
    add(2, [])                                            # nothing boxed: unknown content, dropped
    add(3, ["0 0.5 0.5 0.3 0.3"])                         # a plain drone frame, for the positives


@pytest.fixture()
def birds(tmp_path):
    root = tmp_path / "birds"
    build_bird_dataset(root)
    return root


def test_negative_mode_keeps_only_clean_birds_that_are_big_enough(birds):
    chosen, stats = plan(birds, CLASS_MAP, cap=100, negative_map=NEG)
    negatives = [f for f in chosen if f.negative]
    assert len(negatives) == 6 and stats["selected_negatives"] == 6
    assert all(f.boxes[0][0] == "birds" for f in negatives)                # filed under our class, whatever the dataset called it
    assert stats["dropped_negative_too_small"] == 2                        # the flock with a tiny bird, and the lone tiny bird
    assert stats["dropped_negative_with_drone_class"] == 1 and stats["dropped_no_boxes"] == 1
    positives = [f for f in chosen if not f.negative]
    assert len(positives) == 1 and positives[0].boxes[0][0] == "multirotor"
    assert summarize_classes(chosen) == {"birds": 6, "multirotor": 1}


def summarize_classes(chosen):
    return import_yolo.summarize(chosen, collections.Counter())["selected_by_class"]


def test_a_bird_next_to_an_unmapped_drone_class_is_dropped_too(birds):
    chosen, stats = plan(birds, {}, cap=100, negative_map=NEG)               # UAV not mapped at all
    assert stats["selected_negatives"] == 6 and stats["dropped_unmapped_or_malformed"] == 2   # the bird+drone frame, the plain drone frame


def test_the_size_floor_is_a_fraction_of_the_images_short_side(birds):
    assert plan(birds, {}, negative_map=NEG, negative_min_size=0.5)[1]["selected_negatives"] == 0        # birds are 40% of the image
    assert plan(birds, {}, negative_map=NEG, negative_min_size=0.0)[1]["selected_negatives"] == 8        # no floor: the tiny birds come back
    assert plan(birds, {}, negative_map=NEG, negative_min_size=0.03)[1]["selected_negatives"] == 8       # 0.04 and 0.05 both pass 3%
    assert plan(birds, {}, negative_map=NEG, negative_min_size=0.045)[1]["selected_negatives"] == 7      # only the 0.04 bird fails


def test_a_class_cannot_be_both_a_drone_and_a_negative(birds):
    with pytest.raises(ImportProblem, match="both --map and --negative"):
        plan(birds, {"bird": "multirotor"}, negative_map=NEG)


def import_with_negatives(client, root, tag="birds-v1"):
    pid = client.post("/projects", json={"name": "p"}).json()["id"]
    chosen, _ = plan(root, CLASS_MAP, cap=100, negative_map=NEG)
    with client.session_factory() as db:
        result = run_import(db, pid, chosen, tag, {**META, "url": "https://example.test/ds/2"})
    return pid, chosen, result


def test_negative_images_are_filed_under_the_negative_class_reviewed_and_without_boxes(client, birds):
    pid, chosen, result = import_with_negatives(client, birds)
    assert result == {"added": 7, "already_imported": 0, "rejected": 0}
    with client.session_factory() as db:
        negatives = db.scalars(select(Image).where(Image.storage_key.contains("/raw/birds/"))).all()
        assert len(negatives) == 6
        for img in negatives:
            assert img.source == Source.imported and img.review_status == ReviewStatus.reviewed and not img.annotations
            assert "/raw/birds/imported/birds-v1/" in img.storage_key and (storage._storage.root / img.storage_key).exists()
            assert img.group_key.startswith("birds-v1:scene-")
            assert img.import_record.dataset == "birds-v1" and img.import_record.dataset_url == "https://example.test/ds/2"
        assert db.scalar(select(func.count()).select_from(Annotation)) == 1            # only the drone frame has a box
    rows = {r["class_name"]: r for r in client.get(f"/projects/{pid}/stats").json()}
    assert rows["birds"]["imported"] == 6 and rows["birds"]["boxes"] == 0 and rows["multirotor"]["boxes"] == 1
    with client.session_factory() as db:
        again = run_import(db, pid, chosen, "birds-v1", META)
    assert again == {"added": 0, "already_imported": 7, "rejected": 0}               # a re-run is harmless here too


def test_negative_images_become_empty_background_images_in_a_dataset_version(client, birds):
    pid, _, _ = import_with_negatives(client, birds)
    v = client.post(f"/projects/{pid}/datasets", json={"imported_datasets": ["birds-v1"]}).json()
    with client.session_factory() as db:
        items = datasets.load_manifest(db.get(DatasetVersion, v["id"]))["images"]
    background = [it for it in items if it["folder_class"] == "birds"]
    assert len(background) == 6 and all(it["boxes"] == [] for it in background) and len(items) == 7
    without = client.post(f"/projects/{pid}/datasets", json={"imported_datasets": ["birds-v1"], "include_negatives": False}).json()
    with client.session_factory() as db:
        assert len(datasets.load_manifest(db.get(DatasetVersion, without["id"]))["images"]) == 1


def test_rollback_removes_the_negatives_and_their_files(client, birds):
    pid, _, _ = import_with_negatives(client, birds)
    with client.session_factory() as db:
        keys = list(db.scalars(select(Image.storage_key)))
        assert rollback(db, "birds-v1")["images_removed"] == 7
        assert db.scalar(select(func.count()).select_from(Image)) == 0
    assert not any((storage._storage.root / k).exists() for k in keys)


def test_a_negative_target_must_be_a_negative_class_and_nothing_is_written_otherwise(client, birds):
    pid = client.post("/projects", json={"name": "p"}).json()["id"]
    for target in ("multirotor", "no-such-class", "mixed"):
        chosen, _ = plan(birds, CLASS_MAP, negative_map={"bird": target})
        with client.session_factory() as db, pytest.raises(ImportProblem, match="hard-negative"):
            run_import(db, pid, chosen, "birds-v1", META)
    chosen, _ = plan(birds, {"UAV": "birds"})                                          # and a drone class may not point at a negative
    with client.session_factory() as db, pytest.raises(ImportProblem, match="trainable"):
        run_import(db, pid, chosen, "birds-v1", META)
    with client.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Image)) == 0


def test_negative_dry_run_reports_the_negatives_and_a_sample_sheet(birds, tmp_path, capsys):
    sheet = tmp_path / "birds.jpg"
    code = import_yolo.main(["import", str(birds), "--dataset", "birds-v1", "--negative", "bird=birds", "--author", "x",
                             "--dry-run", "--sample-sheet", str(sheet)])
    out = capsys.readouterr().out
    assert code == 0 and "selected_negatives: 6" in out and "'birds': 6" in out and sheet.exists()
    assert import_yolo.main(["import", str(birds), "--dataset", "birds-v1", "--negative", "bird=birds", "--author", "x",
                             "--negative-min-size", "2", "--dry-run"]) == 1
    assert "between 0 and 1" in capsys.readouterr().err
