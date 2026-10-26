import io
import zipfile
from collections import defaultdict

import pytest

from app.datasets import assign_splits, load_manifest, split_quota
from app.models import DatasetVersion, Image, ScrapeRecord, Source
from app.storage import get_storage
from app.yolo_export import SPLITS, data_yaml, label_text

from .helpers import photo


# ---------- split logic (pure) ----------

@pytest.mark.parametrize("n, val, test, expected", [
    (0, 0.1, 0.1, (0, 0)),
    (1, 0.1, 0.1, (0, 0)),    # a lone group stays in train
    (2, 0.1, 0.1, (1, 0)),    # a validation group exists as soon as it can, train keeps one
    (3, 0.1, 0.1, (1, 1)),
    (10, 0.1, 0.1, (1, 1)),
    (100, 0.1, 0.2, (10, 20)),
    (50, 0.0, 0.0, (0, 0)),   # ratios of zero mean no such split
    (4, 0.5, 0.5, (2, 1)),    # never starve train
])
def test_split_quota(n, val, test, expected):
    assert split_quota(n, val, test) == expected


def make_items(spec):
    """spec: list of (group, stratum, source)."""
    return [{"group": g, "stratum": s, "source": src} for g, s, src in spec]


def splits_by_group(items):
    out = defaultdict(set)
    for it in items:
        out[it["group"]].add(it["split"])
    return out


def test_assign_splits_keeps_groups_together_and_is_deterministic():
    spec = [(f"page-{i // 3}", "multirotor", "scraped") for i in range(60)] + [(f"solo-{i}", "fixed-wing", "uploaded") for i in range(20)]
    a, b = make_items(spec), make_items(spec)
    assign_splits(a, 0.15, 0.15, seed=7)
    assign_splits(b, 0.15, 0.15, seed=7)
    assert [i["split"] for i in a] == [i["split"] for i in b]
    assert all(len(s) == 1 for s in splits_by_group(a).values()), "a group was split across train/val/test"
    c = make_items(spec)
    assign_splits(c, 0.15, 0.15, seed=8)
    assert [i["split"] for i in a] != [i["split"] for i in c]  # a different seed gives a different split


def test_assign_splits_stratifies_by_class_and_keeps_test_real_only():
    spec = ([(f"m{i}", "multirotor", "uploaded") for i in range(20)] + [(f"f{i}", "fixed-wing", "scraped") for i in range(10)]
            + [(f"s{i}", "delta-wing", "uploaded") for i in range(4)] + [(f"g{i}", "multirotor", "generated") for i in range(30)])
    items = make_items(spec)
    assign_splits(items, 0.1, 0.1, seed=1)
    for cls in ("multirotor", "fixed-wing", "delta-wing"):
        seen = {i["split"] for i in items if i["stratum"] == cls and i["source"] != "generated"}
        assert {"train", "val"} <= seen, f"{cls} missing from val"
    assert not any(i["split"] != "train" for i in items if i["source"] == "generated")  # synthetic data never measures the model
    assert any(i["split"] == "test" for i in items)


# ---------- manifest, stats, export through the API ----------

@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


_seed = iter(range(1000, 10_000))


def add(client, pid, class_id, boxes=None, status="reviewed", group=None):
    """Upload one fresh image and optionally review it with `boxes` = [(class_id, cx, cy, w, h)]."""
    params = {"group_key": group} if group else {}
    img = client.post(f"/projects/{pid}/classes/{class_id}/images", params=params,
                      files=[("files", ("x.jpg", photo(next(_seed)), "image/jpeg"))]).json()["added"][0]
    if status != "pending":
        body = {"boxes": [{"class_id": c, "cx": cx, "cy": cy, "w": w, "h": h} for c, cx, cy, w, h in (boxes or [])], "review_status": status}
        assert client.put(f"/images/{img['id']}/annotations", json=body).status_code == 200
    return img


def box(cls_id, cx=0.5, cy=0.5, w=0.2, h=0.1):
    return (cls_id, cx, cy, w, h)


def populate(client, pid, ids):
    for _ in range(14):
        add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    for _ in range(8):
        add(client, pid, ids["fixed-wing"], [box(ids["fixed-wing"], 0.3, 0.6)])
    for _ in range(4):
        add(client, pid, ids["birds"])                                  # hard negatives: reviewed automatically
    add(client, pid, ids["multirotor"], status="pending")               # not reviewed: excluded
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])], status="rejected")  # rejected: excluded
    add(client, pid, ids["delta-wing"], [], status="reviewed")              # reviewed, nothing in it: a background image


def unzip(resp):
    return zipfile.ZipFile(io.BytesIO(resp.content))


def test_preview_counts_what_would_be_included(client, project):
    pid, ids = project
    populate(client, pid, ids)
    p = client.get(f"/projects/{pid}/datasets/preview").json()
    assert (p["reviewed_images"], p["reviewed_with_boxes"], p["reviewed_empty"], p["negative_images"]) == (27, 22, 1, 4)
    assert (p["pending"], p["rejected"]) == (1, 1)
    assert p["boxes_per_class"] == {"multirotor": 14, "fixed-wing": 8, "delta-wing": 0}


def test_create_version_stats_and_warnings(client, project):
    pid, ids = project
    populate(client, pid, ids)
    r = client.post(f"/projects/{pid}/datasets", json={})
    assert r.status_code == 201
    v = r.json()
    assert v["name"] == "v1" and v["config"]["classes"] == [{"index": 0, "name": "multirotor"}, {"index": 1, "name": "fixed-wing"}, {"index": 2, "name": "delta-wing"}]
    st = v["stats"]
    assert sum(st["images"].values()) == 27 and sum(st["boxes"].values()) == 22
    assert st["excluded"] == {"pending": 1, "rejected": 1, "not_selected": 0}
    assert sum(st["background_images"].values()) == 5  # 4 hard negatives + the empty delta-wing image
    assert st["images"]["train"] > st["images"]["val"] > 0
    warnings = " ".join(st["warnings"])
    assert "'delta-wing' has no boxes at all" in warnings          # can't learn a class that has no examples
    assert "'fixed-wing' has only 8 boxes" in warnings
    assert client.post(f"/projects/{pid}/datasets", json={}).json()["name"] == "v2"
    assert [d["name"] for d in client.get(f"/projects/{pid}/datasets").json()] == ["v2", "v1"]


def test_download_is_a_valid_yolo_dataset(client, project):
    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={"name": "first", "seed": 3}).json()
    z = unzip(client.get(f"/datasets/{v['id']}/download"))
    names = z.namelist()

    yaml = z.read("data.yaml").decode()
    assert 'names:\n  0: "multirotor"\n  1: "fixed-wing"\n  2: "delta-wing"' in yaml
    assert "train: images/train" in yaml and "val: images/val" in yaml and "test: images/test" in yaml

    images = [n for n in names if n.startswith("images/")]
    labels = [n for n in names if n.startswith("labels/")]
    assert len(images) == len(labels) == 27
    for n in images:  # every image has a same-named label file in the same split
        split, file = n.split("/")[1:]
        assert f"labels/{split}/{file.rsplit('.', 1)[0]}.txt" in names
    per_split = {s: sum(1 for n in images if n.startswith(f"images/{s}/")) for s in ("train", "val", "test")}
    assert per_split == v["stats"]["images"]

    # label content: class index + normalized box, and background images get an empty file
    texts = [z.read(n).decode() for n in labels]
    assert sum(1 for t in texts if t == "") == 5
    assert "0 0.500000 0.500000 0.200000 0.100000\n" in texts
    assert "1 0.300000 0.600000 0.200000 0.100000\n" in texts
    for t in texts:
        for line in t.splitlines():
            c, *nums = line.split()
            assert int(c) in (0, 1, 2) and all(0 <= float(x) <= 1 for x in nums)

    # exported bytes are exactly what was uploaded
    sha_in_zip = {__import__("hashlib").sha256(z.read(n)).hexdigest() for n in images}
    with client.session_factory() as db:
        assert sha_in_zip == {i.sha256 for i in db.query(Image).filter(Image.review_status == "reviewed")}


def test_version_is_a_frozen_snapshot(client, project):
    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    before = unzip(client.get(f"/datasets/{v['id']}/download"))
    labels_before = {n: before.read(n) for n in before.namelist() if n.startswith("labels/")}

    # keep editing the project afterwards: change boxes, review another image, add a class
    first = client.get(f"/projects/{pid}/images", params={"class_id": ids["multirotor"], "review_status": "reviewed"}).json()[0]
    client.put(f"/images/{first['id']}/annotations", json={"boxes": [], "review_status": "reviewed"})
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    client.post(f"/projects/{pid}/classes/bulk", json={"names": ["vtol"]})

    after = unzip(client.get(f"/datasets/{v['id']}/download"))
    assert {n: after.read(n) for n in after.namelist() if n.startswith("labels/")} == labels_before
    assert client.get(f"/datasets/{v['id']}").json()["stats"] == v["stats"]


def test_groups_never_straddle_splits(client, project):
    pid, ids = project
    for g in range(6):  # 6 "videos" of 5 frames each
        for _ in range(5):
            add(client, pid, ids["multirotor"], [box(ids["multirotor"])], group=f"video-{g}")
    for _ in range(6):
        add(client, pid, ids["fixed-wing"], [box(ids["fixed-wing"])])
    v = client.post(f"/projects/{pid}/datasets", json={"val_ratio": 0.2, "test_ratio": 0.2}).json()
    z = unzip(client.get(f"/datasets/{v['id']}/download"))
    split_of = {int(n.rsplit("/", 1)[1].split(".")[0]): n.split("/")[1] for n in z.namelist() if n.startswith("images/")}
    with client.session_factory() as db:
        by_group = defaultdict(set)
        for img in db.query(Image).filter(Image.group_key.isnot(None)):
            by_group[img.group_key].add(split_of[img.id])
    assert len(by_group) == 6 and all(len(s) == 1 for s in by_group.values()), by_group


def test_photos_of_one_commons_series_share_a_group_but_ordinary_pages_stay_apart(client, project):
    """Near-identical frames of one sortie live on separate File: pages; the split must not scatter them. Other pages are not merged."""
    pid, ids = project
    scraped = {}

    def attach(img, page):
        with client.session_factory() as db:
            db.add(ScrapeRecord(image_id=img["id"], query="q", original_url=page + ".jpg", page_url=page))
            db.commit()
        scraped[img["id"]] = page

    for s in range(8):  # 8 sorties of 4 photos each, one File: page per photo
        for k in range(4):
            attach(add(client, pid, ids["fixed-wing"], [box(ids["fixed-wing"])]),
                   f"https://commons.wikimedia.org/wiki/File:MQ-9_Sortie_{s}_Over_Town_16101{k}-Z-QU230-00{k}.jpg")
    for n in range(6):  # ordinary pages whose address merely ends in a number
        attach(add(client, pid, ids["multirotor"], [box(ids["multirotor"])]), f"https://news.test/a/story-{n}")
    v = client.post(f"/projects/{pid}/datasets", json={"val_ratio": 0.25, "test_ratio": 0.25}).json()
    manifest = __import__("json").loads(get_storage().get(f"projects/{pid}/datasets/{v['id']}/manifest.json"))
    groups = {it["image_id"]: it["group"] for it in manifest["images"]}
    commons = {i for i, page in scraped.items() if "wikimedia" in page}
    assert len({groups[i] for i in commons}) == 8  # one group per sortie, not per photo
    assert len({groups[i] for i in scraped if i not in commons}) == 6  # a news article per group, as before
    split_of = {it["image_id"]: it["split"] for it in manifest["images"]}
    by_series = defaultdict(set)
    for i in commons:
        by_series[groups[i]].add(split_of[i])
    assert all(len(sp) == 1 for sp in by_series.values()), by_series
    assert len({sp for group in by_series.values() for sp in group}) > 1  # the check is not vacuous: sorties do land in different splits


def test_negatives_can_be_left_out(client, project):
    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={"include_negatives": False}).json()
    assert sum(v["stats"]["images"].values()) == 23  # the 4 hard negatives are gone


def test_no_test_split_when_ratio_is_zero(client, project):
    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={"test_ratio": 0}).json()
    assert v["stats"]["images"]["test"] == 0
    z = unzip(client.get(f"/datasets/{v['id']}/download"))
    assert "test:" not in z.read("data.yaml").decode()
    assert any("test split is empty" in w for w in v["stats"]["warnings"])


def test_create_rejects_unusable_requests(client, project):
    pid, ids = project
    r = client.post(f"/projects/{pid}/datasets", json={})
    assert r.status_code == 422 and "No reviewed images" in r.json()["detail"]

    add(client, pid, ids["birds"])  # only a background image: still nothing to learn from
    r = client.post(f"/projects/{pid}/datasets", json={})
    assert r.status_code == 422 and "has a box" in r.json()["detail"]

    for _ in range(3):
        add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    assert client.post(f"/projects/{pid}/datasets", json={"name": "a"}).status_code == 201
    assert client.post(f"/projects/{pid}/datasets", json={"name": "a"}).status_code == 409
    assert client.post(f"/projects/{pid}/datasets", json={"val_ratio": 0.4, "test_ratio": 0.4}).status_code == 422
    assert client.post(f"/projects/{pid}/datasets", json={"val_ratio": -0.1}).status_code == 422
    assert client.get("/datasets/999").status_code == 404
    assert client.get("/datasets/999/download").status_code == 404


def test_download_after_an_image_was_deleted_explains_itself(client, project):
    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    victim = client.get(f"/projects/{pid}/images", params={"class_id": ids["fixed-wing"]}).json()[0]
    client.delete(f"/images/{victim['id']}")
    r = client.get(f"/datasets/{v['id']}/download")
    assert r.status_code == 409 and "no longer exist" in r.json()["detail"] and "create a new version" in r.json()["detail"]


def test_delete_version_removes_its_manifest(client, project):
    pid, ids = project
    populate(client, pid, ids)
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    from app import storage

    key = f"projects/{pid}/datasets/{v['id']}/manifest.json"
    assert storage.get_storage().get(key)
    assert client.delete(f"/datasets/{v['id']}").status_code == 204
    with pytest.raises(FileNotFoundError):
        storage.get_storage().get(key)
    assert client.get(f"/projects/{pid}/datasets").json() == []


def test_label_and_yaml_formatting():
    item = {"image_id": 7, "boxes": [[2, 0.123456789, 0.5, 1.0, 0.25]]}
    assert label_text(item) == "2 0.123457 0.500000 1.000000 0.250000\n"
    assert label_text({"boxes": []}) == ""
    m = {"classes": [{"index": 0, "name": 'a "b"'}], "images": [{"split": "train"}]}
    y = data_yaml(m)
    assert 'names:\n  0: "a \\"b\\""' in y and "test:" not in y  # names are quoted/escaped, test omitted when empty


@pytest.mark.parametrize("seed", range(1, 9))
def test_split_is_balanced_by_the_class_of_the_boxes_not_the_class_the_image_was_filed_under(client, project, seed):
    pid, ids = project
    # One mixed search, so every image is filed under multirotor even though 10 of them show fixed-wing drones.
    for _ in range(14):
        add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    for _ in range(10):
        add(client, pid, ids["multirotor"], [box(ids["fixed-wing"], 0.3, 0.6)])
    per_class = client.post(f"/projects/{pid}/datasets", json={"seed": seed}).json()["stats"]["per_class"]
    for cls in ("multirotor", "fixed-wing"):
        for split in ("val", "test"):
            assert per_class[cls][split]["boxes"] > 0, f"{cls} is missing from {split} with seed {seed}"


def test_an_image_is_balanced_by_its_most_common_box_class_and_background_images_by_their_filing(client, project):
    pid, ids = project
    for _ in range(6):                                                    # 2 fixed-wing boxes + 1 multirotor box: fixed-wing wins
        add(client, pid, ids["multirotor"], [box(ids["fixed-wing"], 0.2), box(ids["fixed-wing"], 0.6), box(ids["multirotor"], 0.9)])
    for _ in range(6):
        add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    for _ in range(4):
        add(client, pid, ids["birds"])                                    # hard negatives keep their own stratum
    v = client.post(f"/projects/{pid}/datasets", json={}).json()
    st = v["stats"]
    assert st["images"]["val"] > 0 and st["background_images"].get("val", 0) + st["background_images"].get("test", 0) >= 1
    assert st["per_class"]["fixed-wing"]["val"]["boxes"] > 0 and st["per_class"]["multirotor"]["val"]["boxes"] > 0


# ---------- treating classes as one ----------

def merged_version(client, pid, merge, **body):
    return client.post(f"/projects/{pid}/datasets", json={"merge_classes": merge, **body})


def test_a_version_can_treat_two_classes_as_one_without_touching_the_labels(client, project):
    pid, ids = project
    populate(client, pid, ids)
    add(client, pid, ids["delta-wing"], [box(ids["delta-wing"], 0.7, 0.2)])
    add(client, pid, ids["fixed-wing"], [box(ids["fixed-wing"], 0.2, 0.2), box(ids["delta-wing"], 0.8, 0.8)])  # both kinds in one picture

    r = merged_version(client, pid, [{"class_ids": [ids["fixed-wing"], ids["delta-wing"]]}])

    assert r.status_code == 201, r.text
    v = r.json()
    assert v["config"]["classes"] == [{"index": 0, "name": "multirotor"}, {"index": 1, "name": "fixed-wing+delta-wing"}]
    assert v["config"]["merged"] == [{"name": "fixed-wing+delta-wing", "classes": ["fixed-wing", "delta-wing"]}]
    per_class = v["stats"]["per_class"]
    assert list(per_class) == ["multirotor", "fixed-wing+delta-wing"]
    assert sum(per_class["fixed-wing+delta-wing"][s]["boxes"] for s in SPLITS) == 8 + 1 + 2  # 8 fixed-wing, 1 delta-wing, and both kinds in one picture
    assert "no boxes at all" not in " ".join(v["stats"]["warnings"])  # delta-wing on its own would have had one
    z = unzip(client.get(f"/datasets/{v['id']}/download"))
    assert 'names:\n  0: "multirotor"\n  1: "fixed-wing+delta-wing"\n' in z.read("data.yaml").decode()
    used = {line.split()[0] for n in z.namelist() if n.startswith("labels/") for line in z.read(n).decode().splitlines()}
    assert used == {"0", "1"}  # no index beyond the two classes
    # the project's own boxes keep their classes
    assert client.get(f"/projects/{pid}/datasets/preview").json()["boxes_per_class"] == {"multirotor": 14, "fixed-wing": 9, "delta-wing": 2}
    # a version made without merging is exactly as before
    plain = client.post(f"/projects/{pid}/datasets", json={}).json()
    assert len(plain["config"]["classes"]) == 3 and "merged" not in plain["config"]


def test_merged_class_takes_its_first_members_place_and_indexes_stay_contiguous(client, project):
    pid, ids = project
    populate(client, pid, ids)
    add(client, pid, ids["delta-wing"], [box(ids["delta-wing"])])
    v = merged_version(client, pid, [{"class_ids": [ids["delta-wing"], ids["multirotor"]], "name": "small-things"}]).json()
    assert v["config"]["classes"] == [{"index": 0, "name": "small-things"}, {"index": 1, "name": "fixed-wing"}]
    z = unzip(client.get(f"/datasets/{v['id']}/download"))
    assert 'names:\n  0: "small-things"\n  1: "fixed-wing"\n' in z.read("data.yaml").decode()


def test_two_separate_merges_in_one_version(client, project):
    pid, ids = project
    other = client.post(f"/projects/{pid}/classes/bulk", json={"names": ["interceptor-test", "vtol-test"]})
    assert other.status_code in (200, 201), other.text
    all_ids = {c["name"]: c["id"] for c in client.get(f"/projects/{pid}").json()["classes"]}
    populate(client, pid, ids)
    add(client, pid, all_ids["interceptor-test"], [box(all_ids["interceptor-test"])])
    add(client, pid, all_ids["vtol-test"], [box(all_ids["vtol-test"])])
    v = merged_version(client, pid, [{"class_ids": [ids["multirotor"], all_ids["interceptor-test"]], "name": "rotors"},
                                     {"class_ids": [ids["fixed-wing"], ids["delta-wing"]], "name": "wings"}]).json()
    assert [c["name"] for c in v["config"]["classes"]] == ["rotors", "wings", "vtol-test"]
    assert [m["name"] for m in v["config"]["merged"]] == ["rotors", "wings"]


def test_merging_is_refused_when_it_does_not_make_sense(client, project):
    pid, ids = project
    populate(client, pid, ids)
    other = client.post("/projects", json={"name": "other"}).json()
    foreign = other["classes"][0]["id"]
    fw, dw, mr, birds = ids["fixed-wing"], ids["delta-wing"], ids["multirotor"], ids["birds"]
    for merge, why in [
        ([{"class_ids": [fw]}], None),                                                     # one class is not a merge (schema)
        ([{"class_ids": [fw, fw]}], "at least two different"),
        ([{"class_ids": [fw, birds]}], "not a drone class"),                               # a hard negative
        ([{"class_ids": [fw, foreign]}], "not a drone class"),                             # another project's class
        ([{"class_ids": [fw, 99999]}], "not a drone class"),
        ([{"class_ids": [fw, dw]}, {"class_ids": [dw, mr]}], "in two merged groups"),
        ([{"class_ids": [fw, dw], "name": "multirotor"}], "both be called 'multirotor'"),  # clashes with a class that stays separate
        ([{"class_ids": [fw, dw], "name": "Has Spaces"}], None),                          # schema: lowercase names only
        ([{"class_ids": [fw, dw], "name": "x"}] * 6, None),                                # schema: at most five groups
    ]:
        r = merged_version(client, pid, merge)
        assert r.status_code == 422, (merge, r.text)
        if why:
            assert why in r.json()["detail"], (merge, r.json())
    assert client.get(f"/projects/{pid}/datasets").json() == []  # nothing half-made


# ---------- leaving out similar pictures (same scene, boxes barely moved) ----------

def test_similar_keeps_one_of_each_run_of_repeats_and_anything_that_moved():
    from app.similar import similar_to_skip
    scene, other = 0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0
    b = lambda cx, cy=0.5, w=0.05, h=0.05, c=0: [[c, cx, cy, w, h]]
    items = [(1, b(0.50)), (2, b(0.52)), (3, b(0.53)),   # a run of repeats: the first stays
             (4, b(0.80)),                               # same scene, drone moved 30%: kept
             (5, b(0.50, w=0.2, h=0.2)),                 # same place but four times the size (zoomed): kept
             (6, b(0.50, c=1)),                          # same place, another class: kept
             (7, b(0.50) + b(0.2)),                      # an extra drone: kept
             (8, []),                                    # no boxes (a negative): never compared
             (9, b(0.50)),                               # repeat of 1 but in another scene: kept
             (10, b(0.50))]                              # no fingerprint: kept
    hashes = {i: scene for i in range(1, 9)} | {9: other}
    hashes[3] = scene ^ 0b1111                           # 4 bits off is still the same scene
    assert similar_to_skip(items, hashes, 5) == {2, 3}
    assert similar_to_skip(items, hashes, 1) == set()     # tighter: 2% and 3% moves are real moves
    assert similar_to_skip(items, {**hashes, 3: scene ^ 0b11111}, 5) == {2}  # 5 bits off: another scene


def test_similar_compares_with_kept_images_only():
    """A slow drift (0.50 -> 0.54 -> 0.58) must not erase the whole run: each image is compared with what was kept."""
    from app.similar import similar_to_skip
    items = [(i, [[0, 0.50 + 0.04 * (i - 1), 0.5, 0.05, 0.05]]) for i in (1, 2, 3)]
    assert similar_to_skip(items, {1: 7, 2: 7, 3: 7}, 5) == {2}  # 3 is 8% from 1, the image it would repeat


def _same_scene(client, image_ids, value=0x0123456789ABCDEF):
    from app.models import ImageHash
    with client.session_factory() as s:
        for iid in image_ids:
            s.get(ImageHash, iid).dhash = f"{value:016x}"
        s.commit()


def test_a_version_can_leave_out_similar_pictures(client, project):
    pid, ids = project
    mr = ids["multirotor"]
    run = [add(client, pid, mr, [box(mr, 0.5 + d, 0.5)])["id"] for d in (0, 0.01, 0.02, 0.3)]  # three repeats, then a move
    others = [add(client, pid, mr, [box(mr)])["id"] for _ in range(6)]
    _same_scene(client, run)
    assert client.get(f"/projects/{pid}/datasets/preview", params={"skip_similar_pct": 5}).json()["similar"] == 2
    assert client.get(f"/projects/{pid}/datasets/preview").json()["similar"] is None

    v = client.post(f"/projects/{pid}/datasets", json={"skip_similar": True, "similar_box_pct": 5}).json()
    assert v["config"]["skip_similar"] == {"box_pct": 5.0, "hash_bits": 4}
    assert v["stats"]["excluded"]["similar"] == 2
    with client.session_factory() as s:
        manifest = load_manifest(s.get(DatasetVersion, v["id"]))
    assert {it["image_id"] for it in manifest["images"]} == {run[0], run[3], *others}
    assert sum(v["stats"]["images"].values()) == 8
    with client.session_factory() as s:  # nothing in the project changed
        assert all(s.get(Image, i).review_status.value == "reviewed" for i in run)


def test_versions_without_the_option_are_unchanged(client, project):
    pid, ids = project
    run = [add(client, pid, ids["multirotor"], [box(ids["multirotor"])])["id"] for _ in range(4)]
    _same_scene(client, run)
    v = client.post(f"/projects/{pid}/datasets", json={"similar_box_pct": 5}).json()  # the box alone does nothing
    assert "skip_similar" not in v["config"] and "similar" not in v["stats"]["excluded"]
    assert sum(v["stats"]["images"].values()) == 4


@pytest.mark.parametrize("pct", [0, 0.4, 25.1, -1])
def test_similar_distance_must_be_sensible(client, project, pct):
    pid, ids = project
    add(client, pid, ids["multirotor"], [box(ids["multirotor"])])
    assert client.post(f"/projects/{pid}/datasets", json={"skip_similar": True, "similar_box_pct": pct}).status_code == 422
    assert client.get(f"/projects/{pid}/datasets/preview", params={"skip_similar_pct": pct}).status_code == 422


# ---------- keeping look-alike scenes in one split ----------

def test_scene_groups_join_frames_of_one_scene_also_across_imported_datasets():
    from app.similar import scene_groups
    a, far = 0x00FF00FF00FF00FF, 0xFFFFFFFF00000000
    items = [(1, "imported:anti-uav:scene-1", "anti-uav", True),
             (2, "imported:cmy-uav:scene-9", "cmy-uav", True),     # the same video copied into another dataset
             (3, "imported:anti-uav:scene-2", "anti-uav", True),   # drifted 3 bits from 2, 6 from 1: chains in through 2
             (4, "scraped:page-a", "scraped", False),               # a look-alike sky on a web page: never joined with imports
             (5, "scraped:page-b", "scraped", False),               # but joined with a scraped look-alike
             (6, "imported:anti-uav:scene-7", "anti-uav", True),   # another scene
             (7, "imported:anti-uav:scene-8", "anti-uav", True)]   # no hash: untouched
    hashes = {1: a, 2: a ^ 0b111, 3: a ^ 0b111000, 4: a, 5: a ^ 0b1, 6: far}
    g = scene_groups(items, hashes)
    assert g[1] == g[2] == g[3]
    assert g[4] == g[5] != g[1]
    assert g[6] == "imported:anti-uav:scene-7" and g[7] == "imported:anti-uav:scene-8"
    assert scene_groups(items, {**hashes, 2: a ^ (0b11111 << 20)})[2] != g[1]  # 5 bits from 1 (and 8 from 3): another scene


def test_a_version_can_keep_look_alike_scenes_in_one_split(client, project):
    pid, ids = project
    mr = ids["multirotor"]
    clip = [add(client, pid, mr, [box(mr, 0.2 + 0.1 * k)])["id"] for k in range(6)]  # six frames, each its own upload group
    for _ in range(14):
        add(client, pid, mr, [box(mr)])
    _same_scene(client, clip)
    v = client.post(f"/projects/{pid}/datasets", json={"group_scenes": True, "seed": 3}).json()
    assert v["config"]["group_scenes"] == {"hash_bits": 4} and v["stats"]["scene_groups_joined"] == 5
    with client.session_factory() as s:
        manifest = load_manifest(s.get(DatasetVersion, v["id"]))
    split = {it["image_id"]: it["split"] for it in manifest["images"]}
    assert len({split[i] for i in clip}) == 1                       # the whole clip on one side
    assert len({it["group"] for it in manifest["images"] if it["image_id"] in clip}) == 1
    plain = client.post(f"/projects/{pid}/datasets", json={"seed": 3}).json()
    assert "group_scenes" not in plain["config"] and "scene_groups_joined" not in plain["stats"]
