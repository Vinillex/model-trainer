import io
import zipfile
from collections import defaultdict

import pytest

from app.datasets import assign_splits, split_quota
from app.models import Image, Source
from app.yolo_export import data_yaml, label_text

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
            + [(f"s{i}", "shahed", "uploaded") for i in range(4)] + [(f"g{i}", "multirotor", "generated") for i in range(30)])
    items = make_items(spec)
    assign_splits(items, 0.1, 0.1, seed=1)
    for cls in ("multirotor", "fixed-wing", "shahed"):
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
    add(client, pid, ids["shahed"], [], status="reviewed")              # reviewed, nothing in it: a background image


def unzip(resp):
    return zipfile.ZipFile(io.BytesIO(resp.content))


def test_preview_counts_what_would_be_included(client, project):
    pid, ids = project
    populate(client, pid, ids)
    p = client.get(f"/projects/{pid}/datasets/preview").json()
    assert (p["reviewed_images"], p["reviewed_with_boxes"], p["reviewed_empty"], p["negative_images"]) == (27, 22, 1, 4)
    assert (p["pending"], p["rejected"]) == (1, 1)
    assert p["boxes_per_class"] == {"multirotor": 14, "fixed-wing": 8, "shahed": 0}


def test_create_version_stats_and_warnings(client, project):
    pid, ids = project
    populate(client, pid, ids)
    r = client.post(f"/projects/{pid}/datasets", json={})
    assert r.status_code == 201
    v = r.json()
    assert v["name"] == "v1" and v["config"]["classes"] == [{"index": 0, "name": "multirotor"}, {"index": 1, "name": "fixed-wing"}, {"index": 2, "name": "shahed"}]
    st = v["stats"]
    assert sum(st["images"].values()) == 27 and sum(st["boxes"].values()) == 22
    assert st["excluded"] == {"pending": 1, "rejected": 1}
    assert sum(st["background_images"].values()) == 5  # 4 hard negatives + the empty shahed image
    assert st["images"]["train"] > st["images"]["val"] > 0
    warnings = " ".join(st["warnings"])
    assert "'shahed' has no boxes at all" in warnings          # can't learn a class that has no examples
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
    assert 'names:\n  0: "multirotor"\n  1: "fixed-wing"\n  2: "shahed"' in yaml
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
