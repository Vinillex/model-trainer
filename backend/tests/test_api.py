from .helpers import png


def test_project_seeds_classes(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    names = {c["name"]: c for c in p["classes"]}
    assert names["multirotor"]["index"] == 0 and names["delta-wing"]["index"] == 2
    assert names["birds"]["is_negative"] and names["birds"]["index"] is None
    assert client.post("/projects", json={"name": "drones"}).status_code == 409


def test_upload_dedup_reject_and_stats(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    cid = next(c["id"] for c in p["classes"] if c["name"] == "multirotor")
    files = [
        ("files", ("a.png", png(), "image/png")),
        ("files", ("dup.png", png(), "image/png")),
        ("files", ("b.png", png((1, 2, 3)), "image/png")),
        ("files", ("bad.txt", b"hello", "text/plain")),
        ("files", ("tiny.png", png(size=(8, 8)), "image/png")),
    ]
    r = client.post(f"/projects/{p['id']}/classes/{cid}/images", files=files).json()
    assert len(r["added"]) == 2
    assert r["duplicates"] == ["dup.png"]
    assert len(r["rejected"]) == 2

    imgs = client.get(f"/projects/{p['id']}/images", params={"class_id": cid, "source": "uploaded"}).json()
    assert len(imgs) == 2
    assert client.get(f"/images/{imgs[0]['id']}/file").headers["content-type"] == "image/png"

    stats = {s["class_name"]: s for s in client.get(f"/projects/{p['id']}/stats").json()}
    assert stats["multirotor"]["uploaded"] == 2 and stats["delta-wing"]["uploaded"] == 0

    assert client.delete(f"/images/{imgs[0]['id']}").status_code == 204
    assert client.get(f"/images/{imgs[0]['id']}/file").status_code == 404


def make_project(client, name="drones"):
    p = client.post("/projects", json={"name": name}).json()
    return p, {c["name"]: c["id"] for c in p["classes"]}


def upload(client, project_id, class_id, color=(10, 20, 30)):
    files = [("files", ("x.png", png(color), "image/png"))]
    return client.post(f"/projects/{project_id}/classes/{class_id}/images", files=files).json()["added"][0]


def test_bulk_classes_skip_dedupe_normalize(client):
    p, ids = make_project(client)
    r = client.post(f"/projects/{p['id']}/classes/bulk", json={
        "names": ["Quad Copter", "quad-copter", "multirotor", "BIRDS", "hexa, bad!", "", "  vtol  "],
        "is_negative": False,
    }).json()
    assert [c["name"] for c in r["added"]] == ["quad-copter", "vtol"]
    assert r["skipped"] == ["multirotor", "birds"]  # existing in either category; "birds" is NOT flipped to a target
    assert r["invalid"] == ["hexa, bad!"]
    assert [c["index"] for c in r["added"]] == [3, 4]  # appended after the 3 seeded target classes

    neg = client.post(f"/projects/{p['id']}/classes/bulk", json={"names": ["drones-in-fog", "vtol"], "is_negative": True}).json()
    assert [c["name"] for c in neg["added"]] == ["drones-in-fog"]
    assert neg["added"][0]["is_negative"] and neg["added"][0]["index"] is None
    assert neg["skipped"] == ["vtol"]
    classes = {c["name"]: c for c in client.get(f"/projects/{p['id']}").json()["classes"]}
    assert not classes["vtol"]["is_negative"]


def test_annotations_roundtrip_and_review_state(client):
    p, ids = make_project(client)
    img = upload(client, p["id"], ids["multirotor"])
    assert img["review_status"] == "pending"
    neg = upload(client, p["id"], ids["birds"], (1, 1, 1))
    assert neg["review_status"] == "reviewed"  # negatives need no boxes

    boxes = [{"class_id": ids["multirotor"], "cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.1},
             {"class_id": ids["delta-wing"], "cx": 0.1, "cy": 0.1, "w": 0.2, "h": 0.2}]  # another type than the folder's
    r = client.put(f"/images/{img['id']}/annotations", json={"boxes": boxes, "review_status": "reviewed"})
    assert r.status_code == 200
    body = r.json()
    assert body["review_status"] == "reviewed" and len(body["annotations"]) == 2
    assert {a["status"] for a in body["annotations"]} == {"reviewed"}
    assert client.get(f"/images/{img['id']}").json()["annotations"][0]["cx"] == 0.5

    # replace semantics, and review_status omitted leaves the status alone
    r = client.put(f"/images/{img['id']}/annotations", json={"boxes": boxes[:1]}).json()
    assert len(r["annotations"]) == 1 and r["review_status"] == "reviewed"

    pending = client.get(f"/projects/{p['id']}/images", params={"review_status": "pending"}).json()
    assert pending == []
    client.put(f"/images/{img['id']}/annotations", json={"boxes": [], "review_status": "rejected"})
    assert len(client.get(f"/projects/{p['id']}/images", params={"review_status": "rejected"}).json()) == 1
    stats = {s["class_name"]: s for s in client.get(f"/projects/{p['id']}/stats").json()}
    assert stats["multirotor"]["rejected"] == 1 and stats["multirotor"]["pending"] == 0 and stats["birds"]["pending"] == 0


def test_annotation_validation(client):
    p, ids = make_project(client)
    other, other_ids = make_project(client, "other")
    img = upload(client, p["id"], ids["multirotor"])
    url = f"/images/{img['id']}/annotations"
    ok = {"class_id": ids["multirotor"], "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}

    assert client.put(url, json={"boxes": [{**ok, "class_id": ids["birds"]}]}).status_code == 422  # negative class
    assert client.put(url, json={"boxes": [{**ok, "class_id": other_ids["multirotor"]}]}).status_code == 422  # other project
    assert client.put(url, json={"boxes": [{**ok, "cx": 0.95}]}).status_code == 422  # sticks out of the image
    assert client.put(url, json={"boxes": [{**ok, "w": 0}]}).status_code == 422  # zero size
    assert client.put(url, json={"boxes": [{**ok, "cy": 1.2}]}).status_code == 422  # out of range
    assert client.put("/images/999/annotations", json={"boxes": []}).status_code == 404
    assert client.get(f"/images/{img['id']}").json()["annotations"] == []  # nothing partially saved
    assert client.put(url, json={"boxes": [{**ok, "cx": 0.9, "w": 0.2}]}).status_code == 200  # exactly touches the edge


def test_delete_image_removes_annotations(client):
    from app.models import Annotation
    from sqlalchemy import select, func

    p, ids = make_project(client)
    img = upload(client, p["id"], ids["multirotor"])
    client.put(f"/images/{img['id']}/annotations", json={"boxes": [
        {"class_id": ids["multirotor"], "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]})
    assert client.delete(f"/images/{img['id']}").status_code == 204
    with client.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Annotation)) == 0
