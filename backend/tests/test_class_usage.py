"""Finding what still uses a class (so it can be removed): the box_class_id filter, and the message that says where."""

import pytest

from .test_datasets import add, box


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def ids_of(client, pid, **params):
    return sorted(i["id"] for i in client.get(f"/projects/{pid}/images", params={"limit": 500, **params}).json())


def test_box_class_filter_finds_images_by_the_classes_of_their_boxes_wherever_they_are_filed(client, project):
    pid, ids = project
    m, f, s = ids["multirotor"], ids["fixed-wing"], ids["shahed"]
    plain = add(client, pid, m, [box(m)])["id"]
    mixed = add(client, pid, m, [box(f), box(s, cx=0.2, cy=0.2)])["id"]            # two classes: the review moves it to the `mixed` folder
    only_s = add(client, pid, m, [box(s)])["id"]                                    # one class: moves to that class's folder
    unlabelled = add(client, pid, m, status="pending")["id"]
    filed = {i["id"]: i["class_id"] for i in client.get(f"/projects/{pid}/images").json()}
    assert filed[mixed] != m and filed[only_s] == s                                  # the images really are filed somewhere else
    assert ids_of(client, pid, box_class_id=s) == sorted([mixed, only_s])           # found even inside `mixed`
    assert ids_of(client, pid, box_class_id=f) == [mixed]
    assert ids_of(client, pid, box_class_id=m) == [plain]
    assert ids_of(client, pid, class_id=m, box_class_id=s) == []                    # the two filters are different things: filed under multirotor AND a shahed box
    assert unlabelled not in ids_of(client, pid, box_class_id=m)
    assert client.get(f"/projects/{pid}/images", params={"box_class_id": "x"}).status_code == 422


def test_the_in_use_message_says_where_the_boxes_are_and_a_relabelled_class_can_then_be_removed(client, project):
    pid, ids = project
    m, f, v = ids["multirotor"], ids["fixed-wing"], None
    v = client.post(f"/projects/{pid}/classes/bulk", json={"names": ["vtol"], "is_negative": False}).json()["added"][0]["id"]
    img = add(client, pid, m, [box(f), box(v, cx=0.2, cy=0.2), box(v, cx=0.8, cy=0.8)])["id"]   # a mixed image with two vtol boxes
    r = client.delete(f"/projects/{pid}/classes/{v}")
    assert r.status_code == 409 and "0 images filed under it, 2 boxes on 1 image)" in r.json()["detail"]
    assert ids_of(client, pid, box_class_id=v) == [img]                              # and the filter finds that image
    fix = {"boxes": [{"class_id": f, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.1}, {"class_id": f, "cx": 0.2, "cy": 0.2, "w": 0.2, "h": 0.1}, {"class_id": f, "cx": 0.8, "cy": 0.8, "w": 0.2, "h": 0.1}], "review_status": "reviewed"}
    assert client.put(f"/images/{img}/annotations", json=fix).status_code == 200  # relabelled in the review screen
    assert ids_of(client, pid, box_class_id=v) == []
    assert client.delete(f"/projects/{pid}/classes/{v}").status_code == 204
