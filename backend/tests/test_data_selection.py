import pytest

from app.models import Annotation, AnnotationStatus, Image, ImportRecord, ReviewStatus, Source

_n = iter(range(1, 100_000))


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def insert(client, pid, class_id, source, dataset=None, box_class=None, group=None, status=ReviewStatus.reviewed):
    """A reviewed image straight into the database (selection only reads rows, never files)."""
    n = next(_n)
    with client.session_factory() as db:
        img = Image(project_id=pid, class_id=class_id, source=source, storage_key=f"projects/{pid}/raw/x/{source.value}/{n}.jpg",
                    sha256=f"{n:064x}", width=64, height=64, review_status=status, group_key=group or f"g{n}")
        if box_class:
            img.annotations.append(Annotation(class_id=box_class, cx=.5, cy=.5, w=.2, h=.2, status=AnnotationStatus.reviewed))
        if dataset:
            img.import_record = ImportRecord(dataset=dataset, original_name=f"{n}.jpg", original_split="train", dataset_url="u",
                                             license="CC BY 4.0", author="a")
        db.add(img)
        db.commit()


@pytest.fixture()
def mixed_data(client, project):
    pid, ids = project
    m = ids["multirotor"]
    for _ in range(12):
        insert(client, pid, m, Source.uploaded, box_class=m)
    for _ in range(14):
        insert(client, pid, m, Source.scraped, box_class=m)
    for _ in range(16):
        insert(client, pid, m, Source.imported, "anti-uav", box_class=m)
    for _ in range(10):
        insert(client, pid, m, Source.imported, "other-set", box_class=m)
    for _ in range(4):
        insert(client, pid, ids["birds"], Source.scraped)                 # hard negatives, scraped
    return pid, ids


def make(client, pid, **body):
    return client.post(f"/projects/{pid}/datasets", json=body)


def test_by_default_a_version_uses_everything(client, mixed_data):
    pid, _ = mixed_data
    v = make(client, pid).json()
    assert v["stats"]["by_source"] == {"imported:anti-uav": 16, "imported:other-set": 10, "scraped": 18, "uploaded": 12}
    assert v["stats"]["excluded"]["not_selected"] == 0 and v["config"]["sources"] is None and v["config"]["imported_datasets"] is None


def test_sources_and_each_imported_dataset_can_be_chosen_separately(client, mixed_data):
    pid, _ = mixed_data
    v = make(client, pid, sources=["uploaded"], imported_datasets=["anti-uav"]).json()
    assert v["stats"]["by_source"] == {"imported:anti-uav": 16, "uploaded": 12}                   # no scraped, no other-set
    assert v["stats"]["excluded"]["not_selected"] == 18 + 10
    assert v["config"]["sources"] == ["uploaded"] and v["config"]["imported_datasets"] == ["anti-uav"]

    only_scraped = make(client, pid, sources=["scraped"], imported_datasets=[]).json()           # [] = no imported dataset at all
    assert only_scraped["stats"]["by_source"] == {"scraped": 18}

    only_other = make(client, pid, sources=[], imported_datasets=["other-set"]).json()
    assert only_other["stats"]["by_source"] == {"imported:other-set": 10}


def test_hard_negatives_follow_their_source(client, mixed_data):
    pid, _ = mixed_data
    without = make(client, pid, sources=["uploaded"], imported_datasets=[]).json()
    assert sum(without["stats"]["background_images"].values()) == 0                              # the scraped negatives are out
    with_scraped = make(client, pid, sources=["uploaded", "scraped"], imported_datasets=[]).json()
    assert sum(with_scraped["stats"]["background_images"].values()) == 4


def test_choosing_nothing_that_has_reviewed_images_is_a_clear_error(client, mixed_data):
    pid, _ = mixed_data
    r = make(client, pid, sources=["generated"], imported_datasets=[])
    assert r.status_code == 422 and "None of the data you selected" in r.json()["detail"]
    r = make(client, pid, sources=[], imported_datasets=["no-such-dataset"])
    assert r.status_code == 422
    assert client.get(f"/projects/{pid}/datasets").json() == []                                  # nothing was created


def test_bad_selection_values_are_rejected(client, mixed_data):
    pid, _ = mixed_data
    assert make(client, pid, sources=["imported"]).status_code == 422                            # imported is chosen by dataset name
    assert make(client, pid, sources=["nonsense"]).status_code == 422
    assert make(client, pid, imported_datasets=["Bad Name!"]).status_code == 422


def test_the_manifest_records_each_images_dataset(client, mixed_data):
    from app import datasets
    from app.models import DatasetVersion
    pid, _ = mixed_data
    v = make(client, pid).json()
    with client.session_factory() as db:
        items = datasets.load_manifest(db.get(DatasetVersion, v["id"]))["images"]
    assert {it["dataset"] for it in items if it["source"] == "imported"} == {"anti-uav", "other-set"}
    assert all(it["dataset"] is None for it in items if it["source"] != "imported")


def test_preview_reflects_the_selection_and_lists_what_can_be_chosen(client, mixed_data):
    pid, _ = mixed_data
    everything = client.get(f"/projects/{pid}/datasets/preview").json()
    assert everything["reviewed_images"] == everything["reviewed_total"] == 12 + 18 + 16 + 10 and everything["not_selected"] == 0
    assert everything["available"] == {"sources": {"uploaded": 12, "scraped": 18, "generated": 0},
                                       "imported_datasets": [{"name": "anti-uav", "reviewed": 16}, {"name": "other-set", "reviewed": 10}]}

    chosen = client.get(f"/projects/{pid}/datasets/preview", params={"sources": "uploaded", "imported": "anti-uav"}).json()
    assert chosen["reviewed_images"] == 12 + 16 and chosen["not_selected"] == 18 + 10
    assert chosen["boxes_per_class"]["multirotor"] == 12 + 16
    assert chosen["available"] == everything["available"]                                        # the options don't shrink with the choice

    nothing = client.get(f"/projects/{pid}/datasets/preview", params={"sources": "", "imported": ""}).json()
    assert nothing["reviewed_images"] == 0 and nothing["not_selected"] == 12 + 18 + 16 + 10
    assert client.get(f"/projects/{pid}/datasets/preview", params={"sources": "bogus"}).status_code == 422


def test_the_picker_offers_only_reviewed_imported_data(client, project):
    pid, ids = project
    insert(client, pid, ids["multirotor"], Source.imported, "pending-only", box_class=ids["multirotor"], status=ReviewStatus.pending)
    insert(client, pid, ids["multirotor"], Source.imported, "real-one", box_class=ids["multirotor"])
    avail = client.get(f"/projects/{pid}/datasets/preview").json()["available"]["imported_datasets"]
    assert avail == [{"name": "real-one", "reviewed": 1}]


def test_a_frozen_version_does_not_change_when_data_is_added_later(client, mixed_data):
    pid, ids = mixed_data
    v1 = make(client, pid, sources=["uploaded"], imported_datasets=[]).json()
    for _ in range(5):
        insert(client, pid, ids["multirotor"], Source.uploaded, box_class=ids["multirotor"])
    assert client.get(f"/datasets/{v1['id']}").json()["stats"]["by_source"] == {"uploaded": 12}


def test_the_image_list_can_show_one_imported_dataset_on_its_own(client, mixed_data):
    pid, _ = mixed_data
    listing = lambda **p: client.get(f"/projects/{pid}/images", params={"limit": 500, **p})
    mine = listing(dataset="anti-uav").json()
    assert len(mine) == 16 and {i["dataset"] for i in mine} == {"anti-uav"} and {i["source"] for i in mine} == {"imported"}
    assert len(listing(dataset="other-set").json()) == 10
    assert listing(dataset="no-such-dataset").json() == []
    everything = listing().json()
    assert {i["dataset"] for i in everything if i["source"] == "imported"} == {"anti-uav", "other-set"}
    assert all(i["dataset"] is None for i in everything if i["source"] != "imported")
    assert listing(dataset="Bad Name!").status_code == 422


def test_the_dataset_filter_combines_with_the_other_filters(client, project):
    pid, ids = project
    m = ids["multirotor"]
    for status in (ReviewStatus.reviewed, ReviewStatus.pending, ReviewStatus.pending):
        insert(client, pid, m, Source.imported, "set-a", status=status)
    insert(client, pid, ids["birds"], Source.imported, "set-a")
    listing = lambda **p: client.get(f"/projects/{pid}/images", params=p).json()
    assert len(listing(dataset="set-a", review_status="pending")) == 2
    assert len(listing(dataset="set-a", class_id=ids["birds"])) == 1
    assert listing(dataset="set-a", source="uploaded") == []


def test_imported_datasets_are_counted_by_review_state(client, project):
    pid, ids = project
    m = ids["multirotor"]
    for status, n in ((ReviewStatus.reviewed, 3), (ReviewStatus.pending, 2), (ReviewStatus.rejected, 1)):
        for _ in range(n):
            insert(client, pid, m, Source.imported, "set-a", status=status)
    insert(client, pid, m, Source.imported, "set-b")
    insert(client, pid, m, Source.scraped)                                  # not imported: never listed
    assert client.get(f"/projects/{pid}/imported-datasets").json() == [
        {"name": "set-a", "total": 6, "reviewed": 3, "pending": 2, "rejected": 1, "checked": 0},
        {"name": "set-b", "total": 1, "reviewed": 1, "pending": 0, "rejected": 0, "checked": 0},
    ]
    other = client.post("/projects", json={"name": "other"}).json()["id"]
    assert client.get(f"/projects/{other}/imported-datasets").json() == []   # per project
    assert client.get("/projects/9999/imported-datasets").status_code == 404


def test_opening_an_imported_image_marks_it_as_manually_checked(client, project):
    pid, ids = project
    m = ids["multirotor"]
    insert(client, pid, m, Source.imported, "set-a")
    insert(client, pid, m, Source.imported, "set-a")
    insert(client, pid, m, Source.imported, "set-b")
    insert(client, pid, m, Source.scraped)
    rows = client.get(f"/projects/{pid}/images", params={"limit": 50}).json()
    imported = [r for r in rows if r["source"] == "imported"]
    scraped = next(r for r in rows if r["source"] == "scraped")
    assert {r["manually_checked"] for r in imported} == {False} and scraped["manually_checked"] is None  # imported images only

    first = next(r["id"] for r in imported if r["dataset"] == "set-a")
    done = client.post(f"/images/{first}/manual-check")
    assert done.status_code == 200 and done.json()["manually_checked"] is True
    assert client.post(f"/images/{first}/manual-check").json()["manually_checked"] is True          # repeating is harmless
    assert client.get(f"/images/{first}").json()["manually_checked"] is True

    listing = lambda **p: sorted(r["id"] for r in client.get(f"/projects/{pid}/images", params=p).json())
    unchecked = sorted(r["id"] for r in imported if r["id"] != first)
    assert listing(manually_checked="true") == [first]
    assert listing(manually_checked="false") == unchecked                                           # never the scraped image
    assert listing(dataset="set-a", manually_checked="true") == [first]
    assert len(listing(dataset="set-a", manually_checked="false")) == 1
    counts = {d["name"]: d["checked"] for d in client.get(f"/projects/{pid}/imported-datasets").json()}
    assert counts == {"set-a": 1, "set-b": 0}

    assert client.post(f"/images/{scraped['id']}/manual-check").status_code == 409
    assert client.post("/images/999999/manual-check").status_code == 404


def test_a_manual_check_goes_with_its_image(client, project):
    from sqlalchemy import func, select

    from app.models import ManualCheck

    pid, ids = project
    insert(client, pid, ids["multirotor"], Source.imported, "set-a")
    image_id = client.get(f"/projects/{pid}/images").json()[0]["id"]
    assert client.post(f"/images/{image_id}/manual-check").status_code == 200
    with client.session_factory() as db:
        assert db.scalar(select(func.count()).select_from(ManualCheck)) == 1
        db.delete(db.get(Image, image_id))                                                          # what a rollback or delete does
        db.commit()
        assert db.scalar(select(func.count()).select_from(ManualCheck)) == 0
