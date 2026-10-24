"""The dataset bucket (Cloudflare R2 or any S3-compatible store): pods download the training zip from there, not through the Mac."""

import threading
import zipfile
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, urlparse

import pytest
from botocore.exceptions import ClientError

from app import dataset_bucket
from app.config import settings

from .test_datasets import populate
from .test_training import claim, hdr, lan  # noqa: F401 - fixtures

BUCKET = "mt-datasets"


class FakeBucket:
    """The four boto3 calls dataset_bucket makes, against a dict. Links point at a real local HTTP server, so agents can use them."""

    def __init__(self):
        self.objects, self.uploads, self.requests = {}, 0, []
        self.modified, self.page_size, self.down = {}, 1000, False
        self.broken = False
        bucket = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                url = urlparse(self.path)
                bucket.requests.append({"path": url.path, "headers": dict(self.headers), "query": parse_qs(url.query)})
                data = bucket.objects.get(url.path.lstrip("/"))
                if bucket.broken or data is None or parse_qs(url.query).get("X-Amz-Signature") != ["sig"]:
                    self.send_response(403)
                    self.end_headers()
                    self.wfile.write(b"<Error><Code>AccessDenied</Code></Error>")
                    return
                if rng := self.headers.get("Range"):
                    start, end = (int(x) for x in rng.split("=")[1].split("-"))
                    data = data[start:end + 1]
                    self.send_response(206)
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def head_object(self, Bucket, Key):
        assert Bucket == BUCKET
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        return {"ContentLength": len(self.objects[Key])}

    def upload_file(self, Filename, Bucket, Key, Callback=None, ExtraArgs=None, Config=None):
        assert Bucket == BUCKET and ExtraArgs == {"ContentType": "application/zip"}
        data = open(Filename, "rb").read()
        self.put(Key, data)
        self.uploads += 1
        if Callback:
            Callback(len(data))

    def generate_presigned_url(self, op, Params, ExpiresIn):
        assert op == "get_object" and Params["Bucket"] == BUCKET and ExpiresIn == dataset_bucket.LINK_SECONDS
        return f"{self.base}/{Params['Key']}?X-Amz-Expires={ExpiresIn}&X-Amz-Signature=sig"

    def put(self, key, data=b"zip"):
        """An object as if uploaded; each one a minute after the one before."""
        self.objects[key] = data
        self.modified[key] = datetime(2026, 10, 1, tzinfo=timezone.utc) + timedelta(minutes=len(self.modified))

    def list_objects_v2(self, Bucket, Prefix="", ContinuationToken=None):
        assert Bucket == BUCKET
        if self.down:
            raise ConnectionError("R2 unreachable")
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + self.page_size]
        out = {"Contents": [{"Key": k, "Size": len(self.objects[k]), "LastModified": self.modified[k]} for k in page], "IsTruncated": start + self.page_size < len(keys)}
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + self.page_size)
        return out

    def delete_object(self, Bucket, Key):
        self.objects.pop(Key, None)

    def close(self):
        self.server.shutdown()


@pytest.fixture()
def bucket(monkeypatch):
    monkeypatch.setattr(settings, "dataset_bucket", BUCKET)
    monkeypatch.setattr(settings, "dataset_bucket_endpoint", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setattr(settings, "dataset_bucket_access_key", "AKID")
    monkeypatch.setattr(settings, "dataset_bucket_secret_key", "SECRET")
    fake = FakeBucket()
    dataset_bucket.set_client(fake)
    yield fake
    dataset_bucket.set_client(None)
    fake.close()


@pytest.fixture()
def version(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    populate(client, p["id"], {c["name"]: c["id"] for c in p["classes"]})
    return client.post(f"/projects/{p['id']}/datasets", json={}).json()["id"]


def get_version(client, vid):
    from app.models import DatasetVersion
    db = client.session_factory()
    return db, db.get(DatasetVersion, vid)


def test_off_unless_all_four_settings_are_there(monkeypatch):
    assert not dataset_bucket.enabled()
    monkeypatch.setattr(settings, "dataset_bucket", BUCKET)
    monkeypatch.setattr(settings, "dataset_bucket_endpoint", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setattr(settings, "dataset_bucket_access_key", "AKID")
    assert not dataset_bucket.enabled()                                   # no secret key yet
    monkeypatch.setattr(settings, "dataset_bucket_secret_key", "SECRET")
    assert dataset_bucket.enabled()


def test_the_zip_is_uploaded_once_per_version_and_is_the_real_dataset(client, bucket, version):
    db, v = get_version(client, version)
    said = []
    key, size = dataset_bucket.ensure_uploaded(db, v, said.append)
    assert bucket.uploads == 1 and size == len(bucket.objects[key]) and key.endswith(".zip")
    with zipfile.ZipFile(BytesIO(bucket.objects[key])) as z:
        names = z.namelist()
    assert "data.yaml" in names and any(n.startswith("images/train/") for n in names) and any(n.startswith("labels/val/") for n in names)
    assert any("uploading" in s for s in said)

    assert dataset_bucket.ensure_uploaded(db, v, said.append) == (key, size)
    assert bucket.uploads == 1 and "already in the bucket" in said[-1]   # the slow upload happens once


def test_a_claim_gets_a_signed_link_only_when_the_copy_is_there(client, bucket, version, lan):
    pid = client.get("/projects").json()[0]["id"]
    first = client.post(f"/projects/{pid}/training-runs", json={"dataset_id": version, "epochs": 1}).json()["id"]
    task = claim(lan).json()
    assert task["run_id"] == first and task["dataset_link"] is None              # not uploaded yet: the backend route only
    lan.post(f"/workers/runs/{first}/fail", json={"error": "x"}, headers=hdr())

    db, v = get_version(client, version)
    key, _ = dataset_bucket.ensure_uploaded(db, v)
    second = client.post(f"/projects/{pid}/training-runs", json={"dataset_id": version, "epochs": 1}).json()["id"]
    task = claim(lan).json()
    assert task["run_id"] == second and task["dataset_link"].startswith(f"{bucket.base}/{key}?")
    assert task["dataset_url"] == f"/workers/runs/{second}/dataset"              # the fallback is still there


def test_a_broken_bucket_never_breaks_a_claim(client, bucket, version):
    db, v = get_version(client, version)

    class Down:
        def head_object(self, **kw):
            raise ConnectionError("R2 unreachable")
    dataset_bucket.set_client(Down())
    assert dataset_bucket.link_if_stored(v) is None


def test_deleting_a_version_deletes_its_copy(client, bucket, version):
    db, v = get_version(client, version)
    key, _ = dataset_bucket.ensure_uploaded(db, v)
    assert client.delete(f"/datasets/{version}").status_code == 204
    assert key not in bucket.objects


def test_the_real_client_signs_an_r2_link_for_one_file_without_any_network(monkeypatch):
    """boto3 signs locally, so this checks the configuration (endpoint, SigV4, expiry) without contacting anything."""
    monkeypatch.setattr(settings, "dataset_bucket", BUCKET)
    monkeypatch.setattr(settings, "dataset_bucket_endpoint", "https://acct123.r2.cloudflarestorage.com")
    monkeypatch.setattr(settings, "dataset_bucket_access_key", "AKIDEXAMPLE")
    monkeypatch.setattr(settings, "dataset_bucket_secret_key", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY")
    dataset_bucket.set_client(None)
    try:
        url = urlparse(dataset_bucket.link("datasets/1-abc.zip"))
        q = parse_qs(url.query)
        assert url.scheme == "https" and url.hostname == "acct123.r2.cloudflarestorage.com"
        assert url.path == f"/{BUCKET}/datasets/1-abc.zip"
        assert q["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"] and q["X-Amz-Expires"] == [str(6 * 3600)] and "X-Amz-Signature" in q
        assert "wJalrXUtnFEMI" not in url.geturl()                             # the secret itself never travels
    finally:
        dataset_bucket.set_client(None)


# ---------- the cloud copies, listed and deleted from the Datasets section ----------

def two_versions(client, version):
    """The fixture's version plus a second one with another split, so its zip (named after the frozen manifest) is another file."""
    pid = client.get("/projects").json()[0]["id"]
    second = client.post(f"/projects/{pid}/datasets", json={"name": "v2", "seed": 2}).json()["id"]
    return pid, version, second


def upload(client, bucket, vid):
    db, v = get_version(client, vid)
    return dataset_bucket.ensure_uploaded(db, v)[0]


def test_the_section_says_nothing_without_a_bucket(client, version):
    pid = client.get("/projects").json()[0]["id"]
    assert client.get(f"/projects/{pid}/datasets/cloud").json() == {"enabled": False, "copies": [], "total_bytes": 0}
    assert client.post(f"/projects/{pid}/datasets/cloud/delete", json={"keys": [f"datasets/{pid}-{'a' * 20}.zip"]}).status_code == 409


def test_the_copies_are_listed_newest_first_with_the_version_each_belongs_to(client, bucket, version):
    pid, v1, v2 = two_versions(client, version)
    k1, k2 = upload(client, bucket, v1), upload(client, bucket, v2)
    assert k1 != k2
    orphan = f"datasets/{pid}-{'a' * 20}.zip"
    bucket.put(orphan, b"x" * 10)
    bucket.put(f"datasets/{pid + 1}-{'b' * 20}.zip")                  # another project's copy: not listed
    bucket.put(f"datasets/{pid}-notes.txt")                           # not a dataset zip: not listed
    bucket.page_size = 2                                              # the listing follows the bucket's pages
    got = client.get(f"/projects/{pid}/datasets/cloud").json()
    assert got["enabled"] is True and [c["key"] for c in got["copies"]] == [orphan, k2, k1]
    assert [(c["dataset_id"], c["dataset_name"], c["in_use"]) for c in got["copies"]] == [(None, None, False), (v2, "v2", False), (v1, client.get(f"/datasets/{v1}").json()["name"], False)]
    assert got["total_bytes"] == sum(len(bucket.objects[k]) for k in (orphan, k1, k2))


def test_a_run_that_is_waiting_or_training_marks_its_copy_in_use(client, bucket, version):
    from app.models import RunStage, TrainingRun
    pid, v1, v2 = two_versions(client, version)
    k1, k2 = upload(client, bucket, v1), upload(client, bucket, v2)
    run = client.post(f"/projects/{pid}/training-runs", json={"dataset_id": v1, "epochs": 1}).json()["id"]
    use = lambda: {c["key"]: c["in_use"] for c in client.get(f"/projects/{pid}/datasets/cloud").json()["copies"]}
    assert use() == {k1: True, k2: False}
    with client.session_factory() as db:
        db.get(TrainingRun, run).stage = RunStage.done
        db.commit()
    assert use() == {k1: False, k2: False}


def test_deleting_removes_only_the_chosen_copies_and_keeps_the_versions(client, bucket, version):
    pid, v1, v2 = two_versions(client, version)
    k1, k2 = upload(client, bucket, v1), upload(client, bucket, v2)
    orphan = f"datasets/{pid}-{'a' * 20}.zip"
    bucket.put(orphan, b"x" * 10)
    freed = len(bucket.objects[k1]) + 10
    r = client.post(f"/projects/{pid}/datasets/cloud/delete", json={"keys": [k1, orphan, f"datasets/{pid}-{'c' * 20}.zip"]})
    assert r.status_code == 200 and r.json() == {"deleted": 2, "freed_bytes": freed}      # the third was never there: not counted
    assert set(bucket.objects) == {k2}
    assert len(client.get(f"/projects/{pid}/datasets").json()) == 2                       # both versions are still there
    # and the next run on the deleted one finds no copy, so it goes through the backend as before
    db, v = get_version(client, v1)
    assert dataset_bucket.link_if_stored(v) is None


def test_deleting_refuses_other_objects_and_copies_a_run_needs_and_then_deletes_nothing(client, bucket, version):
    pid, v1, v2 = two_versions(client, version)
    k1, k2 = upload(client, bucket, v1), upload(client, bucket, v2)
    other = f"datasets/{pid + 1}-{'b' * 20}.zip"
    bucket.put(other)
    for bad in (other, "datasets/other.zip", "../datasets/x.zip", f"datasets/{pid}-{'a' * 20}.zip.bak", "images/1.jpg"):
        r = client.post(f"/projects/{pid}/datasets/cloud/delete", json={"keys": [k2, bad]})
        assert r.status_code == 422 and "not a dataset copy of this project" in r.json()["detail"], bad
    client.post(f"/projects/{pid}/training-runs", json={"dataset_id": v1, "epochs": 1})
    r = client.post(f"/projects/{pid}/datasets/cloud/delete", json={"keys": [k2, k1]})
    assert r.status_code == 409 and "may need the zip" in r.json()["detail"]
    assert client.post(f"/projects/{pid}/datasets/cloud/delete", json={"keys": []}).status_code == 422
    assert set(bucket.objects) == {k1, k2, other}


def test_a_bucket_that_does_not_answer_is_reported_not_hidden(client, bucket, version):
    pid = client.get("/projects").json()[0]["id"]
    k = upload(client, bucket, version)
    bucket.down = True
    assert client.get(f"/projects/{pid}/datasets/cloud").status_code == 502
    assert client.post(f"/projects/{pid}/datasets/cloud/delete", json={"keys": [k]}).status_code == 502
    bucket.down = False

    class Refusing(FakeBucket):
        def delete_object(self, Bucket, Key):
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "DeleteObject")
    refusing = Refusing()
    refusing.objects, refusing.modified = bucket.objects, bucket.modified
    dataset_bucket.set_client(refusing)
    try:
        r = client.post(f"/projects/{pid}/datasets/cloud/delete", json={"keys": [k]})
        assert r.status_code == 502 and "AccessDenied" in r.json()["detail"] and k in bucket.objects
    finally:
        dataset_bucket.set_client(bucket)
        refusing.close()
