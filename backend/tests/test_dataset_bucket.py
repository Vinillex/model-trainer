"""The dataset bucket (Cloudflare R2 or any S3-compatible store): pods download the training zip from there, not through the Mac."""

import threading
import zipfile
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
        self.objects[Key] = data
        self.uploads += 1
        if Callback:
            Callback(len(data))

    def generate_presigned_url(self, op, Params, ExpiresIn):
        assert op == "get_object" and Params["Bucket"] == BUCKET and ExpiresIn == dataset_bucket.LINK_SECONDS
        return f"{self.base}/{Params['Key']}?X-Amz-Expires={ExpiresIn}&X-Amz-Signature=sig"

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
