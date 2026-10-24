"""A copy of each dataset zip in a private S3-compatible bucket (Cloudflare R2), for RunPod pods to download from directly.

The Mac's upload link is slow (about 0.15-0.35 MB/s measured on 2026-10-22), so a pod pulling the zip through the tunnel spends
20-40 paid minutes on it, and the home connection can cut it off. Instead the zip is uploaded once per dataset version, with no
pod running, and each pod gets a link that is signed for a few hours and opens only that one file. The bucket stays private.

Separate from MT_STORAGE_BACKEND: images and artifacts stay where they are; only the training zips are copied here. Unset
(MT_DATASET_BUCKET empty) = everything goes through the backend as before.
"""

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

from sqlalchemy.orm import Session

from . import datasets
from .config import settings
from .models import DatasetVersion
from .storage import get_storage
from .yolo_export import make_zip_file

LINK_SECONDS = 6 * 3600  # long enough for a pod's setup plus every retry, short enough that a leaked link soon stops working
_client = None


def enabled() -> bool:
    return bool(settings.dataset_bucket and settings.dataset_bucket_endpoint
                and settings.dataset_bucket_access_key and settings.dataset_bucket_secret_key)


def client():
    """The boto3 client for the bucket. Tests replace it with set_client()."""
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        _client = boto3.client(
            "s3",
            endpoint_url=settings.dataset_bucket_endpoint,
            aws_access_key_id=settings.dataset_bucket_access_key,
            aws_secret_access_key=settings.dataset_bucket_secret_key,
            region_name="auto",  # what R2 expects
            config=Config(signature_version="s3v4", retries={"max_attempts": 10, "mode": "adaptive"},
                          connect_timeout=30, read_timeout=120),
        )
    return _client


def set_client(c) -> None:
    global _client
    _client = c


def key_for(version: DatasetVersion) -> str:
    """Named after the frozen manifest's hash, not the version id: ids are reused after a database reset, the bucket is not."""
    digest = hashlib.sha256(get_storage().get(version.manifest_key)).hexdigest()[:20]
    return f"datasets/{version.project_id}-{digest}.zip"


def stored_size(key: str) -> int | None:
    """Size of the copy in the bucket, or None if there is none."""
    from botocore.exceptions import ClientError

    try:
        return int(client().head_object(Bucket=settings.dataset_bucket, Key=key)["ContentLength"])
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def ensure_uploaded(db: Session, version: DatasetVersion, say: Callable[[str], None] = lambda s: None) -> tuple[str, int]:
    """Upload the version's zip unless the bucket already has it. Returns (key, size in bytes)."""
    from boto3.s3.transfer import TransferConfig

    key = key_for(version)
    size = stored_size(key)
    if size is not None:
        say(f"  dataset {version.name} is already in the bucket ({size / 2**20:.0f} MB)")
        return key, size
    say(f"  building the zip of dataset {version.name}...")
    path = make_zip_file(datasets.resolve_manifest(db, datasets.load_manifest(version)), get_storage().get)
    try:
        total = path.stat().st_size
        say(f"  uploading it to the bucket ({total / 2**20:.0f} MB; at this Mac's upload speed that can take a while, once per version)")
        done, shown = [0], [0]

        def progress(n):
            done[0] += n
            if done[0] - shown[0] >= 25 * 2**20 or done[0] == total:  # a line every 25 MB
                shown[0] = done[0]
                say(f"    {done[0] / 2**20:.0f} of {total / 2**20:.0f} MB")
        # 8 MB parts, retried one by one: a dropped connection costs one part, not the whole upload.
        client().upload_file(str(path), settings.dataset_bucket, key, Callback=progress,
                             ExtraArgs={"ContentType": "application/zip"},
                             Config=TransferConfig(multipart_threshold=8 * 2**20, multipart_chunksize=8 * 2**20, max_concurrency=2))
    finally:
        Path(path).unlink(missing_ok=True)
    size = stored_size(key)
    if size != total:
        raise RuntimeError(f"the bucket holds {size} bytes of {key}, but {total} were uploaded")
    say("  uploaded.")
    return key, size


def link(key: str) -> str:
    """A signed https link to one object, valid for LINK_SECONDS. Whoever has it can download that file, nothing else."""
    return client().generate_presigned_url("get_object", Params={"Bucket": settings.dataset_bucket, "Key": key}, ExpiresIn=LINK_SECONDS)


def link_if_stored(version: DatasetVersion) -> str | None:
    """For a worker's claim: a link to the version's copy if the bucket has it. Never fails the claim: any problem with the
    bucket means None, and the worker downloads through the backend as before."""
    if not enabled():
        return None
    try:
        key = key_for(version)
        return link(key) if stored_size(key) is not None else None
    except Exception:  # noqa: BLE001 - the bucket is an optimisation; the backend route always works
        return None


def project_prefix(project_id: int) -> str:
    return f"datasets/{project_id}-"


def is_copy_key(project_id: int, key: str) -> bool:
    """Only keys that key_for() could have made for this project: a request can never name some other object of the bucket."""
    return re.fullmatch(rf"datasets/{project_id}-[0-9a-f]{{20}}\.zip", key) is not None


def list_copies(project_id: int) -> list[dict]:
    """Every dataset zip the bucket holds for this project, newest first: {key, size, modified}."""
    out, token = [], None
    while True:
        args = {"Bucket": settings.dataset_bucket, "Prefix": project_prefix(project_id)}
        if token:
            args["ContinuationToken"] = token
        page = client().list_objects_v2(**args)
        out += [{"key": o["Key"], "size": int(o["Size"]), "modified": o["LastModified"]} for o in page.get("Contents", []) if is_copy_key(project_id, o["Key"])]
        if not page.get("IsTruncated"):
            return sorted(out, key=lambda c: c["modified"], reverse=True)
        token = page["NextContinuationToken"]


def delete_copies(keys: list[str]) -> None:
    """Delete these objects. Not best effort like forget(): a failure is raised, so the person is told it is still there."""
    for key in keys:
        client().delete_object(Bucket=settings.dataset_bucket, Key=key)


def forget(version: DatasetVersion) -> None:
    """Delete the version's copy (best effort), when the version itself is deleted."""
    if not enabled():
        return
    try:
        client().delete_object(Bucket=settings.dataset_bucket, Key=key_for(version))
    except Exception:  # noqa: BLE001
        pass
