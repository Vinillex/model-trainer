import io
import zipfile
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update

from app import training
from app.config import settings
from app.main import app
from app.models import TrainingRun
from app.security import is_loopback
from app.storage import get_storage

from .helpers import from_ip
from .test_datasets import add, box, populate

TOKEN = "worker-token-xyz"


def hdr(name="gpu-1", token=TOKEN):
    return {"X-Worker-Token": token, "X-Worker-Name": name}


@pytest.fixture()
def lan(client, monkeypatch):
    """A client on another machine of the LAN (e.g. the GPU PC), with the worker API enabled."""
    monkeypatch.setattr(settings, "worker_token", TOKEN)
    training._workers.clear()
    training._last_served = None
    return TestClient(from_ip(app, "192.168.1.50"))


@pytest.fixture()
def ready(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    populate(client, p["id"], {c["name"]: c["id"] for c in p["classes"]})
    return p["id"], client.post(f"/projects/{p['id']}/datasets", json={}).json()["id"]


def new_run(client, ready, **cfg):
    pid, did = ready
    r = client.post(f"/projects/{pid}/training-runs", json={"dataset_id": did, **cfg})
    assert r.status_code == 201, r.text
    return r.json()


def claim(lan, kind="gpu", name="gpu-1"):
    return lan.post("/workers/claim", json={"kind": kind}, headers=hdr(name))


def upload(lan, run_id, name, data=b"x", worker="gpu-1"):
    return lan.put(f"/workers/runs/{run_id}/artifacts/{name}", content=data, headers=hdr(worker))


def finish_gpu(lan, run_id, worker="gpu-1"):
    assert lan.post(f"/workers/runs/{run_id}/heartbeat", json={"stage": "exporting"}, headers=hdr(worker)).status_code == 200
    for name, data in (("best_pt", b"weights"), ("imx_zip", b"PK-packer"), ("labels", b"multirotor\nfixed-wing\n")):
        assert upload(lan, run_id, name, data, worker).status_code == 200
    r = lan.post(f"/workers/runs/{run_id}/complete", json={"metrics": {"map50": 0.5}}, headers=hdr(worker))
    assert r.status_code == 200, r.text


def expire_lease(client, run_id):
    with client.session_factory() as db:
        db.execute(update(TrainingRun).where(TrainingRun.id == run_id).values(lease_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
        db.commit()


def stage(client, run_id):
    return client.get(f"/training-runs/{run_id}").json()["stage"]


# ---------- who may call what ----------

@pytest.mark.parametrize("host, expected", [("127.0.0.1", True), ("::1", True), ("::ffff:127.0.0.1", True), ("localhost", False),
                                            ("192.168.1.5", False), ("10.0.0.2", False), ("testclient", False), (None, False)])
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


def test_lan_clients_reach_only_worker_routes_and_need_the_token(client, lan):
    assert client.get("/projects").status_code == 200                      # the local browser is unaffected
    assert lan.get("/projects").status_code == 403
    assert lan.get("/health").status_code == 403
    assert lan.delete("/datasets/1", headers=hdr()).status_code == 403     # a valid token doesn't open the rest of the API
    assert lan.get("/workers/ping").status_code == 401
    assert lan.get("/workers/ping", headers=hdr(token="wrong")).status_code == 401
    assert lan.get("/workers/ping", headers=hdr()).json() == {"ok": True}
    assert client.get("/workers/ping").status_code == 401                  # even locally the worker API needs the token


def test_worker_api_is_disabled_without_a_token(client, monkeypatch):
    monkeypatch.setattr(settings, "worker_token", None)
    lan = TestClient(from_ip(app, "192.168.1.50"))
    assert lan.get("/workers/ping", headers=hdr(token="anything")).status_code == 503


# ---------- creating runs ----------

def test_create_run_defaults_and_validation(client, ready):
    run = new_run(client, ready)
    assert run["stage"] == "queued" and run["name"] == f"run-{run['id']}"
    assert run["config"] == {"model": "yolo11n", "epochs": 100, "imgsz": 640, "batch": 16, "export_imx": True}
    pid, did = ready
    for bad in ({"model": "yolo11x"}, {"imgsz": 650}, {"imgsz": 1280}, {"epochs": 0}, {"batch": 0}):
        assert client.post(f"/projects/{pid}/training-runs", json={"dataset_id": did, **bad}).status_code == 422, bad
    assert client.post(f"/projects/{pid}/training-runs", json={"dataset_id": 9999}).status_code == 404
    other = client.post("/projects", json={"name": "other"}).json()["id"]
    assert client.post(f"/projects/{other}/training-runs", json={"dataset_id": did}).status_code == 404  # another project's version


def test_create_run_refuses_a_dataset_that_cannot_train(client):
    p = client.post("/projects", json={"name": "tiny"}).json()
    ids = {c["name"]: c["id"] for c in p["classes"]}
    add(client, p["id"], ids["multirotor"], [box(ids["multirotor"])])           # one image: everything lands in train, val is empty
    did = client.post(f"/projects/{p['id']}/datasets", json={}).json()["id"]
    r = client.post(f"/projects/{p['id']}/training-runs", json={"dataset_id": did})
    assert r.status_code == 422 and "no validation images" in r.json()["detail"]

    q = client.post("/projects", json={"name": "boxless"}).json()
    qids = {c["name"]: c["id"] for c in q["classes"]}
    for _ in range(4):
        add(client, q["id"], qids["birds"])                                      # only hard negatives: no boxes anywhere
    add(client, q["id"], qids["multirotor"], [box(qids["multirotor"])])
    add(client, q["id"], qids["multirotor"], [box(qids["multirotor"])])
    stats = client.post(f"/projects/{q['id']}/datasets", json={}).json()
    assert stats["stats"]["images"]["val"] > 0                                    # val exists, so the refusal below is about boxes
    with client.session_factory() as db:                                          # force "no training boxes" on that version
        from app.models import DatasetVersion
        v = db.get(DatasetVersion, stats["id"])
        v.stats = {**v.stats, "boxes": {**v.stats["boxes"], "train": 0}}
        db.commit()
    r = client.post(f"/projects/{q['id']}/training-runs", json={"dataset_id": stats["id"]})
    assert r.status_code == 422 and "no training boxes" in r.json()["detail"]


# ---------- the whole pipeline ----------

def test_full_pipeline_queued_to_done(client, lan, ready):
    run = new_run(client, ready, epochs=2)
    rid = run["id"]
    assert claim(lan, "pi", "pi-1").status_code == 204                            # nothing to package yet

    task = claim(lan).json()
    assert (task["run_id"], task["stage"], task["config"]["epochs"]) == (rid, "training", 2)
    assert claim(lan, name="gpu-2").status_code == 204                            # already taken
    assert stage(client, rid) == "training" and client.get(f"/training-runs/{rid}").json()["worker"] == "gpu-1"

    z = lan.get(task["dataset_url"], headers=hdr())
    assert "data.yaml" in zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert lan.get(task["dataset_url"], headers=hdr("gpu-2")).status_code == 409  # only the lease holder gets the dataset

    hb = lan.post(f"/workers/runs/{rid}/heartbeat", json={"progress": {"epoch": 1, "epochs": 2, "map50": 0.1}}, headers=hdr())
    assert hb.json() == {"cancel": False}
    assert client.get(f"/training-runs/{rid}").json()["progress"] == {"epoch": 1, "epochs": 2, "map50": 0.1}

    r = lan.post(f"/workers/runs/{rid}/complete", json={}, headers=hdr())        # nothing uploaded yet
    assert r.status_code == 422 and "best_pt" in r.json()["detail"]
    finish_gpu(lan, rid)
    out = client.get(f"/training-runs/{rid}").json()
    assert out["stage"] == "awaiting_package" and out["worker"] is None and out["metrics"] == {"map50": 0.5}
    assert set(out["artifacts"]) == {"best_pt", "imx_zip", "labels"} and "key" not in str(out["artifacts"])
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr()).status_code == 409  # the GPU is done with it

    assert claim(lan, name="gpu-1").status_code == 204                            # a GPU worker can't take packaging
    ptask = claim(lan, "pi", "pi-1").json()
    assert (ptask["run_id"], ptask["stage"]) == (rid, "packaging") and ptask["artifacts"] == ["best_pt", "imx_zip", "labels"]
    assert lan.get(f"/workers/runs/{rid}/artifacts/imx_zip", headers=hdr("pi-1")).content == b"PK-packer"
    assert upload(lan, rid, "best_pt", worker="pi-1").status_code == 422          # the Pi may only upload the .rpk
    assert lan.post(f"/workers/runs/{rid}/complete", json={}, headers=hdr("pi-1")).status_code == 422
    assert upload(lan, rid, "rpk", b"RPK!", worker="pi-1").status_code == 200
    assert lan.post(f"/workers/runs/{rid}/complete", json={}, headers=hdr("pi-1")).status_code == 200

    done = client.get(f"/training-runs/{rid}").json()
    assert done["stage"] == "done" and done["finished_at"] and done["error"] is None
    for name, content in (("best_pt", b"weights"), ("imx_zip", b"PK-packer"), ("labels", b"multirotor\nfixed-wing\n"), ("rpk", b"RPK!")):
        r = client.get(f"/training-runs/{rid}/artifacts/{name}")
        assert r.content == content and f"{done['name']}-" in r.headers["content-disposition"]
    assert client.get(f"/training-runs/{rid}/artifacts/nope").status_code == 404


def test_without_imx_export_the_run_ends_after_training(client, lan, ready):
    rid = new_run(client, ready, export_imx=False)["id"]
    claim(lan)
    assert upload(lan, rid, "best_pt", b"weights").status_code == 200
    assert lan.post(f"/workers/runs/{rid}/complete", json={"metrics": {"map50": 0.2}}, headers=hdr()).status_code == 200
    assert stage(client, rid) == "done"
    assert claim(lan, "pi", "pi-1").status_code == 204


def test_artifact_upload_rules(client, lan, ready, monkeypatch):
    rid = new_run(client, ready)["id"]
    assert upload(lan, rid, "best_pt").status_code == 409                         # not claimed yet
    claim(lan)
    assert upload(lan, rid, "mystery").status_code == 422
    assert upload(lan, rid, "rpk").status_code == 422                             # GPU workers don't make the .rpk
    assert upload(lan, rid, "best_pt", worker="gpu-2").status_code == 409         # not the lease holder
    monkeypatch.setattr(training, "MAX_ARTIFACT_BYTES", 8)
    assert upload(lan, rid, "best_pt", b"123456789").status_code == 413
    assert upload(lan, rid, "best_pt", b"12345678").status_code == 200
    assert lan.get(f"/workers/runs/{rid}/artifacts/best_pt", headers=hdr()).content == b"12345678"


def test_heartbeat_can_only_move_training_to_exporting(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan)
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={"stage": "bogus"}, headers=hdr()).status_code == 422
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={"stage": "exporting"}, headers=hdr()).status_code == 200
    assert stage(client, rid) == "exporting"
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={"stage": "exporting"}, headers=hdr()).status_code == 422


def test_fail_records_the_error_and_frees_the_run(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan)
    assert lan.post(f"/workers/runs/{rid}/fail", json={"error": "CUDA out of memory"}, headers=hdr("gpu-2")).status_code == 409
    assert lan.post(f"/workers/runs/{rid}/fail", json={"error": "CUDA out of memory"}, headers=hdr()).status_code == 200
    out = client.get(f"/training-runs/{rid}").json()
    assert (out["stage"], out["error"], out["worker"]) == ("failed", "CUDA out of memory", None)
    assert claim(lan).status_code == 204


# ---------- leases ----------

def test_silent_worker_loses_the_run_and_cannot_touch_it_again(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan, name="gpu-1")
    expire_lease(client, rid)
    task = claim(lan, name="gpu-2").json()                                        # reaped, requeued and claimed in one go
    assert task["run_id"] == rid and client.get(f"/training-runs/{rid}").json()["attempts"] == 1
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr("gpu-1")).status_code == 409
    assert upload(lan, rid, "best_pt", worker="gpu-1").status_code == 409
    assert lan.post(f"/workers/runs/{rid}/fail", json={"error": "x"}, headers=hdr("gpu-1")).status_code == 409
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr("gpu-2")).status_code == 200


def test_a_heartbeat_renews_the_lease(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan, name="gpu-1")
    expire_lease(client, rid)                                                     # lease ran out but nobody reaped it yet
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr("gpu-1")).status_code == 200
    assert claim(lan, name="gpu-2").status_code == 204                            # so the run is still gpu-1's
    assert stage(client, rid) == "training"


def test_a_run_whose_workers_keep_dying_fails_instead_of_looping(client, lan, ready):
    rid = new_run(client, ready)["id"]
    for i in range(training.MAX_ATTEMPTS):
        assert claim(lan, name=f"gpu-{i}").status_code == 200
        expire_lease(client, rid)
    assert claim(lan, name="gpu-x").status_code == 204
    out = client.get(f"/training-runs/{rid}").json()
    assert out["stage"] == "failed" and "stopped responding" in out["error"]


def test_a_dead_packager_puts_the_run_back_in_the_packaging_queue(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan)
    finish_gpu(lan, rid)
    claim(lan, "pi", "pi-1")
    expire_lease(client, rid)
    assert claim(lan, "pi", "pi-2").json()["run_id"] == rid
    assert stage(client, rid) == "packaging"


# ---------- cancel and delete ----------

def test_cancel_queued_run_is_immediate(client, lan, ready):
    rid = new_run(client, ready)["id"]
    assert client.post(f"/training-runs/{rid}/cancel").json()["stage"] == "cancelled"
    assert claim(lan).status_code == 204
    assert client.post(f"/training-runs/{rid}/cancel").status_code == 409        # already over


def test_cancel_running_run_stops_it_at_the_next_heartbeat(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan)
    out = client.post(f"/training-runs/{rid}/cancel").json()
    assert out["stage"] == "training" and out["cancel_requested"] is True        # the worker hasn't heard yet
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={"progress": {"epoch": 9}}, headers=hdr()).json() == {"cancel": True}
    out = client.get(f"/training-runs/{rid}").json()
    assert out["stage"] == "cancelled" and out["worker"] is None and out["finished_at"]
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr()).status_code == 409


def test_cancelled_run_whose_worker_died_is_not_requeued(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan)
    client.post(f"/training-runs/{rid}/cancel")
    expire_lease(client, rid)
    assert claim(lan, name="gpu-2").status_code == 204
    assert stage(client, rid) == "cancelled"


def test_delete_run_and_dataset_dependency(client, lan, ready):
    pid, did = ready
    rid = new_run(client, ready)["id"]
    assert client.delete(f"/datasets/{did}").status_code == 409                  # a run still needs the frozen version
    claim(lan)
    assert client.delete(f"/training-runs/{rid}").status_code == 409             # in progress
    finish_gpu(lan, rid)
    claim(lan, "pi", "pi-1")
    upload(lan, rid, "rpk", b"RPK!", "pi-1")
    lan.post(f"/workers/runs/{rid}/complete", json={}, headers=hdr("pi-1"))
    key = None
    with client.session_factory() as db:
        key = db.get(TrainingRun, rid).artifacts["rpk"]["key"]
    assert get_storage().get(key) == b"RPK!"
    assert client.delete(f"/training-runs/{rid}").status_code == 204
    with pytest.raises(FileNotFoundError):
        get_storage().get(key)                                                    # artifacts go with the run
    assert client.get(f"/training-runs/{rid}").status_code == 404
    assert client.delete(f"/datasets/{did}").status_code == 204


def test_workers_list_shows_who_called_in(client, lan, ready):
    new_run(client, ready)
    claim(lan, name="gpu-1")
    claim(lan, "pi", "pi-1")
    workers = {w["name"]: w for w in client.get("/training-workers").json()}
    assert workers["gpu-1"]["kind"] == "gpu" and workers["gpu-1"]["online"] and workers["gpu-1"]["run_id"] is not None
    assert workers["pi-1"]["kind"] == "pi" and workers["pi-1"]["run_id"] is None


def test_a_worker_keeps_its_kind_and_is_freed_when_its_run_is_cancelled(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan, name="gpu-1")
    lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr("gpu-1"))
    client.post(f"/training-runs/{rid}/cancel")
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr("gpu-1")).json() == {"cancel": True}
    w = {w["name"]: w for w in client.get("/training-workers").json()}["gpu-1"]
    assert w["kind"] == "gpu" and w["run_id"] is None          # not "pi" (the stage is no longer a GPU one), not busy with a dead run


def test_time_the_backend_was_not_serving_never_counts_against_a_worker(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan, name="gpu-1")
    expire_lease(client, rid)                                                      # the Mac slept for 20 minutes...
    training._last_served = datetime.now(timezone.utc) - timedelta(minutes=20)     # ...so no worker could heartbeat
    assert claim(lan, "pi", "pi-1").status_code == 204                             # the Pi is first to call in afterwards
    assert stage(client, rid) == "training"                                        # and does NOT take the run from gpu-1
    assert claim(lan, name="gpu-2").status_code == 204
    assert lan.post(f"/workers/runs/{rid}/heartbeat", json={}, headers=hdr("gpu-1")).status_code == 200


def test_a_lease_that_expired_while_the_backend_was_serving_is_still_reaped(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan, name="gpu-1")
    expire_lease(client, rid)                                                      # gpu-1 went silent, but the server kept answering
    training._last_served = datetime.now(timezone.utc) - timedelta(seconds=5)
    assert claim(lan, name="gpu-2").json()["run_id"] == rid                        # so the run is handed over as before


def test_a_restart_counts_as_a_gap(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan, name="gpu-1")
    expire_lease(client, rid)
    training._last_served = None                                                   # what a freshly started backend has
    assert claim(lan, name="gpu-2").status_code == 204
    assert stage(client, rid) == "training"


def test_unauthenticated_requests_do_not_count_as_the_server_serving_workers(client, lan, ready):
    rid = new_run(client, ready)["id"]
    claim(lan, name="gpu-1")
    expire_lease(client, rid)
    training._last_served = datetime.now(timezone.utc) - timedelta(minutes=20)
    assert lan.get("/workers/ping").status_code == 401                             # no token
    assert training._last_served < datetime.now(timezone.utc) - timedelta(minutes=19)
