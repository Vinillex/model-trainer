"""The GPU and Pi agents against a real backend over HTTP (uvicorn in a thread), with fake training and packaging."""

import io
import socket
import stat
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import uvicorn

from app.config import settings
from app.main import app

AGENTS = Path(__file__).resolve().parents[2] / "agents"
sys.path.insert(0, str(AGENTS))

import common  # noqa: E402
import gpu_agent  # noqa: E402
import pi_agent  # noqa: E402

REAL_PREFLIGHT = gpu_agent.preflight  # the autouse fixture below stubs it out for the other tests

from .test_datasets import populate  # noqa: E402
from .test_dataset_bucket import bucket, get_version  # noqa: E402, F401 - fixture

TOKEN = "agent-test-token"


@pytest.fixture()
def server(client, monkeypatch):
    """The real app on a real socket, using the same test database and storage as `client`."""
    monkeypatch.setattr(settings, "worker_token", TOKEN)
    monkeypatch.setenv("MT_WORKER_TOKEN", TOKEN)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))  # no startup hooks
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.02)
    assert srv.started
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=10)


@pytest.fixture()
def run_id(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    populate(client, p["id"], {c["name"]: c["id"] for c in p["classes"]})
    did = client.post(f"/projects/{p['id']}/datasets", json={}).json()["id"]
    return client.post(f"/projects/{p['id']}/training-runs", json={"dataset_id": did, "epochs": 3}).json()["id"]


class FakeBackend:
    """Stands in for Ultralytics: checks what the agent hands over and writes dummy outputs."""

    def __init__(self, epoch_seconds=0.0, fail_with=None, export_fails=None):
        self.epoch_seconds, self.fail_with, self.export_fails = epoch_seconds, fail_with, export_fails
        self.seen, self.epochs_run, self.exports = {}, 0, 0

    def train(self, cfg, data_yaml, run_dir, on_progress, should_stop):
        text = data_yaml.read_text()
        self.seen["yaml"] = text
        self.seen["dirs"] = sorted(p.name for p in (data_yaml.parent / "images").iterdir())
        if self.fail_with:
            raise self.fail_with
        for epoch in range(1, cfg["epochs"] + 1):
            on_progress(epoch=epoch, epochs=cfg["epochs"], map50=epoch / 10)
            self.epochs_run += 1
            time.sleep(self.epoch_seconds)
            if should_stop():
                break
        best = run_dir / "best.pt"
        best.write_bytes(b"weights")
        return gpu_agent.TrainResult(best, {"map50": 0.3, "per_class": {"multirotor": {"ap50": 0.3}}})

    def export_imx(self, best_pt, data_yaml, cfg, run_dir):
        self.exports += 1
        if self.export_fails:
            raise self.export_fails
        (run_dir / "packerOut.zip").write_bytes(b"PK-packer")
        (run_dir / "labels.txt").write_text("multirotor\nfixed-wing\nshahed\n")
        return run_dir / "packerOut.zip", run_dir / "labels.txt"


def gpu(server, backend, *extra):
    return gpu_agent.main(["--server", server, "--name", "gpu-test", "--workdir", "work-gpu", "--once", "--heartbeat", "0.1", *extra], backend)


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpu_agent, "preflight", lambda export_imx, **kw: None)  # this machine is a Mac without Java


@pytest.fixture()
def fake_packager(tmp_path):
    script = tmp_path / "fake-imx500-package"
    script.write_text('#!/bin/sh\n# usage: -i <packerOut.zip> -o <dir>\nunzip -l "$2" >/dev/null 2>&1 || true\nprintf "RPK-from-%s" "$(wc -c < "$2" | tr -d " ")" > "$4/network.rpk"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


# ---------- the pipeline ----------

def test_gpu_then_pi_agent_take_a_run_from_queued_to_done(client, server, run_id, fake_packager):
    backend = FakeBackend()
    assert gpu(server, backend) == 0
    run = client.get(f"/training-runs/{run_id}").json()
    assert run["stage"] == "awaiting_package" and set(run["artifacts"]) == {"best_pt", "imx_zip", "labels"}
    assert run["metrics"]["map50"] == 0.3 and run["worker"] is None
    assert client.get(f"/training-runs/{run_id}/artifacts/best_pt").content == b"weights"
    assert client.get(f"/training-runs/{run_id}/artifacts/labels").text.startswith("multirotor")

    # The dataset it trained on: data.yaml points at the extracted folder (not "."), and the splits are there.
    path_line = next(l for l in backend.seen["yaml"].splitlines() if l.startswith("path:"))
    assert path_line != "path: ." and Path(path_line.split(":", 1)[1].strip().strip('"')).is_absolute()
    assert backend.seen["dirs"] == ["train", "val"] or backend.seen["dirs"] == ["test", "train", "val"]

    assert pi_agent.main(["--server", server, "--name", "pi-test", "--workdir", "work-pi", "--once", "--packager", fake_packager]) == 0
    done = client.get(f"/training-runs/{run_id}").json()
    assert done["stage"] == "done" and done["error"] is None
    assert client.get(f"/training-runs/{run_id}/artifacts/rpk").content == b"RPK-from-9"   # packerOut.zip was 9 bytes
    assert not (Path("work-gpu") / f"run-{run_id}" / "dataset").exists()                    # the image copy is cleaned up


def test_heartbeats_carry_training_progress(client, server, run_id):
    gpu(server, FakeBackend(epoch_seconds=0.15))
    # The run is finished by now, but progress from the last beats is kept on it.
    assert client.get(f"/training-runs/{run_id}").json()["progress"].get("epoch") in (1, 2, 3)


def test_a_slow_upload_shows_up_in_the_progress_message(client, server, run_id, monkeypatch):
    """The second RunPod pod sat in a failing upload for most of an hour while the run still said "training"."""
    seen, real_upload = [], common.Api.upload

    def slow_upload(self, rid, name, file):
        time.sleep(0.4)                                                                      # beats go out every 0.1 s meanwhile
        seen.append((name, client.get(f"/training-runs/{rid}").json()["progress"].get("message")))
        real_upload(self, rid, name, file)

    monkeypatch.setattr(common.Api, "upload", slow_upload)
    assert gpu(server, FakeBackend()) == 0
    assert seen == [("best_pt", "uploading the trained weights"), ("imx_zip", "uploading the IMX500 package"),
                    ("labels", "uploading the IMX500 package")]


def cannot_upload_best_pt(monkeypatch):
    """Uploads of best.pt give up (the network path to the backend is dead for big bodies); everything else works."""
    real_upload = common.Api.upload

    def upload(self, rid, name, file):
        if name == "best_pt":
            raise common.ApiError(0, "gave up on PUT /workers/runs/1/artifacts/best_pt after 900s: can't reach the backend (TimeoutError)")
        real_upload(self, rid, name, file)
    monkeypatch.setattr(common.Api, "upload", upload)


def test_a_best_pt_upload_that_cannot_get_through_still_runs_the_export_and_says_so(client, server, run_id, monkeypatch):
    """Pod 4 spent its hour retrying this upload and never reached the export, the one step nobody had seen work."""
    cannot_upload_best_pt(monkeypatch)
    backend = FakeBackend()
    assert gpu(server, backend) == 0
    run = client.get(f"/training-runs/{run_id}").json()
    assert backend.exports == 1
    assert run["stage"] == "failed" and run["artifacts"] == {}
    assert "best.pt could not be uploaded" in run["error"] and "IMX export itself worked" in run["error"] and "packerOut.zip" in run["error"]


def test_an_export_that_fails_after_a_lost_upload_reports_both(client, server, run_id, monkeypatch):
    cannot_upload_best_pt(monkeypatch)
    assert gpu(server, FakeBackend(export_fails=RuntimeError("matplotlib"))) == 0
    error = client.get(f"/training-runs/{run_id}").json()["error"]
    assert "could not be uploaded" in error and "export failed too: matplotlib" in error


def test_a_lost_upload_without_an_export_fails_the_run_at_once(client, server, monkeypatch):
    cannot_upload_best_pt(monkeypatch)
    p = client.post("/projects", json={"name": "d"}).json()
    populate(client, p["id"], {c["name"]: c["id"] for c in p["classes"]})
    did = client.post(f"/projects/{p['id']}/datasets", json={}).json()["id"]
    rid = client.post(f"/projects/{p['id']}/training-runs", json={"dataset_id": did, "epochs": 1, "export_imx": False}).json()["id"]
    backend = FakeBackend()
    gpu(server, backend)
    run = client.get(f"/training-runs/{rid}").json()
    assert run["stage"] == "failed" and "gave up" in run["error"] and backend.exports == 0


def test_the_patience_setting_reaches_the_api(server, run_id, monkeypatch):
    seen = []

    class Spy(gpu_agent.Api):
        def __init__(self, *a, **kw):
            seen.append(kw.get("max_wait"))
            super().__init__(*a, **kw)
    monkeypatch.setattr(gpu_agent, "Api", Spy)
    gpu(server, FakeBackend(), "--patience", "42")
    monkeypatch.setenv("MT_PATIENCE", "900")
    gpu(server, FakeBackend())
    assert seen == [42.0, 900.0]


def test_the_dataset_comes_from_the_bucket_and_the_worker_token_stays_home(client, server, run_id, bucket, caplog):
    from app import dataset_bucket
    from app.models import TrainingRun
    db, v = get_version(client, db_run(client, run_id).dataset_id)
    dataset_bucket.ensure_uploaded(db, v)
    backend = FakeBackend()
    assert gpu(server, backend) == 0
    assert client.get(f"/training-runs/{run_id}").json()["stage"] == "awaiting_package"
    assert len(bucket.requests) == 1                                                   # fetched from the bucket, once
    sent = {k.lower() for k in bucket.requests[0]["headers"]}
    assert "x-worker-token" not in sent and "x-worker-name" not in sent                # the token never goes to the bucket
    assert backend.seen["dirs"]                                                        # and it trained on what came from there


def test_a_bad_bucket_link_falls_back_to_the_backend_without_logging_the_signature(client, server, run_id, bucket, caplog):
    from app import dataset_bucket
    db, v = get_version(client, db_run(client, run_id).dataset_id)
    dataset_bucket.ensure_uploaded(db, v)
    bucket.broken = True                                                               # an expired link answers 403
    assert gpu(server, FakeBackend()) == 0
    assert client.get(f"/training-runs/{run_id}").json()["stage"] == "awaiting_package"
    assert len(bucket.requests) == 2                                                   # two quick tries, then the backend
    assert "X-Amz-Signature" not in caplog.text and "through the backend instead" in caplog.text


def db_run(client, run_id):
    from app.models import TrainingRun
    with client.session_factory() as db:
        return db.get(TrainingRun, run_id)


def test_training_only_run_skips_export_and_needs_no_pi(client, server):
    p = client.post("/projects", json={"name": "d"}).json()
    populate(client, p["id"], {c["name"]: c["id"] for c in p["classes"]})
    did = client.post(f"/projects/{p['id']}/datasets", json={}).json()["id"]
    rid = client.post(f"/projects/{p['id']}/training-runs", json={"dataset_id": did, "epochs": 1, "export_imx": False}).json()["id"]
    gpu(server, FakeBackend())
    run = client.get(f"/training-runs/{rid}").json()
    assert run["stage"] == "done" and set(run["artifacts"]) == {"best_pt"}


def test_cancelling_stops_training_and_uploads_nothing(client, server, run_id):
    backend = FakeBackend(epoch_seconds=0.3)
    t = threading.Thread(target=gpu, args=(server, backend), daemon=True)
    t.start()
    deadline = time.time() + 10
    while client.get(f"/training-runs/{run_id}").json()["stage"] != "training" and time.time() < deadline:
        time.sleep(0.05)
    client.post(f"/training-runs/{run_id}/cancel")
    t.join(timeout=15)
    assert not t.is_alive()
    run = client.get(f"/training-runs/{run_id}").json()
    assert run["stage"] == "cancelled" and run["artifacts"] == {}
    assert backend.epochs_run < 3, "training ran to the end even though the run was cancelled"  # the agent really stopped it


def test_a_crash_in_training_is_reported_as_a_failed_run(client, server, run_id):
    gpu(server, FakeBackend(fail_with=RuntimeError("CUDA out of memory")))
    run = client.get(f"/training-runs/{run_id}").json()
    assert run["stage"] == "failed" and "RuntimeError: CUDA out of memory" in run["error"]


def test_a_packaging_failure_is_reported(client, server, run_id, tmp_path):
    gpu(server, FakeBackend())
    broken = tmp_path / "broken-package"
    broken.write_text("#!/bin/sh\necho 'bad model' >&2\nexit 3\n")
    broken.chmod(broken.stat().st_mode | stat.S_IEXEC)
    pi_agent.main(["--server", server, "--workdir", "work-pi", "--once", "--packager", str(broken)])
    run = client.get(f"/training-runs/{run_id}").json()
    assert run["stage"] == "failed" and "exited with 3" in run["error"] and "bad model" in run["error"]


def test_agents_exit_cleanly_when_nothing_is_waiting(server):
    assert gpu(server, FakeBackend()) == 0


def test_bad_token_and_unreachable_server_are_reported(server, monkeypatch):
    monkeypatch.setenv("MT_WORKER_TOKEN", "wrong")
    assert gpu(server, FakeBackend()) == 2
    monkeypatch.setenv("MT_WORKER_TOKEN", TOKEN)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}"
    assert gpu(dead, FakeBackend()) == 2


# ---------- pieces ----------

def test_preflight_catches_a_missing_java_or_wrong_os_before_training():
    p = REAL_PREFLIGHT
    assert p(False, "darwin") is None                                           # no export, no requirements
    assert "WSL2" in p(True, "win32", which=lambda c: "/usr/bin/java")
    assert "Java 17" in p(True, "linux", which=lambda c: None)
    assert p(True, "linux", which=lambda c: "/usr/bin/java") is None


def test_a_run_is_failed_up_front_when_the_machine_cannot_export(client, server, run_id, monkeypatch):
    monkeypatch.setattr(gpu_agent, "preflight", lambda export_imx, **kw: "needs Java 17" if export_imx else None)
    backend = FakeBackend()
    gpu(server, backend)
    run = client.get(f"/training-runs/{run_id}").json()
    assert run["stage"] == "failed" and run["error"] == "needs Java 17"
    assert backend.epochs_run == 0 and not backend.seen                          # no hours wasted training first


def test_rewrite_data_path(tmp_path):
    y = tmp_path / "data.yaml"
    y.write_text('# Generated\npath: .\ntrain: images/train\nval: images/val\nnames:\n  0: "multirotor"\n')
    gpu_agent.rewrite_data_path(y, tmp_path)
    lines = y.read_text().splitlines()
    assert lines[1] == f'path: "{tmp_path.resolve()}"' and lines[2] == "train: images/train" and lines[-1] == '  0: "multirotor"'
    y.write_text("train: images/train\n")                                           # no path line at all: add one
    gpu_agent.rewrite_data_path(y, tmp_path)
    assert y.read_text().startswith("path: ")


class Flaky(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):
        Flaky.hits += 1
        code = 503 if Flaky.hits <= 2 else 200
        body = b'{"detail":"restarting"}' if code == 503 else b'{"ok": true}'
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_api_retries_through_an_outage_but_not_forever_and_not_on_client_errors():
    srv = HTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        api = common.Api(url, "t", "w", backoff_base=0.01)
        assert api.json("GET", "/anything") == {"ok": True} and Flaky.hits == 3        # two 503s, then success
        Flaky.hits = 0
        with pytest.raises(common.ApiError, match="HTTP 503"):
            common.Api(url, "t", "w", backoff_base=0.01).call("GET", "/x", retry=False)  # no retry asked for
        Flaky.hits = 0
        with pytest.raises(common.ApiError, match="gave up"):
            common.Api(url, "t", "w", backoff_base=0.05, max_wait=0.1).call("GET", "/x")  # budget exhausted
    finally:
        srv.shutdown()


def test_api_retries_when_the_backend_is_down_then_comes_up():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    api = common.Api(f"http://127.0.0.1:{port}", "t", "w", backoff_base=0.1)
    Flaky.hits = 10                                                                   # the server will answer 200 straight away

    def start_later():
        time.sleep(0.35)
        srv = HTTPServer(("127.0.0.1", port), Flaky)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        start_later.srv = srv

    threading.Thread(target=start_later, daemon=True).start()
    try:
        assert api.json("GET", "/anything") == {"ok": True}                          # connection refused a few times, then works
    finally:
        time.sleep(0.05)
        getattr(start_later, "srv", None) and start_later.srv.shutdown()


class CutShort(BaseHTTPRequestHandler):
    """Announces 1000 bytes but, the first time, sends 400 and hangs up: what a connection dropped mid-download looks like."""

    hits = 0

    def do_GET(self):
        CutShort.hits += 1
        self.send_response(200)
        self.send_header("Content-Length", "1000")
        self.end_headers()
        self.wfile.write(b"x" * (400 if CutShort.hits == 1 else 1000))

    def log_message(self, *args):
        pass


def test_a_download_that_is_cut_short_is_noticed_and_fetched_again(tmp_path):
    CutShort.hits = 0
    srv = HTTPServer(("127.0.0.1", 0), CutShort)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        dest = tmp_path / "dataset.zip"
        common.Api(f"http://127.0.0.1:{srv.server_port}", "t", "w", backoff_base=0.01).download("/file", dest)
        assert dest.stat().st_size == 1000 and CutShort.hits == 2           # the 400-byte file was not kept as if it were complete
        assert list(tmp_path.glob("*.part")) == []
    finally:
        srv.shutdown()


class ScriptedApi:
    """Stands in for the backend connection: each download writes the next prepared payload."""

    def __init__(self, *payloads):
        self.payloads, self.calls = list(payloads), 0

    def download(self, url, dest):
        self.calls += 1
        dest.write_bytes(self.payloads.pop(0))


def a_dataset_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("data.yaml", "path: .\ntrain: images/train\nval: images/val\n")
    return buf.getvalue()


def test_a_dataset_that_is_not_a_zip_is_fetched_again(tmp_path):
    api, said = ScriptedApi(b"<html>502 Bad Gateway</html>", a_dataset_zip()), []
    data_yaml = gpu_agent.fetch_dataset(api, "/x", tmp_path, said.append, pause=0)
    assert data_yaml.name == "data.yaml" and api.calls == 2
    assert said == ["downloading the dataset", "downloading the dataset (try 2 of 3)"]


def test_a_dataset_that_never_arrives_intact_fails_saying_what_arrived(tmp_path):
    junk = b"<html>nope</html>"
    api = ScriptedApi(junk, junk, junk, junk)
    with pytest.raises(RuntimeError) as e:
        gpu_agent.fetch_dataset(api, "/x", tmp_path, pause=0)
    assert api.calls == 3 and "after 3 tries" in str(e.value)                       # gives up, but not at the first bad file
    assert f"{len(junk)} bytes" in str(e.value) and "<html>nope" in str(e.value)    # so the next failure explains itself


def test_the_export_calibrates_on_about_150_validation_images_not_all_of_them(tmp_path):
    def make(n):
        d = tmp_path / f"ds{n}"
        (d / "images" / "val").mkdir(parents=True)
        for i in range(n):
            (d / "images" / "val" / f"{i}.jpg").write_bytes(b"x")
        return d / "data.yaml"
    assert gpu_agent.calibration_fraction(make(520)) == pytest.approx(150 / 520)          # run 5: 520 images, one at a time, was killed
    assert gpu_agent.calibration_fraction(make(23)) == 1.0                                # small sets are used whole
    assert gpu_agent.calibration_fraction(tmp_path / "empty" / "data.yaml") == 1.0        # no val folder: leave it to Ultralytics
