"""The IMX500 export on this Mac: a run made with export_where "mac" is only trained on the pod, then exported by a job here."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from app import imx_export, jobs, training
from app.models import Job, JobStatus, TrainingRun
from app.storage import get_storage

from .test_training import claim, hdr, lan, new_run, ready  # noqa: F401 - fixtures

FAKE_EXPORT = textwrap.dedent('''
    """Stands in for agents/imx_export_mac.py: checks what it was given, writes the two files the real export makes."""
    import json, sys
    from pathlib import Path
    best, yaml, imgsz = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    assert best.read_bytes() == b"weights", "not the run's best.pt"
    text = yaml.read_text()
    n = len(list((yaml.parent / "images" / "val").iterdir()))
    out = best.parent / "best_imx_model"
    out.mkdir()
    (out / "packerOut.zip").write_bytes(b"PK-packer-" + imgsz.encode())
    (out / "labels.txt").write_text(json.dumps({"yaml": text, "calibration_images": n}))
    print("quantizing...")
    print("MT-RESULT " + json.dumps({"packer": str(out / "packerOut.zip"), "labels": str(out / "labels.txt")}))
''')


@pytest.fixture()
def tools(tmp_path, monkeypatch):
    """Export tools that are "set up": this test's Python plus a stand-in for the export script."""
    agent = tmp_path / "fake_export.py"
    agent.write_text(FAKE_EXPORT)
    monkeypatch.setattr(imx_export, "AGENT", agent)
    monkeypatch.setattr(imx_export, "env_python", lambda: Path(sys.executable))
    monkeypatch.setattr(imx_export, "tools_status", lambda: {"ready": True, "java": 17, "path": "-"})
    return agent


def trained_on_pod(client, lan, ready, **cfg):  # noqa: F811
    """A run the pod has trained and completed with best.pt only."""
    run = new_run(client, ready, export_where="mac", epochs=1, **cfg)
    task = claim(lan).json()
    assert task["config"]["export_imx"] is False                         # the pod is told not to export (nor set up the tools)
    assert lan.put(f"/workers/runs/{run['id']}/artifacts/best_pt", content=b"weights", headers=hdr()).status_code == 200
    done = lan.post(f"/workers/runs/{run['id']}/complete", json={"metrics": {"map50": 0.7}}, headers=hdr())
    assert done.status_code == 200, done.text
    return done.json()


def run_jobs(client):
    for job_id in sorted(j.id for j in client.session_factory().query(Job).filter(Job.status == JobStatus.queued)):
        jobs.run_job(job_id, client.session_factory)


def test_the_pod_only_trains_and_the_export_is_queued_here(client, lan, ready):  # noqa: F811
    run = trained_on_pod(client, lan, ready)
    assert run["stage"] == "awaiting_export"                             # complete accepted best.pt alone
    with client.session_factory() as db:
        queued = db.query(Job).filter(Job.type == "imx_export").all()
    assert [(j.params, j.status) for j in queued] == [({"run_id": run["id"]}, JobStatus.queued)]
    assert claim(lan, kind="pi", name="pi").status_code == 204          # the Pi waits for the export
    assert client.post(f"/training-runs/{run['id']}/export-imx").json()["id"] == queued[0].id  # no second job while one waits


def test_the_export_job_stores_the_files_and_hands_the_run_to_the_pi(client, lan, ready, tools):  # noqa: F811
    run = trained_on_pod(client, lan, ready)
    run_jobs(client)
    after = client.get(f"/training-runs/{run['id']}").json()
    assert after["stage"] == "awaiting_package" and after["error"] is None
    assert set(after["artifacts"]) == {"best_pt", "imx_zip", "labels"}
    assert client.get(f"/training-runs/{run['id']}/artifacts/imx_zip").content == b"PK-packer-640"
    seen = json.loads(client.get(f"/training-runs/{run['id']}/artifacts/labels").content)
    assert "names:" in seen["yaml"] and "multirotor" in seen["yaml"]   # the version's classes, for the model's labels
    assert 0 < seen["calibration_images"] <= imx_export.CALIBRATION_IMAGES
    assert claim(lan, kind="pi", name="pi").json()["run_id"] == run["id"]


def test_a_failed_export_keeps_best_pt_and_can_be_retried(client, lan, ready, tools):  # noqa: F811
    good = tools.read_text()
    tools.write_text("import sys\nprint('Java crashed')\nsys.exit(3)\n")
    run = trained_on_pod(client, lan, ready)
    run_jobs(client)
    after = client.get(f"/training-runs/{run['id']}").json()
    assert after["stage"] == "awaiting_export" and "Java crashed" in after["error"]
    assert set(after["artifacts"]) == {"best_pt"}
    with client.session_factory() as db:
        assert db.query(Job).filter(Job.type == "imx_export").one().status == JobStatus.failed
    tools.write_text(good)                                               # "Export on this Mac" again
    assert client.post(f"/training-runs/{run['id']}/export-imx").status_code == 202
    run_jobs(client)
    assert client.get(f"/training-runs/{run['id']}").json()["stage"] == "awaiting_package"


def test_an_export_without_the_tools_says_what_to_press(client, lan, ready):  # noqa: F811
    run = trained_on_pod(client, lan, ready)
    run_jobs(client)
    with client.session_factory() as db:
        job = db.query(Job).filter(Job.type == "imx_export").one()
        assert job.status == JobStatus.failed and "Set up the IMX export tools" in job.error
    assert client.get(f"/training-runs/{run['id']}").json()["stage"] == "awaiting_export"


def test_a_new_export_of_a_packaged_run_drops_the_old_rpk(client, lan, ready, tools):  # noqa: F811
    run = trained_on_pod(client, lan, ready)
    run_jobs(client)
    task = claim(lan, kind="pi", name="pi").json()
    assert lan.put(f"/workers/runs/{task['run_id']}/artifacts/rpk", content=b"rpk", headers=hdr("pi")).status_code == 200
    assert lan.post(f"/workers/runs/{task['run_id']}/complete", json={}, headers=hdr("pi")).json()["stage"] == "done"
    with client.session_factory() as db:
        rpk_key = db.get(TrainingRun, run["id"]).artifacts["rpk"]["key"]
    assert client.post(f"/training-runs/{run['id']}/export-imx").status_code == 202
    run_jobs(client)
    after = client.get(f"/training-runs/{run['id']}").json()
    assert after["stage"] == "awaiting_package" and "rpk" not in after["artifacts"]  # the Pi packages the new export
    with pytest.raises(FileNotFoundError):
        get_storage().get(rpk_key)


def test_runs_that_cannot_be_exported_are_refused(client, lan, ready):  # noqa: F811
    queued = new_run(client, ready, export_where="mac")
    r = client.post(f"/training-runs/{queued['id']}/export-imx")
    assert r.status_code == 409 and "queued" in r.json()["detail"]
    assert client.post("/training-runs/999/export-imx").status_code == 404


def test_a_run_waiting_for_its_export_can_be_cancelled(client, lan, ready):  # noqa: F811
    run = trained_on_pod(client, lan, ready)
    assert client.post(f"/training-runs/{run['id']}/cancel").json()["stage"] == "cancelled"


def test_old_runs_without_the_setting_still_export_on_the_pod():
    assert training.exports_on_mac({"export_imx": True}) is False           # made before export_where existed
    assert training.exports_on_mac({"export_imx": True, "export_where": "mac"}) is True
    assert training.exports_on_mac({"export_imx": False, "export_where": "mac"}) is False  # no export at all
    assert training.worker_config({"export_imx": True, "export_where": "mac"})["export_imx"] is False
    assert training.worker_config({"export_imx": True})["export_imx"] is True


def test_tools_status_follows_the_lock_file(tmp_path, monkeypatch):
    monkeypatch.setattr(imx_export.settings, "imx_export_dir", str(tmp_path))
    lock = tmp_path / "lock.txt"
    lock.write_text("ultralytics==8.4.183\n")
    monkeypatch.setattr(imx_export, "LOCK", lock)
    assert imx_export.tools_status()["ready"] is False
    (tmp_path / "env" / "bin").mkdir(parents=True)
    (tmp_path / "env" / "bin" / "python").write_text("")
    (tmp_path / "ready.json").write_text(json.dumps({"wanted": "ultralytics==8.4.183\n"}))
    assert imx_export.tools_status()["ready"] is True
    lock.write_text("ultralytics==8.5.0\n")
    assert imx_export.tools_status()["ready"] is False                    # a changed lock needs setting up again


def test_the_committed_lock_pins_every_package_exactly():
    """Setup installs this file with `uv pip sync`: a loose pin would let a new release in and break the export."""
    lines = [ln for ln in imx_export.LOCK.read_text().splitlines() if ln and not ln.startswith("#")]
    assert len(lines) > 50 and all("==" in ln for ln in lines), [ln for ln in lines if "==" not in ln]
    pinned = dict(ln.split("==", 1) for ln in lines)
    assert pinned["ultralytics"] == "8.4.183"                               # imx_export_mac.py lifts a check inside this version
    assert pinned["torch"] == "2.9.1" and "imx500-converter" in pinned and "model-compression-toolkit" in pinned


def test_the_setup_button_queues_one_job(client):
    pid = client.post("/projects", json={"name": "p"}).json()["id"]
    first = client.post(f"/projects/{pid}/imx-tools/setup")
    assert first.status_code == 202
    assert client.post(f"/projects/{pid}/imx-tools/setup").json()["id"] == first.json()["id"]
    setup = client.get("/imx-tools").json()["setup"]
    assert (setup["job_id"], setup["status"]) == (first.json()["id"], "queued")
