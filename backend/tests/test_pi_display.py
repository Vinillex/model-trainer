"""Run on Pi: the page's request, what the Pi agent is told and reports, and the agent's viewer handling (fake processes, no Pi)."""

import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import pi_display as store
from app import training
from app.config import settings
from app.main import app

from .helpers import from_ip
from .test_training import TOKEN, claim, finish_gpu, hdr, lan, new_run, ready, upload  # noqa: F401 - fixtures

AGENTS = Path(__file__).resolve().parents[2] / "agents"
sys.path.insert(0, str(AGENTS))

import pi_display as agent  # noqa: E402  (the Pi agent's part)
from common import ApiError  # noqa: E402

PI = {"X-MT-Pi": "1"}
RPK, LABELS = b"RPK-BYTES", b"multirotor\nfixed-wing\n"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "pi_display_file", str(tmp_path / "pi_display.json"))


@pytest.fixture()
def done_run(client, lan, ready):  # noqa: F811
    rid = new_run(client, ready)["id"]
    claim(lan)
    finish_gpu(lan, rid)
    assert claim(lan, "pi", "pi-1").status_code == 200
    assert upload(lan, rid, "rpk", RPK, "pi-1").status_code == 200
    assert upload(lan, rid, "labels", LABELS, "pi-1").status_code in (200, 422)  # labels came from the GPU worker already
    assert lan.post(f"/workers/runs/{rid}/complete", json={}, headers=hdr("pi-1")).status_code == 200
    training.touch_worker("pi-1", "pi")  # the Pi is online
    return rid


def show(client, rid, **body):
    return client.post(f"/training-runs/{rid}/pi-display", json=body, headers=PI)


# ---------- the page ----------

def test_a_finished_run_can_be_shown_and_the_page_sees_it_waiting(client, done_run):
    r = show(client, done_run)
    assert r.status_code == 200
    v = r.json()
    assert v["state"] == "waiting" and v["run_id"] == done_run and v["threshold"] == 0.25 and v["fps"] == 15.0 and v["pi_online"] is True
    assert client.get("/pi-display").json()["request_id"] == v["request_id"]


def test_the_frame_rate_follows_the_models_input_size_unless_given(client, done_run):
    assert show(client, done_run).json()["fps"] == 15.0                                            # 640: measured 15
    with client.session_factory() as db:
        from app.models import TrainingRun
        run = db.get(TrainingRun, done_run)
        run.config = {**run.config, "imgsz": 416}
        db.commit()
    assert show(client, done_run).json()["fps"] == 23.0                                            # 416: measured 23
    assert show(client, done_run, fps=12, threshold=0.5).json()["fps"] == 12
    assert store.default_fps(512) == 15.0 and store.default_fps(320) == 15.0                       # not measured: the safe rate


@pytest.mark.parametrize("body", [{"threshold": 0.1}, {"threshold": 1.0}, {"fps": 0}, {"fps": 31}])
def test_bad_numbers_are_refused(client, done_run, body):
    assert show(client, done_run, **body).status_code == 422
    assert client.get("/pi-display").json()["state"] == "none"


def test_only_finished_runs_with_a_camera_file_can_be_shown(client, done_run, ready, lan):  # noqa: F811
    queued = new_run(client, ready)["id"]
    r = show(client, queued)
    assert r.status_code == 409 and "not finished" in r.json()["detail"]
    no_export = new_run(client, ready, export_imx=False)["id"]
    claim(lan)                                                                                      # the queued run goes first; finish them in turn
    assert upload(lan, queued, "best_pt", b"w").status_code == 200
    lan.post(f"/workers/runs/{queued}/fail", json={"error": "x"}, headers=hdr("gpu-1"))
    claim(lan)
    assert upload(lan, no_export, "best_pt", b"w").status_code == 200
    assert lan.post(f"/workers/runs/{no_export}/complete", json={}, headers=hdr("gpu-1")).status_code == 200
    r = show(client, no_export)
    assert r.status_code == 409 and "Export for IMX500 was off" in r.json()["detail"]
    assert show(client, 9999).status_code == 404
    assert client.get("/pi-display").json()["state"] == "none"


def test_nothing_is_requested_when_no_pi_is_online(client, done_run):
    training._workers.clear()
    r = show(client, done_run)
    assert r.status_code == 409 and "No Raspberry Pi is online" in r.json()["detail"]
    assert client.get("/pi-display").json()["state"] == "none"


def test_switching_the_screen_on_needs_the_app_header_and_a_local_caller(client, done_run):
    assert client.post(f"/training-runs/{done_run}/pi-display", json={}).status_code == 403          # a web page in the same browser
    assert client.delete("/pi-display").status_code == 403
    lan_client = TestClient(from_ip(app, "192.168.1.50"))
    assert lan_client.post(f"/training-runs/{done_run}/pi-display", json={}, headers=PI).status_code == 403
    for proxy in ({"CF-Connecting-IP": "1.2.3.4"}, {"X-Forwarded-For": "1.2.3.4"}):
        assert client.post(f"/training-runs/{done_run}/pi-display", json={}, headers={**PI, **proxy}).status_code == 403
    assert client.get("/pi-display").json()["state"] == "none"


# ---------- what the Pi agent is told ----------

def test_the_request_is_handed_out_once_and_the_token_is_needed(client, lan, done_run):  # noqa: F811
    assert lan.get("/workers/display").status_code == 401
    assert lan.get("/workers/display", headers=hdr("pi-1")).json() == {"active": None, "request": None}
    rid = show(client, done_run).json()["request_id"]
    first = lan.get("/workers/display", headers=hdr("pi-1")).json()
    assert first["active"] == rid and first["request"]["run_name"] and first["request"]["threshold"] == 0.25
    assert lan.get("/workers/display", headers=hdr("pi-1")).json() == {"active": rid, "request": None}   # used up: not sent again


def test_the_pi_gets_the_files_of_the_requested_run_and_nothing_else(client, lan, done_run):  # noqa: F811
    assert lan.get("/workers/display/files/rpk", headers=hdr("pi-1")).status_code == 404            # nothing is wanted
    show(client, done_run)
    assert lan.get("/workers/display/files/rpk").status_code == 401
    assert lan.get("/workers/display/files/rpk", headers=hdr("pi-1")).content == RPK
    assert lan.get("/workers/display/files/labels", headers=hdr("pi-1")).content == b"multirotor\nfixed-wing\n"
    for other in ("best_pt", "imx_zip", "../rpk", "nope"):
        assert lan.get(f"/workers/display/files/{other}", headers=hdr("pi-1")).status_code == 404   # only the camera file and the labels
    client.delete("/pi-display", headers=PI)
    assert lan.get("/workers/display/files/rpk", headers=hdr("pi-1")).status_code == 404            # nothing is wanted again


def report(lan, rid, state, **extra):
    return lan.post("/workers/display/status", json={"request_id": rid, "state": state, **extra}, headers=hdr("pi-1"))


def test_reports_move_the_page_from_waiting_to_running_to_closed(client, lan, done_run):  # noqa: F811
    rid = show(client, done_run).json()["request_id"]
    lan.get("/workers/display", headers=hdr("pi-1"))
    assert report(lan, rid, "starting", message="loading").status_code == 204
    v = client.get("/pi-display").json()
    assert v["state"] == "starting" and v["message"] == "loading"
    report(lan, rid, "running", message="Showing", camera_fps=15.0, results_per_s=14.8, detections=2)
    v = client.get("/pi-display").json()
    assert (v["state"], v["camera_fps"], v["results_per_s"], v["detections"]) == ("running", 15.0, 14.8, 2)
    report(lan, rid, "stopped", message="The viewer ended on the Pi.")
    v = client.get("/pi-display").json()
    assert v["state"] == "stopped" and "ended on the Pi" in v["message"] and v["run_id"] == done_run
    assert lan.get("/workers/display", headers=hdr("pi-1")).json() == {"active": None, "request": None}   # the request is over, never re-sent


def test_a_failure_stays_on_the_page_with_its_reason(client, lan, done_run):  # noqa: F811
    rid = show(client, done_run).json()["request_id"]
    lan.get("/workers/display", headers=hdr("pi-1"))
    report(lan, rid, "failed", message="the user program is running on the Pi and holds the camera")
    v = client.get("/pi-display").json()
    assert v["state"] == "failed" and "holds the camera" in v["message"]


def test_stopping_from_the_page_tells_the_agent_to_end_the_viewer_and_it_confirms(client, lan, done_run):  # noqa: F811
    rid = show(client, done_run).json()["request_id"]
    lan.get("/workers/display", headers=hdr("pi-1"))
    report(lan, rid, "running", camera_fps=15, results_per_s=15, detections=0)
    assert client.delete("/pi-display", headers=PI).json()["state"] == "none"
    assert lan.get("/workers/display", headers=hdr("pi-1")).json() == {"active": None, "request": None}   # nothing should be showing now
    assert report(lan, rid, "running", camera_fps=15, results_per_s=15, detections=0).status_code == 204   # a late report is dropped...
    assert client.get("/pi-display").json()["state"] == "none"
    report(lan, rid, "stopped", message="Stopped from the Training page.")                              # ...but how it ended is kept
    assert client.get("/pi-display").json()["state"] == "stopped"


def test_a_new_request_replaces_the_old_one_and_old_reports_are_ignored(client, lan, done_run):  # noqa: F811
    first = show(client, done_run).json()["request_id"]
    lan.get("/workers/display", headers=hdr("pi-1"))
    second = show(client, done_run).json()["request_id"]
    assert first != second
    report(lan, first, "running", camera_fps=1, results_per_s=1, detections=9)                          # the old viewer still talking
    v = client.get("/pi-display").json()
    assert v["request_id"] == second and v["state"] == "waiting" and v["detections"] is None
    got = lan.get("/workers/display", headers=hdr("pi-1")).json()
    assert got["active"] == second and got["request"]["request_id"] == second


def test_bad_reports_are_refused(client, lan, done_run):  # noqa: F811
    assert lan.post("/workers/display/status", json={"request_id": "x", "state": "exploded"}, headers=hdr("pi-1")).status_code == 422
    assert lan.post("/workers/display/status", json={"state": "running"}, headers=hdr("pi-1")).status_code == 422
    assert lan.post("/workers/display/status", json={"request_id": "x", "state": "running"}).status_code == 401


# ---------- a Pi that goes quiet ----------

class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


@pytest.fixture()
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(store, "_now", c)
    return c


def test_a_picked_up_request_whose_reports_stop_is_shown_as_ended_not_running_forever(client, lan, done_run, clock):  # noqa: F811
    rid = show(client, done_run).json()["request_id"]
    lan.get("/workers/display", headers=hdr("pi-1"))
    report(lan, rid, "running", camera_fps=15, results_per_s=15, detections=0)
    clock.advance(seconds=20)
    assert client.get("/pi-display").json()["state"] == "running"                                    # reports every ~5 s: 20 s of quiet is still fine
    clock.advance(seconds=10)
    v = client.get("/pi-display").json()
    assert v["state"] == "ended" and "stopped reporting" in v["message"]
    assert lan.get("/workers/display", headers=hdr("pi-1")).json() == {"active": None, "request": None}   # and it is not started again by itself


def test_a_request_the_pi_never_picks_up_is_given_up_on(client, lan, done_run, clock):  # noqa: F811
    show(client, done_run)
    clock.advance(seconds=60)
    assert client.get("/pi-display").json()["state"] == "waiting"
    clock.advance(seconds=40)
    v = client.get("/pi-display").json()
    assert v["state"] == "failed" and "did not pick this up" in v["message"]


def test_a_request_picked_up_but_not_yet_reported_gets_time_to_start(client, lan, done_run, clock):  # noqa: F811
    show(client, done_run)
    lan.get("/workers/display", headers=hdr("pi-1"))
    clock.advance(seconds=60)                                                                         # checking, downloading, starting
    assert client.get("/pi-display").json()["state"] == "waiting" or client.get("/pi-display").json()["state"] == "starting"
    clock.advance(seconds=40)
    assert client.get("/pi-display").json()["state"] == "ended"


def test_a_finished_result_is_forgotten_after_ten_minutes(client, lan, done_run, clock):  # noqa: F811
    rid = show(client, done_run).json()["request_id"]
    lan.get("/workers/display", headers=hdr("pi-1"))
    report(lan, rid, "failed", message="no screen")
    clock.advance(minutes=9)
    assert client.get("/pi-display").json()["state"] == "failed"
    clock.advance(minutes=2)
    assert client.get("/pi-display").json()["state"] == "none"


def test_the_state_survives_a_backend_restart(client, lan, done_run):  # noqa: F811
    rid = show(client, done_run).json()["request_id"]
    lan.get("/workers/display", headers=hdr("pi-1"))
    report(lan, rid, "running", camera_fps=15, results_per_s=15, detections=1)
    assert Path(settings.pi_display_file).exists()                                                    # it lives in a file, not in memory
    assert client.get("/pi-display").json()["state"] == "running"


# ================= the Pi agent's side =================

class FakeApi:
    def __init__(self, info=None, files=True):
        self.info, self.reports, self.downloads, self.files, self.down = info or {"active": None, "request": None}, [], [], files, False

    def display_request(self):
        if self.down:
            raise ApiError(0, "can't reach the Mac")
        return self.info

    def display_status(self, body):
        self.reports.append(body)

    def display_file(self, name, dest):
        if not self.files:
            raise ApiError(404, "gone")
        self.downloads.append(name)
        dest.write_bytes(b"x")


class FakeProc:
    def __init__(self, pid=777, polls_until_exit=None, ignores=()):
        self.pid, self.returncode, self._n, self.ignores = pid, None, polls_until_exit, set(ignores)
        self.signals = []

    def poll(self):
        return self.returncode

    def die(self, code=0):
        self.returncode = code


class Env:
    """The process table and the clock of a Pi, both fake."""

    def __init__(self, tmp_path, request=None, **api_kw):
        self.request = {"request_id": "r1", "run_id": 1, "run_name": "v2-cmy-100ep", "threshold": 0.25, "fps": 15.0} if request is None else request
        self.api = FakeApi({"active": self.request["request_id"], "request": self.request}, **api_kw)
        self.popen_calls, self.procs, self.t, self.busy_now, self.screen = [], [], 0.0, False, True
        self.signals, self.ignores = [], set()
        self.mgr = agent.DisplayManager(self.api, tmp_path, Path("/agent/pi_viewer.py"), popen=self.popen, killpg=self.killpg, busy=lambda: self.busy_now,
                                        env_for=self.env_for, now=lambda: self.t, sleep=self.sleep)

    def env_for(self):
        if not self.screen:
            raise agent.NoScreen("no desktop session")
        return {"WAYLAND_DISPLAY": "wayland-0", "QT_QPA_PLATFORM": "wayland", "XDG_RUNTIME_DIR": "/run/user/1000"}

    def popen(self, cmd, **kw):
        self.popen_calls.append((cmd, kw))
        p = FakeProc(pid=700 + len(self.procs))
        self.procs.append(p)
        return p

    def killpg(self, pid, sig):
        self.signals.append((pid, sig))
        p = next(p for p in self.procs if p.pid == pid)
        if sig not in self.ignores:
            p.die(-int(sig))

    def sleep(self, s):
        self.t += s

    def viewer_says(self, text):
        (self.mgr.dir / "viewer.log").write_text(text)

    def states(self):
        return [(r["state"]) for r in self.api.reports]


LINE = "[viewer] camera fps 15.0, network results/s 14.9, detections now 3\n"


def test_a_request_starts_the_viewer_with_exactly_the_wayland_environment(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    (cmd, kw), = e.popen_calls
    assert cmd[0] == sys.executable and cmd[1] == "/agent/pi_viewer.py"
    assert dict(zip(cmd[2::2], cmd[3::2])) == {"--model": str(e.mgr.dir / "network.rpk"), "--labels": str(e.mgr.dir / "labels.txt"), "--threshold": "0.25",
                                               "--fps": "15.0", "--title": "v2-cmy-100ep"}
    assert kw["env"] == {"WAYLAND_DISPLAY": "wayland-0", "QT_QPA_PLATFORM": "wayland", "XDG_RUNTIME_DIR": "/run/user/1000"}    # and no DISPLAY (xcb fails)
    assert kw["start_new_session"] is True and kw["stdin"] is not None
    assert e.api.downloads == ["rpk", "labels"] and e.states() == ["starting"]
    assert (e.mgr.dir / "network.rpk").exists()


def test_paths_given_to_the_viewer_are_absolute_because_it_runs_in_another_folder(tmp_path, monkeypatch):
    """The service's --workdir is the relative "work"; the viewer starts inside the request's folder, so a relative path would not be found."""
    monkeypatch.chdir(tmp_path)
    e = Env(Path("work"))
    e.mgr.viewer = Path("pi_viewer.py")
    e.mgr = agent.DisplayManager(e.api, Path("work"), Path("pi_viewer.py"), popen=e.popen, killpg=e.killpg, busy=lambda: False, env_for=e.env_for,
                                 now=lambda: e.t, sleep=e.sleep)
    e.mgr.sync()
    cmd, kw = e.popen_calls[0]
    opts = dict(zip(cmd[2::2], cmd[3::2]))
    assert Path(cmd[1]).is_absolute() and Path(opts["--model"]).is_absolute() and Path(opts["--labels"]).is_absolute()
    assert Path(opts["--model"]).exists() and Path(opts["--labels"]).exists() and str(kw["cwd"]).startswith(str(tmp_path.resolve()))


def test_the_real_environment_is_wayland_without_display(tmp_path):
    runtime = tmp_path / "run" / "1000"
    runtime.mkdir(parents=True)
    with pytest.raises(agent.NoScreen) as e:
        agent.session_env(tmp_path / "run", uid=1000)                                                  # a folder but no desktop socket
    assert "no desktop session" in str(e.value)
    (runtime / "wayland-0").touch()
    (runtime / "wayland-0.lock").touch()
    env = agent.session_env(tmp_path / "run", uid=1000, environ={"HOME": "/home/p", "PATH": "/usr/bin", "DISPLAY": ":0", "SECRET": "x"})
    assert env["WAYLAND_DISPLAY"] == "wayland-0" and env["QT_QPA_PLATFORM"] == "wayland" and env["XDG_RUNTIME_DIR"] == str(runtime)
    assert "DISPLAY" not in env and "SECRET" not in env
    with pytest.raises(agent.NoScreen):
        agent.session_env(tmp_path / "nowhere", uid=1000)


def test_progress_comes_from_the_viewers_log_and_is_reported_every_pass(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": "r1", "request": None}                                                      # handed out once
    e.mgr.sync()
    assert e.states() == ["starting", "starting"]                                                       # still loading the network onto the chip
    e.viewer_says("Network Firmware Upload: 40%\n" + LINE)
    e.mgr.sync()
    last = e.api.reports[-1]
    assert (last["state"], last["camera_fps"], last["results_per_s"], last["detections"]) == ("running", 15.0, 14.9, 3)
    assert len(e.popen_calls) == 1                                                                      # never started twice


def test_loading_the_model_onto_the_chip_shows_its_percentage(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": "r1", "request": None}
    e.viewer_says("Network Firmware Upload:  64%|######4   | 2.00M/3.10M [00:45<00:24]\rNetwork Firmware Upload:  66%|######6   | 2.05M/3.10M [00:47<00:25]\r")
    e.mgr.sync()
    assert e.api.reports[-1]["state"] == "starting" and "66%" in e.api.reports[-1]["message"]


def test_a_qt_failure_is_noticed_even_though_the_process_hangs(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": "r1", "request": None}
    e.viewer_says('qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.\nThis application failed to start because no Qt platform plugin could be initialized.\n')
    e.ignores = {signal.SIGINT}                                                                         # the hang: it ignores Ctrl-C
    e.mgr.sync()
    assert e.api.reports[-1]["state"] == "failed" and "Qt platform plugin" in e.api.reports[-1]["message"]
    assert [s for _, s in e.signals] == [signal.SIGINT, signal.SIGTERM]                                 # polite first, then firmer
    assert e.mgr.proc is None


def test_stopping_escalates_from_ctrl_c_to_terminate_to_kill_and_waits_for_the_exit(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": None, "request": None}                                                      # the page pressed Stop
    e.ignores = {signal.SIGINT, signal.SIGTERM}
    t0 = e.t
    e.mgr.sync()
    assert [s for _, s in e.signals] == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert e.signals[0][0] == e.procs[0].pid and e.t - t0 >= 5 + 3                                      # it really waited between the steps
    assert e.api.reports[-1]["state"] == "stopped" and e.mgr.proc is None
    assert not (tmp_path / "display").exists()                                                          # the model files are gone too


def test_a_polite_viewer_gets_only_ctrl_c(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": None, "request": None}
    e.mgr.sync()
    assert [s for _, s in e.signals] == [signal.SIGINT]


def test_a_new_request_waits_for_the_old_viewer_to_be_gone_before_the_camera_is_used_again(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    order = []
    e.killpg_orig = e.killpg
    e.mgr.killpg = lambda pid, sig: (order.append(("kill", sig)), e.killpg_orig(pid, sig))
    real_popen = e.popen
    e.mgr.popen = lambda cmd, **kw: (order.append(("start", None)), real_popen(cmd, **kw))[1]
    second = {"request_id": "r2", "run_id": 2, "run_name": "v1-100ep", "threshold": 0.4, "fps": 23.0}
    e.api.info = {"active": "r2", "request": second}
    e.mgr.sync()
    assert order == [("kill", signal.SIGINT), ("start", None)] and e.procs[0].returncode is not None
    assert e.popen_calls[-1][0][e.popen_calls[-1][0].index("--threshold") + 1] == "0.4"


def test_a_viewer_that_ends_by_itself_is_reported_once_and_not_restarted(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": "r1", "request": None}
    e.viewer_says(LINE)
    e.mgr.sync()
    e.procs[0].die(0)
    e.mgr.sync()
    assert e.api.reports[-1]["state"] == "stopped" and "ended on the Pi" in e.api.reports[-1]["message"]
    n = len(e.api.reports)
    e.mgr.sync()
    e.mgr.sync()
    assert len(e.api.reports) == n and len(e.popen_calls) == 1                                          # nothing more, and no relaunch


def test_a_crashing_viewer_is_reported_as_failed(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": "r1", "request": None}
    e.viewer_says("Traceback (most recent call last):\n  File x\nRuntimeError: no camera\n")
    e.procs[0].die(1)
    e.mgr.sync()
    assert e.api.reports[-1]["state"] == "failed" and "RuntimeError: no camera" in e.api.reports[-1]["message"]   # the error itself, not just "Traceback"


def test_an_outage_never_stops_a_working_viewer(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.down = True
    for _ in range(5):
        e.mgr.sync()
    assert e.signals == [] and e.mgr.proc is not None and e.mgr.proc.poll() is None
    e.api.down = False
    e.api.info = {"active": "r1", "request": None}
    e.mgr.sync()
    assert e.signals == []                                                                              # still wanted: still showing


def test_the_user_camera_program_is_never_touched_and_is_reported(tmp_path):
    e = Env(tmp_path)
    e.busy_now = True
    e.mgr.sync()
    assert e.popen_calls == [] and e.signals == []
    assert e.api.reports[-1]["state"] == "failed" and agent.PROGRAM in e.api.reports[-1]["message"] and "will not stop it" in e.api.reports[-1]["message"]


def test_no_desktop_session_and_a_failed_download_are_reported_without_starting_anything(tmp_path):
    e = Env(tmp_path)
    e.screen = False
    e.mgr.sync()
    assert e.popen_calls == [] and e.api.reports[-1]["state"] == "failed" and "no desktop session" in e.api.reports[-1]["message"]
    e = Env(tmp_path / "b", files=False)
    e.mgr.sync()
    assert e.popen_calls == [] and e.api.reports[-1]["state"] == "failed" and "download" in e.api.reports[-1]["message"]
    assert not (tmp_path / "b" / "display").exists()                                                    # no half-downloaded files left behind


def test_a_model_that_never_starts_is_given_up_on_after_ten_minutes(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.api.info = {"active": "r1", "request": None}
    e.t += agent.STARTUP_LIMIT - 1
    e.mgr.sync()
    assert e.api.reports[-1]["state"] == "starting"
    e.t += 2
    e.mgr.sync()
    assert e.api.reports[-1]["state"] == "failed" and "10 minutes" in e.api.reports[-1]["message"]


def test_the_user_program_is_found_by_its_script_name_and_never_by_a_search_string(tmp_path):
    def proc(pid, *argv):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    proc(10, "bash", "-c", "pgrep -f main_ai.py; echo main_ai.py")                                      # something that merely mentions it
    proc(11, "/usr/bin/python3", "/home/p/agent/pi_agent.py")
    assert agent.camera_program_running(tmp_path) is False
    proc(12, "/usr/bin/python3", "/home/prateek/Dev/rpi_ai/main_ai.py")
    assert agent.camera_program_running(tmp_path) is True


def test_shutdown_ends_the_viewer(tmp_path):
    e = Env(tmp_path)
    e.mgr.sync()
    e.mgr.shutdown()
    assert [s for _, s in e.signals] == [signal.SIGINT] and e.mgr.proc is None


# ---------- inside the agent's loop ----------

import pi_agent  # noqa: E402

from .test_pi_setup import FakeApi as LoopApi  # noqa: E402


class Stop(Exception):
    pass


def run_loop(monkeypatch, argv, display):
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 3:
            raise Stop

    monkeypatch.setenv("MT_WORKER_TOKEN", "x")
    monkeypatch.setattr(pi_agent, "Api", lambda *a, **k: LoopApi())
    monkeypatch.setattr(pi_agent, "DisplayManager", lambda *a, **k: display)
    monkeypatch.setattr(pi_agent.shutil, "which", lambda c: "/usr/bin/imx500-package")
    monkeypatch.setattr(pi_agent.time, "sleep", fake_sleep)
    with pytest.raises(Stop):
        pi_agent.main(["--server", "http://mac.local:8000", *argv])


class SpyDisplay:
    def __init__(self, fail=False):
        self.syncs, self.shut, self.fail = 0, False, fail

    def sync(self):
        self.syncs += 1
        if self.fail:
            raise RuntimeError("boom")

    def shutdown(self):
        self.shut = True


def test_the_live_view_is_synced_on_every_pass_and_ended_when_the_agent_stops(monkeypatch):
    d = SpyDisplay()
    run_loop(monkeypatch, ["--display"], d)
    assert d.syncs == 3 and d.shut is True


def test_a_failing_live_view_never_takes_the_packaging_worker_down(monkeypatch, caplog):
    d = SpyDisplay(fail=True)
    with caplog.at_level(logging.ERROR, logger="agent"):
        run_loop(monkeypatch, ["--display"], d)
    assert d.syncs == 3                                                                                 # the loop carried on
    assert any("the live view failed" in r.getMessage() for r in caplog.records)


def test_without_the_flag_there_is_no_live_view(monkeypatch):
    made = []
    monkeypatch.setattr(pi_agent, "DisplayManager", lambda *a, **k: made.append(1))
    monkeypatch.setenv("MT_WORKER_TOKEN", "x")
    monkeypatch.setattr(pi_agent, "Api", lambda *a, **k: LoopApi())
    monkeypatch.setattr(pi_agent.shutil, "which", lambda c: "/usr/bin/imx500-package")
    assert pi_agent.main(["--server", "http://mac.local:8000", "--once"]) == 0
    assert made == []
