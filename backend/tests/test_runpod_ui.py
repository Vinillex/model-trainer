"""Training on RunPod from the UI: the price check, the confirmed launch (a detached process), its log, stopping, and the launcher
options that bind a launch to what the person confirmed. Nothing here reaches RunPod or starts a real launcher."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import runpod as ui
from app.config import settings
from app.main import app
from scripts import runpod as cli

from .helpers import from_ip
from .test_runpod_launch import KEY, FakeApi, Harness, http
from .test_runpod_launch import configured, queued  # noqa: F401 - fixtures
from .test_training import TOKEN, lan, new_run, ready  # noqa: F401 - fixtures

SPEND = {"X-MT-Spend": "1"}
BACKEND = Path(__file__).resolve().parents[1]


class FakeProc:
    def __init__(self, pid=4242, exit_code=None):
        self.pid, self._code, self.done = pid, exit_code, threading.Event()
        if exit_code is not None:
            self.done.set()

    def finish(self, code):
        self._code = code
        self.done.set()

    def wait(self):
        self.done.wait(10)
        return self._code


class World:
    """The launcher, the clock of processes and RunPod, all fake."""

    def __init__(self):
        self.api = FakeApi()
        self.spawned, self.killed, self.running = [], [], set()
        self.next_proc = FakeProc()
        self.cloudflared = True

    def spawn(self, cmd, log):
        self.spawned.append(cmd)
        log.write(b"Starting a tunnel to this backend...\nTunnel: https://x.trycloudflare.com\n")
        log.flush()
        self.running.add(self.next_proc.pid)
        return self.next_proc

    def alive(self, pid):
        return pid in self.running

    def kill(self, pid, sig):
        self.killed.append((pid, sig))


@pytest.fixture()
def world(client, configured, monkeypatch, tmp_path):  # noqa: F811
    w = World()
    monkeypatch.setattr(settings, "runpod_launch_dir", str(tmp_path / "runpod"))
    monkeypatch.setattr(ui, "make_api", lambda: w.api)
    monkeypatch.setattr(ui, "make_http", lambda: http())
    monkeypatch.setattr(ui, "spawn", w.spawn)
    monkeypatch.setattr(ui, "alive", w.alive)
    monkeypatch.setattr(ui.os, "kill", w.kill)
    monkeypatch.setattr(ui.shutil, "which", lambda name: "/usr/local/bin/cloudflared" if w.cloudflared else None)
    return w


def launch_body(run, **over):
    return {"run_id": run, "max_hours": 3, "cloud": "secure", "max_cost": 1.5, **over}


# ---------- reading: status and the price check ----------

def test_status_lists_only_this_projects_pods_and_the_account(client, world):
    world.api.pods = {"p1": {"id": "p1", "name": "mt-1024-1000", "desiredStatus": "RUNNING", "costPerHr": 0.59},
                      "p2": {"id": "p2", "name": "someone-elses", "desiredStatus": "RUNNING", "costPerHr": 2.0}}
    s = client.get("/runpod/status").json()
    assert s["key_set"] and s["token_set"] and s["cloudflared"] and s["balance"] == 5.0
    assert [p["id"] for p in s["pods"]] == ["p1"] and s["pods"][0]["cost_per_hr"] == 0.59 and s["other_pods"] == 1
    assert KEY not in json.dumps(s)


def test_status_without_a_key_says_so_and_does_not_call_runpod(client, world, monkeypatch):
    monkeypatch.setattr(settings, "runpod_api_key", None)
    monkeypatch.setattr(ui, "make_api", lambda: pytest.fail("no key, no RunPod call"))
    s = client.get("/runpod/status").json()
    assert s["key_set"] is False and s["balance"] is None and s["pods"] == []


def test_plan_shows_gpus_prices_and_the_worst_case_and_starts_nothing(client, world, queued):
    p = client.get(f"/runpod/plan?run_id={queued}&max_hours=3").json()
    assert p["run_id"] == queued and p["max_hours"] == 3 and p["cloud"] == "secure"
    assert [g["id"] for g in p["gpus"]] == ["NVIDIA RTX A4000", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]   # preference order, only those offered
    assert [g["price"] for g in p["gpus"]] == [0.25, 0.27, 0.50]
    assert p["worst_price"] == 0.50 and p["worst_case"] == 1.5 and p["balance"] == 5.0
    assert world.spawned == [] and world.api.created == []


def test_plan_prices_the_community_cloud_separately(client, world, queued):
    p = client.get(f"/runpod/plan?run_id={queued}&max_hours=2&cloud=community").json()
    assert p["cloud"] == "community" and p["worst_price"] == 0.25 and p["worst_case"] == 0.5


def test_plan_names_the_run_it_was_asked_about_not_the_oldest(client, world, ready, queued):  # noqa: F811
    second = new_run(client, ready)["id"]
    assert client.get(f"/runpod/plan?run_id={second}").json()["run_id"] == second


@pytest.mark.parametrize("make, expect", [
    (lambda w, c, q: f"/runpod/plan?run_id={q + 99}", "not waiting"),
    (lambda w, c, q: f"/runpod/plan?run_id={q}&max_hours=3", "balance is $0.00"),
])
def test_plan_refusals_arrive_with_their_reason(client, world, queued, make, expect):
    world.api.balance = 0.0
    r = client.get(make(world, client, queued))
    assert r.status_code == 422 and expect in r.json()["detail"]


def test_plan_limits_and_setup_are_checked(client, world, queued, monkeypatch):
    assert client.get(f"/runpod/plan?run_id={queued}&max_hours=13").status_code == 422
    assert client.get(f"/runpod/plan?run_id={queued}&max_hours=0").status_code == 422
    assert client.get(f"/runpod/plan?run_id={queued}&cloud=moon").status_code == 422
    monkeypatch.setattr(settings, "runpod_api_key", None)
    assert client.get(f"/runpod/plan?run_id={queued}").status_code == 409


# ---------- the launch ----------

def test_launch_starts_the_detached_launcher_with_exactly_what_was_confirmed(client, world, queued):
    r = client.post("/runpod/launch", json=launch_body(queued, max_cost=1.5, max_hours=2.5, cloud="community"), headers=SPEND)
    assert r.status_code == 202 and r.json()["state"] == "running" and r.json()["run_id"] == queued
    (cmd,) = world.spawned
    assert cmd[0] == sys.executable and cmd[1:6] == ["-u", "-m", "scripts.runpod", "launch", "--yes"]    # -u: lines reach the log as they happen
    opts = dict(zip(cmd[6::2], cmd[7::2]))
    assert opts == {"--run": str(queued), "--max-hours": "2.5", "--cloud": "community", "--max-cost": "1.50", "--backend-url": "http://127.0.0.1:8000"}
    assert KEY not in " ".join(cmd) and TOKEN not in " ".join(cmd)                                     # secrets are never on a command line
    assert any("Tunnel:" in line for line in r.json()["lines"])


def test_a_launch_needs_the_app_header_and_a_local_caller(client, world, queued):
    assert client.post("/runpod/launch", json=launch_body(queued)).status_code == 403                  # a web page in the same browser
    assert client.post("/runpod/launch", json=launch_body(queued), headers={"X-MT-Spend": "0"}).status_code == 403
    assert world.spawned == []
    lan_client = TestClient(from_ip(app, "192.168.1.50"))
    assert lan_client.post("/runpod/launch", json=launch_body(queued), headers=SPEND).status_code == 403
    for proxy in ({"CF-Connecting-IP": "1.2.3.4"}, {"X-Forwarded-For": "1.2.3.4"}):                    # the pod's own tunnel is public while it runs
        assert client.post("/runpod/launch", json=launch_body(queued), headers={**SPEND, **proxy}).status_code == 403
    assert client.post("/runpod/launch/stop", headers=proxy).status_code == 403
    assert world.spawned == [] and world.killed == []


def test_the_cors_allow_list_only_lists_this_apps_own_pages(client):
    pre = client.options("/runpod/launch", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST",
                                                    "Access-Control-Request-Headers": "x-mt-spend,content-type"})
    assert "access-control-allow-origin" not in pre.headers                                             # the browser would stop the real request
    ok = client.options("/runpod/launch", headers={"Origin": "http://127.0.0.1:3000", "Access-Control-Request-Method": "POST",
                                                   "Access-Control-Request-Headers": "x-mt-spend,content-type"})
    assert ok.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"


@pytest.mark.parametrize("over", [{"max_cost": None}, {"max_cost": 0}, {"max_hours": 13}, {"max_hours": 0}, {"cloud": "moon"}, {"run_id": "x"}])
def test_a_launch_with_bad_numbers_starts_nothing(client, world, queued, over):
    body = {k: v for k, v in launch_body(queued, **over).items() if v is not None}
    assert client.post("/runpod/launch", json=body, headers=SPEND).status_code == 422
    assert world.spawned == []


def test_a_launch_needs_a_waiting_run_the_setup_and_cloudflared(client, world, queued, monkeypatch):
    assert client.post("/runpod/launch", json=launch_body(queued + 99), headers=SPEND).status_code == 404
    world.cloudflared = False
    r = client.post("/runpod/launch", json=launch_body(queued), headers=SPEND)
    assert r.status_code == 409 and "brew install cloudflared" in r.json()["detail"]
    world.cloudflared = True
    monkeypatch.setattr(settings, "runpod_api_key", None)
    assert client.post("/runpod/launch", json=launch_body(queued), headers=SPEND).status_code == 409
    monkeypatch.setattr(settings, "runpod_api_key", KEY)
    monkeypatch.setattr(settings, "worker_token", None)
    assert client.post("/runpod/launch", json=launch_body(queued), headers=SPEND).status_code == 409
    assert world.spawned == []


def test_a_run_that_is_not_waiting_is_not_launched(client, world, queued, lan):  # noqa: F811
    from .test_training import claim
    assert claim(lan).status_code == 200                                                                                  # the run is now being trained
    r = client.post("/runpod/launch", json=launch_body(queued), headers=SPEND)
    assert r.status_code == 409 and "not waiting" in r.json()["detail"] and world.spawned == []


def test_only_one_launch_at_a_time(client, world, queued, ready):  # noqa: F811
    other = new_run(client, ready)["id"]
    assert client.post("/runpod/launch", json=launch_body(queued), headers=SPEND).status_code == 202
    r = client.post("/runpod/launch", json=launch_body(other), headers=SPEND)
    assert r.status_code == 409 and "already running" in r.json()["detail"] and len(world.spawned) == 1


def test_a_new_launch_is_allowed_once_the_last_one_ended(client, world, queued):
    assert client.post("/runpod/launch", json=launch_body(queued), headers=SPEND).status_code == 202
    world.next_proc.finish(1)
    for _ in range(100):                                                                                 # the watcher thread records the exit
        if client.get("/runpod/launch").json()["state"] == "finished":
            break
        time.sleep(0.02)
    world.next_proc = FakeProc(pid=5555)
    assert client.post("/runpod/launch", json=launch_body(queued), headers=SPEND).status_code == 202 and len(world.spawned) == 2


def test_the_launch_page_state_follows_the_process(client, world, queued):
    assert client.get("/runpod/launch").json() == {"state": "none", "lines": []}
    client.post("/runpod/launch", json=launch_body(queued), headers=SPEND)
    s = client.get("/runpod/launch").json()
    assert s["state"] == "running" and s["exit_code"] is None and s["stopping"] is False and s["lost_track"] is False
    world.running.clear()                                                                                # the backend restarted: nobody saw the exit code
    s = client.get("/runpod/launch").json()
    assert s["state"] == "finished" and s["lost_track"] is True and s["exit_code"] is None
    assert any("Tunnel:" in line for line in s["lines"])                                                 # the log is still readable


def test_the_exit_code_is_recorded_when_the_launcher_ends(client, world, queued):
    world.next_proc = FakeProc(exit_code=0)
    client.post("/runpod/launch", json=launch_body(queued), headers=SPEND)
    for _ in range(100):
        s = client.get("/runpod/launch").json()
        if s["exit_code"] is not None:
            break
        time.sleep(0.02)
    assert s["state"] == "finished" and s["exit_code"] == 0 and s["lost_track"] is False


# ---------- stopping ----------

def test_stop_sends_ctrl_c_so_the_launcher_deletes_the_pod_itself(client, world, queued):
    client.post("/runpod/launch", json=launch_body(queued), headers=SPEND)
    assert client.post("/runpod/launch/stop").status_code == 403                                         # the app header again
    r = client.post("/runpod/launch/stop", headers=SPEND)
    assert r.status_code == 200 and r.json()["stopping"] is True
    assert world.killed == [(4242, signal.SIGINT)]                                                       # SIGINT: only that path cleans up


def test_stop_with_nothing_running_is_refused_and_signals_nobody(client, world):
    assert client.post("/runpod/launch/stop", headers=SPEND).status_code == 409
    assert world.killed == []


# ---------- deleting a pod that was left running ----------

def test_a_stray_pod_of_this_project_can_be_deleted_but_never_someone_elses(client, world):
    world.api.pods = {"p1": {"id": "p1", "name": "mt-1024-1000", "desiredStatus": "RUNNING"},
                      "p2": {"id": "p2", "name": "someone-elses", "desiredStatus": "RUNNING"}}
    assert client.post("/runpod/pods/p1/delete").status_code == 403                                      # the app header
    assert client.post("/runpod/pods/p2/delete", headers=SPEND).status_code == 404
    assert world.api.deleted == []
    assert client.post("/runpod/pods/p1/delete", headers=SPEND).json() == {"deleted": True}
    assert world.api.deleted == ["p1"] and "p2" in world.api.pods


def test_a_pod_that_cannot_be_deleted_is_reported(client, world, monkeypatch):
    monkeypatch.setattr(cli, "DELETE_TRIES", 1)
    world.api.undeletable = True
    world.api.pods = {"p1": {"id": "p1", "name": "mt-1024-1000", "desiredStatus": "RUNNING"}}
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    r = client.post("/runpod/pods/p1/delete", headers=SPEND)
    assert r.status_code == 502 and "console.runpod.io" in r.json()["detail"]


# ---------- the process helpers, for real ----------

def test_the_launcher_runs_detached_from_the_backend_in_the_backend_folder(monkeypatch, tmp_path):
    """Detached = it survives a backend restart (the pod is still deleted at its end); the folder = it finds backend/.env and scripts/."""
    seen = {}
    monkeypatch.setattr(ui.subprocess, "Popen", lambda cmd, **kw: seen.update(cmd=cmd, **kw) or FakeProc())
    with open(tmp_path / "log", "ab") as log:
        ui.spawn(["python", "-m", "scripts.runpod"], log)
    assert seen["start_new_session"] is True and seen["cwd"] == BACKEND and seen["stdin"] == subprocess.DEVNULL
    assert seen["stderr"] == subprocess.STDOUT and seen["stdout"] is log



def test_alive_means_the_launcher_and_not_any_process_with_that_pid():
    assert ui.alive(os.getpid()) is False                                                                # this test process is not a launcher
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "scripts.runpod"])
    try:
        assert ui.alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait()
    assert ui.alive(proc.pid) is False


def test_tail_returns_the_last_lines_and_survives_a_missing_file(tmp_path):
    f = tmp_path / "log"
    f.write_text("\n".join(f"line {i}" for i in range(500)) + "\n")
    got = ui.tail(str(f), 3)
    assert got == ["line 497", "line 498", "line 499"]
    assert ui.tail(str(tmp_path / "nope")) == []


# ---------- the launcher options the UI relies on ----------

def test_the_launcher_trains_the_run_it_was_told_to_not_the_oldest(client, configured, queued, ready):  # noqa: F811
    second = new_run(client, ready)["id"]
    h = Harness(client)
    assert cli.main(["launch", "--run", str(second)], deps=h.deps) == 0
    assert f"Run {second} " in h.said and f"Run {queued} " not in h.said


def test_the_launcher_refuses_a_run_that_is_not_waiting(client, configured, queued, lan):  # noqa: F811
    from .test_training import claim
    assert claim(lan).status_code == 200
    h = Harness(client)
    assert cli.main(["launch", "--yes", "--run", str(queued)], deps=h.deps) == 1
    assert "not waiting for a GPU" in h.said and h.api.created == [] and h.tunnels_made == []
    h = Harness(client)
    assert cli.main(["launch", "--yes", "--run", "999"], deps=h.deps) == 1
    assert "not waiting for a GPU" in h.said and h.api.created == []


def test_the_launcher_refuses_a_price_above_what_was_confirmed(client, configured, queued):  # noqa: F811
    h = Harness(client)                                                                                  # worst case is 3 h x $0.50 = $1.50
    assert cli.main(["launch", "--yes", "--max-cost", "1.00"], deps=h.deps) == 1
    assert "price changed" in h.said and "$1.50" in h.said and "$1.00" in h.said
    assert h.api.created == [] and h.tunnels_made == []                                                  # refused before a tunnel, a pod or an upload
    h = Harness(client)
    assert cli.main(["launch", "--max-cost", "1.50"], deps=h.deps) == 0                                  # exactly what was confirmed is fine
    assert "Dry run" in h.said


STOP_CHILD = ("import sys, time\nsys.path.insert(0, {backend!r})\nfrom scripts import runpod\nrunpod.install_stop_handlers()\n"
              "print('ready', flush=True)\ntry:\n    time.sleep(30)\nexcept KeyboardInterrupt:\n    print('interrupted', flush=True)\n")


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_ctrl_c_and_terminate_both_take_the_careful_path_even_if_sigint_was_inherited_as_ignored(sig):
    proc = subprocess.Popen([sys.executable, "-c", STOP_CHILD.format(backend=str(BACKEND))], stdout=subprocess.PIPE, text=True, cwd=BACKEND,
                            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))            # what a background shell can hand down
    try:
        assert proc.stdout.readline().strip() == "ready"
        proc.send_signal(sig)
        assert "interrupted" in proc.communicate(timeout=15)[0]
    finally:
        proc.kill()
