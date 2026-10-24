"""The always-on Pi worker: the setup command the page copies, the files the Pi downloads, and the agent's service mode."""

import io
import logging
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import training as training_api
from app.config import settings
from app.main import app

from .helpers import from_ip

AGENTS = Path(__file__).resolve().parents[2] / "agents"
sys.path.insert(0, str(AGENTS))

import pi_agent  # noqa: E402
from common import ApiError  # noqa: E402

TOKEN = "pi-setup-token-xyz"


@pytest.fixture()
def with_token(monkeypatch):
    monkeypatch.setattr(settings, "worker_token", TOKEN)
    monkeypatch.setattr(training_api, "lan_hostname", lambda: "test-mac.local")


# ---------- the setup command ----------

def test_setup_command_carries_the_address_and_token_and_installs_from_the_backend(client, with_token):
    r = client.get("/pi-setup")
    assert r.status_code == 200
    body = r.json()
    assert body["server"] == "http://test-mac.local:8000"   # a name, not an IP: it survives a new DHCP address
    cmd = body["command"]
    assert f"T={TOKEN}" in cmd and "$T" in cmd               # the token is set once and used by variable
    assert "http://test-mac.local:8000/workers/pi-agent" in cmd and "bash pi_install.sh" in cmd
    assert shlex.split(cmd)[0] == f"T={TOKEN}"                # parses as the shell would read it


def test_setup_command_quotes_odd_tokens_and_runs_in_a_shell(client, monkeypatch, tmp_path):
    token = f"a b;$(touch {tmp_path}/pwned)'"
    monkeypatch.setattr(settings, "worker_token", token)
    monkeypatch.setattr(training_api, "lan_hostname", lambda: "test-mac.local")
    cmd = client.get("/pi-setup").json()["command"]
    # Evaluate just the assignment part in a real shell: the token must come out exactly as it went in.
    out = subprocess.run(["bash", "-c", cmd.split(" && ")[0] + '; printf %s "$T"'], capture_output=True, text=True).stdout
    assert out == token and not (tmp_path / "pwned").exists()


def test_setup_command_needs_a_worker_token(client, monkeypatch):
    monkeypatch.setattr(settings, "worker_token", None)
    assert client.get("/pi-setup").status_code == 503


def test_setup_command_is_never_served_off_this_machine(client, with_token):
    """It contains the token: another machine, or anything behind a tunnel (it connects from loopback), must get a 403."""
    assert TestClient(from_ip(app, "192.168.1.50")).get("/pi-setup").status_code == 403
    assert client.get("/pi-setup", headers={"X-Forwarded-For": "1.2.3.4"}).status_code == 403
    assert client.get("/pi-setup", headers={"CF-Connecting-IP": "1.2.3.4"}).status_code == 403


def test_lan_hostname_is_a_dot_local_name(monkeypatch):
    monkeypatch.setattr(training_api.sys, "platform", "linux")
    monkeypatch.setattr(training_api.socket, "gethostname", lambda: "box")
    assert training_api.lan_hostname() == "box.local"
    monkeypatch.setattr(training_api.socket, "gethostname", lambda: "box.example.org")
    assert training_api.lan_hostname() == "box.example.org"


# ---------- what the Pi downloads ----------

def test_pi_bundle_has_the_agent_and_installer_and_needs_the_token(client, with_token):
    lan = TestClient(from_ip(app, "192.168.1.60"))
    assert lan.get("/workers/pi-agent").status_code == 401
    r = lan.get("/workers/pi-agent", headers={"X-Worker-Token": TOKEN, "X-Worker-Name": "pi"})
    assert r.status_code == 200 and r.headers["content-type"] == "application/gzip"
    with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
        assert sorted(m.name for m in tar.getmembers()) == ["common.py", "pi_agent.py", "pi_display.py", "pi_install.sh", "pi_viewer.py"]   # nothing else leaves the Mac
        assert tar.extractfile("pi_install.sh").read() == (AGENTS / "pi_install.sh").read_bytes()
        assert tar.extractfile("pi_agent.py").read() == (AGENTS / "pi_agent.py").read_bytes()


def test_installer_script_is_valid_bash_and_keeps_the_token_out_of_the_unit_file():
    script = (AGENTS / "pi_install.sh").read_text()
    assert subprocess.run(["bash", "-n", str(AGENTS / "pi_install.sh")]).returncode == 0
    unit = script[script.index("[Unit]"):script.index("EOF\n\nsudo systemctl")]
    assert "MT_WORKER_TOKEN" not in unit and "EnvironmentFile=" in unit    # the secret is only in the 0600 env file
    assert "umask 077" in script                                            # ... which only the user can read
    assert "--keep-trying" in unit and "--display" in unit and "Restart=always" in unit and "network-online.target" in unit


# ---------- the agent's service mode ----------

class FakeApi:
    def __init__(self, ping_failures=0, claim_failures=0):
        self.ping_failures, self.claim_failures, self.claims = ping_failures, claim_failures, 0

    def ping(self):
        if self.ping_failures:
            self.ping_failures -= 1
            raise ApiError(0, "can't reach the Mac")

    def claim(self, kind):
        self.claims += 1
        if self.claim_failures:
            self.claim_failures -= 1
            raise ApiError(0, "can't reach the Mac")
        return None


class Stop(Exception):
    pass


def test_wait_for_backend_retries_and_complains_once(monkeypatch, caplog):
    sleeps = []
    monkeypatch.setattr(pi_agent.time, "sleep", lambda s: sleeps.append(s))
    with caplog.at_level(logging.INFO, logger="agent"):
        pi_agent.wait_for_backend(FakeApi(ping_failures=3), retry_every=7)
    assert sleeps == [7, 7, 7]
    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 1          # not one line per try
    assert any("reachable again" in r.getMessage() for r in caplog.records)


def run_service_loop(monkeypatch, api, stop_after_sleeps, argv):
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= stop_after_sleeps:
            raise Stop

    monkeypatch.setenv("MT_WORKER_TOKEN", "x")
    monkeypatch.setattr(pi_agent, "Api", lambda *a, **k: api)
    monkeypatch.setattr(pi_agent.shutil, "which", lambda c: "/usr/bin/imx500-package")
    monkeypatch.setattr(pi_agent.time, "sleep", fake_sleep)
    with pytest.raises(Stop):
        pi_agent.main(["--server", "http://mac.local:8000", *argv])
    return sleeps


def test_service_mode_waits_quietly_through_an_outage_then_polls_normally(monkeypatch, caplog):
    api = FakeApi(claim_failures=3)
    with caplog.at_level(logging.INFO, logger="agent"):
        sleeps = run_service_loop(monkeypatch, api, 5, ["--keep-trying", "--poll", "5"])
    assert sleeps == [30.0, 30.0, 30.0, 5.0, 5.0]                        # slow while the Mac is away, normal once it is back
    assert [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING] == ["claim failed: can't reach the Mac"]
    assert any("reachable again" in r.getMessage() for r in caplog.records)


def test_without_keep_trying_the_old_behaviour_is_unchanged(monkeypatch, caplog):
    api = FakeApi(claim_failures=3)
    with caplog.at_level(logging.INFO, logger="agent"):
        sleeps = run_service_loop(monkeypatch, api, 4, ["--poll", "5"])
    assert sleeps == [5.0, 5.0, 5.0, 5.0]
    assert [r.levelno for r in caplog.records].count(logging.WARNING) == 3   # every failed poll is logged, as before


def test_service_mode_does_not_exit_when_the_backend_is_down_at_boot(monkeypatch):
    api = FakeApi(ping_failures=2)
    sleeps = run_service_loop(monkeypatch, api, 4, ["--keep-trying", "--poll", "5"])
    assert sleeps[:2] == [30.0, 30.0] and api.ping_failures == 0 and api.claims >= 1
