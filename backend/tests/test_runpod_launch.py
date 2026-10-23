"""The RunPod launcher: everything that can spend money or leave a pod running, tested with fakes (no network, no pods)."""

import os
import stat
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from app import runpod_client as rp
from app.config import settings
from app.runpod_pod import DEFAULT_GPUS, IMAGE, POD_PREFIX, build_pod_spec, start_script
from scripts import runpod as cli

from .test_training import TOKEN, claim, expire_lease, finish_gpu, hdr, lan, new_run, ready  # noqa: F401 - fixtures

KEY = "rpa_SECRET-KEY-123"
TUNNEL = "https://quiet-fox-1234.trycloudflare.com"

# Copied from RunPod's OpenAPI spec (https://rest.runpod.io/v1/openapi.json, 2026-10-22): the fields POST /pods accepts.
POD_CREATE_KEYS = {
    "allowedCudaVersions", "cloudType", "computeType", "containerDiskInGb", "containerRegistryAuthId", "countryCodes",
    "cpuFlavorIds", "cpuFlavorPriority", "dataCenterIds", "dataCenterPriority", "dockerEntrypoint", "dockerStartCmd", "env",
    "globalNetworking", "gpuCount", "gpuTypeIds", "gpuTypePriority", "imageName", "interruptible", "locked",
    "minDiskBandwidthMBps", "minDownloadMbps", "minRAMPerGPU", "minUploadMbps", "minVCPUPerGPU", "name", "networkVolumeId",
    "ports", "supportPublicIp", "templateId", "vcpuCount", "volumeInGb", "volumeMountPath"}
GPU_TYPE_IDS = {
    "NVIDIA A40", "NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA L4", "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX A5000"}  # the subset of the spec's list this project names


# ---------- the pod spec ----------

def test_the_pod_spec_uses_only_fields_and_values_runpod_documents():
    spec = build_pod_spec(name="mt-1022-0900", server_url=TUNNEL, worker_token="tok", gpu_ids=DEFAULT_GPUS)
    assert set(spec) <= POD_CREATE_KEYS
    assert set(spec["gpuTypeIds"]) <= GPU_TYPE_IDS and set(DEFAULT_GPUS) <= GPU_TYPE_IDS
    assert "allowedCudaVersions" not in spec                              # asking for CUDA 12.8+ hosts left RunPod with no instances
    assert spec["env"]["MT_PATIENCE"] == "900"                            # a pod gives up on an unreachable backend after 15 minutes
    assert spec["cloudType"] in ("SECURE", "COMMUNITY") and spec["computeType"] == "GPU" and spec["gpuCount"] == 1
    assert all(isinstance(v, str) for v in spec["env"].values())
    assert spec["interruptible"] is False and spec["ports"] == []        # no spot pod, nothing exposed
    assert spec["imageName"] == IMAGE and spec["gpuTypePriority"] == "custom"


def test_the_pod_is_told_where_and_how_long_and_runs_the_start_script_inline():
    spec = build_pod_spec(name="mt-x", server_url=TUNNEL, worker_token="tok", gpu_ids=["NVIDIA RTX A4000"], max_hours=2.5)
    assert spec["env"]["MT_SERVER"] == TUNNEL and spec["env"]["MT_WORKER_TOKEN"] == "tok"
    assert spec["env"]["MT_MAX_SECONDS"] == "9000" and spec["env"]["MT_WORKER_NAME"] == "runpod-mt-x"
    assert spec["dockerEntrypoint"] == ["bash", "-c"] and spec["dockerStartCmd"] == [start_script()]
    with pytest.raises(ValueError):
        build_pod_spec(name="my-own-pod", server_url=TUNNEL, worker_token="tok", gpu_ids=[])      # never a name `stop --all` would miss


# ---------- fakes ----------

class FakeApi:
    def __init__(self, balance=5.0, cost=0.25, create_error=None, lose_answer=False, undeletable=False, get_errors=0, blind=False):
        self.balance, self.cost, self.create_error, self.lose_answer, self.undeletable = balance, cost, create_error, lose_answer, undeletable
        self.get_errors, self.blind = get_errors, blind  # RunPod unreachable: for the first N looks, or until a delete is tried
        self.pods, self.created, self.deleted = {}, [], []
        self.gpu_rows = [{"id": g, "name": g, "memory_gb": 16, "secure_price": p, "community_price": p / 2, "stock": "High"}
                         for g, p in (("NVIDIA RTX A4000", 0.25), ("NVIDIA RTX A5000", 0.27), ("NVIDIA GeForce RTX 3090", 0.50))]

    def account(self):
        return {"balance": self.balance, "spend_per_hr": 0, "spend_limit_per_hr": 80, "pod_count": len(self.pods)}

    def gpus(self):
        return self.gpu_rows

    def list_pods(self):
        return list(self.pods.values())

    def get_pod(self, pod_id):
        if not self.deleted and (self.blind or self.get_errors > 0):
            self.get_errors -= 1
            raise rp.RunPodError("RunPod request failed (ConnectError)")
        return self.pods.get(pod_id)

    def create_pod(self, spec):
        self.created.append(spec)
        if self.lose_answer:  # RunPod made the pod, but the answer never arrived
            self.pods["pod-lost"] = {"id": "pod-lost", "name": spec["name"], "desiredStatus": "RUNNING"}
            raise rp.RunPodError("RunPod request failed (ReadTimeout)")
        if self.create_error:
            raise rp.RunPodError(self.create_error)
        self.pods["pod-1"] = {"id": "pod-1", "name": spec["name"], "desiredStatus": "RUNNING", "costPerHr": self.cost}
        return dict(self.pods["pod-1"])

    def delete_pod(self, pod_id):
        self.deleted.append(pod_id)
        if self.undeletable:
            raise rp.RunPodError("RunPod HTTP 500: try later")
        return self.pods.pop(pod_id, None) is not None


class FakeTunnel:
    def __init__(self, url=TUNNEL, fail=None):
        self.url, self.fail, self.stopped = url, fail, False

    def start(self):
        if self.fail:
            raise cli.Refusal(self.fail)
        return self.url

    def stop(self):
        self.stopped = True


class FakeAwake:
    terminated = False

    def terminate(self):
        self.terminated = True


def http(health=403, projects=403, ping=200, hub=200, downloads=200):
    def handler(request: httpx.Request):
        host, path = request.url.host, request.url.path
        if host == "hub.docker.com":
            return httpx.Response(hub, json={})
        if host in ("download.pytorch.org", "astral.sh", "pypi.org"):          # what the pod downloads while it sets up
            return httpx.Response(downloads)
        status = {"/health": health, "/projects": projects, "/workers/ping": ping}.get(path, 404)
        if status == 403:  # the backend's own refusal, which is what the self-test looks for
            return httpx.Response(403, json={"detail": "Only this machine may use this endpoint; workers use /workers/*."})
        if status == 200:
            return httpx.Response(200, json=[] if path == "/projects" else {"ok": True})
        return httpx.Response(status, json={"detail": "no"})
    return httpx.Client(transport=httpx.MockTransport(handler))


class Harness:
    """Fake clock and fake everything, with a hook that runs on every sleep so a test can move the run along."""

    def __init__(self, client, api=None, tunnel=None, http_client=None, on_sleep=None):
        self.t, self.on_sleep, self.messages = 1000.0, on_sleep, []
        self.api, self.tunnels_made, self.awake = api or FakeApi(), [], FakeAwake()
        self.tunnel_objs = list(tunnel) if isinstance(tunnel, (list, tuple)) else [tunnel or FakeTunnel()]
        self.tunnel_obj = self.tunnel_objs[0]
        self.deps = cli.Deps(self.api, http_client or http(), client.session_factory, tunnel=self._tunnel, keep_awake=lambda: self.awake,
                             sleep=self._sleep, now=lambda: self.t, say=self.messages.append)

    def _tunnel(self, url):
        self.tunnels_made.append(url)
        return self.tunnel_objs[min(len(self.tunnels_made), len(self.tunnel_objs)) - 1]

    def _sleep(self, seconds):
        self.t += seconds
        if self.on_sleep:
            self.on_sleep(self)

    @property
    def said(self):
        return "\n".join(self.messages)


@pytest.fixture()
def configured(monkeypatch, lan):
    monkeypatch.setattr(settings, "runpod_api_key", KEY)
    monkeypatch.setattr(settings, "worker_token", TOKEN)


@pytest.fixture()
def queued(client, ready):  # noqa: F811
    return new_run(client, ready)["id"]


# ---------- checks before anything is started ----------

def test_a_dry_run_checks_everything_and_starts_nothing(client, configured, queued):
    h = Harness(client)
    assert cli.main(["launch"], deps=h.deps) == 0
    assert h.api.created == [] and h.tunnels_made == []                  # no pod, no tunnel
    assert "Dry run" in h.said and f"Run {queued}" in h.said
    assert "$1.50" in h.said                                                # worst case: 3 h x the dearest GPU it may pick ($0.50)
    assert "worker token" in h.said and "Cloudflare" in h.said            # says what leaves the Mac


@pytest.mark.parametrize("setup, expect", [
    (lambda m, s: m.setattr(s, "runpod_api_key", None), "MT_RUNPOD_API_KEY"),
    (lambda m, s: m.setattr(s, "worker_token", None), "MT_WORKER_TOKEN"),
])
def test_it_refuses_without_the_key_or_the_worker_token(client, configured, queued, monkeypatch, setup, expect):
    setup(monkeypatch, settings)
    h = Harness(client)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert expect in h.said and h.api.created == [] and h.tunnels_made == []


def test_it_refuses_when_no_run_is_waiting(client, configured):
    h = Harness(client)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "No training run is waiting" in h.said and h.api.created == []


def test_it_refuses_when_the_balance_cannot_cover_the_worst_case(client, configured, queued):
    h = Harness(client, api=FakeApi(balance=0.0))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "balance is $0.00" in h.said and "billing" in h.said and h.api.created == []
    h = Harness(client, api=FakeApi(balance=1.0))                                       # 3 h x $0.50/h = $1.50 > $1.00
    assert cli.main(["launch", "--yes", "--gpu", "NVIDIA GeForce RTX 3090"], deps=h.deps) == 1
    assert "up to $1.50" in h.said and h.api.created == []
    h = Harness(client, api=FakeApi(balance=1.0))                                       # 1 h x $0.50/h = $0.50 fits
    assert cli.main(["launch", "--max-hours", "1", "--gpu", "NVIDIA GeForce RTX 3090"], deps=h.deps) == 0
    assert "Dry run" in h.said


@pytest.mark.parametrize("argv, expect", [
    (["--max-hours", "0"], "--max-hours"), (["--max-hours", "13"], "--max-hours"),
    (["--gpu", "NVIDIA H100 PCIe"], "Not offered"),
])
def test_it_refuses_bad_limits_and_unknown_gpus(client, configured, queued, argv, expect):
    h = Harness(client)
    assert cli.main(["launch", "--yes", *argv], deps=h.deps) == 1
    assert expect in h.said and h.api.created == []


def test_it_refuses_when_the_pinned_image_is_gone(client, configured, queued):
    h = Harness(client, http_client=http(hub=404))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "no longer exists" in h.said and h.api.created == []


def test_the_tunnel_is_checked_and_a_leaky_one_stops_everything(client, configured, queued):
    h = Harness(client, http_client=http(health=200, projects=200))                  # the whole API answers through the tunnel
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "SECURITY" in h.said and h.api.created == []                                # no pod, and no waiting around to retry
    assert h.tunnel_obj.stopped and h.awake.terminated


def test_a_tunnel_that_never_works_is_given_up_on(client, configured, queued):
    h = Harness(client, http_client=http(ping=502))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "doesn't work" in h.said and h.api.created == [] and h.tunnel_obj.stopped


def test_a_machine_whose_dns_cannot_find_the_new_tunnel_name_still_gets_the_security_check(client, configured, queued):
    # Seen on the real Mac: its ISP resolvers never learn a new trycloudflare.com name, Cloudflare's own resolver does.
    seen = []

    def handler(request: httpx.Request):
        host = request.url.host
        seen.append((host, request.url.path, request.headers.get("host"), request.extensions.get("sni_hostname")))
        if host == "quiet-fox-1234.trycloudflare.com":
            raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
        if host == "1.1.1.1":
            return httpx.Response(200, json={"Answer": [{"type": 5, "data": "alias."}, {"type": 1, "data": "104.16.230.132"}]})
        if host == "104.16.230.132":  # the tunnel, reached by address with the real name presented
            status = {"/health": 403, "/projects": 403, "/workers/ping": 200}[request.url.path]
            return httpx.Response(status, json={"detail": "Only this machine may use this endpoint; workers use /workers/*."} if status == 403 else {"ok": True})
        return httpx.Response(404)

    h = Harness(client, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    cli.selftest(TUNNEL, TOKEN, h.deps)                                                  # passes, no Refusal
    via_ip = [s for s in seen if s[0] == "104.16.230.132"]
    assert [s[1] for s in via_ip] == ["/health", "/projects", "/workers/ping"]
    assert all(s[2] == "quiet-fox-1234.trycloudflare.com" and s[3] == "quiet-fox-1234.trycloudflare.com" for s in via_ip)

    leaky = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}) if r.url.host == "104.16.230.132"
                                                       else httpx.Response(200, json={"Answer": [{"type": 1, "data": "104.16.230.132"}]}) if r.url.host == "1.1.1.1"
                                                       else (_ for _ in ()).throw(httpx.ConnectError("no dns"))))
    with pytest.raises(cli.Refusal, match="SECURITY"):                                   # a leak is caught on this path too
        cli.selftest(TUNNEL, TOKEN, Harness(client, http_client=leaky).deps)


def test_when_no_resolver_knows_the_tunnel_the_launcher_gives_up_without_a_pod(client, configured, queued):
    def handler(request: httpx.Request):
        if request.url.host == "1.1.1.1":
            return httpx.Response(200, json={"Status": 3})                                # NXDOMAIN: no Answer at all
        raise httpx.ConnectError("no dns")

    h = Harness(client, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "doesn't work" in h.said and h.api.created == [] and h.tunnel_obj.stopped


def test_a_placeholder_page_is_not_mistaken_for_a_leak_but_is_not_a_pass_either(client, configured, queued):
    placeholder = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>tunnel coming up</html>")))
    with pytest.raises(cli.TunnelNotReady, match="placeholder page"):                  # a 200 that is not this API: keep waiting, then give up
        cli.selftest(TUNNEL, TOKEN, Harness(client, http_client=placeholder).deps)


def test_only_this_apis_own_answer_counts_as_a_leak(client, configured, queued):
    for body in ({"ok": True}, [{"id": 1}]):                                          # /health's answer, and a project list
        leaky = httpx.Client(transport=httpx.MockTransport(lambda r, b=body: httpx.Response(200, json=b)))
        with pytest.raises(cli.TunnelLeaks):
            cli.selftest(TUNNEL, TOKEN, Harness(client, http_client=leaky).deps)


def test_a_tunnel_that_cannot_be_reached_is_replaced_by_a_fresh_one(client, configured, queued, lan):
    dead, alive = "https://dead-name-1.trycloudflare.com", "https://alive-name-2.trycloudflare.com"

    def handler(request: httpx.Request):
        if request.url.host in ("dead-name-1.trycloudflare.com", "1.1.1.1"):
            raise httpx.ConnectError("no dns for this one yet")
        if request.url.host in ("hub.docker.com", "download.pytorch.org", "astral.sh", "pypi.org"):
            return httpx.Response(200, json={})
        status = {"/health": 403, "/projects": 403, "/workers/ping": 200}.get(request.url.path, 404)
        return httpx.Response(status, json={"detail": "Only this machine may use this endpoint; workers use /workers/*."} if status == 403 else {"ok": True})

    tunnels = [FakeTunnel(url=dead), FakeTunnel(url=alive)]
    h = Harness(client, tunnel=tunnels, http_client=httpx.Client(transport=httpx.MockTransport(handler)), on_sleep=moves(lan, queued, {1: "claim", 2: "finish"}))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 0
    assert len(h.tunnels_made) == 2 and tunnels[0].stopped and tunnels[1].stopped    # the first was closed before the second opened
    assert h.api.created[0]["env"]["MT_SERVER"] == alive                              # the pod got the one that works
    assert "fresh tunnel" in h.said


def test_it_gives_up_after_three_unreachable_tunnels_without_starting_a_pod(client, configured, queued):
    def handler(request: httpx.Request):
        if request.url.host == "hub.docker.com":
            return httpx.Response(200, json={})
        raise httpx.ConnectError("no dns")

    tunnels = [FakeTunnel(), FakeTunnel(), FakeTunnel(), FakeTunnel()]
    h = Harness(client, tunnel=tunnels, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert len(h.tunnels_made) == cli.TUNNEL_TRIES and all(t.stopped for t in tunnels[:3]) and h.api.created == []
    assert "doesn't work" in h.said


def test_a_leaky_tunnel_is_never_retried(client, configured, queued):
    tunnels = [FakeTunnel(), FakeTunnel()]
    h = Harness(client, tunnel=tunnels, http_client=http(health=200, projects=200))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert len(h.tunnels_made) == 1 and "SECURITY" in h.said and h.api.created == []   # one look was enough


def test_a_missing_cloudflared_is_explained(client, configured, queued):
    h = Harness(client, tunnel=FakeTunnel(fail="cloudflared isn't installed. Install it with: brew install cloudflared"))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "brew install cloudflared" in h.said and h.api.created == []


# ---------- a run, and every way it can end ----------

def moves(lan, run_id, steps):
    """An on_sleep hook that plays the pod's worker: claim the run on the first sleep, finish it on the second."""
    count = {"n": 0}

    def hook(_h):
        count["n"] += 1
        step = steps.get(count["n"])
        if step == "claim":
            assert claim(lan, name="runpod-pod").status_code == 200
        elif step == "finish":
            finish_gpu(lan, run_id, worker="runpod-pod")
    return hook


def test_a_normal_run_ends_with_the_pod_deleted_and_the_tunnel_closed(client, configured, queued, lan):
    h = Harness(client, on_sleep=moves(lan, queued, {1: "claim", 2: "finish"}))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 0
    spec = h.api.created[0]
    assert spec["name"].startswith(POD_PREFIX) and spec["env"]["MT_SERVER"] == TUNNEL and spec["env"]["MT_WORKER_TOKEN"] == TOKEN
    assert h.api.deleted == ["pod-1"] and h.api.pods == {}
    assert h.tunnel_obj.stopped and h.awake.terminated
    assert "The GPU part of the run is over" in h.said and "pod deleted" in h.said


def test_a_server_address_can_be_given_instead_of_a_tunnel(client, configured, queued, lan):
    h = Harness(client, on_sleep=moves(lan, queued, {1: "claim", 2: "finish"}))
    assert cli.main(["launch", "--yes", "--server", "https://quiet-fox-1234.trycloudflare.com/"], deps=h.deps) == 0
    assert h.tunnels_made == [] and h.api.created[0]["env"]["MT_SERVER"] == TUNNEL      # no tunnel of ours, address tidied


def ctrl_c_once():
    """An on_sleep hook: the person presses Ctrl-C during the first wait (and not again while the pod is being deleted)."""
    pressed = []

    def hook(_h):
        if not pressed:
            pressed.append(1)
            raise KeyboardInterrupt
    return hook


def test_ctrl_c_deletes_the_pod(client, configured, queued):
    h = Harness(client, on_sleep=ctrl_c_once())
    assert cli.main(["launch", "--yes"], deps=h.deps) == 130
    assert h.api.deleted == ["pod-1"] and h.tunnel_obj.stopped and h.awake.terminated


def test_a_pod_that_never_picks_the_run_up_is_deleted_after_twenty_minutes(client, configured, queued):
    h = Harness(client)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert h.api.deleted == ["pod-1"] and "too slowly or got stuck" in h.said and "Logs" in h.said
    assert h.t - 1000 < cli.CLAIM_TIMEOUT + 3 * cli.POLL + 60                              # it did not wait forever


def test_a_pod_that_vanishes_with_its_run_unfinished_is_reported_as_a_failure(client, configured, queued, lan):
    """Run 4: the pod's own hour ran out in the middle of an upload; the launcher called that "ended by itself" and exited 0."""
    sleeps = []

    def hook(h):
        sleeps.append(1)
        if len(sleeps) == 1:
            assert claim(lan, name="runpod-pod").status_code == 200
        else:
            h.api.pods.clear()                                                             # killed by its time limit, say
    h = Harness(client, on_sleep=hook)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "did not finish" in h.said and "ended by itself" not in h.said and h.tunnel_obj.stopped
    assert "RunPod reports the pod as gone" in h.said and "already gone" in h.said        # so the next failure can say whether the pod deleted itself


def test_a_pod_that_vanishes_just_after_finishing_its_run_is_a_success(client, configured, queued, lan):
    claimed = []

    def hook(_h):
        if not claimed:
            claimed.append(1)
            assert claim(lan, name="runpod-pod").status_code == 200
    h = Harness(client, on_sleep=hook)
    look = h.api.get_pod

    def get_pod(pod_id):                                                                    # the run ends between the launcher's two looks
        if claimed and h.api.pods:
            finish_gpu(lan, queued, worker="runpod-pod")
            h.api.pods.clear()
        return look(pod_id)
    h.api.get_pod = get_pod
    assert cli.main(["launch", "--yes"], deps=h.deps) == 0
    assert "The GPU part of the run is over" in h.said and "did not finish" not in h.said


def test_the_time_limit_ends_a_run_that_never_finishes(client, configured, queued, lan):
    h = Harness(client, on_sleep=moves(lan, queued, {1: "claim"}))                          # claimed, then silence
    assert cli.main(["launch", "--yes", "--max-hours", "1"], deps=h.deps) == 1
    assert h.api.deleted == ["pod-1"] and "time limit" in h.said
    assert h.t - 1000 <= 3600 + cli.GRACE + cli.POLL + 60


def test_a_pod_that_costs_far_more_than_quoted_is_deleted_at_once(client, configured, queued):
    h = Harness(client, api=FakeApi(cost=3.0))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert h.api.deleted == ["pod-1"] and "much more" in h.said


def test_a_refused_create_leaves_nothing_behind(client, configured, queued):
    h = Harness(client, api=FakeApi(create_error="RunPod HTTP 400: no GPUs available"))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "no GPUs available" in h.said and h.api.pods == {} and h.tunnel_obj.stopped


def test_a_pod_whose_creation_answer_was_lost_is_found_by_name_and_deleted(client, configured, queued):
    h = Harness(client, api=FakeApi(lose_answer=True))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert h.api.deleted == ["pod-lost"] and h.api.pods == {}


def test_a_pod_that_cannot_be_deleted_is_reported_loudly(client, configured, queued):
    h = Harness(client, api=FakeApi(undeletable=True), on_sleep=ctrl_c_once())
    assert cli.main(["launch", "--yes"], deps=h.deps) == 130
    assert len(h.api.deleted) == cli.DELETE_TRIES and "STILL THERE AND CHARGING" in h.said and "POD NOT DELETED" in h.said


def test_a_short_loss_of_contact_with_runpod_does_not_end_a_working_run(client, configured, queued, lan):
    # The third real run: this Mac's connection blinked, one failed look at the pod ended the launcher, and the pod was deleted
    # in the middle of the IMX export.
    h = Harness(client, api=FakeApi(get_errors=5), on_sleep=moves(lan, queued, {1: "claim", 8: "finish"}))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 0
    assert "can't reach RunPod just now" in h.said and "reachable again" in h.said
    assert "The GPU part of the run is over" in h.said and h.api.deleted == ["pod-1"]       # deleted once, at the end, not at the blip


def test_a_long_loss_of_contact_ends_the_pod_unseen_rather_than_leaving_it_to_bill(client, configured, queued, lan):
    h = Harness(client, api=FakeApi(blind=True), on_sleep=moves(lan, queued, {1: "claim"}))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "unreachable for 10 minutes" in h.said and h.api.deleted == ["pod-1"] and h.tunnel_obj.stopped
    assert h.t - 1000 < cli.API_PATIENCE + 3 * cli.POLL + 60                                  # it waited out the patience, not forever


def test_a_run_left_in_progress_by_a_dead_pod_can_be_launched_again(client, configured, queued, lan):
    assert claim(lan, name="runpod-dead").status_code == 200                                  # a pod took the run, then vanished
    h = Harness(client)
    assert cli.main(["launch"], deps=h.deps) == 1 and "No training run is waiting" in h.said   # its lease has not run out yet
    expire_lease(client, queued)
    h = Harness(client)
    assert cli.main(["launch"], deps=h.deps) == 0 and f"Run {queued}" in h.said               # now it is released and can be retried
    assert client.get(f"/training-runs/{queued}").json()["stage"] == "queued"


# ---------- stop, status, gpus ----------

def test_stop_only_ever_touches_pods_this_project_started(client, configured):
    api = FakeApi()
    api.pods = {"a": {"id": "a", "name": "mt-1022-0900", "desiredStatus": "RUNNING"},
                "b": {"id": "b", "name": "my-jupyter", "desiredStatus": "RUNNING"}}
    h = Harness(client, api=api)
    assert cli.main(["stop", "b"], deps=h.deps) == 1 and api.deleted == []                  # not ours: refused
    assert cli.main(["stop"], deps=h.deps) == 1 and api.deleted == []                       # must say which
    assert cli.main(["stop", "--all"], deps=h.deps) == 0
    assert api.deleted == ["a"] and list(api.pods) == ["b"]


def test_status_and_gpus_are_read_only(client, configured, queued):
    h = Harness(client)
    assert cli.main(["status"], deps=h.deps) == 0 and "Runs waiting:  1" in h.said and "$5.00" in h.said
    assert cli.main(["gpus"], deps=h.deps) == 0 and "NVIDIA RTX A4000" in h.said
    assert h.api.created == [] and h.api.deleted == []


# ---------- the tunnel process ----------

class FakeProc:
    def __init__(self, lines, alive=True):
        self.stdout, self.alive, self.terminated = iter(lines), alive, False

    def poll(self):
        return None if self.alive and not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


def test_the_tunnel_address_is_read_from_cloudflareds_output_and_the_process_is_stopped():
    proc, commands = FakeProc(["2026 INF Requesting new quick Tunnel\n", f"2026 INF |  {TUNNEL}  |\n", "more output\n"]), []
    t = cli.QuickTunnel("http://127.0.0.1:8000", popen=lambda args, **k: commands.append(args) or proc, which=lambda n: "/usr/bin/cloudflared")
    assert t.start() == TUNNEL
    t.stop()
    assert proc.terminated
    cmd = commands[0]
    assert cmd[cmd.index("--protocol") + 1] == "http2"                        # QUIC dropped big uploads on this Mac's connection


def test_the_tunnel_start_fails_clearly():
    with pytest.raises(cli.Refusal, match="brew install cloudflared"):
        cli.QuickTunnel("http://127.0.0.1:8000", which=lambda n: None).start()
    proc = FakeProc(["no address here\n"], alive=False)
    with pytest.raises(cli.Refusal, match="didn't report"):
        cli.QuickTunnel("http://127.0.0.1:8000", popen=lambda *a, **k: proc, which=lambda n: "/x", wait=1).start()
    assert proc.terminated or not proc.alive


# ---------- the script the pod runs, in a real bash with stubbed tools ----------

SCRIPT = Path(__file__).resolve().parents[2] / "agents" / "runpod_start.sh"


def run_script(tmp_path, apt_body, env=None, extra_stubs=None, timeout=60):
    stubs = tmp_path / "bin"
    stubs.mkdir()
    log = tmp_path / "calls.log"

    def stub(name, body):
        p = stubs / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)

    stub("apt-get", apt_body)
    stub("curl", f'echo "curl $*" >> {log}\nexit 0')
    # Stands in for GNU timeout, which the pod has and macOS doesn't. The watcher drops its output so it can't hold the pipe open.
    stub("timeout", "if [ \"$1\" = -k ]; then shift 2; fi\nsecs=$1; shift\n\"$@\" &\npid=$!\n"
                    "( exec >/dev/null 2>&1; sleep \"$secs\"; kill -TERM $pid ) &\nw=$!\nwait $pid; rc=$?\nkill $w 2>/dev/null\nexit $rc")
    for name, body in (extra_stubs or {}).items():
        stub(name, body.replace("LOG", str(log)))
    environ = {"PATH": f"{stubs}:/usr/bin:/bin", "MT_SERVER": TUNNEL, "MT_WORKER_TOKEN": "tok", "MT_WORKER_NAME": "runpod-mt-x",
               "MT_MAX_SECONDS": "30", "RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "rpa_pod_key", "MT_RUNPOD_REST": "https://rest.test/v1"}
    environ.update(env or {})
    started = time.monotonic()
    r = subprocess.run(["bash", str(SCRIPT)], env=environ, capture_output=True, text=True, timeout=timeout)
    return r, (log.read_text() if log.exists() else ""), time.monotonic() - started


DELETE_CALL = "curl -fsS -m 30 -X DELETE -H Authorization: Bearer rpa_pod_key https://rest.test/v1/pods/pod-123"


def test_the_pod_deletes_itself_when_setup_fails(tmp_path):
    r, calls, _ = run_script(tmp_path, "echo 'E: no network' >&2; exit 100")
    assert r.returncode != 0 and DELETE_CALL in calls
    assert "stopped with exit code" in r.stdout and "deleting this pod" in r.stdout


def test_the_pod_deletes_itself_when_the_time_limit_hits(tmp_path):
    r, calls, took = run_script(tmp_path, "exec sleep 8 >/dev/null 2>&1", env={"MT_MAX_SECONDS": "1"})   # exec: no leftover process holds the pipe
    assert DELETE_CALL in calls and r.returncode != 0
    assert took < 7                                                                       # the limit ended it, not the stuck step


def test_the_pod_prefers_runpodctl_when_it_exists(tmp_path):
    r, calls, _ = run_script(tmp_path, "exit 1", extra_stubs={"runpodctl": 'echo "runpodctl $*" >> LOG\nexit 0'})
    assert "runpodctl remove pod pod-123" in calls and "-X DELETE" not in calls


def test_a_pod_that_cannot_delete_itself_says_so_and_names_the_fallbacks(tmp_path):
    r, calls, _ = run_script(tmp_path, "exit 1", env={"RUNPOD_API_KEY": ""})
    assert "-X DELETE" not in calls and "COULD NOT DELETE THIS POD" in r.stdout and "stop --all" in r.stdout
    assert r.returncode != 0                                                              # the failure is not hidden behind the clean-up


def test_the_start_script_is_valid_bash_and_what_the_launcher_sends():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    assert start_script() == SCRIPT.read_text() and "trap end_pod EXIT" in start_script()
    assert os.access(SCRIPT, os.R_OK)


# ---------- the torch build is chosen from the host's driver ----------

def cuda_index(tmp_path, smi_line):
    """Run the start script's cuda_index function in real bash, with an nvidia-smi that prints `smi_line` (None = no nvidia-smi)."""
    import re

    stubs = tmp_path / "bin"
    stubs.mkdir()
    if smi_line is not None:
        smi = stubs / "nvidia-smi"
        smi.write_text(f"#!/bin/bash\necho '{smi_line}'\n")
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    func = re.search(r"^cuda_index\(\) \{.*?^\}", SCRIPT.read_text(), re.S | re.M).group(0)
    prog = 'say() { echo "[mt-pod] $*"; }\n' + func + "\ncuda_index"
    r = subprocess.run(["bash", "-c", prog], env={"PATH": f"{stubs}:/usr/bin:/bin"}, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def banner(cuda):
    return f"| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: {cuda}     |"


@pytest.mark.parametrize("cuda, expected", [("13.0", "cu128"), ("12.9", "cu128"), ("12.8", "cu128"), ("12.6", "cu126"),
                                            ("12.4", "cu126"), ("12.0", "cu126")])
def test_the_pod_installs_the_torch_build_that_fits_its_driver(tmp_path, cuda, expected):
    code, out, _ = cuda_index(tmp_path, banner(cuda))
    assert (code, out) == (0, expected)


@pytest.mark.parametrize("smi_line, why", [(banner("11.8"), "CUDA 11.8"), (None, "cannot read")])
def test_a_host_torch_cannot_use_fails_fast_with_a_reason(tmp_path, smi_line, why):
    code, out, err = cuda_index(tmp_path, smi_line)
    assert code != 0 and out == "" and why in err                                   # nothing for the caller to install


# ---------- the whole start script, start to finish, with stubbed tools ----------

UV_INSTALLER = r'''mkdir -p "$UV_UNMANAGED_INSTALL"
cat > "$UV_UNMANAGED_INSTALL/uv" <<'STUB'
#!/bin/bash
echo "uv $*" >> "$STUB_LOG"
eval "$UV_HOOK"
if [ "$1" = venv ]; then
  dir="${@: -1}"
  mkdir -p "$dir/bin"
  printf '#!/bin/bash\necho "python $*" >> "%s"\n%s\nexit 0\n' "$STUB_LOG" "$PYTHON_HOOK" > "$dir/bin/python"
  chmod +x "$dir/bin/python"
fi
# Installing the export tools puts the converter next to python, as the real imx500-converter wheel does.
if [ "$1" = pip ] && [[ "$*" == *imx500-converter* ]] && [ -z "$NO_IMXCONV" ]; then
  prev=""
  for a in "$@"; do
    if [ "$prev" = --python ]; then printf '#!/bin/bash\n' > "$(dirname "$a")/imxconv-pt"; chmod +x "$(dirname "$a")/imxconv-pt"; fi
    prev="$a"
  done
fi
STUB
chmod +x "$UV_UNMANAGED_INSTALL/uv"
'''


def run_whole_script(tmp_path, cuda="12.4", prep="1", import_check_fails=False, java="openjdk 17.0.12 2024-07-16", imxconv=True,
                     speed_bytes=None, uv_hook="", install_seconds=None, cgroup=None, python_hook=""):
    """Run agents/runpod_start.sh end to end in real bash. Everything it installs or downloads is a stub that logs its
    arguments, so the log shows what the pod would do, in order. (/opt is replaced by a temp folder through MT_ROOT.)"""
    import tarfile

    stubs, src = tmp_path / "bin", tmp_path / "agent-src"
    stubs.mkdir()
    src.mkdir()
    (src / "gpu_agent.py").write_text("# stub\n")
    (src / "requirements-gpu.txt").write_text("torch>=2.9,<2.12\nultralytics==8.4.183\n")
    installer = tmp_path / "uv-installer.sh"
    installer.write_text(UV_INSTALLER)
    log = tmp_path / "calls.log"

    def stub(name, body):
        p = stubs / name
        p.write_text("#!/bin/bash\n" + body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)

    stub("apt-get", f'echo "apt-get $*" >> {log}')
    stub("nvidia-smi", f"echo '| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: {cuda}     |'")
    stub("curl", f'echo "curl $*" >> {log}\ncase "$*" in\n'
                 f'  */workers/agent*) cd "$AGENT_SRC" && tar czf - gpu_agent.py requirements-gpu.txt ;;\n'
                 f'  *astral.sh*) cat "$UV_INSTALLER" ;;\n'
                 f'  *size_download*download.pytorch.org*) printf "%s" "${{SPEED_BYTES:-41943040}}" ;;\nesac\nexit 0')
    if install_seconds:  # a timeout that really times out (the plain stub below never does)
        stub("timeout", "if [ \"$1\" = -k ]; then shift 2; fi\nsecs=$1; shift\n\"$@\" &\npid=$!\n"
                        "( exec >/dev/null 2>&1; sleep \"$secs\"; kill -TERM $pid ) &\nw=$!\nwait $pid; rc=$?\nkill $w 2>/dev/null\nexit $rc")
    else:
        stub("timeout", "if [ \"$1\" = -k ]; then shift 2; fi\nshift\nexec \"$@\"")
    if java:
        stub("java", f"echo '{java}'")
    # What the venv's python does when asked to run an import check: it fails if the test says so.
    hook = 'case "$*" in *model_compression_toolkit*) exit 1 ;; esac' if import_check_fails else ""
    hook += "\n" + python_hook
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "MT_SERVER": TUNNEL, "MT_WORKER_TOKEN": "tok", "MT_WORKER_NAME": "runpod-mt-x",
           "MT_MAX_SECONDS": "60", "MT_ROOT": str(tmp_path / "opt"), "MT_PREP_IMX": prep, "STUB_LOG": str(log),
           "AGENT_SRC": str(src), "UV_INSTALLER": str(installer), "PYTHON_HOOK": hook, "NO_IMXCONV": "" if imxconv else "1",
           "RUNPOD_POD_ID": "pod-123", "RUNPOD_API_KEY": "rpa_pod_key", "MT_RUNPOD_REST": "https://rest.test/v1",
           "UV_HOOK": uv_hook}
    if cgroup is not None:
        env["MT_CGROUP"] = str(cgroup)
    if speed_bytes is not None:
        env["SPEED_BYTES"] = str(speed_bytes)
    if install_seconds:
        env["MT_INSTALL_SECONDS"] = str(install_seconds)
    r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    return r, [ln for ln in log.read_text().splitlines()] if log.exists() else []


def test_a_normal_pod_run_installs_the_export_tools_before_any_python_runs(tmp_path):
    r, calls = run_whole_script(tmp_path, cuda="12.4")
    assert r.returncode == 0, r.stdout + r.stderr
    tools = next(i for i, c in enumerate(calls) if "model-compression-toolkit>=2.4.1" in c)
    first_python = next(i for i, c in enumerate(calls) if c.startswith("python "))
    assert tools < first_python                       # the first real-pod failure: packages were swapped under a running python
    install = calls[tools]
    for want in ("torch==2.9.1", "torchvision==0.24.1", "edge-mdt-cl<1.1.0", "edge-mdt-tpc>=1.2.0", "pydantic<2.12", "imx500-converter[pt]>=3.17.3"):
        assert want in install
    order = [next(i for i, c in enumerate(calls) if key in c) for key in
             ("whl/cu126", "-r requirements-gpu.txt", "model-compression-toolkit>=2.4.1", "import matplotlib.figure",
              "import torch", "gpu_agent.py --once --name runpod-mt-x --device 0")]
    assert order == sorted(order)                     # torch for this driver, requirements, export tools, import check, GPU check, agent
    assert any("-X DELETE" in c and "/pods/pod-123" in c for c in calls)         # and then the pod deletes itself


def test_a_cuda_13_host_still_gets_the_cu128_torch_build(tmp_path):
    """A pod whose driver said CUDA 13 picked cu130 and sat in the install for 13 minutes; cu128 is what ever worked."""
    r, calls = run_whole_script(tmp_path, cuda="13.0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert any("whl/cu128" in c for c in calls) and not any("cu130" in c for c in calls)


def test_a_slow_route_to_pytorch_is_found_in_a_minute_and_the_pod_gives_up_before_installing(tmp_path):
    r, calls = run_whole_script(tmp_path, speed_bytes=3 * 1048576)             # only 3 of the 40 MB arrived in time
    assert r.returncode != 0 and "SLOW DOWNLOAD" in r.stdout
    assert not any(c.startswith("uv ") for c in calls) and not any("gpu_agent.py" in c for c in calls)   # nothing was installed
    assert any("-X DELETE" in c for c in calls)                                       # and the pod deleted itself


def test_the_speed_test_names_the_wheel_about_to_be_installed(tmp_path):
    r, calls = run_whole_script(tmp_path, cuda="12.4")
    test = next(c for c in calls if "size_download" in c)
    assert "-r 0-41943039" in test and "torch-2.9.1%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl" in test


def test_a_stuck_install_is_stopped_retried_once_and_then_given_up_on(tmp_path):
    hook = 'case "$*" in *"pip install"*"torch=="*"--index-url"*) exec sleep 20 ;; esac'      # the PyTorch download hangs
    r, calls = run_whole_script(tmp_path, uv_hook=hook, install_seconds=1)
    assert r.returncode != 0
    assert r.stdout.count("installing torch 2.9.1 (cu126), about 3 GB (try") == 2 and r.stdout.count("did not finish") == 2
    assert "SLOW DOWNLOAD" in r.stdout and not any("gpu_agent.py" in c for c in calls)
    assert any("-X DELETE" in c for c in calls)


def test_a_run_that_does_not_export_skips_the_export_tools(tmp_path):
    r, calls = run_whole_script(tmp_path, cuda="12.9", prep="0")
    assert r.returncode == 0, r.stdout + r.stderr
    assert not any("model-compression-toolkit" in c for c in calls)
    assert any("whl/cu128" in c for c in calls) and any("gpu_agent.py --once" in c for c in calls)


def test_broken_export_tools_stop_the_pod_before_any_training_time_is_spent(tmp_path):
    r, calls = run_whole_script(tmp_path, import_check_fails=True)
    assert r.returncode != 0
    assert not any("gpu_agent.py" in c for c in calls)                                # the agent never started
    assert any("-X DELETE" in c for c in calls)                                       # the pod still deleted itself


@pytest.mark.parametrize("problem", [{"imxconv": False}, {"java": "openjdk 11.0.2 2019-01-15"}, {"java": None}])
def test_a_missing_converter_or_java_stops_the_pod_before_any_training_time_is_spent(tmp_path, problem):
    """Without them Ultralytics would fail, or apt-install Java, only after training: check them during setup."""
    r, calls = run_whole_script(tmp_path, **problem)
    assert r.returncode != 0
    assert not any("gpu_agent.py" in c for c in calls) and any("-X DELETE" in c for c in calls)


def test_the_pod_is_told_whether_the_run_exports(client, configured, ready):
    for export, flag in ((True, "1"), (False, "0")):
        new_run(client, ready, export_imx=export)
        h = Harness(client)
        assert cli.main(["launch", "--yes"], deps=h.deps) == 1                       # nobody claims it here; only the request matters
        assert h.api.created[0]["env"]["MT_PREP_IMX"] == flag
        with client.session_factory() as db:                                          # clear the queue for the next round
            from app.models import RunStage, TrainingRun
            for run in db.query(TrainingRun).all():
                run.stage = RunStage.cancelled
            db.commit()


# ---------- the dataset bucket ----------

from .test_dataset_bucket import bucket  # noqa: E402, F401 - fixture


def http_with_bucket(fake_bucket, **kw):
    """The usual fake web, plus real requests to the fake bucket's local server (the launcher checks a signed link there)."""
    mock, real = http(**kw), httpx.Client()

    def handler(request: httpx.Request):
        if request.url.host == "127.0.0.1":
            r = real.request(request.method, str(request.url), headers={k: v for k, v in request.headers.items() if k.lower() == "range"})
            return httpx.Response(r.status_code, content=r.content)
        return mock.send(request)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_dataset_goes_to_the_bucket_before_any_tunnel_or_pod_exists(client, configured, queued, lan, bucket):
    order = []

    class Watching(FakeApi):
        def create_pod(self, spec):
            order.append(("pod", bucket.uploads))
            return super().create_pod(spec)
    h = Harness(client, api=Watching(), http_client=http_with_bucket(bucket), on_sleep=moves(lan, queued, {1: "claim", 2: "finish"}))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 0
    assert order == [("pod", 1)]                                                        # uploaded first, then the pod
    assert "private dataset bucket" in h.said and "a signed link serves the zip" in h.said


def test_the_dry_run_uploads_nothing_but_says_where_the_data_goes(client, configured, queued, bucket):
    h = Harness(client, http_client=http_with_bucket(bucket))
    assert cli.main(["launch"], deps=h.deps) == 0
    assert bucket.uploads == 0 and "private dataset bucket" in h.said and "through a Cloudflare tunnel" not in h.said


def test_a_bucket_link_that_does_not_work_starts_nothing(client, configured, queued, bucket):
    bucket.broken = True
    h = Harness(client, http_client=http_with_bucket(bucket))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert h.api.created == [] and h.tunnels_made == [] and "Not starting" in h.said and "HTTP 403" in h.said


def test_a_dataset_can_be_uploaded_ahead_of_time(client, configured, queued, bucket):
    h = Harness(client, http_client=http_with_bucket(bucket))
    assert cli.main(["upload-dataset"], deps=h.deps) == 0 and bucket.uploads == 1
    assert cli.main(["upload-dataset"], deps=h.deps) == 0 and bucket.uploads == 1        # already there: nothing sent again
    assert h.api.created == [] and h.tunnels_made == []


def test_upload_dataset_explains_the_settings_when_there_is_no_bucket(client, configured, queued):
    h = Harness(client)
    assert cli.main(["upload-dataset"], deps=h.deps) == 1 and "MT_DATASET_BUCKET_SECRET_KEY" in h.said


# ---------- the pod's setup lines ----------

def test_the_launcher_prints_what_the_pod_reports_while_it_sets_up(client, configured, queued, lan):
    sleeps = []

    def hook(h):
        sleeps.append(1)
        worker = "runpod-" + h.api.created[0]["name"]
        if len(sleeps) == 1:
            lan.post("/workers/log", content=b"[mt-pod 10:00:01] installing system packages", headers=hdr(worker))
            lan.post("/workers/log", content=b"[mt-pod 10:02:40] installing Python 3.11 and torch 2.9.1 (cu128)", headers=hdr(worker))
    h = Harness(client, on_sleep=hook)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1                                 # nobody claims: the 20-minute give-up
    said = h.said
    assert "pod: [mt-pod 10:00:01] installing system packages" in said and "(cu128)" in said
    assert said.index("(cu128)") < said.index("too slowly or got stuck")                    # seen while waiting, before giving up
    assert said.count("installing system packages") == 1                                   # each line once


def test_the_pod_script_reports_each_step_to_the_backend(tmp_path):
    r, calls = run_whole_script(tmp_path, cuda="12.4")
    assert r.returncode == 0, r.stdout + r.stderr
    reports = [c for c in calls if "/workers/log" in c]
    assert any("installing system packages" in c for c in reports) and any("starting the agent" in c for c in reports)
    assert all("X-Worker-Name: runpod-mt-x" in c for c in reports)


# ---------- checks on the pod's downloads, from the Mac ----------

def test_the_launcher_refuses_when_a_file_the_pod_needs_is_missing(client, configured, queued):
    h = Harness(client, http_client=http(downloads=404))
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "Not starting" in h.said and "HTTP 404 for download.pytorch.org/torch-2.9.1%2Bcu126" in h.said
    assert h.api.created == [] and h.tunnels_made == []                                       # nothing started, nothing spent


def test_a_download_server_that_errors_also_stops_the_launch_but_an_unreachable_one_does_not(client, configured, queued):
    assert cli.main(["launch"], deps=Harness(client, http_client=http(downloads=503)).deps) == 1

    def offline(request):                                                                     # this Mac's connection is down: say nothing
        if request.url.host in ("download.pytorch.org", "astral.sh", "pypi.org"):
            raise httpx.ConnectError("no route", request=request)
        return http().send(request)
    h = Harness(client, http_client=httpx.Client(transport=httpx.MockTransport(offline)))
    assert cli.main(["launch"], deps=h.deps) == 0 and "Dry run" in h.said


def test_the_downloads_checked_are_exactly_the_wheels_the_pod_script_can_pick():
    script = SCRIPT.read_text()
    for build in cli.TORCH_BUILDS:
        assert f"echo {build}" in script                                                       # the script can pick it ...
        assert any(f"/{build}/torch-2.9.1%2B{build}-" in u for u in cli.DOWNLOADS)             # ... so it is checked
    assert "echo cu130" not in script


def test_a_pod_that_reports_steps_but_then_goes_quiet_is_given_up_on_after_twelve_minutes(client, configured, queued, lan):
    sent = []

    def hook(h):
        if not sent:
            sent.append(1)
            worker = "runpod-" + h.api.created[0]["name"]
            lan.post("/workers/log", content=b"[mt-pod 10:00:00] installing torch 2.9.1 (cu128)", headers=hdr(worker))
    h = Harness(client, on_sleep=hook)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "too slowly or got stuck" in h.said and "installing torch" in h.said and h.api.deleted == ["pod-1"]
    assert h.t - 1000 < cli.STALL_TIMEOUT + 3 * cli.POLL + 60                                  # not the 20 minutes of total silence


def test_a_pod_that_keeps_reporting_gets_longer_but_not_forever(client, configured, queued, lan):
    count = []

    def hook(h):
        count.append(1)
        if len(count) % 40 == 1:                                                                # a line every ~10 minutes
            worker = "runpod-" + h.api.created[0]["name"]
            lan.post("/workers/log", content=f"[mt-pod step {len(count)}]".encode(), headers=hdr(worker))
    h = Harness(client, on_sleep=hook)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert h.t - 1000 > cli.CLAIM_TIMEOUT and h.t - 1000 <= cli.CLAIM_CAP + 3 * cli.POLL + 60  # past 20 minutes, stopped at the cap


def test_a_slow_download_report_from_the_pod_is_shown_before_it_vanishes(client, configured, queued, lan):
    sleeps = []

    def hook(h):
        sleeps.append(1)
        if len(sleeps) == 2:
            worker = "runpod-" + h.api.created[0]["name"]
            lan.post("/workers/log", content=b"[mt-pod 10:01:00] SLOW DOWNLOAD: this host reaches PyTorch's server too slowly", headers=hdr(worker))
            h.api.pods.clear()                                                                 # it deletes itself right after saying so
    h = Harness(client, on_sleep=hook)
    assert cli.main(["launch", "--yes"], deps=h.deps) == 1
    assert "SLOW DOWNLOAD" in h.said and "did not finish" in h.said


# ---------- memory reports and the cause of a kill ----------

def fake_cgroup(tmp_path, current_mb=1024, max_mb=16384, oom_kill=0):
    d = tmp_path / "cgroup"
    d.mkdir()
    (d / "memory.current").write_text(str(current_mb * 1048576) + "\n")
    (d / "memory.max").write_text(str(max_mb * 1048576) + "\n")
    (d / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom 0\noom_kill {oom_kill}\n")
    return d


def test_the_pod_reports_its_memory_while_the_agent_runs_and_when_it_stops(tmp_path):
    r, calls = run_whole_script(tmp_path, cgroup=fake_cgroup(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "memory: 1024 MB used of 16384 MB" in r.stdout
    assert any("/workers/log" in c and "memory: 1024 MB" in c for c in calls)            # also sent to the Mac, where it outlives the pod


def test_a_killed_agent_is_explained_and_the_oom_counter_is_reported(tmp_path):
    """Run 5's export died with exit 137 and nothing else to go on."""
    r, calls = run_whole_script(tmp_path, cgroup=fake_cgroup(tmp_path, oom_kill=1),
                                python_hook='case "$*" in *gpu_agent.py*) kill -9 $$ ;; esac')
    assert r.returncode == 137, r.stdout + r.stderr
    assert "exit 137 means the process was killed" in r.stdout and "oom_kill 1" in r.stdout
    assert any("-X DELETE" in c for c in calls)                                           # still deleted itself


def test_a_pod_without_a_readable_cgroup_still_runs(tmp_path):
    r, _ = run_whole_script(tmp_path, cgroup=tmp_path / "nowhere")
    assert r.returncode == 0 and "memory:" not in r.stdout
