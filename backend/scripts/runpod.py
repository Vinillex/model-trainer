"""Train on a rented RunPod GPU instead of a GPU of your own, and make sure the pod is gone afterwards.

    python -m scripts.runpod status                 what is set up, your balance, any pods that are running
    python -m scripts.runpod gpus                   the GPUs and what they cost per hour
    python -m scripts.runpod launch                 a DRY RUN: checks everything and prints the plan, starts nothing
    python -m scripts.runpod launch --yes           spends money: starts a tunnel and a pod, watches, then deletes the pod
    python -m scripts.runpod stop --all             deletes every pod this project started (name starts with "mt-")

Create the training run in the UI first ("Train & export"); `launch` attaches a GPU to the run that is waiting.
How it works, what leaves your Mac and what it costs: agents/RUNPOD.md.
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from sqlalchemy import select

from app import dataset_bucket, training
from app import runpod_client as rp
from app.config import settings
from app.models import DatasetVersion, RunStage, TrainingRun
from app.runpod_pod import DEFAULT_GPUS, IMAGE, POD_PREFIX, build_pod_spec

MAX_HOURS_LIMIT = 12
TUNNEL_TRIES = 3  # fresh tunnels to try when one can't be reached (each gets a new name, and DNS for new names is slow and uneven)
CLAIM_TIMEOUT = 20 * 60  # seconds the pod gets to start up and pick the run up before it is given up on
POLL = 15  # seconds between looks at the pod and the run
DELETE_TRIES = 8  # attempts to delete the pod, about a minute and a half in all
API_PATIENCE = 10 * 60  # how long RunPod may be unreachable from this Mac before the pod is ended unseen
ENDED = (RunStage.awaiting_export, RunStage.awaiting_package, RunStage.done, RunStage.failed, RunStage.cancelled)  # the GPU part is over
STALL_TIMEOUT = 12 * 60  # a pod that has reported setup steps but then says nothing for this long is stuck in one
CLAIM_CAP = 40 * 60  # the longest a pod may take to set up, however steadily it reports
GRACE = 10 * 60  # how far past --max-hours this watcher waits for the pod's own time limit before ending it
CONSOLE = "https://console.runpod.io/pods"
BILLING = "https://console.runpod.io/user/billing"


class Refusal(Exception):
    """A reason not to go ahead. The message is written for the person at the keyboard."""


class TunnelNotReady(Refusal):
    """The tunnel can't be reached (yet). Often just slow DNS for a brand-new name; a fresh tunnel may do better."""


class TunnelLeaks(Refusal):
    """Through the tunnel the internet can reach more than the worker routes. Never retried."""


# ---------- the real things the launcher talks to (tests replace them) ----------

class RunPodApi:
    def __init__(self, key: str, http: httpx.Client | None = None):
        self.key, self.http = key, http or httpx.Client(timeout=30)

    def account(self):
        return rp.fetch_account(self.key, self.http)

    def gpus(self):
        return rp.fetch_gpus(self.key, self.http)

    def list_pods(self):
        return rp.list_pods(self.key, self.http)

    def get_pod(self, pod_id):
        return rp.get_pod(self.key, pod_id, self.http)

    def create_pod(self, spec):
        return rp.create_pod(self.key, spec, self.http)

    def delete_pod(self, pod_id):
        return rp.delete_pod(self.key, pod_id, self.http)


class QuickTunnel:
    """A free Cloudflare "quick tunnel" (no account): a public https address that forwards to the backend on this Mac for as
    long as this object is running, and no longer."""

    URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

    def __init__(self, backend_url: str, popen=subprocess.Popen, which=shutil.which, wait: float = 45):
        self.backend_url, self.popen, self.which, self.wait = backend_url, popen, which, wait
        self.proc = None

    def start(self) -> str:
        exe = self.which("cloudflared")
        if not exe:
            raise Refusal("cloudflared isn't installed. Install it with: brew install cloudflared")
        # --protocol http2: over this Mac's connection cloudflared's default (QUIC) dropped big request bodies. Pod 4's upload of best.pt
        # failed 13 times in a row, and a 5.5 MB PUT from the Mac through a quick tunnel failed 3 of 8 times with QUIC (the server saw 0
        # bytes) and 0 of 8 with http2 (2026-10-22).
        self.proc = self.popen([exe, "tunnel", "--protocol", "http2", "--url", self.backend_url, "--no-autoupdate"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        found: list[str] = []

        def read():  # keeps draining the output afterwards, or cloudflared would block on a full pipe
            for line in self.proc.stdout:
                if not found and (m := self.URL.search(line)):
                    found.append(m.group(0))

        threading.Thread(target=read, daemon=True).start()
        deadline = time.monotonic() + self.wait
        while not found and time.monotonic() < deadline and self.proc.poll() is None:
            time.sleep(0.2)
        if not found:
            self.stop()
            raise Refusal("cloudflared didn't report a tunnel address. Run `cloudflared tunnel --url http://127.0.0.1:8000` "
                          "yourself to see why, or pass its address with --server.")
        return found[0]

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@dataclass
class Deps:
    api: object
    http: httpx.Client
    session: object  # sessionmaker for the backend's database
    tunnel: object = None  # called with the backend URL, returns something with start() -> url and stop()
    keep_awake: object = None  # returns something with terminate(), or None
    sleep: object = time.sleep
    now: object = time.monotonic
    say: object = print


def real_deps(key: str) -> Deps:
    from app.db import SessionLocal

    def awake():
        exe = shutil.which("caffeinate")  # stops the Mac sleeping (and dropping the tunnel) while this runs
        return subprocess.Popen([exe, "-i", "-w", str(os.getpid())]) if exe else None

    http = httpx.Client(timeout=30)
    return Deps(RunPodApi(key, http), http, SessionLocal, tunnel=lambda url: QuickTunnel(url), keep_awake=awake)


# ---------- checks, all before anything costs money ----------

@dataclass
class Plan:
    run_id: int
    run_name: str
    images: int
    gpus: list[str]
    cloud: str
    max_hours: float
    worst_price: float
    worst_case: float
    balance: float
    image: str
    export_imx: bool = True  # whether the queued run exports for the IMX500 (the pod then installs the export tools)
    dataset_id: int | None = None
    bucket: bool = False  # the pod downloads the dataset from the dataset bucket (MT_DATASET_BUCKET) instead of through the tunnel
    gpu_prices: dict = field(default_factory=dict)  # per hour, for each of `gpus` in this cloud


def image_exists(image: str, http: httpx.Client) -> bool | None:
    """Is the pinned image still on Docker Hub? None when that can't be told (another registry, or no network)."""
    repo, _, tag = image.partition(":")
    if repo.split("/")[0].count(".") or not tag:  # ghcr.io/..., quay.io/... : not checked
        return None
    try:
        r = http.get(f"https://hub.docker.com/v2/repositories/{repo if '/' in repo else 'library/' + repo}/tags/{tag}")
    except httpx.HTTPError:
        return None
    return {200: True, 404: False}.get(r.status_code)


# What the pod downloads while it sets up (agents/runpod_start.sh): the exact PyTorch wheels for the builds it can pick, the uv
# installer and PyPI. The Mac can't tell how fast the pod's own route is (the pod times that itself), but it can tell whether
# these files exist and their servers answer. A missing file or a server error ends here, before any money is spent.
TORCH_BUILDS = ("cu126", "cu128")  # runpod_start.sh never picks anything newer
DOWNLOADS = [f"https://download.pytorch.org/whl/{c}/{pkg}-{v}%2B{c}-cp311-cp311-manylinux_2_28_x86_64.whl"
             for c in TORCH_BUILDS for pkg, v in (("torch", "2.9.1"), ("torchvision", "0.24.1"))]
DOWNLOADS += ["https://astral.sh/uv/0.8.0/install.sh", "https://pypi.org/simple/ultralytics/"]


def download_problems(http: httpx.Client) -> list[str]:
    """Only a definite answer counts (missing file, server error). A request that can't be made at all says something about this
    Mac's connection, not the servers, and is not a reason to refuse."""
    problems = []
    for url in DOWNLOADS:
        try:
            r = http.head(url, follow_redirects=True, timeout=20)
        except httpx.HTTPError:
            continue
        if r.status_code >= 400:
            problems.append(f"HTTP {r.status_code} for {urlparse(url).hostname}/{url.rsplit('/', 1)[-1] or 'index'}")
    return problems


def make_plan(args, d: Deps) -> Plan:
    key, token = settings.runpod_api_key, settings.worker_token
    if not key:
        raise Refusal("MT_RUNPOD_API_KEY isn't set in backend/.env.")
    if not token:
        raise Refusal("MT_WORKER_TOKEN isn't set in backend/.env (the pod signs in to the backend with it).")
    if not 0 < args.max_hours <= MAX_HOURS_LIMIT:
        raise Refusal(f"--max-hours must be between 0 and {MAX_HOURS_LIMIT}.")

    with d.session() as db:
        # A run whose pod died stays "training" until a worker next asks for work, and then only if its lease has run out.
        # This launcher is that worker's only caller, so it does the release itself, or a stranded run could never be retried.
        training.reap_expired(db)
        wanted_run = getattr(args, "run", None)
        if wanted_run is not None:  # the UI names the run the person confirmed; never "whichever is first" then
            run = db.get(TrainingRun, wanted_run, populate_existing=True)
            if run is None or run.stage != RunStage.queued:
                raise Refusal(f"Run {wanted_run} is not waiting for a GPU (it must be queued). Nothing was started.")
        else:
            run = db.scalars(select(TrainingRun).where(TrainingRun.stage == RunStage.queued).order_by(TrainingRun.id).limit(1)).first()
        if run is None:
            raise Refusal("No training run is waiting. Create one in the UI (Train & export), then run this again. "
                          "A pod with nothing to do would only cost money.")
        version = db.get(DatasetVersion, run.dataset_id)
        images = sum((version.stats or {}).get("images", {}).values()) if version else 0
        run_id, run_name, dataset_id = run.id, run.name, run.dataset_id
        export_imx = bool(training.worker_config(run.config or {}).get("export_imx", True))  # False when the Mac exports

    cloud = args.cloud.upper()
    price_key = "secure_price" if cloud == "SECURE" else "community_price"
    offered = {g["id"]: g for g in d.api.gpus() if g[price_key] > 0}
    if unknown := [g for g in args.gpu or [] if g not in offered]:
        raise Refusal(f"Not offered in the {cloud.title()} cloud: {', '.join(unknown)}. See: python -m scripts.runpod gpus")
    wanted = args.gpu or list(DEFAULT_GPUS)
    chosen = [g for g in wanted if g in offered]
    if not chosen:
        raise Refusal(f"None of {', '.join(wanted)} is offered in the {cloud.title()} cloud. See: python -m scripts.runpod gpus")
    worst_price = max(offered[g][price_key] for g in chosen)
    worst_case = round(worst_price * args.max_hours, 2)
    agreed = getattr(args, "max_cost", None)
    if agreed is not None and worst_case > agreed + 0.005:  # the person confirmed a price; a dearer one needs a new confirmation
        raise Refusal(f"The price changed: this could now cost up to ${worst_case:.2f}, more than the ${agreed:.2f} that was agreed. "
                      "Nothing was started; check the price again.")

    balance = d.api.account().get("balance") or 0
    if balance < worst_case:
        raise Refusal(f"Your RunPod balance is ${balance:.2f}, and this could cost up to ${worst_case:.2f} "
                      f"({args.max_hours:g} h at ${worst_price:.2f}/h). Add credit at {BILLING}, or lower --max-hours. "
                      "A normal run costs far less than the limit.")

    if image_exists(args.image, d.http) is False:
        raise Refusal(f"The pod image {args.image} no longer exists on Docker Hub. Pass a current one with --image.")
    if problems := download_problems(d.http):
        raise Refusal("A download the pod needs is not available right now, so it would only sit and bill: " + "; ".join(problems)
                      + ". Try again in a little while.")
    return Plan(run_id, run_name, images, chosen, cloud, args.max_hours, worst_price, worst_case, balance, args.image, export_imx,
                dataset_id, dataset_bucket.enabled(), {g: offered[g][price_key] for g in chosen})


def describe(p: Plan) -> str:
    return "\n".join([
        f"Run {p.run_id} ({p.run_name}) will train on one RunPod GPU: first of {', '.join(p.gpus)} with stock, {p.cloud.title()} cloud.",
        f"Cost: the pod is deleted when the run ends. The limit is {p.max_hours:g} h, so at most ${p.worst_case:.2f} "
        f"(up to ${p.worst_price:.2f}/h). Your balance is ${p.balance:.2f}.",
        "What leaves your Mac:",
        (f"  - the {p.images:,} images of the run's dataset version, as a zip in your private dataset bucket (uploaded once per "
         "version, before the pod starts); the pod gets a link to that one file, valid for 6 hours" if p.bucket else
         f"  - the {p.images:,} images of the run's dataset version, through a Cloudflare tunnel to RunPod"),
        "  - the worker token, as a setting of the pod (it only opens /workers/* on this backend; the tunnel is closed afterwards)",
        "While it runs, keep this window open and the Mac awake. Ctrl-C ends the pod.",
    ])


def doh_resolve(host: str, http: httpx.Client) -> str | None:
    """Ask Cloudflare's own resolver (over https, by address) for a host. A brand-new trycloudflare.com name is often unknown to a
    home or ISP resolver for a long while, even though Cloudflare, and so the pod, resolves it within seconds."""
    try:
        r = http.get("https://1.1.1.1/dns-query", params={"name": host, "type": "A"}, headers={"accept": "application/dns-json"},
                     timeout=10)
        return next((a["data"] for a in r.json().get("Answer", []) if a.get("type") == 1), None)
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def fetch(http: httpx.Client, url: str, headers: dict | None = None) -> httpx.Response:
    """GET that falls back to Cloudflare's resolver when this machine's DNS can't find the host (see doh_resolve)."""
    try:
        return http.get(url, headers=headers)
    except httpx.ConnectError:
        host = urlparse(url).hostname
        ip = doh_resolve(host, http)
        if not ip:
            raise
        # Connect to the address, but still present the real name (Host header, TLS name) so the certificate is checked as usual.
        return http.get(url.replace(host, ip, 1), headers={**(headers or {}), "Host": host}, extensions={"sni_hostname": host})


def _json(r: httpx.Response):
    try:
        return r.json()
    except ValueError:
        return None


def _closed(r: httpx.Response) -> bool:
    """The backend's own refusal, not just any 403."""
    return r.status_code == 403 and "workers use /workers/" in r.text


def _leaks(health: httpx.Response, projects: httpx.Response) -> bool:
    """A real answer from this API. A 200 of any other kind (a placeholder page while the tunnel comes up) is not one."""
    return (health.status_code == 200 and _json(health) == {"ok": True}) or (projects.status_code == 200 and isinstance(_json(projects), list))


def selftest(url: str, token: str, d: Deps, timeout: float = 75) -> None:
    """Before a pod is started: through the tunnel, the API must be closed to the internet (the backend's own refusal comes back)
    and the worker route must open with the token. If the API answers instead, the launcher stops for good, whatever else works."""
    deadline = d.now() + timeout
    last = "no answer"
    while True:
        try:
            health = fetch(d.http, url + "/health")
            projects = fetch(d.http, url + "/projects")
            ping = fetch(d.http, url + "/workers/ping", {"X-Worker-Token": token})
            if _leaks(health, projects):
                raise TunnelLeaks("SECURITY: through the tunnel the internet can reach the whole API (not just /workers/*). Nothing was "
                                  "started. Restart the backend so it runs the current app/security.py, and try again.")
            if _closed(health) and _closed(projects) and ping.status_code == 200:
                return
            last = f"/health {health.status_code}, /projects {projects.status_code}, /workers/ping {ping.status_code}"
            if 200 in (health.status_code, projects.status_code):
                last += " (a 200 that is not this API, probably a placeholder page)"
        except httpx.HTTPError as e:
            last = type(e).__name__
        if d.now() > deadline:
            raise TunnelNotReady(f"The tunnel doesn't work: {last}. Is the backend running, with MT_WORKER_TOKEN set?")
        d.sleep(3)


# ---------- the run itself ----------

def end_pod(d: Deps, pod_id: str, name: str) -> bool:
    """Make sure the pod is gone. Tries several times; says so loudly if it can't."""
    for attempt in range(DELETE_TRIES):
        try:
            if not d.api.delete_pod(pod_id) and attempt == 0:
                d.say("  (RunPod no longer knew the pod: it was already gone before the launcher deleted it)")
            pod = d.api.get_pod(pod_id)
            if pod is None or pod.get("desiredStatus") == "TERMINATED":
                return True
        except rp.RunPodError as e:
            d.say(f"  could not delete the pod yet: {e}")
        d.sleep(min(2 * (attempt + 1), 20))  # a dropped connection can take a while to come back
    d.say(f"!! POD {pod_id} ({name}) IS STILL THERE AND CHARGING. Delete it now: {CONSOLE}  or  python -m scripts.runpod stop --all")
    return False


def find_by_name(d: Deps, name: str) -> list[str]:
    try:
        return [p["id"] for p in d.api.list_pods() if p.get("name") == name]
    except rp.RunPodError:
        return []


def progress_line(run: TrainingRun) -> str:
    p = run.progress or {}
    bits = [run.stage.value]
    if p.get("message"):
        bits.append(str(p["message"]))
    if p.get("epoch") and p.get("epochs"):
        bits.append(f"epoch {p['epoch']}/{p['epochs']}")
    if p.get("map50") is not None:
        bits.append(f"mAP50 {p['map50']:.3f}")
    return f"run {run.id}: " + ", ".join(bits)


class PodLog:
    """The setup lines a pod reports to the backend (POST /workers/log), read as they arrive."""

    def __init__(self, worker: str):
        from app.api.workers import worker_log_path
        self.path, self.offset = worker_log_path(worker), 0

    def new_lines(self) -> list[str]:
        try:
            with self.path.open("rb") as f:
                f.seek(self.offset)
                data = f.read()
        except FileNotFoundError:
            return []
        end = data.rfind(b"\n") + 1  # only whole lines; the rest comes next time
        self.offset += end
        return data[:end].decode("utf-8", "replace").splitlines()


def watch(plan: Plan, pod_id: str, d: Deps, worker: str | None = None) -> str:
    """Returns why it stopped: finished | pod_gone | never_claimed | time_limit | lost_contact. The caller deletes the pod either way."""
    started, last, blind_since = d.now(), None, None
    limit = plan.max_hours * 3600 + GRACE
    pod_log = PodLog(worker) if worker else None
    last_line_at = None
    while True:
        for line in pod_log.new_lines() if pod_log else []:
            d.say("  pod: " + line)
            last_line_at = d.now()
        with d.session() as db:
            run = db.get(TrainingRun, plan.run_id, populate_existing=True)
            stage, line = run.stage, progress_line(run)
        if line != last:
            d.say("  " + line)
            last = line
        if stage in ENDED:
            return "finished"
        pod, seen = None, False
        try:
            pod, seen = d.api.get_pod(pod_id), True
            if blind_since is not None:
                d.say("  RunPod is reachable again.")
                blind_since = None
        except rp.RunPodError as e:
            # A blip in this Mac's connection says nothing about the pod. Ending it over one failed look would throw away a run
            # that is working (this happened in the third real run, in the middle of the export), so keep watching.
            if blind_since is None:
                blind_since = d.now()
                d.say(f"  can't reach RunPod just now ({e}); still watching")
            elif d.now() - blind_since > API_PATIENCE:
                return "lost_contact"
        if seen and (pod is None or pod.get("desiredStatus") in ("EXITED", "TERMINATED")):
            for line in pod_log.new_lines() if pod_log else []:  # what it said on its way out (it deletes itself right after)
                d.say("  pod: " + line)
            # A pod that did its job completes the run first and disappears after, so look once more: the stage read above can be
            # a poll old. A pod that is gone with the run still open was killed (its own time limit, a crash), not finished.
            with d.session() as db:
                stage = db.get(TrainingRun, plan.run_id, populate_existing=True).stage
            if stage in ENDED:
                return "finished"
            d.say("  RunPod reports the pod as " + ("gone" if pod is None else str(pod.get("desiredStatus")).lower()))
            return "pod_gone"
        waited = d.now() - started
        if stage == RunStage.queued:
            # Nothing heard at all: the image may still be loading, so allow CLAIM_TIMEOUT. Once it reports, it is stuck if it
            # goes quiet for STALL_TIMEOUT, but a pod that keeps making progress may take up to CLAIM_CAP.
            silent_for = d.now() - (last_line_at if last_line_at is not None else started)
            if silent_for > (STALL_TIMEOUT if last_line_at is not None else CLAIM_TIMEOUT) or waited > CLAIM_CAP:
                return "never_claimed"
        if waited > limit:
            return "time_limit"
        d.sleep(POLL)


def run_on_runpod(args, plan: Plan, d: Deps) -> int:
    name = POD_PREFIX + datetime.now(timezone.utc).strftime("%m%d-%H%M%S")
    tunnel = awake = None
    pod_id = None
    cost_per_hr = pod_started = None
    create_attempted = False
    try:
        awake = d.keep_awake() if d.keep_awake else None
        url = args.server.rstrip("/") if args.server else None
        for attempt in range(1, TUNNEL_TRIES + 1):
            if not args.server:
                if tunnel:
                    tunnel.stop()
                tunnel = d.tunnel(args.backend_url)
                d.say("Starting a tunnel to this backend..." if attempt == 1 else f"Starting a new tunnel ({attempt} of {TUNNEL_TRIES})...")
                url = tunnel.start()
            d.say(f"Tunnel: {url}. Checking that only the worker routes are open through it...")
            try:
                selftest(url, settings.worker_token, d)
                break
            except TunnelNotReady as e:
                if args.server or attempt == TUNNEL_TRIES:
                    raise
                d.say(f"  not reachable yet ({str(e).split(': ', 1)[-1].split('. Is the')[0]}); trying a fresh tunnel")
        d.say("  ok: the rest of the API is closed to the internet.")

        spec = build_pod_spec(name=name, server_url=url, worker_token=settings.worker_token, gpu_ids=plan.gpus,
                              cloud=plan.cloud, max_hours=plan.max_hours, image=plan.image, export_imx=plan.export_imx)
        d.say(f"Starting pod {name}...")
        create_attempted = True  # from here a pod may exist even if we never learn its id
        try:
            pod = d.api.create_pod(spec)
        except rp.RunPodError as e:
            raise Refusal(f"RunPod refused to start the pod: {e}")
        pod_id = pod["id"]
        pod_started = d.now()
        cost_per_hr = pod.get("costPerHr")
        d.say(f"  pod {pod_id} started" + (f" at ${cost_per_hr:.2f}/h" if cost_per_hr else "") + f". Console: {CONSOLE}")
        if cost_per_hr and cost_per_hr > plan.worst_price * 1.25 + 0.05:
            raise Refusal(f"The pod costs ${cost_per_hr:.2f}/h, much more than the ${plan.worst_price:.2f}/h expected. Deleting it.")

        d.say("Waiting for the pod to install its tools and pick the run up (a few minutes)...")
        why = watch(plan, pod_id, d, worker="runpod-" + name)
        notes = {"finished": "The GPU part of the run is over.",
                 "pod_gone": "The pod is gone but its run did not finish: the pod's own time limit probably ran out, or it crashed. "
                             "The run goes back to the queue once its lease runs out (15 minutes).",
                 "never_claimed": "The pod set up too slowly or got stuck (its last report is above), so it was given up on. "
                                  f"Its own Logs are at {CONSOLE} while it exists. Run the same command again: a new pod gets another host.",
                 "time_limit": f"The time limit of {plan.max_hours:g} h passed.",
                 "lost_contact": f"RunPod has been unreachable for {API_PATIENCE // 60} minutes, so the pod can't be watched. Ending it."}
        d.say(notes[why])
        return 0 if why == "finished" else 1
    except KeyboardInterrupt:
        d.say("Interrupted.")
        return 130
    except Refusal as e:
        d.say(str(e))
        return 1
    finally:
        # Without an id (the answer to the create request was lost, or Ctrl-C came in between) look for the pod by its name.
        for stray in [pod_id] if pod_id else (find_by_name(d, name) if create_attempted else []):
            d.say("Deleting the pod...")
            gone = end_pod(d, stray, name)
            hours = (d.now() - pod_started) / 3600 if pod_started else 0
            d.say(("  pod deleted." if gone else "  POD NOT DELETED.")
                  + (f" Roughly ${cost_per_hr * hours:.2f} was spent." if cost_per_hr else ""))
        if tunnel:
            tunnel.stop()
            d.say("Tunnel closed.")
        if awake:
            awake.terminate()


# ---------- commands ----------

def cmd_status(args, d: Deps) -> int:
    say = d.say
    say(f"RunPod key:    {'set' if settings.runpod_api_key else 'MISSING (MT_RUNPOD_API_KEY in backend/.env)'}")
    say(f"Worker token:  {'set' if settings.worker_token else 'MISSING (MT_WORKER_TOKEN in backend/.env)'}")
    say(f"cloudflared:   {shutil.which('cloudflared') or 'not installed (brew install cloudflared)'}")
    say(f"Dataset bucket: {settings.dataset_bucket if dataset_bucket.enabled() else 'not set up (pods download through the tunnel)'}")
    if not settings.runpod_api_key:
        return 1
    acct = d.api.account()
    say(f"Balance:       ${acct['balance'] or 0:.2f}   spending now: ${acct['spend_per_hr'] or 0:.2f}/h")
    with d.session() as db:
        queued = len(db.scalars(select(TrainingRun.id).where(TrainingRun.stage == RunStage.queued)).all())
    say(f"Runs waiting:  {queued}")
    pods = d.api.list_pods()
    ours = [p for p in pods if str(p.get("name", "")).startswith(POD_PREFIX)]
    say(f"Pods:          {len(ours)} started by this project, {len(pods) - len(ours)} others")
    for p in ours:
        say(f"  {p['id']}  {p.get('name')}  {p.get('desiredStatus')}  ${p.get('costPerHr') or 0:.2f}/h")
    return 0


def cmd_gpus(args, d: Deps) -> int:
    key = "secure_price" if args.cloud == "secure" else "community_price"
    rows = sorted((g for g in d.api.gpus() if g[key] > 0 and g["memory_gb"] >= 16), key=lambda g: g[key])
    d.say(f"GPUs with 16 GB or more in the {args.cloud} cloud, per hour (stock is RunPod's hint for the cheapest offer):")
    for g in rows:
        mark = "*" if g["id"] in DEFAULT_GPUS else " "
        d.say(f" {mark} ${g[key]:.2f}  {g['memory_gb']:>3} GB  {g['id']}  stock: {g['stock'] or 'unknown'}")
    d.say("* = tried by default, in this order: " + ", ".join(DEFAULT_GPUS))
    return 0


def cmd_launch(args, d: Deps) -> int:
    try:
        plan = make_plan(args, d)
    except Refusal as e:
        d.say(f"Not starting: {e}")
        return 1
    d.say(describe(plan))
    if not args.yes:
        d.say("\nDry run: nothing was started. Run the same command with --yes to start the tunnel and the pod.")
        return 0
    if plan.bucket:
        try:
            put_in_bucket(plan.dataset_id, d)
        except Refusal as e:
            d.say(f"Not starting: {e}")
            return 1
    return run_on_runpod(args, plan, d)


def put_in_bucket(dataset_id: int, d: Deps) -> None:
    """Upload the dataset to the bucket if it isn't there yet, then check from here that a signed link really serves it. All of
    this happens before any pod exists, so the slow upload costs nothing."""
    d.say("Dataset bucket:")
    try:
        with d.session() as db:
            version = db.get(DatasetVersion, dataset_id)
            key, size = dataset_bucket.ensure_uploaded(db, version, d.say)
        r = d.http.get(dataset_bucket.link(key), headers={"Range": "bytes=0-3"}, timeout=30)
    except Refusal:
        raise
    except Exception as e:  # noqa: BLE001 - boto3/botocore and network errors: say what it was and start nothing
        raise Refusal(f"the dataset bucket didn't work ({type(e).__name__}: {e}). Check the MT_DATASET_BUCKET settings in "
                      "backend/.env, or remove MT_DATASET_BUCKET to send the dataset through the tunnel instead.") from None
    if r.status_code not in (200, 206) or not r.content.startswith(b"PK"):
        raise Refusal(f"a signed link to the dataset answered HTTP {r.status_code} instead of the zip. Nothing was started.")
    d.say(f"  ok: a signed link serves the zip ({size / 2**20:.0f} MB).")


def cmd_upload_dataset(args, d: Deps) -> int:
    """Upload ahead of time, for example overnight, so a later launch finds the zip already there."""
    if not dataset_bucket.enabled():
        d.say("The dataset bucket isn't set up: MT_DATASET_BUCKET, MT_DATASET_BUCKET_ENDPOINT, MT_DATASET_BUCKET_ACCESS_KEY and "
              "MT_DATASET_BUCKET_SECRET_KEY in backend/.env (see agents/RUNPOD.md).")
        return 1
    dataset_id = args.dataset_id
    if dataset_id is None:
        with d.session() as db:
            run = db.scalars(select(TrainingRun).where(TrainingRun.stage == RunStage.queued).order_by(TrainingRun.id).limit(1)).first()
            if run is None:
                d.say("No run is waiting; name a dataset version id instead.")
                return 1
            dataset_id = run.dataset_id
    with d.session() as db:
        if db.get(DatasetVersion, dataset_id) is None:
            d.say(f"There is no dataset version {dataset_id}.")
            return 1
    try:
        put_in_bucket(dataset_id, d)
    except Refusal as e:
        d.say(str(e))
        return 1
    return 0


def cmd_stop(args, d: Deps) -> int:
    pods = [p for p in d.api.list_pods() if str(p.get("name", "")).startswith(POD_PREFIX)]
    if args.pod_id:
        pods = [p for p in pods if p["id"] == args.pod_id]
        if not pods:
            d.say(f"No pod {args.pod_id} whose name starts with '{POD_PREFIX}'. Others are never touched here; use the RunPod console.")
            return 1
    elif not args.all:
        d.say("Say which: a pod id, or --all for every pod this project started.")
        return 1
    if not pods:
        d.say("No pods of this project are running.")
        return 0
    failed = 0
    for p in pods:
        ok = end_pod(d, p["id"], p.get("name", ""))
        d.say(f"  {p['id']} ({p.get('name')}): {'deleted' if ok else 'NOT deleted'}")
        failed += not ok
    return 1 if failed else 0


def install_stop_handlers() -> None:
    """Make SIGINT and SIGTERM end the launch the careful way (KeyboardInterrupt runs the `finally` that deletes the pod and closes
    the tunnel). A launcher started by the backend may have inherited SIGINT as "ignored", which Python would then leave alone."""
    def stop(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
    except ValueError:  # not the main thread
        pass


def main(argv=None, deps: Deps | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    g = sub.add_parser("gpus")
    g.add_argument("--cloud", choices=("secure", "community"), default="secure")
    ln = sub.add_parser("launch")
    ln.add_argument("--yes", action="store_true", help="really start (spends money). Without it this is a dry run.")
    ln.add_argument("--gpu", action="append", help="a GPU id, repeat for several in order of preference (see `gpus`)")
    ln.add_argument("--cloud", choices=("secure", "community"), default="secure", help="community is cheaper, secure is steadier")
    ln.add_argument("--run", type=int, help="train this queued run (default: the oldest one waiting)")
    ln.add_argument("--max-cost", type=float, help="refuse if the worst case now costs more than this many dollars (what you agreed to)")
    ln.add_argument("--max-hours", type=float, default=3.0, help=f"hard limit on the pod's life, at most {MAX_HOURS_LIMIT} (default 3)")
    ln.add_argument("--image", default=IMAGE)
    ln.add_argument("--backend-url", default="http://127.0.0.1:8000", help="where the backend listens on this Mac")
    ln.add_argument("--server", help="use this public address of the backend instead of starting a tunnel")
    up = sub.add_parser("upload-dataset", help="put a dataset version's zip in the dataset bucket now (no pod, no cost)")
    up.add_argument("dataset_id", type=int, nargs="?", help="dataset version id (default: the one of the next waiting run)")
    st = sub.add_parser("stop")
    st.add_argument("pod_id", nargs="?")
    st.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    if deps is None:
        if not settings.runpod_api_key and args.cmd not in ("status", "upload-dataset"):
            print("MT_RUNPOD_API_KEY isn't set in backend/.env.")
            return 1
        deps = real_deps(settings.runpod_api_key or "")
        install_stop_handlers()
    try:
        return {"status": cmd_status, "gpus": cmd_gpus, "launch": cmd_launch, "stop": cmd_stop,
                "upload-dataset": cmd_upload_dataset}[args.cmd](args, deps)
    except rp.RunPodError as e:
        deps.say(f"RunPod said no: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
