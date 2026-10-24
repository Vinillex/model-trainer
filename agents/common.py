"""Shared by the GPU agent and the Pi agent. Standard library only: the Pi's system Python can't `pip install`."""

import http.client
import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("agent")


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail
        super().__init__(f"HTTP {status}: {detail}" if status else detail)


class LeaseLost(ApiError):
    """The backend gave this run to someone else, or it was cancelled. Stop working on it."""


# Anything in this family means "the backend isn't reachable right now" (restarting, asleep, Wi-Fi blip).
_NETWORK_ERRORS = (urllib.error.URLError, http.client.HTTPException, socket.timeout, ConnectionError, TimeoutError, OSError)


class Api:
    def __init__(self, server: str, token: str, name: str, backoff_base: float = 2.0, backoff_max: float = 60.0,
                 max_wait: float = 6 * 3600):
        self.server = server.rstrip("/")
        self.headers = {"X-Worker-Token": token, "X-Worker-Name": name}
        self.name = name
        self.backoff_base, self.backoff_max, self.max_wait = backoff_base, backoff_max, max_wait

    def _once(self, method: str, path: str, body, file: Path | None, timeout: float, out: Path | None) -> bytes:
        # A full https address is a signed link to somewhere else (the dataset bucket): it must never get the worker token.
        external = path.startswith(("https://", "http://"))
        headers = {} if external else dict(self.headers)
        data = None
        if body is not None:
            data, headers["Content-Type"] = json.dumps(body).encode(), "application/json"
        fh = open(file, "rb") if file else None
        try:
            if fh:
                data, headers["Content-Length"] = fh, str(file.stat().st_size)
            req = urllib.request.Request(path if external else self.server + path, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if out is None:
                    return resp.read()
                expected = resp.headers.get("Content-Length")
                part = out.with_name(out.name + ".part")
                got = 0
                with open(part, "wb") as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                if expected is not None and got != int(expected):
                    # A connection that dies mid-transfer just ends the stream: without this check a cut-off file is kept as if complete.
                    part.unlink(missing_ok=True)
                    raise http.client.IncompleteRead(b"", int(expected) - got)
                os.replace(part, out)
                return b""
        finally:
            if fh:
                fh.close()

    def call(self, method: str, path: str, *, body=None, file: Path | None = None, out: Path | None = None,
             retry: bool = True, timeout: float = 120) -> bytes:
        """One API call. With retry=True, outages and 5xx responses are retried with backoff (up to max_wait)."""
        started, attempt = time.monotonic(), 0
        external = path.startswith(("https://", "http://"))
        # A signed link is a password for one file until it expires: logs and errors show it without its signature.
        shown, where = (path.split("?", 1)[0], "the dataset bucket") if external else (path, self.server)
        while True:
            try:
                return self._once(method, path, body, file, timeout, out)
            except urllib.error.HTTPError as e:  # before URLError: HTTPError is a subclass of it
                detail = _detail(e.read())
                if e.code == 409 and not external:
                    raise LeaseLost(e.code, detail) from None
                if e.code < 500 or not retry:
                    raise ApiError(e.code, f"{shown}: {detail}" if external else detail) from None
                problem = f"HTTP {e.code}: {detail}"
            except _NETWORK_ERRORS as e:
                if not retry:
                    raise ApiError(0, f"can't reach {where} ({type(e).__name__}: {e})") from None
                problem = f"can't reach {where} ({type(e).__name__})"
            wait = min(self.backoff_base * 2 ** attempt, self.backoff_max)
            if time.monotonic() - started + wait > self.max_wait:
                raise ApiError(0, f"gave up on {method} {shown} after {self.max_wait:.0f}s: {problem}")
            log.warning("%s %s: %s; retrying in %.0fs", method, shown, problem, wait)
            time.sleep(wait)
            attempt += 1

    def json(self, method: str, path: str, **kw):
        raw = self.call(method, path, **kw)
        return json.loads(raw) if raw else None

    # ---- the worker API ----
    def ping(self) -> None:
        self.call("GET", "/workers/ping", retry=False, timeout=15)

    def claim(self, kind: str) -> dict | None:
        return self.json("POST", "/workers/claim", body={"kind": kind}, retry=False, timeout=30)

    def heartbeat(self, run_id: int, progress: dict | None = None, stage: str | None = None, retry: bool = False) -> bool:
        """Returns True if the run was cancelled and the worker should stop."""
        body = {"progress": progress or None, "stage": stage}
        return bool(self.json("POST", f"/workers/runs/{run_id}/heartbeat", body=body, retry=retry, timeout=30)["cancel"])

    def download(self, path: str, dest: Path) -> None:
        self.call("GET", path, out=dest)

    # ---- the live view on the Pi's screen ----
    def display_request(self) -> dict:
        """{"active": the request id that should be showing or None, "request": that request the first time it is asked for}"""
        return self.json("GET", "/workers/display", retry=False, timeout=15)

    def display_status(self, body: dict) -> None:
        self.call("POST", "/workers/display/status", body=body, retry=False, timeout=15)

    def display_file(self, name: str, dest: Path) -> None:
        self.call("GET", f"/workers/display/files/{name}", out=dest, retry=False, timeout=120)

    def upload(self, run_id: int, name: str, file: Path) -> None:
        self.call("PUT", f"/workers/runs/{run_id}/artifacts/{name}", file=file)
        log.info("uploaded %s (%.1f MB)", name, file.stat().st_size / 2**20)

    def complete(self, run_id: int, metrics: dict | None = None) -> None:
        self.call("POST", f"/workers/runs/{run_id}/complete", body={"metrics": metrics})

    def fail(self, run_id: int, error: str) -> None:
        try:
            self.call("POST", f"/workers/runs/{run_id}/fail", body={"error": error[-3900:]})
        except LeaseLost:
            pass  # someone else owns the run now; nothing to report


def _detail(raw: bytes) -> str:
    try:
        d = json.loads(raw)["detail"]
        return d if isinstance(d, str) else json.dumps(d)
    except (ValueError, KeyError, TypeError):
        return raw[:200].decode("utf-8", "replace") or "no details"


class Heartbeat:
    """Renews the lease from a background thread, so a slow step (a long epoch, the IMX export, packaging) never loses it.

    `update()` stores progress that goes out with the next beat. `cancelled` is set when the user cancels the run,
    `lost` when the backend gave the run to someone else; long steps should poll `stopped()` and wind down.
    """

    def __init__(self, api: Api, run_id: int, interval: float = 20.0):
        self.api, self.run_id, self.interval = api, run_id, interval
        self.cancelled, self.lost = threading.Event(), threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._progress: dict = {}
        self._thread = threading.Thread(target=self._run, name="heartbeat", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    def update(self, **progress) -> None:
        with self._lock:
            self._progress.update(progress)

    def stopped(self) -> bool:
        return self.cancelled.is_set() or self.lost.is_set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                progress, self._progress = self._progress, {}
            try:
                if self.api.heartbeat(self.run_id, progress):
                    log.info("run %s was cancelled", self.run_id)
                    self.cancelled.set()
                    return
            except LeaseLost as e:
                log.warning("lost run %s: %s", self.run_id, e.detail)
                self.lost.set()
                return
            except ApiError as e:  # an outage: put the progress back and try again next beat
                log.warning("heartbeat failed: %s", e)
                with self._lock:
                    self._progress = {**progress, **self._progress}


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
