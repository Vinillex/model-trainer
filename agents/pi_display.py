"""The Pi agent's display part: shows a finished run's model live on the Pi's screen when the Training page asks for it.

`DisplayManager.sync()` runs once per pass of the agent's loop. The backend says which request should be showing (`active`) and hands
each new request out once (`request`). The viewer (pi_viewer.py) is a child process in its own process group, so one signal ends the
viewer and anything it started. A request is used up when picked up: after a Pi or agent restart nothing starts by itself, and while the
backend can't be reached the viewer is left alone (only an answer from the backend changes what is showing).

The user's own camera program (main_ai.py) is never touched: if it is running, the request fails with a message and that's all.
"""

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from common import Api, ApiError

log = logging.getLogger("agent")

PROGRAM = "main_ai.py"            # the user's own camera program, which holds the camera while it runs
STARTUP_LIMIT = 10 * 60           # loading the network onto the chip "can take several minutes"
GRACE = (5.0, 3.0)                # seconds to wait after SIGINT, then after SIGTERM, before SIGKILL
UPLOAD_LINE = re.compile(r"Network Firmware Upload:\s+(\d+)%")  # the chip's loading bar, which the viewer's log carries
VIEWER_LINE = re.compile(r"\[viewer\] camera fps ([\d.]+), network results/s ([\d.]+), detections now (\d+)")
# What the viewer's log says when it can't work. A Qt failure does not end the process (it hangs, ignoring SIGINT), so the log is watched.
FAILURES = (re.compile(r"Could not load the Qt platform plugin", re.I), re.compile(r"no Qt platform plugin could be initialized", re.I),
            re.compile(r"Device or resource busy", re.I), re.compile(r"Pipeline handler in use", re.I),
            re.compile(r"failed to acquire camera", re.I), re.compile(r"Traceback \(most recent call last\)"))


class NoScreen(Exception):
    """There is nowhere to show anything."""


def camera_program_running(proc_root: Path = Path("/proc"), name: str = PROGRAM) -> bool:
    """Is the user's camera program running? Read from /proc so that no search string can match the searcher itself."""
    for entry in proc_root.glob("[0-9]*"):
        try:
            argv = [a.decode("utf-8", "replace") for a in (entry / "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        if argv and Path(argv[0]).name.startswith("python") and any(Path(a).name == name for a in argv[1:3]):
            return True
    return False


def session_env(runtime_root: Path = Path("/run/user"), uid: int | None = None, environ=os.environ) -> dict:
    """The environment that lets the viewer open a window on the Pi's desktop (Wayland). DISPLAY is left out on purpose: with it, Qt
    picks OpenCV's bundled xcb plugin and fails. Raises NoScreen when nobody is logged in on the Pi's screen."""
    runtime = runtime_root / str(os.getuid() if uid is None else uid)
    sockets = sorted(s for s in runtime.glob("wayland-[0-9]*") if not s.name.endswith(".lock")) if runtime.is_dir() else []
    if not sockets:
        raise NoScreen("The Pi has no desktop session on its screen (nobody is logged in there), so there is nowhere to show the picture.")
    return {"HOME": environ.get("HOME", str(Path.home())), "PATH": environ.get("PATH", "/usr/bin:/bin"), "XDG_RUNTIME_DIR": str(runtime),
            "WAYLAND_DISPLAY": sockets[0].name, "QT_QPA_PLATFORM": "wayland", "PYTHONUNBUFFERED": "1"}


class DisplayManager:
    def __init__(self, api: Api, workdir: Path, viewer: Path, popen=subprocess.Popen, killpg=os.killpg, busy=camera_program_running,
                 env_for=session_env, now=time.monotonic, sleep=time.sleep):
        self.api, self.workdir, self.viewer = api, Path(workdir).resolve(), Path(viewer).resolve()  # absolute: the viewer runs in another folder
        self.popen, self.killpg, self.busy, self.env_for, self.now, self.sleep = popen, killpg, busy, env_for, now, sleep
        self.proc = None
        self.request: dict | None = None
        self.dir: Path | None = None
        self.started = 0.0

    # ---- one pass ----
    def sync(self) -> None:
        try:
            info = self.api.display_request()
        except ApiError:
            return  # the backend can't be reached: leave whatever is showing alone
        active, req = info.get("active"), info.get("request")
        self._notice_exit()
        if self.proc and self.request["request_id"] != active:
            self._stop("stopped", "Stopped from the Training page.")
        if req and not (self.proc and self.request["request_id"] == req["request_id"]):
            self._start(req)  # reports "starting" itself
        elif self.proc:
            self._progress()

    def shutdown(self) -> None:
        if self.proc:
            self._stop("stopped", "The Pi agent is shutting down.")

    # ---- starting ----
    def _start(self, req: dict) -> None:
        rid = req["request_id"]
        if self.busy():
            return self._report(rid, "failed", f"{PROGRAM} is running on the Pi and holds the camera. Stop it there first; this will not stop it for you.")
        try:
            env = self.env_for()
        except NoScreen as e:
            return self._report(rid, "failed", str(e))
        self.dir = self.workdir / "display" / rid
        shutil.rmtree(self.dir.parent, ignore_errors=True)  # older requests' files
        self.dir.mkdir(parents=True)
        try:
            self.api.display_file("rpk", self.dir / "network.rpk")
            self.api.display_file("labels", self.dir / "labels.txt")
        except ApiError as e:
            self._cleanup()
            return self._report(rid, "failed", f"Could not download the model from the backend: {e}")
        log_file = open(self.dir / "viewer.log", "wb")
        try:
            self.proc = self.popen([sys.executable, str(self.viewer), "--model", str(self.dir / "network.rpk"), "--labels", str(self.dir / "labels.txt"),
                                    "--threshold", str(req["threshold"]), "--fps", str(req["fps"]), "--title", req["run_name"]],
                                   cwd=self.dir, env=env, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            log_file.close()  # the child has its own copy
        self.request, self.started = req, self.now()
        log.info("showing %s on the Pi's screen (request %s)", req["run_name"], rid[:8])
        self._report(rid, "starting", "Loading the model onto the camera chip. This takes a minute or two.")

    # ---- watching ----
    def _log_text(self) -> str:
        try:
            with open(self.dir / "viewer.log", "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 16 * 1024))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    @staticmethod
    def _failure(text: str) -> str | None:
        """The line of the log that says why the viewer can't work. For a Python traceback that is its last line (the error itself)."""
        lines = [re.sub(r"\x1b\[[0-9;]*m", "", l).strip() for l in text.splitlines()]
        lines = [l for l in lines if l]
        for i in range(len(lines) - 1, -1, -1):
            if any(p.search(lines[i]) for p in FAILURES):
                if lines[i].startswith("Traceback"):
                    return " ... ".join(lines[-2:])[:300]  # the file line and the error line at the end of the traceback
                return lines[i][:300]
        return None

    def _progress(self) -> None:
        rid, text = self.request["request_id"], self._log_text()
        if (bad := self._failure(text)) is not None:
            self._stop("failed", f"The viewer could not start: {bad}")
            return
        lines = VIEWER_LINE.findall(text)
        if lines:
            fps, results, dets = lines[-1]
            self._report(rid, "running", "Showing on the Pi's screen.", camera_fps=float(fps), results_per_s=float(results), detections=int(dets))
        elif self.now() - self.started > STARTUP_LIMIT:
            self._stop("failed", "The model did not start within 10 minutes.")
        else:
            loaded = UPLOAD_LINE.findall(text)
            self._report(rid, "starting", "Loading the model onto the camera chip" + (f": {loaded[-1]}%" if loaded else "") + ". This takes a minute or two.")

    def _notice_exit(self) -> None:
        """The viewer ended by itself (crashed, or somebody killed it)."""
        if self.proc and self.proc.poll() is not None:
            code, text = self.proc.returncode, self._log_text()
            bad = self._failure(text)
            self._finish("failed" if bad or code else "stopped",
                         f"The viewer could not run: {bad}" if bad else f"The viewer ended on the Pi (exit code {code})." if code else "The viewer ended on the Pi.")

    # ---- stopping ----
    def _stop(self, state: str, message: str) -> None:
        """Ctrl-C to the viewer's process group, then SIGTERM, then SIGKILL. Returns only when it is gone, so the camera is free again."""
        for sig, wait in ((signal.SIGINT, GRACE[0]), (signal.SIGTERM, GRACE[1]), (signal.SIGKILL, 5.0)):
            try:
                self.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                break
            deadline = self.now() + wait
            while self.proc.poll() is None and self.now() < deadline:
                self.sleep(0.2)
            if self.proc.poll() is not None:
                break
        self._finish(state, message)

    def _finish(self, state: str, message: str) -> None:
        rid = self.request["request_id"]
        self.proc, self.request = None, None
        self._cleanup()
        self._report(rid, state, message)

    def _cleanup(self) -> None:
        if self.dir:
            shutil.rmtree(self.dir.parent, ignore_errors=True)
            self.dir = None

    def _report(self, rid: str, state: str, message: str, **numbers) -> None:
        try:
            self.api.display_status({"request_id": rid, "state": state, "message": message, **numbers})
        except ApiError as e:  # the next pass reports again
            log.warning("could not report the display status: %s", e)
