"""What the Raspberry Pi's screen should show: the Training page's "Run on Pi" request, and what the Pi agent reports back.

The Pi agent is the side that knows what is really on the screen, so a request is *used up* once the agent has picked it up: if
the Pi restarts, or the agent does, the viewer is not started again by itself. The page sees that as the Pi going quiet.

State is one small JSON file (`pi_display_file`), so it survives a backend restart:
  desired  the request that should be showing: {request_id, run_id, run_name, threshold, fps, requested_at, picked_up}, or None
  status   the agent's last report: {request_id, run_id, run_name, state, message, camera_fps, results_per_s, detections, updated_at}
"""

import json
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import settings

_lock = threading.Lock()
FRESH = timedelta(seconds=25)        # the agent reports about every 5 s; silence for this long means the viewer is probably gone
FIRST_REPORT_LIMIT = timedelta(seconds=90)  # after picking a request up the agent checks, downloads and starts before its first report
PICKUP_LIMIT = timedelta(seconds=90)  # a request nobody picked up in this time is given up on (the Pi is off, or its agent is stuck)
KEEP_RESULT = timedelta(minutes=10)   # how long "closed" / "failed" stays on the page
ENDED = ("stopped", "failed", "ended")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _read() -> dict:
    try:
        data = json.loads(Path(settings.pi_display_file).read_text())
    except (OSError, ValueError):
        data = {}
    return {"desired": data.get("desired"), "status": data.get("status")}


def _write(state: dict) -> None:
    path = Path(settings.pi_display_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, path)


def _status(request: dict | None, state: str, message: str, **numbers) -> dict:
    return {"request_id": (request or {}).get("request_id"), "run_id": (request or {}).get("run_id"), "run_name": (request or {}).get("run_name"),
            "state": state, "message": message, "camera_fps": None, "results_per_s": None, "detections": None, **numbers,
            "updated_at": _now().isoformat()}


# ---------- the page ----------

def default_fps(imgsz: int) -> float:
    """The camera frame rate the chip sustains for a model of this input size (measured on the Pi 5 + AI Camera: 640 -> 15, 416 -> 23);
    sizes that were not measured get the slower, safe rate."""
    return {416: 23.0}.get(imgsz, 15.0)


def request(run_id: int, run_name: str, threshold: float, fps: float) -> dict:
    with _lock:
        desired = {"request_id": uuid.uuid4().hex, "run_id": run_id, "run_name": run_name, "threshold": threshold, "fps": fps,
                   "requested_at": _now().isoformat(), "picked_up": False, "picked_up_at": None}
        _write({"desired": desired, "status": None})
        return desired


def stop() -> None:
    """The agent sees that nothing should be showing and ends the viewer (and reports "stopped")."""
    with _lock:
        _write({**_read(), "desired": None})


def view(pi_online: bool) -> dict:
    """What the page shows. A request that went quiet is turned into an ended one here, so it never hangs on "running"."""
    with _lock:
        st = _read()
        desired, status, now = st["desired"], st["status"], _now()
        if desired:
            mine = status if status and status["request_id"] == desired["request_id"] else None
            if mine is None and not desired["picked_up"] and now - _parse(desired["requested_at"]) > PICKUP_LIMIT:
                status = _status(desired, "failed", "The Pi did not pick this up. Is the Pi on, and is its agent running (it shows as online above)?")
                desired = None
            elif desired["picked_up"] and now - (_parse(mine["updated_at"]) if mine else _parse(desired["picked_up_at"])) > (FRESH if mine else FIRST_REPORT_LIMIT):
                status = _status(desired, "ended", "The Pi stopped reporting, so the live view is probably gone (the Pi may have restarted).")
                desired = None
            if desired is None:
                _write({"desired": None, "status": status})
        out = {"state": "none", "pi_online": pi_online, "request_id": None, "run_id": None, "run_name": None, "message": "", "camera_fps": None,
               "results_per_s": None, "detections": None, "threshold": None, "fps": None}
        if desired:
            mine = status if status and status["request_id"] == desired["request_id"] else None
            out.update(state=(mine["state"] if mine else "waiting"), request_id=desired["request_id"], run_id=desired["run_id"],
                       run_name=desired["run_name"], threshold=desired["threshold"], fps=desired["fps"],
                       message=(mine["message"] if mine else "Waiting for the Pi to pick this up…"))
            if mine:
                out.update(camera_fps=mine["camera_fps"], results_per_s=mine["results_per_s"], detections=mine["detections"])
        elif status and status["state"] in ENDED and now - _parse(status["updated_at"]) < KEEP_RESULT:
            out.update(state=status["state"], request_id=status["request_id"], run_id=status["run_id"], run_name=status["run_name"],
                       message=status["message"])
        return out


# ---------- the Pi agent ----------

def for_worker() -> dict:
    """`active` is what should be showing; `request` is the request itself, handed out once."""
    with _lock:
        st = _read()
        desired = st["desired"]
        if desired is None:
            return {"active": None, "request": None}
        fresh = not desired["picked_up"]
        if fresh:
            _write({"desired": {**desired, "picked_up": True, "picked_up_at": _now().isoformat()}, "status": st["status"]})
        return {"active": desired["request_id"], "request": desired if fresh else None}


def report(body: dict) -> None:
    """The agent's report. Reports about a request that is no longer the wanted one are dropped, except how an old one ended."""
    with _lock:
        st = _read()
        desired = st["desired"]
        rid, state = body["request_id"], body["state"]
        if desired and rid != desired["request_id"]:
            return
        if desired is None and state not in ENDED:
            return
        ref = desired or {"request_id": rid, "run_id": (st["status"] or {}).get("run_id"), "run_name": (st["status"] or {}).get("run_name")}
        status = _status(ref, state, body.get("message") or "", camera_fps=body.get("camera_fps"), results_per_s=body.get("results_per_s"),
                         detections=body.get("detections"))
        _write({"desired": None if state in ENDED else desired, "status": status})


def wanted_run() -> int | None:
    """The run whose files the agent may download: only the one in the current request."""
    with _lock:
        desired = _read()["desired"]
        return desired["run_id"] if desired else None
