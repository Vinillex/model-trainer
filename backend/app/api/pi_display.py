"""The Training page's "Run on Pi": show a finished run's model live on the Raspberry Pi's screen, and stop it again.

Only asks; the Pi agent does the showing (app/pi_display.py). Switching the camera and a screen on needs the `X-MT-Pi: 1` header, which
a web page open in the same browser cannot send without a CORS preflight the allow-list refuses.
"""

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import pi_display, training
from ..db import get_db
from ..models import RunStage
from .training import get_run

router = APIRouter(tags=["pi-display"])


def pi_header(x_mt_pi: str | None = Header(None)) -> None:
    if x_mt_pi != "1":
        raise HTTPException(403, "this switches on the Pi's camera and screen, so it must come from the app (missing X-MT-Pi header)")


def pi_online() -> bool:
    return any(w["kind"] == "pi" and w["online"] for w in training.list_workers())


class ShowIn(BaseModel):
    threshold: float = Field(0.25, ge=0.25, le=0.99)  # the export bakes in 0.25, so lower would change nothing
    fps: float | None = Field(None, gt=0, le=30)      # None = what the model's input size sustains


@router.get("/pi-display")
def pi_display_state():
    return pi_display.view(pi_online())


@router.post("/training-runs/{run_id}/pi-display", dependencies=[Depends(pi_header)])
def show_on_pi(run_id: int, body: ShowIn | None = None, db: Session = Depends(get_db)):
    body = body or ShowIn()
    run = get_run(db, run_id)
    if run.stage != RunStage.done:
        raise HTTPException(409, f"{run.name} is not finished yet (it is {run.stage.value}).")
    if "rpk" not in (run.artifacts or {}):
        raise HTTPException(409, f"{run.name} has no camera file (.rpk): Export for IMX500 was off for this run.")
    if not pi_online():
        raise HTTPException(409, "No Raspberry Pi is online. Switch it on and wait for it to show as online above.")
    fps = body.fps or pi_display.default_fps(int((run.config or {}).get("imgsz", 640)))
    pi_display.request(run.id, run.name, body.threshold, fps)
    return pi_display.view(True)


@router.delete("/pi-display", dependencies=[Depends(pi_header)])
def stop_on_pi():
    pi_display.stop()
    return pi_display.view(pi_online())
