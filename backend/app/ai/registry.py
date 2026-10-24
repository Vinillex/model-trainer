"""Which models a run can use: the detector (a finished training run, or a .pt file the person uploaded) and the SAM size.

Uploaded weights are only ever stored here, never opened: an Ultralytics .pt is a pickle, and loading one runs whatever code is inside it.
So a file is loaded only when the person picks it for a run (see service.get_models), never when it arrives."""

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..models import RunStage, TrainingRun


class AiUnavailable(RuntimeError):
    """The AI extras are not installed, or there is no detector yet. The message says what to do."""


SAM_FILES = {"t": "sam2.1_t.pt", "s": "sam2.1_s.pt", "b": "sam2.1_b.pt", "l": "sam2.1_l.pt"}
SAM_HINTS = {"t": "tiny: fastest, roughest boxes", "s": "small: the default", "b": "base: tighter boxes, slower", "l": "large: tightest boxes, slowest"}
FILE_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,76}\.pt"
YOLOE_ID = "yoloe:26s"  # the text-prompt detector (app/ai/yoloe.py); its files live in models_dir()/yoloe and are downloaded on first use
YOLOE_LABEL = "YOLOE-26s (text prompts)"
YOLOE_DOWNLOAD_MB = 285
DETECTOR_PATTERN = rf"^(run:\d+|file:{FILE_NAME}|{YOLOE_ID})$"
MAX_UPLOAD = 300 * 1024 * 1024  # the biggest YOLO11 model is ~120 MB


def models_dir() -> Path:
    return Path(settings.ai_dir) / "models"


def custom_dir() -> Path:
    return models_dir() / "custom"


def yoloe_dir() -> Path:
    return models_dir() / "yoloe"


def yoloe_downloaded() -> bool:
    from .yoloe import ENCODER_FILE, WEIGHTS_FILE

    return all((yoloe_dir() / f).is_file() for f in (WEIGHTS_FILE, ENCODER_FILE))


def clean_model_name(name: str) -> str | None:
    """A weights file name that is safe on disk: letters, digits, dot, dash, underscore, ending .pt. None when nothing usable is left."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name.replace("\\", "/")).name.strip()).strip(".-_")
    return base if re.fullmatch(FILE_NAME, base) else None


def run_weights(run: TrainingRun) -> Path | None:
    """best.pt of a finished run, copied out of storage once; None if the run has none."""
    from .. import storage  # not at the top: the app's storage module is swapped in tests

    key = (run.artifacts or {}).get("best_pt", {}).get("key")
    if run.stage != RunStage.done or not key:
        return None
    path = models_dir() / f"detector-run{run.id}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(storage.get_storage().get(key))
    return path


def list_detectors(db: Session) -> list[dict]:
    """Every detector a run can use: finished training runs (newest first), uploaded files, then the text-prompt detector."""
    out = []
    for run in db.scalars(select(TrainingRun).where(TrainingRun.stage == RunStage.done).order_by(TrainingRun.id.desc())):
        if (run.artifacts or {}).get("best_pt", {}).get("key"):
            out.append({"id": f"run:{run.id}", "label": f"yolo run {run.id} ({run.name})", "kind": "run", "map50": (run.metrics or {}).get("map50")})
    if custom_dir().is_dir():
        out += [{"id": f"file:{p.name}", "label": f"{p.name} (uploaded)", "kind": "file", "map50": None}
                for p in sorted(custom_dir().glob("*.pt")) if re.fullmatch(FILE_NAME, p.name)]
    out.append({"id": YOLOE_ID, "label": YOLOE_LABEL, "kind": "prompt", "map50": None, "downloaded": yoloe_downloaded()})
    return out


def resolve_detector(db: Session, detector: str | None, default) -> tuple[str, Path, str]:
    """(cache key, weights path, label) for a detector choice. None = the default (`default(db)` gives (path, label) or None)."""
    if detector is None:
        found = default(db)
        if found is None:
            raise AiUnavailable("No detector yet: finish a training run first (its best.pt is the detector), or set MT_AI_DETECTOR_WEIGHTS.")
        return f"default:{found[0]}", found[0], found[1]
    if not re.fullmatch(DETECTOR_PATTERN, detector):
        raise AiUnavailable(f"unknown detector {detector!r}")
    if detector == YOLOE_ID:
        return detector, yoloe_dir(), YOLOE_LABEL  # a folder, not a file: it holds the weights and the text encoder
    kind, _, name = detector.partition(":")
    if kind == "run":
        run = db.get(TrainingRun, int(name))
        path = run_weights(run) if run else None
        if path is None:
            raise AiUnavailable(f"training run {name} has no finished model to use as a detector")
        return detector, path, f"yolo run {run.id} ({run.name})"
    path = custom_dir() / name
    if not path.is_file():
        raise AiUnavailable(f"the uploaded model {name} is not there any more; upload it again")
    return detector, path, f"{name} (uploaded)"


def save_upload(stream, filename: str) -> str:
    """Store an uploaded .pt without opening it. Returns its detector id. Raises ValueError (message for the person) when it is refused.
    Written to a temporary name and renamed, so a refused or aborted upload leaves nothing behind."""
    name = clean_model_name(filename or "")
    if name is None:
        raise ValueError("the file name must end in .pt (letters, digits, dot, dash and underscore only)")
    target = custom_dir() / name
    if target.exists():
        raise FileExistsError(f"a model called {name} is already there; rename the file")
    custom_dir().mkdir(parents=True, exist_ok=True)
    tmp = custom_dir() / f".{name}.part"
    size = 0
    try:
        with tmp.open("wb") as out:
            first = True
            while chunk := stream.read(1 << 20):
                if first and not chunk.startswith(b"PK\x03\x04"):  # PyTorch saves a zip archive; anything else is not a current .pt file
                    raise ValueError("this does not look like a PyTorch .pt file (it is not a zip archive)")
                first = False
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise OverflowError(f"the file is larger than {MAX_UPLOAD // (1024 * 1024)} MB")
                out.write(chunk)
        if size == 0:
            raise ValueError("the file is empty")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return f"file:{name}"
