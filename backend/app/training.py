"""Training runs: the rules for who may do what to a run, and when.

Stages: queued -> training -> exporting -> awaiting_package -> packaging -> done (or failed / cancelled).
GPU workers do the first three; the Raspberry Pi does packaging. A worker holds a *lease* on its run and
renews it with heartbeats; if it goes silent the run goes back in the queue (after MAX_ATTEMPTS it fails).
"""

import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .config import settings
from .models import DatasetVersion, Project, RunStage, TrainingRun
from .schemas import TrainingRunCreate
from .storage import get_storage

GPU_STAGES = (RunStage.training, RunStage.exporting)
PI_STAGES = (RunStage.packaging,)
ACTIVE = GPU_STAGES + PI_STAGES
TERMINAL = (RunStage.done, RunStage.failed, RunStage.cancelled)
MAX_ATTEMPTS = 3

ARTIFACT_FILES = {"best_pt": "best.pt", "imx_zip": "packerOut.zip", "labels": "labels.txt", "rpk": "network.rpk"}
UPLOADABLE = {"gpu": ("best_pt", "imx_zip", "labels"), "pi": ("rpk",)}
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


class NotFound(LookupError):
    pass


class LeaseLost(Exception):
    """The caller no longer holds this run (lease expired and was reassigned, run cancelled, wrong worker)."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _lease() -> datetime:
    return _now() + timedelta(seconds=settings.worker_lease_seconds)


def kind_of(stage: RunStage) -> str:
    return "gpu" if stage in GPU_STAGES else "pi"


# ---------- creating, cancelling, deleting (the person using the UI) ----------

def create_run(db: Session, project: Project, body: TrainingRunCreate) -> TrainingRun:
    version = db.get(DatasetVersion, body.dataset_id)
    if version is None or version.project_id != project.id:
        raise NotFound("dataset version not found in this project")
    images, boxes = version.stats.get("images", {}), version.stats.get("boxes", {})
    # data.yaml always names images/val, and a model with no training boxes can't learn anything.
    if images.get("val", 0) == 0:
        raise ValueError(f"Dataset version '{version.name}' has no validation images, so training couldn't measure accuracy. "
                         "Review more images and create a new version.")
    if boxes.get("train", 0) == 0:
        raise ValueError(f"Dataset version '{version.name}' has no training boxes. Review and box some images, then create a new version.")
    run = TrainingRun(project_id=project.id, dataset_id=version.id, name=body.name or "",
                      config=body.model_dump(exclude={"dataset_id", "name"}))
    db.add(run)
    db.flush()
    if not run.name:
        run.name = f"run-{run.id}"
    return run


def cancel_run(db: Session, run: TrainingRun) -> TrainingRun:
    if run.stage in TERMINAL:
        raise ValueError(f"The run is already {run.stage.value}.")
    if run.stage in (RunStage.queued, RunStage.awaiting_package):
        run.stage, run.finished_at = RunStage.cancelled, _now()  # nobody is working on it
    else:
        run.cancel_requested = True  # the worker learns on its next heartbeat
    return run


def delete_run(db: Session, run: TrainingRun) -> None:
    if run.stage in ACTIVE:
        raise ValueError("The run is in progress. Cancel it first.")
    storage = get_storage()
    for art in run.artifacts.values():
        storage.delete(art["key"])
    db.delete(run)


# ---------- leases (the workers) ----------

def reap_expired(db: Session) -> None:
    """Take runs back from workers that went silent."""
    now = _now()
    for run in db.scalars(select(TrainingRun).where(TrainingRun.stage.in_(ACTIVE), TrainingRun.lease_expires_at < now)).all():
        values: dict = {"worker": None, "lease_expires_at": None}
        if run.cancel_requested:
            values |= {"stage": RunStage.cancelled, "finished_at": now}
        elif run.attempts + 1 >= MAX_ATTEMPTS:
            values |= {"stage": RunStage.failed, "finished_at": now, "attempts": run.attempts + 1,
                       "error": f"The worker stopped responding {MAX_ATTEMPTS} times; giving up."}
        else:
            values |= {"stage": RunStage.queued if run.stage in GPU_STAGES else RunStage.awaiting_package,
                       "attempts": run.attempts + 1,
                       "progress": {"message": f"Requeued: worker '{run.worker}' stopped responding"}}
        # Conditional on the lease we saw, so a heartbeat that renewed it in the meantime wins.
        db.execute(update(TrainingRun).where(TrainingRun.id == run.id, TrainingRun.lease_expires_at == run.lease_expires_at).values(**values))
    db.commit()


def claim(db: Session, kind: str, worker: str) -> TrainingRun | None:
    reap_expired(db)
    waiting, active = (RunStage.queued, RunStage.training) if kind == "gpu" else (RunStage.awaiting_package, RunStage.packaging)
    run_id = db.scalar(select(TrainingRun.id).where(TrainingRun.stage == waiting).order_by(TrainingRun.id).limit(1))
    if run_id is None:
        return None
    values: dict = {"stage": active, "worker": worker, "lease_expires_at": _lease(),
                    "started_at": func.coalesce(TrainingRun.started_at, _now())}
    if kind == "gpu":
        values["progress"] = {}
    # The conditional UPDATE makes the claim atomic when two workers ask at once.
    res = db.execute(update(TrainingRun).where(TrainingRun.id == run_id, TrainingRun.stage == waiting).values(**values))
    db.commit()
    return db.get(TrainingRun, run_id, populate_existing=True) if res.rowcount == 1 else None


def owned(db: Session, run_id: int, worker: str, stages=ACTIVE) -> TrainingRun:
    run = db.get(TrainingRun, run_id, populate_existing=True)
    if run is None:
        raise NotFound("training run not found")
    if run.worker != worker or run.stage not in stages:
        raise LeaseLost(f"'{worker}' no longer holds run {run_id} (it is {run.stage.value}"
                        f"{', held by ' + run.worker if run.worker else ''})")
    return run


def heartbeat(db: Session, run_id: int, worker: str, progress: dict | None, stage: str | None) -> bool:
    """Renew the lease and record progress. Returns True if the run was cancelled and the worker should stop."""
    run = owned(db, run_id, worker)
    if run.cancel_requested:
        run.stage, run.finished_at, run.worker, run.lease_expires_at = RunStage.cancelled, _now(), None, None
        db.commit()
        return True
    if stage:
        if run.stage != RunStage.training:
            raise ValueError(f"Can't move to {stage} from {run.stage.value}.")
        run.stage = RunStage.exporting
    if progress:
        run.progress = {**run.progress, **progress}
    run.lease_expires_at = _lease()
    db.commit()
    return False


def store_artifact(db: Session, run_id: int, worker: str, name: str, data: bytes) -> TrainingRun:
    run = owned(db, run_id, worker)
    if name not in UPLOADABLE[kind_of(run.stage)]:
        raise ValueError(f"A {kind_of(run.stage)} worker can't upload '{name}' (allowed: {', '.join(UPLOADABLE[kind_of(run.stage)])}).")
    key = f"projects/{run.project_id}/training/{run.id}/{ARTIFACT_FILES[name]}"
    get_storage().put(key, data)
    run.artifacts = {**run.artifacts, name: {"key": key, "size": len(data)}}
    run.lease_expires_at = _lease()  # an upload is a sign of life
    db.commit()
    return run


def complete(db: Session, run_id: int, worker: str, metrics: dict | None) -> TrainingRun:
    run = owned(db, run_id, worker)
    if run.stage in GPU_STAGES:
        export = run.config.get("export_imx", True)
        need = ["best_pt"] + (["imx_zip", "labels"] if export else [])
        nxt = RunStage.awaiting_package if export else RunStage.done
        run.metrics = metrics or {}
    else:
        need, nxt = ["rpk"], RunStage.done
    missing = [n for n in need if n not in run.artifacts]
    if missing:
        raise ValueError(f"Can't complete: missing artifact(s) {', '.join(missing)}.")
    run.stage, run.worker, run.lease_expires_at = nxt, None, None
    if nxt == RunStage.done:
        run.finished_at = _now()
    db.commit()
    return run


def fail(db: Session, run_id: int, worker: str, error: str) -> TrainingRun:
    run = owned(db, run_id, worker)
    run.stage, run.error, run.finished_at = RunStage.failed, error[:2000], _now()
    run.worker = run.lease_expires_at = None
    db.commit()
    return run


# ---------- which workers have been seen (in memory: it only drives an "online" light in the UI) ----------

_workers: dict[str, dict] = {}
_workers_lock = threading.Lock()
ONLINE_SECONDS = 30


def touch_worker(name: str, kind: str, run_id: int | None = None) -> None:
    with _workers_lock:
        _workers[name] = {"name": name, "kind": kind, "last_seen": _now(), "run_id": run_id}


def list_workers() -> list[dict]:
    now = _now()
    with _workers_lock:
        for stale in [k for k, v in _workers.items() if now - v["last_seen"] >= timedelta(days=1)]:
            del _workers[stale]
        return [{**w, "online": now - w["last_seen"] < timedelta(seconds=ONLINE_SECONDS)}
                for w in sorted(_workers.values(), key=lambda w: w["name"])]
