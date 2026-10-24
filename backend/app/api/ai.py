from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..ai import job as ai_job, registry, service
from ..config import settings
from ..db import get_db
from ..ai.prompts import DRONE_PROMPT, default_object_prompt
from ..models import DetClass, Image, Job
from ..schemas import (AiBoxIn, AiBoxOut, AiCheckOut, AiDetectorOut, AiJobCreate, AiModelsOut, AiObjectPrompt, AiPreviewOut, AiPromptsOut, AiSamOut,
                       AiStatus, JobOut)
from .projects import get_project

# Loopback-only like the rest of the UI API (the LAN guard only opens /workers/*): nobody else on the network can run a check.
router = APIRouter(tags=["ai"])


def _image(db: Session, image_id: int) -> Image:
    image = db.get(Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return image


@router.get("/ai/status", response_model=AiStatus)
def ai_status(db: Session = Depends(get_db)):
    """Whether the AI check can run here, and what it uses. Cheap: loads no model."""
    reason = service.missing_extras()
    found = service.detector_weights(db)  # None until a training run has finished: the text-prompt detector needs none, so the AI is available anyway
    live = service.device_summary() or {}
    return AiStatus(available=reason is None, reason=reason, sam_weights=settings.ai_sam_weights, detector=found[1] if found else None,
                    device=live.get("device") or settings.ai_device, fallbacks=live.get("fallbacks", 0), last_fallback=live.get("last_error"))


@router.get("/ai/models", response_model=AiModelsOut)
def ai_models(db: Session = Depends(get_db)):
    """What a run can choose from: detectors (finished training runs and uploaded .pt files) and the SAM sizes. Loads nothing."""
    found = service.detector_weights(db)
    sams = [AiSamOut(size=k, file=f, hint=registry.SAM_HINTS[k], downloaded=(registry.models_dir() / f).exists()) for k, f in registry.SAM_FILES.items()]
    return AiModelsOut(detectors=[AiDetectorOut(**d) for d in registry.list_detectors(db)], default_detector=found[1] if found else None,
                       sams=sams, default_sam=settings.ai_sam_weights)


@router.post("/ai/models/upload", response_model=AiDetectorOut, status_code=201)
def upload_ai_model(file: UploadFile = File(...), x_mt_upload: str | None = Header(None)):
    """Store a .pt the person wants to use as a detector. It is NOT opened here: a .pt is a pickle and opening one runs code, so it is only
    loaded when a run picks it. The header is what keeps other web pages out: a cross-site form can send a file but not this header (the browser
    would first ask the CORS allow-list, which only lists this app's own pages)."""
    if x_mt_upload != "1":
        raise HTTPException(403, "model uploads must come from the app (missing X-MT-Upload header)")
    try:
        detector_id = registry.save_upload(file.file, file.filename or "")
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except OverflowError as e:
        raise HTTPException(413, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))
    name = detector_id.partition(":")[2]
    return AiDetectorOut(id=detector_id, label=f"{name} (uploaded)", kind="file")


@router.post("/images/{image_id}/ai-check", response_model=AiCheckOut)
def run_ai_check(image_id: int, db: Session = Depends(get_db)):
    """Check one image now (the first call loads the models and takes a while). Only stores a suggestion row: no box or status changes."""
    image = _image(db, image_id)
    try:
        return service.check_image(db, image)
    except service.AiUnavailable as e:
        raise HTTPException(503, str(e))


@router.get("/images/{image_id}/ai-check", response_model=AiCheckOut)
def read_ai_check(image_id: int, db: Session = Depends(get_db)):
    image = _image(db, image_id)
    if image.ai_check is None:
        raise HTTPException(404, "this image has not been checked by the AI")
    return image.ai_check


@router.delete("/images/{image_id}/ai-dismiss", response_model=AiCheckOut)
def undo_ai_dismiss(image_id: int, db: Session = Depends(get_db)):
    """Put the image back in "AI: needs a look": forget that a person reviewed its boxes despite the AI's doubts. Harmless if it was not dismissed."""
    image = _image(db, image_id)
    if image.ai_check is None:
        raise HTTPException(404, "this image has not been checked by the AI")
    image.ai_dismissal = None
    db.commit()
    db.refresh(image)
    return image.ai_check


@router.post("/images/{image_id}/ai-box", response_model=AiBoxOut)
def ai_box(image_id: int, body: AiBoxIn, db: Session = Depends(get_db)):
    """Click-to-box: SAM's tight box around a click and/or a rough box. Nothing is stored."""
    image = _image(db, image_id)
    if (body.x is None) != (body.y is None):
        raise HTTPException(422, "give both x and y, or a box")
    if body.x is None and body.box is None:
        raise HTTPException(422, "give a point (x, y), a box, or both")
    try:
        return service.suggest_box(db, image, body.x, body.y, body.box)
    except service.AiUnavailable as e:
        raise HTTPException(503, str(e))


@router.post("/projects/{project_id}/jobs/ai-check", response_model=JobOut, status_code=201)
def create_ai_job(project_id: int, body: AiJobCreate, db: Session = Depends(get_db)):
    get_project(db, project_id)
    if (reason := service.missing_extras()) is not None:
        raise HTTPException(503, reason)  # fail now rather than queue a job that is certain to fail
    prompted = body.detector == registry.YOLOE_ID
    if body.mode == "negatives":
        if body.detector not in (None, registry.YOLOE_ID):
            raise HTTPException(422, "looking inside the negative classes needs the text-prompt detector")
        prompted = True
    elif prompted and body.mode == "check":
        raise HTTPException(422, "the box check was tuned on the trained detector and is not offered with the text-prompt detector")
    if not prompted and body.min_score is not None and body.min_score < 0.25:
        raise HTTPException(422, "a trained detector is only used from 25% sure (its own threshold); lower values are for the text-prompt detector")
    if body.detector is not None:  # a choice that can't work is refused now, not after the job was queued
        try:
            registry.resolve_detector(db, body.detector, service.detector_weights)
        except service.AiUnavailable as e:
            raise HTTPException(422, str(e))
    params = body.model_dump(mode="json")
    if prompted:
        params["detector"] = registry.YOLOE_ID
        try:
            ai_job.build_prompts(db, project_id, params)  # the words are checked now: same phrase twice, a class that is not a negative one
        except service.AiUnavailable as e:
            raise HTTPException(422, str(e))
    job = Job(project_id=project_id, type="ai_check", params=params)
    db.add(job)
    db.commit()
    return job


@router.get("/projects/{project_id}/ai/prompts", response_model=AiPromptsOut)
def ai_prompts(project_id: int, db: Session = Depends(get_db)):
    """The words the text-prompt detector gets by default: one phrase that finds drones, and for each hard-negative class the words for its own thing."""
    get_project(db, project_id)
    negatives = db.query(DetClass).filter(DetClass.project_id == project_id, DetClass.is_negative.is_(True)).order_by(DetClass.id).all()
    return AiPromptsOut(drone_prompt=DRONE_PROMPT,
                        negative_classes=[AiObjectPrompt(class_id=c.id, name=c.name, prompt=default_object_prompt(c.name)) for c in negatives])


@router.post("/projects/{project_id}/jobs/ai-check/preview", response_model=AiPreviewOut)
def preview_ai_job(project_id: int, body: AiJobCreate, db: Session = Depends(get_db)):
    """How many images this job would process, by the same selection the job uses. Starts nothing."""
    get_project(db, project_id)
    return AiPreviewOut(count=len(ai_job.select_images(db, project_id, body.model_dump(mode="json"))))
