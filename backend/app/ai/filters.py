"""The AI filters as SQL, written once. The gallery list, the per-dataset counts and the drawer badge all use these, so "needs a look" means
the same thing everywhere. Image.ai_status (Python) is the same rule for one image: tests keep the two in agreement."""

from sqlalchemy import and_, exists

from ..models import AiCheck, AiDismissal, Annotation, AnnotationStatus, Image
from .verdict import FLAGGED

AI_FILTER_VALUES = ("unchecked", "stale", "ok", "flagged", "dismissed", "too_small", "autolabelled")


def ai_condition(ai: str):
    """The WHERE condition (on Image) for one filter value: unchecked, stale, ok, flagged, dismissed, too_small or autolabelled."""
    if ai == "unchecked":
        return ~Image.ai_check.has()
    if ai == "stale":
        return Image.ai_check.has(AiCheck.stale.is_(True))
    if ai in ("flagged", "dismissed"):
        # a current, flagged check; "dismissed" = a person saved exactly those boxes after seeing the doubts
        dismissed = exists().where(AiDismissal.image_id == AiCheck.image_id, AiDismissal.fingerprint == AiCheck.fingerprint)
        return Image.ai_check.has(and_(AiCheck.stale.is_(False), AiCheck.verdict.in_(FLAGGED), dismissed if ai == "dismissed" else ~dismissed))
    if ai == "autolabelled":
        return exists().where(Annotation.image_id == Image.id, Annotation.status == AnnotationStatus.auto)
    verdicts = {"ok": ("ok", "none"), "too_small": ("too_small",)}[ai]
    return Image.ai_check.has(and_(AiCheck.stale.is_(False), AiCheck.verdict.in_(verdicts)))
