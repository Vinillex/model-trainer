"""The AI check itself: compare an image's boxes with what the detector sees, and let SAM tighten the suggestions.

Pure logic over two injected models, so it is tested with fakes and the real ones (app/ai/service.py) only load when a check runs:
  * the detector finds drones anywhere in the image (the trained YOLO now; RF-DETR could take its place),
  * SAM turns a rough box into a tight one. It cannot tell a right box from a wrong one (it segments whatever the box points at), so it
    only polishes: the verdict about a label comes from the detector.
Everything returned is a suggestion for a person to accept or ignore. Boxes in results are normalized x1,y1,x2,y2.
"""

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from .geometry import Box, inside, iou, side_on_input, to_normalized, to_pixels
from .sam import Refined
from .verdict import fingerprint, worst


class Detection(NamedTuple):
    box: Box  # normalized
    score: float
    class_name: str | None = None


class Models(Protocol):
    def detect(self, image_rgb: np.ndarray) -> list[Detection]: ...

    def refine(self, image_rgb: np.ndarray, prompt: Box | None = None, point: tuple[float, float] | None = None) -> Refined | None: ...


# Bump when the rules or thresholds change: it is part of every stored check's `model`, so the batch job redoes older results by itself.
#   v1: tight_iou 0.55.  v2: tight_iou 0.45 (on scraped, human-drawn labels 83% of the flags SAM raised alone sat at IoU 0.45-0.55 with the
#   detector agreeing: ordinary looseness, not an error).
POLICY_VERSION = 2


@dataclass(frozen=True)
class Policy:
    """Thresholds, from scripts/ai_spike.py (SAM on 97 exact labels, 0-8 px to 64+ px: none fell below IoU 0.6 against SAM's box), the
    first project-wide runs (human-drawn labels are looser than exact ones, hence 0.45 not 0.55 for tight_iou) and the size table in HANDOVER.md (the detector finds 48% of drones under 8 px, 69% at 8-12, 82% at 12-16,
    88% at 16-24, 90%+ above). Re-run the spike before changing them."""

    floor_px: float = 16.0  # a label smaller than this (side on the 640 px input) that nothing confirms is "too small to judge", not "wrong"
    detect_score: float = 0.25  # detections below this are ignored (it is also the threshold baked into the IMX500 export)
    confident_score: float = 0.5  # a detection this sure, with no label, is reported as a missing label
    agree_iou: float = 0.6  # a label and a detection overlapping this much are the same drone
    related_iou: float = 0.3  # overlapping this much they are still the same drone drawn sloppily: suggest the tighter box
    loose_inside: float = 0.8  # a detection with this share inside a label (and some overlap) means the label is a loose box around it
    tight_iou: float = 0.45  # SAM's tight box against the label: below this the label is clearly loose or off, suggest SAM's box
    sam_sane_iou: float = 0.5  # SAM's box must overlap the detector's, or SAM segmented something else and its box is not offered
    sam_object_score: float = 0.5  # SAM's own score for "there is a compact object in this box"


class LabelBox(NamedTuple):
    annotation_id: int | None
    box: Box  # normalized


def suggest(models: Models, image: np.ndarray, det: Detection, width: int, height: int, policy: Policy) -> tuple[Box, str]:
    """SAM's tightened version of a detection if it is believable, else the detection itself."""
    px = to_pixels(det.box, width, height)
    ref = models.refine(image, prompt=px)
    if ref is not None and not ref.touches_edge and iou(ref.box, px) >= policy.sam_sane_iou:
        return to_normalized(ref.box, width, height), "sam"
    return det.box, "detector"


def run_check(image: np.ndarray, labels: list[LabelBox], models: Models, policy: Policy = Policy()) -> dict:
    height, width = image.shape[:2]
    dets = [d for d in models.detect(image) if d.score >= policy.detect_score]
    claimed: set[int] = set()  # detections already explained by a label
    items = []
    for label in labels:
        side = side_on_input(label.box, width, height)
        overlaps = [iou(d.box, label.box) for d in dets]
        # the same drone even if the label is sloppy: overlapping enough, or sitting inside a loose label
        related = [i for i, d in enumerate(dets) if overlaps[i] >= policy.related_iou or (overlaps[i] >= 0.1 and inside(d.box, label.box) >= policy.loose_inside)]
        best = max(related, key=lambda i: (overlaps[i], dets[i].score), default=None)
        item = {"annotation_id": label.annotation_id, "box": list(label.box), "side_px": round(side, 1), "verdict": "ok", "reason": "",
                "suggested": None, "detector_iou": None, "detector_score": None, "sam_object": None}
        if best is not None:
            claimed.add(best)
            item |= {"detector_iou": round(overlaps[best], 3), "detector_score": round(dets[best].score, 3)}
            tight, how = suggest(models, image, dets[best], width, height, policy)
            if overlaps[best] >= policy.agree_iou and iou(tight, label.box) >= policy.tight_iou:
                item["reason"] = "the model finds a drone here and the box fits it"
            else:
                shape = ("loose around it" if inside(dets[best].box, label.box) >= policy.loose_inside
                         else "off it" if overlaps[best] >= policy.agree_iou else "only partly on it")
                item |= {"verdict": "adjust", "suggested": list(tight), "reason": f"the model finds the drone here but the box is {shape} ({how} box suggested)"}
        else:
            elsewhere = next((i for i in sorted(range(len(dets)), key=lambda i: -dets[i].score)
                              if dets[i].score >= policy.confident_score and i not in claimed and overlaps[i] < 0.1), None)
            if elsewhere is not None:
                claimed.add(elsewhere)
                tight, how = suggest(models, image, dets[elsewhere], width, height, policy)
                item |= {"verdict": "suspect", "suggested": list(tight), "detector_score": round(dets[elsewhere].score, 3),
                         "reason": f"the model sees a drone elsewhere in the image, not at this box ({how} box suggested)"}
            elif side < policy.floor_px:
                item |= {"verdict": "too_small", "reason": f"about {side:.0f} px across on the 640 px input: too small for the model to confirm either way"}
            else:
                # The detector does not confirm it. SAM, asked about the label's own box, says whether there is any compact object there at all.
                ref = models.refine(image, prompt=to_pixels(label.box, width, height))
                sam_object = ref is not None and not ref.touches_edge and ref.score >= policy.sam_object_score
                item |= {"verdict": "suspect", "sam_object": sam_object,
                         "reason": "the model finds no drone at this box" + (", though SAM does see an object there (a hard or unusual drone?)" if sam_object else ", and SAM sees no object there either")}
        items.append(item)

    missing = []
    for i, det in enumerate(dets):
        if i in claimed or det.score < policy.confident_score or any(iou(det.box, lb.box) >= 0.1 or inside(det.box, lb.box) >= 0.5 for lb in labels):
            continue
        tight, how = suggest(models, image, det, width, height, policy)
        missing.append({"box": list(tight), "detector_box": list(det.box), "score": round(det.score, 3), "class_name": det.class_name,
                        "side_px": round(side_on_input(tight, width, height), 1),
                        "reason": f"the model finds a drone with no label here ({how} box)"})

    summary = {k: sum(1 for it in items if it["verdict"] == k) for k in ("ok", "adjust", "suspect", "too_small")} | {"missing": len(missing), "labels": len(items)}
    verdict = worst([it["verdict"] for it in items] + ["missing"] * len(missing))
    return {"labels": items, "missing": missing, "summary": summary, "verdict": verdict,
            "fingerprint": fingerprint(lb.box for lb in labels)}
