"""Box maths for the AI check. Plain Python + numpy, no torch."""

import math

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2

INPUT_SIDE = 640  # sizes are quoted on the 640 px network input, like the size tables in HANDOVER.md


def inter_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Box, b: Box) -> float:
    inter = inter_area(a, b)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(inner: Box, outer: Box) -> float:
    """Share of `inner` that lies within `outer`: 1 when a detection sits entirely inside a (loose) label."""
    a = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter_area(inner, outer) / a if a > 0 else 0.0


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def side_on_input(norm_box: Box, width: int, height: int) -> float:
    """Side (square root of the area) of a normalized box once its image is letterboxed to the 640 px network input."""
    scale = INPUT_SIDE / max(width, height)
    return math.sqrt(area(norm_box) * width * height) * scale


def to_pixels(norm_box: Box, width: int, height: int) -> Box:
    return (norm_box[0] * width, norm_box[1] * height, norm_box[2] * width, norm_box[3] * height)


def to_normalized(box: Box, width: int, height: int) -> Box:
    return (box[0] / width, box[1] / height, box[2] / width, box[3] / height)


def clip(box: Box, width: float, height: float) -> Box:
    return (min(max(box[0], 0.0), width), min(max(box[1], 0.0), height), min(max(box[2], 0.0), width), min(max(box[3], 0.0), height))


def crop_window(box: Box, width: int, height: int, context: float, min_side: int) -> tuple[int, int, int, int]:
    """A square window around a pixel box: `context` times its longer side but at least `min_side`, moved to stay inside the image.

    SAM sees its input at 1024 px, so cropping a tiny drone with its surroundings gives it far more pixels than the whole image would."""
    side = int(round(max(context * max(box[2] - box[0], box[3] - box[1]), min_side)))
    side = max(1, min(side, width, height))
    x1 = int(round((box[0] + box[2]) / 2 - side / 2))
    y1 = int(round((box[1] + box[3]) / 2 - side / 2))
    x1 = min(max(x1, 0), width - side)
    y1 = min(max(y1, 0), height - side)
    return x1, y1, x1 + side, y1 + side


def shift_box(box: Box, dx: float, dy: float) -> Box:
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def mask_to_box(mask: np.ndarray) -> Box | None:
    """Tight box (x1, y1, x2, y2; x2/y2 exclusive) around the True pixels of a mask, None if it is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
