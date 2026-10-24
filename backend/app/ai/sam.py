"""SAM 2.1 (through Ultralytics) turns a rough box or a click into a tight box. Imported only when a check runs: needs requirements-ai.txt."""

from pathlib import Path
from typing import NamedTuple

import numpy as np

from .device import GPU_LOCK, Device
from .geometry import Box, crop_window, mask_to_box, shift_box


class Refined(NamedTuple):
    box: Box  # tight box around SAM's mask, in the full image's pixels
    score: float  # SAM's own estimate of how good the mask is (0..1)
    mask_area: int  # pixels
    touches_edge: bool  # the mask runs to the edge of the crop: SAM segmented something bigger than the crop (the sky, a wall)


class SamRefiner:
    """One SAM model, loaded on first use. Calls are serialized (GPU_LOCK, shared with the detector): the model is not thread-safe and the Mac has one GPU."""

    def __init__(self, weights: Path, device: str | None = None, context: float = 4.0, min_crop: int = 96):
        self.weights, self.device, self.context, self.min_crop = Path(weights), Device(device), context, min_crop
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import SAM  # the heavy import; the rest of the backend never needs it

            self.weights.parent.mkdir(parents=True, exist_ok=True)
            self._model = SAM(str(self.weights))  # a missing official weights file (sam2.1_t.pt ...) is downloaded to this path
        return self._model

    def _predict(self, crop_bgr: np.ndarray, **prompt):
        model = self._load()
        return self.device.call(lambda dev: model(crop_bgr, verbose=False, device=dev, **prompt))

    def refine(self, image_rgb: np.ndarray, prompt: Box | None = None, point: tuple[float, float] | None = None) -> Refined | None:
        """Tight box for what a rough pixel box and/or one click point at, or None when SAM finds nothing there."""
        if prompt is None and point is None:
            raise ValueError("give a box, a point or both")
        height, width = image_rgb.shape[:2]
        anchor = prompt if prompt is not None else (point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8)
        x1, y1, x2, y2 = window = crop_window(anchor, width, height, self.context, self.min_crop)
        crop = np.ascontiguousarray(image_rgb[y1:y2, x1:x2, ::-1])  # Ultralytics wants BGR arrays, as cv2 gives them
        args = {}
        if prompt is not None:
            args["bboxes"] = [[float(v) for v in shift_box(prompt, -x1, -y1)]]
        if point is not None:
            args |= {"points": [[float(point[0] - x1), float(point[1] - y1)]], "labels": [1]}
        with GPU_LOCK:  # reading the mask back is GPU work too
            result = self._predict(crop, **args)[0]
            if result.masks is None or len(result.masks.data) == 0:
                return None
            mask = result.masks.data[0].cpu().numpy() > 0.5
            score = float(result.boxes.conf[0]) if result.boxes is not None and len(result.boxes) else 0.0
        box = mask_to_box(mask)
        if box is None:
            return None
        h, w = mask.shape
        touches = bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())
        return Refined(shift_box(box, x1, y1), score, int(mask.sum()), touches)
