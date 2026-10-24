"""The detector half of the AI check: the trained YOLO (best.pt of a finished run) finds drones anywhere in an image. Imported lazily."""

from pathlib import Path

import numpy as np

from .check import Detection
from .device import GPU_LOCK, Device


class Detector:
    def __init__(self, weights: Path, device: str | None = None, imgsz: int = 640, conf: float = 0.15):
        self.weights, self.device, self.imgsz, self.conf = Path(weights), Device(device), imgsz, conf
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
        return self._model

    def class_names(self) -> list[str]:
        """What the weights were trained to find (loads them)."""
        return [str(n) for n in self._load().names.values()]

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        model = self._load()
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])  # Ultralytics wants BGR arrays, as cv2 gives them
        with GPU_LOCK:  # reading the boxes back is GPU work too
            result = self.device.call(lambda dev: model(bgr, imgsz=self.imgsz, conf=self.conf, device=dev, verbose=False)[0])
            return [Detection(tuple(float(v) for v in box), float(score), result.names.get(int(cls)))
                    for box, score, cls in zip(result.boxes.xyxyn, result.boxes.conf, result.boxes.cls)]
