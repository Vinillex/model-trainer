"""A detector that is told what to find in words (YOLOE, "yoloe-26s-seg"): "quadcopter drone", "bird". Imported lazily, needs requirements-ai.txt.

It was never trained on this project's classes, so it is a second opinion for images nobody has labelled and a way to look inside the negative
classes, not a replacement for the trained detector (it is weakest on tiny objects, and most drones here are tiny).

Its text encoder (MobileCLIP, 254 MB) is only needed to turn the prompts into numbers once, when the detector loads; it is dropped right after.
Ultralytics would look for it by bare file name in the working directory and download it THERE (outside data/), so it is built here from the
models folder instead."""

import threading
from pathlib import Path

import numpy as np

from .check import Detection
from .device import GPU_LOCK, Device

WEIGHTS_FILE = "yoloe-26s-seg.pt"
ENCODER_FILE = "mobileclip2_b.ts"


class YoloeDetector:
    """Finds the things described by `prompts`, a list of (label, text): detections come back named by the label (a project class name), so
    the rest of the AI code treats them like any detector's. One instance serves one prompt set (set_classes changes the model)."""

    def __init__(self, folder: Path, prompts: list[tuple[str, str]], device: str | None = None, imgsz: int = 640, conf: float = 0.10):
        if not prompts:
            raise ValueError("give at least one prompt")
        self.folder, self.prompts, self.device, self.imgsz, self.conf = Path(folder), list(prompts), Device(device), imgsz, conf
        self._model = None
        self._load_lock = threading.Lock()

    def class_names(self) -> list[str]:
        return [label for label, _ in self.prompts]

    def _load(self):
        with self._load_lock:
            if self._model is None:
                self._model = self._build()
        return self._model

    def _build(self):
        from ultralytics import YOLOE
        from ultralytics.nn.text_model import MobileCLIPTS
        from ultralytics.utils.downloads import attempt_download_asset

        self.folder.mkdir(parents=True, exist_ok=True)
        # A missing file is downloaded to exactly this path (the Ultralytics release asset); a present one is just used.
        weights = attempt_download_asset(str(self.folder / WEIGHTS_FILE))
        encoder = attempt_download_asset(str(self.folder / ENCODER_FILE))
        model = YOLOE(weights)
        texts = [text for _, text in self.prompts]
        with GPU_LOCK:  # torch work; the encoder runs where the weights were loaded (the CPU) and is gone before the first prediction
            inner = model.model
            inner.clip_model = MobileCLIPTS(next(inner.parameters()).device, weight=encoder)
            try:
                embeddings = inner.get_text_pe(texts, cache_clip_model=True)
            finally:
                inner.clip_model = None
            model.set_classes(texts, embeddings)
        return model

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        model = self._load()
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])  # Ultralytics wants BGR arrays, as cv2 gives them
        with GPU_LOCK:  # reading the boxes back is GPU work too
            result = self.device.call(lambda dev: model.predict(bgr, imgsz=self.imgsz, conf=self.conf, device=dev, verbose=False)[0])
            return [Detection(tuple(float(v) for v in box), float(score), self.prompts[int(cls)][0])
                    for box, score, cls in zip(result.boxes.xyxyn, result.boxes.conf, result.boxes.cls)]
