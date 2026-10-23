"""Which device the AI models run on, and what happens when the GPU refuses a call."""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
GIVE_UP_AFTER = 5  # failures in a row on the GPU before the CPU is used for good


class Device:
    """The Mac's GPU ("mps") when torch has it, else the CPU. A call the GPU fails is retried on the CPU for that call only: one odd image
    must not slow a batch of thousands for good. Only GIVE_UP_AFTER failures in a row move everything to the CPU. Every fallback is counted
    and its reason kept, so a slow batch can be explained."""

    def __init__(self, wanted: str | None = None):
        self.wanted = wanted
        self.name: str | None = None  # known once resolved
        self.fallbacks = 0
        self.last_error: str | None = None
        self._streak = 0

    def resolve(self) -> str:
        if self.name is None:
            if self.wanted:
                self.name = self.wanted
            else:
                import torch  # lazy: the backend runs without torch installed

                self.name = "mps" if torch.backends.mps.is_available() else "cpu"
        return self.name

    def call(self, fn: Callable[[str], T]) -> T:
        name = self.resolve()
        if name == "cpu":
            return fn("cpu")
        try:
            out = fn(name)
            self._streak = 0
            return out
        except Exception as e:  # noqa: BLE001 - whatever the GPU did, the CPU gets one try
            self.fallbacks += 1
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            self._streak += 1
            if self._streak >= GIVE_UP_AFTER:
                self.name = "cpu"
            return fn("cpu")

    def summary(self) -> dict:
        return {"device": self.name or self.wanted, "fallbacks": self.fallbacks, "last_error": self.last_error}
