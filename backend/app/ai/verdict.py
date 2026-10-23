"""What the AI check concludes about an image's boxes. Plain Python: the models run elsewhere, this only judges their output."""

import hashlib
from collections.abc import Iterable

from .geometry import Box

# Worst first. An image's verdict is the worst of its items.
ORDER = ("suspect", "missing", "adjust", "too_small", "ok", "none")


# The verdicts that mean "a person should look": what the gallery's "AI: needs a look" lists, unless the person has dismissed it.
FLAGGED = ("suspect", "missing", "adjust")


def worst(verdicts: Iterable[str]) -> str:
    seen = set(verdicts)
    return next((v for v in ORDER if v in seen), "none")


def fingerprint(boxes: Iterable[Box]) -> str:
    """Identifies a set of normalized boxes (order does not matter), to tell whether they changed since an image was checked."""
    parts = sorted(",".join(f"{v:.4f}" for v in b) for b in boxes)
    return hashlib.sha1(";".join(parts).encode()).hexdigest()[:16]
