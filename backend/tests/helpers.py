import io
import random

from PIL import Image as PILImage


def png(color=(10, 20, 30), size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def photo_image(seed: int, size=(320, 240)) -> PILImage.Image:
    """Random-noise 'photo': different seeds give clearly different images (and different dHashes)."""
    rnd = random.Random(seed)
    im = PILImage.new("L", (40, 30))
    im.putdata([rnd.randint(0, 255) for _ in range(40 * 30)])
    return im.resize(size).convert("RGB")


def encode(im: PILImage.Image, fmt="JPEG", **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def photo(seed: int, size=(320, 240), **kw) -> bytes:
    return encode(photo_image(seed, size), **kw)
