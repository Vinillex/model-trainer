"""Difference hash (dHash): 64 bits describing which neighbouring pixels of a 9x8 greyscale thumbnail are brighter.

Resized or re-compressed copies of one photo land within a few bits of each other. Caveat: near-featureless
images (e.g. clear sky with a tiny drone) also hash alike, so near-duplicate skipping is tunable per job.
"""

import io

from PIL import Image as PILImage


def dhash(data: bytes) -> int:
    with PILImage.open(io.BytesIO(data)) as im:
        # draft() lets JPEG decode at reduced size, which makes hashing large photos cheap.
        im.draft("L", (64, 64))
        small = im.convert("L").resize((9, 8), PILImage.Resampling.LANCZOS)
    px = list(small.getdata())
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (px[row * 9 + col] > px[row * 9 + col + 1])
    return bits


def to_hex(h: int) -> str:
    return f"{h:016x}"


def distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()
