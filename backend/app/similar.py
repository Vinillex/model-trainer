"""Leave near-repeats out of a dataset version: the same scene with the drone box in (nearly) the same place.

Video datasets hold runs of frames where nothing moves but the drone, a few pixels at a time. Training on all of them costs
time and teaches the same thing again. Two images count as similar when their dHashes (the scene) are at most HASH_BITS
apart AND their boxes pair up one to one: same class, centre within `box_pct` % of the picture, width and height within
SIZE_RATIO of each other. The dHash alone is not enough: it is computed on a 9x8 thumbnail, so a small drone does not show
in it, and the same sky with the drone somewhere else would look identical.

Only images with boxes take part: a box-less image (a hard negative) has nothing to compare but its scene, and featureless
sky hashes alike. Images are visited in id order (import and frame order); an image is left out when an image already kept
is similar to it, so exactly one image of each run of repeats stays.
"""

import numpy as np

HASH_BITS = 4  # dHash distance for "same scene"; 5-8 bits is where look-alikes start (HANDOVER Gotchas)
SIZE_RATIO = 1.5  # a box at most this much wider or taller than its partner, or it has zoomed: not a repeat


def _popcount(x: np.ndarray) -> np.ndarray:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def boxes_match(a: list, b: list, tol: float) -> bool:
    """Boxes [cls, cx, cy, w, h] (normalised) of two images pair up one to one within `tol` (a fraction of the picture)."""
    if len(a) != len(b):
        return False
    free = list(b)
    for c, cx, cy, w, h in a:
        for k, (c2, cx2, cy2, w2, h2) in enumerate(free):
            if (c == c2 and abs(cx - cx2) <= tol and abs(cy - cy2) <= tol
                    and w <= w2 * SIZE_RATIO and w2 <= w * SIZE_RATIO and h <= h2 * SIZE_RATIO and h2 <= h * SIZE_RATIO):
                del free[k]
                break
        else:
            return False
    return True


def scene_groups(items: list[tuple[int, str, str, bool]], hashes: dict[int, int]) -> dict[int, str]:
    """Join split groups whose images show the same scene, so a scene never sits in train and in val/test at once.

    `items` = [(image id, its split group, its source label, is imported)]. Two images are the same scene when their dHashes are at
    most HASH_BITS apart and they come from the same source, or both from imported datasets: public drone datasets copy each other's
    videos (cmy-uav holds frames of the very scenes in Anti-UAV), while scraped pictures already group by their page. Joining is
    transitive, so a video's frames chain into one group. Measured on the real data (2026-10-26): cross-split near-copies 9,368 -> 1,
    largest group 320 of 11,971. Wider (6 bits, or scraped joined with imported) lets plain skies clump into groups of 1,000+.
    Returns {image id: group}; an image's group is unchanged unless it was joined with another.
    """
    groups = {iid: g for iid, g, _, _ in items}
    parent = {g: g for g in groups.values()}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    hashed = [(iid, src, imp) for iid, _, src, imp in items if iid in hashes]
    h = np.array([hashes[iid] for iid, _, _ in hashed], dtype=np.uint64)
    for i, (iid, src, imp) in enumerate(hashed):
        for j in np.nonzero(_popcount(h[:i] ^ h[i]) <= HASH_BITS)[0]:
            jid, jsrc, jimp = hashed[j]
            if src == jsrc or (imp and jimp):
                a, b = find(groups[iid]), find(groups[jid])
                if a != b:
                    parent[max(a, b)] = min(a, b)  # the name is stable whichever image comes first
    return {iid: find(g) for iid, g in groups.items()}


def similar_to_skip(items: list[tuple[int, list]], hashes: dict[int, int], box_pct: float) -> set[int]:
    """The image ids to leave out. `items` = [(image id, boxes)] in the order to visit; `hashes` = {image id: dHash}.
    Images without boxes or without a hash are always kept."""
    tol = box_pct / 100
    ids = [iid for iid, boxes in items if boxes and iid in hashes]
    boxes_of = dict(items)
    if len(ids) < 2:
        return set()
    h = np.array([hashes[i] for i in ids], dtype=np.uint64)
    kept = np.zeros(len(ids), dtype=bool)
    skip: set[int] = set()
    for i, iid in enumerate(ids):
        close = np.nonzero((_popcount(h[:i] ^ h[i]) <= HASH_BITS) & kept[:i])[0]
        if any(boxes_match(boxes_of[iid], boxes_of[ids[j]], tol) for j in close):
            skip.add(iid)
        else:
            kept[i] = True
    return skip
