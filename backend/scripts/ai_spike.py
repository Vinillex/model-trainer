"""Calibration spike for the AI check: how well does SAM 2.1 refine boxes of each size, and what separates a wrong box from a sloppy one?

Takes reviewed single-box images, builds four prompts from each true label (exact / jittered / loose / wrong place), crops around the
prompt, asks SAM for a tight box and compares it with the true label. Read-only on the database. Results: backend/data/ai/spike.json.
    cd backend && .venv/bin/python -m scripts.ai_spike --weights sam2.1_s.pt --per-bin 14
The thresholds in app/ai/verdict.py come from running this; run it again if the model, the crop settings or the data change."""

import argparse
import itertools
import json
import random
import sqlite3
import statistics as st
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.geometry import Box, area, clip, iou, shift_box, side_on_input, to_pixels
from app.ai.sam import SamRefiner

STRATEGIES = ("box", "inner", "point", "boxpoint")
BINS = [(0, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 10_000)]
BACKEND = Path(__file__).resolve().parents[1]
OUT = BACKEND / "data" / "ai"


def variants(label: Box, width: int, height: int, rng: random.Random) -> dict[str, Box]:
    """Prompts a person (or a bad import) could have drawn instead of the true box."""
    bw, bh = label[2] - label[0], label[3] - label[1]
    side = max(bw, bh)
    cx, cy = (label[0] + label[2]) / 2, (label[1] + label[3]) / 2

    def around(cx, cy, w, h):
        return clip((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), width, height)

    ang = rng.uniform(0, 6.2832)
    far = rng.uniform(2.5, 4.0) * side
    jx, jy = rng.uniform(-0.3, 0.3) * side, rng.uniform(-0.3, 0.3) * side
    return {
        "exact": label,
        "jitter": around(cx + jx, cy + jy, bw * rng.uniform(0.8, 1.3), bh * rng.uniform(0.8, 1.3)),
        "loose": around(cx, cy, bw * 2.0, bh * 2.0),
        "wrong": around(cx + far * np.cos(ang), cy + far * np.sin(ang), bw, bh),
    }


def pick_images(db: sqlite3.Connection, per_bin: int, dataset: str, seed: int):
    rows = db.execute("""
        select i.id, i.storage_key, i.width, i.height, a.cx, a.cy, a.w, a.h
        from images i join import_records r on r.image_id = i.id join annotations a on a.image_id = i.id
        where r.dataset = ? and i.review_status = 'reviewed' and a.class_id = 1
          and (select count(*) from annotations b where b.image_id = i.id) = 1""", (dataset,)).fetchall()
    rng = random.Random(seed)
    rng.shuffle(rows)
    chosen = {b: [] for b in BINS}
    for row in rows:
        _, _, w, h, cx, cy, bw, bh = row
        side = side_on_input((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), w, h)
        for lo, hi in BINS:
            if lo <= side < hi and len(chosen[(lo, hi)]) < per_bin:
                chosen[(lo, hi)].append(row)
                break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="sam2.1_s.pt")
    ap.add_argument("--per-bin", type=int, default=14)
    ap.add_argument("--dataset", default="anti-uav")
    ap.add_argument("--context", type=float, default=4.0)
    ap.add_argument("--min-crop", type=int, default=96)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{BACKEND / 'data' / 'model_trainer.db'}?mode=ro", uri=True)
    sam = SamRefiner(OUT / "models" / args.weights, args.device, args.context, args.min_crop)
    rng = random.Random(args.seed)
    rows = []
    for (lo, hi), images in pick_images(db, args.per_bin, args.dataset, args.seed).items():
        for image_id, key, width, height, cx, cy, bw, bh in images:
            rgb = np.array(Image.open(BACKEND / "data" / "storage" / key).convert("RGB"))
            label = to_pixels((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), width, height)
            side = side_on_input((label[0] / width, label[1] / height, label[2] / width, label[3] / height), width, height)
            for (name, prompt), strategy in itertools.product(variants(label, width, height, rng).items(), STRATEGIES):
                cx_, cy_ = (prompt[0] + prompt[2]) / 2, (prompt[1] + prompt[3]) / 2
                inner = (cx_ - (prompt[2] - prompt[0]) / 4, cy_ - (prompt[3] - prompt[1]) / 4, cx_ + (prompt[2] - prompt[0]) / 4, cy_ + (prompt[3] - prompt[1]) / 4)
                t = time.time()
                r = sam.refine(rgb, **{"box": dict(prompt=prompt), "inner": dict(prompt=inner), "point": dict(point=(cx_, cy_)),
                                       "boxpoint": dict(prompt=prompt, point=(cx_, cy_))}[strategy])
                ms = (time.time() - t) * 1000
                row = {"image": image_id, "bin": f"{lo}-{hi}", "side": round(side, 1), "variant": name, "strategy": strategy, "ms": round(ms),
                       "iou_prompt_label": round(iou(prompt, label), 3)}
                if r is not None:
                    row |= {"found": True, "iou_ref_label": round(iou(r.box, label), 3), "iou_ref_prompt": round(iou(r.box, prompt), 3),
                            "score": round(r.score, 3), "edge": r.touches_edge,
                            "area_ratio": round(r.mask_area / max(area(prompt), 1), 3),
                            "inside": round(max(0.0, (min(r.box[2], prompt[2]) - max(r.box[0], prompt[0]))) * max(0.0, (min(r.box[3], prompt[3]) - max(r.box[1], prompt[1]))) / max(area(r.box), 1e-9), 3)}
                else:
                    row["found"] = False
                rows.append(row)
        print(f"bin {lo}-{hi}: {len(images)} images done", flush=True)
    (OUT / "spike.json").write_text(json.dumps({"args": vars(args), "device": sam.device, "rows": rows}))

    def med(xs):
        return f"{st.median(xs):.2f}" if xs else "  - "

    print(f"\ndevice {sam.device}, median {st.median(r['ms'] for r in rows):.0f} ms per box, weights {args.weights}")
    print("\nmedian IoU of SAM's box with the TRUE label (share of boxes with IoU >= 0.7 in brackets); 'prompt' = the drawn box itself")
    for variant in ("exact", "jitter", "loose", "wrong"):
        print(f"\n== {variant}")
        print(f"{'size on 640':>12} {'n':>3} {'prompt':>7} " + " ".join(f"{s:>16}" for s in STRATEGIES))
        for lo, hi in BINS:
            cells = []
            for strategy in STRATEGIES:
                sel = [r for r in rows if r["variant"] == variant and r["bin"] == f"{lo}-{hi}" and r["strategy"] == strategy]
                ok = [r for r in sel if r["found"]]
                cells.append(f"{med([r['iou_ref_label'] for r in ok]):>6} ({sum(r['iou_ref_label'] >= 0.7 for r in ok):>2}/{len(sel):>2})" if sel else "")
            sel0 = [r for r in rows if r["variant"] == variant and r["bin"] == f"{lo}-{hi}" and r["strategy"] == STRATEGIES[0]]
            if sel0:
                print(f"{f'{lo}-{hi}':>12} {len(sel0):>3} {med([r['iou_prompt_label'] for r in sel0]):>7} " + " ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
