"""Make a throwaway project of synthetic "drone" images with exact boxes, so the whole train -> export -> .rpk pipeline can be
proven without any real data. It only ever creates its own project (default "synthetic-smoke"); it never touches other projects.

    python -m scripts.seed_synthetic http://127.0.0.1:8000                      # project + a dataset version
    python -m scripts.seed_synthetic http://127.0.0.1:8000 --start-run --epochs 3 --export-imx

The images are sky gradients with a dark X-shaped "multirotor" or a white cross-shaped "fixed-wing" plane: enough for a model
to learn something in a few epochs, which is all a plumbing test needs. The numbers it reaches mean nothing for real drones.
"""

import argparse
import io
import random
import sys

import httpx
from PIL import Image, ImageDraw

W, H = 320, 240


def _sky(rnd: random.Random):
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        d.line([(0, y), (W, y)], fill=(90 + y // 4, 140 + y // 5, 220))
    for _ in range(400):  # speckle, so no two images are byte-identical
        d.point((rnd.randrange(W), rnd.randrange(H)), fill=(rnd.randrange(255),) * 3)
    return im, d


def _multirotor(rnd: random.Random):
    im, d = _sky(rnd)
    w = rnd.randrange(40, 90)
    h = w * 2 // 3
    x, y = rnd.randrange(10, W - w - 10), rnd.randrange(10, H - h - 10)
    d.line([(x, y), (x + w, y + h)], fill=(20, 20, 20), width=6)
    d.line([(x + w, y), (x, y + h)], fill=(20, 20, 20), width=6)
    for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h)):
        d.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(40, 40, 40))
    return im, (x - 8, y - 8, x + w + 8, y + h + 8)


def _fixed_wing(rnd: random.Random):
    im, d = _sky(rnd)
    w = rnd.randrange(60, 110)
    h = w // 4
    x, y = rnd.randrange(10, W - w - 10), rnd.randrange(30, H - h - 30)
    white, edge = (235, 235, 235), (30, 30, 30)
    d.ellipse([x, y, x + w, y + h], fill=white, outline=edge)
    d.polygon([(x + w * 0.4, y + h / 2), (x + w * 0.6, y + h / 2), (x + w * 0.5, y - 20), (x + w * 0.45, y - 20)], fill=white, outline=edge)
    d.polygon([(x + w * 0.4, y + h / 2), (x + w * 0.6, y + h / 2), (x + w * 0.5, y + h + 20)], fill=white, outline=edge)
    return im, (x, y - 20, x + w, y + h + 20)


def seed(client, name: str = "synthetic-smoke", per_class: int = 40, seed_value: int = 7, start_run: bool = False,
         epochs: int = 3, imgsz: int = 320, batch: int = 8, export_imx: bool = False) -> dict:
    """`client` is anything httpx-like pointed at the backend (an httpx.Client or a Starlette TestClient)."""
    rnd = random.Random(seed_value)
    project = client.post("/projects", json={"name": name}).json()
    ids = {c["name"]: c["id"] for c in project["classes"]}
    for cls, make in (("multirotor", _multirotor), ("fixed-wing", _fixed_wing)):
        for _ in range(per_class):
            im, (x0, y0, x1, y1) = make(rnd)
            buf = io.BytesIO()
            im.save(buf, "JPEG")
            img = client.post(f"/projects/{project['id']}/classes/{ids[cls]}/images",
                              files=[("files", ("synthetic.jpg", buf.getvalue(), "image/jpeg"))]).json()["added"][0]
            box = {"class_id": ids[cls], "cx": (x0 + x1) / 2 / W, "cy": (y0 + y1) / 2 / H, "w": (x1 - x0) / W, "h": (y1 - y0) / H}
            r = client.put(f"/images/{img['id']}/annotations", json={"boxes": [box], "review_status": "reviewed"})
            r.raise_for_status()
    version = client.post(f"/projects/{project['id']}/datasets", json={})
    version.raise_for_status()
    out = {"project_id": project["id"], "dataset": version.json()}
    if start_run:
        run = client.post(f"/projects/{project['id']}/training-runs",
                          json={"dataset_id": version.json()["id"], "epochs": epochs, "imgsz": imgsz, "batch": batch, "export_imx": export_imx})
        run.raise_for_status()
        out["run"] = run.json()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("api", help="backend URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--name", default="synthetic-smoke", help="name of the project to create")
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--start-run", action="store_true", help="also queue a training run on the new dataset version")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--export-imx", action="store_true", help="the run also exports for the IMX500 (needs the Linux GPU worker + a Pi)")
    a = ap.parse_args(argv)
    with httpx.Client(base_url=a.api, timeout=120) as client:
        out = seed(client, a.name, a.per_class, start_run=a.start_run, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, export_imx=a.export_imx)
    st = out["dataset"]["stats"]
    print(f"project {out['project_id']} '{a.name}': dataset {out['dataset']['name']} images {st['images']} boxes {st['boxes']}")
    if "run" in out:
        print(f"queued run {out['run']['id']} ({out['run']['name']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
