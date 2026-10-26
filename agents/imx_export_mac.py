"""IMX500 export of a trained best.pt on this Mac. Run by the backend's `imx_export` job inside the export tools' own Python
(backend/data/imx-export/env, made by the "Set up the IMX export tools" button), never inside the backend's Python.

    python imx_export_mac.py <best.pt> <data.yaml> <imgsz>

Ultralytics refuses the IMX export anywhere but Linux with one `assert LINUX` in Exporter.export_imx. Everything that export
uses (model-compression-toolkit on the CPU, Sony's Java converter imxconv-pt) runs on macOS, so this lifts that one check and
calls the normal export. It is tied to the pinned Ultralytics version (app/imx_export.py TOOLS): check it again when changing it.
The last line printed is `MT-RESULT {"packer": ..., "labels": ...}` for the backend to pick the two files up.
"""

import json
import sys
from pathlib import Path

import ultralytics.engine.exporter as exporter
from ultralytics import YOLO

exporter.LINUX = True  # the only Linux-specific thing in export_imx is this assert


def main() -> None:
    best, data_yaml, imgsz = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    # data.yaml's val images (at most 150, picked by the backend) are the int8 calibration set, so all of them are used.
    exported = YOLO(str(best)).export(format="imx", data=data_yaml, imgsz=imgsz, device="cpu", fraction=1.0)
    out = Path(exported)  # the <weights>_imx_model/ folder imxconv-pt wrote packerOut.zip and labels.txt into
    if not out.is_dir():
        out = best.with_name(best.stem + "_imx_model")
    packer, labels = next(out.rglob("packerOut.zip"), None), next(out.rglob("labels.txt"), None)
    if packer is None or labels is None:
        found = sorted(str(p.relative_to(out)) for p in out.rglob("*")) if out.exists() else "(no output folder)"
        sys.exit(f"The IMX export finished but packerOut.zip / labels.txt are missing in {out}: {found}")
    print("MT-RESULT " + json.dumps({"packer": str(packer), "labels": str(labels)}), flush=True)


if __name__ == "__main__":
    main()
