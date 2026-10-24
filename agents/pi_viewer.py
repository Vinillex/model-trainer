"""Live preview on the Pi's screen with an IMX500 network's detections drawn on it. Press Ctrl-C (or Stop on the Training page) to end it:
closing the window alone leaves the camera running.

    python3 pi_viewer.py --model network.rpk --labels labels.txt [--threshold 0.25] [--fps 15] [--title "run name"]

The Pi agent starts this when the Training page's "Run on Pi" button is pressed; it can also be run by hand on the Pi's desktop. It prints
a "[viewer] ..." line every 2 seconds (camera fps, network results per second, detections now) that the agent reports to the page.

--fps is the camera's frame rate. The chip returns a result on every frame only if the frame period is longer than its cycle, so a
640 model sustains 15 and a 416 model 23 (measured on the Pi 5 + AI Camera); above that it skips frames in whole steps.
The export bakes in a score threshold of 0.25, so --threshold below that changes nothing.
"""

import argparse
import signal
import time

import cv2
from picamera2 import MappedArray, Picamera2, Preview
from picamera2.devices import IMX500

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="network.rpk")
ap.add_argument("--labels", default="labels.txt")
ap.add_argument("--threshold", type=float, default=0.25)
ap.add_argument("--fps", type=float, default=15.0)
ap.add_argument("--title", default="", help="shown on the preview, e.g. the run's name")
args = ap.parse_args()

LABELS = [l.strip() for l in open(args.labels) if l.strip()]
COLORS = [(0, 200, 255), (0, 255, 0), (255, 80, 80), (255, 0, 255), (255, 255, 0)]
imx500 = IMX500(args.model)
W, H = imx500.get_input_size()
picam2 = Picamera2(imx500.camera_num)
detections, stats = [], {"frames": 0, "outputs": 0, "t": time.time(), "rate": 0.0, "fps": 0.0}


def parse(request):
    md = request.get_metadata()
    stats["frames"] += 1
    out = imx500.get_outputs(md, add_batch=True)
    if out is None:
        return
    stats["outputs"] += 1
    dets = []
    for b, s, c in zip(out[0][0], out[1][0], out[2][0]):
        if s < args.threshold:
            continue
        x0, y0, x1, y1 = (float(v) for v in b)  # pixels of the network input, x1,y1,x2,y2 (Ultralytics' IMX export)
        dets.append((float(s), int(c), imx500.convert_inference_coords((y0 / H, x0 / W, y1 / H, x1 / W), md, picam2)))
    detections[:] = dets


def draw(request):
    parse(request)
    now = time.time()
    if now - stats["t"] >= 2:
        span = now - stats["t"]
        stats["rate"], stats["fps"] = stats["outputs"] / span, stats["frames"] / span
        stats["outputs"], stats["frames"], stats["t"] = 0, 0, now
        print(f"[viewer] camera fps {stats['fps']:.1f}, network results/s {stats['rate']:.1f}, detections now {len(detections)}", flush=True)
    with MappedArray(request, "main") as m:
        for s, c, (x, y, w, h) in detections:
            col = COLORS[c % len(COLORS)]
            cv2.rectangle(m.array, (x, y), (x + w, y + h), col + (255,), 3)
            cv2.putText(m.array, f"{LABELS[c] if c < len(LABELS) else c} {s:.2f}", (x + 4, max(24, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col + (255,), 2)
        cv2.putText(m.array, f"{args.title + '   ' if args.title else ''}detections >= {args.threshold}: {len(detections)}",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
        cv2.putText(m.array, f"camera fps: {stats['fps']:.1f}   network results/s: {stats['rate']:.1f}", (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)


def stop(signum, frame):
    raise KeyboardInterrupt


signal.signal(signal.SIGTERM, stop)  # the agent asks politely (SIGINT, then SIGTERM) before it kills

picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 960)}, controls={"FrameRate": args.fps}, buffer_count=12))
imx500.show_network_fw_progress_bar()  # loading the network onto the chip can take a few minutes
picam2.pre_callback = draw
picam2.start_preview(Preview.QT, x=0, y=0, width=1280, height=960)  # the GL preview can't get an EGL display here
picam2.start()

print("[viewer] preview running; press Ctrl-C to stop", flush=True)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    print("[viewer] stopped", flush=True)
