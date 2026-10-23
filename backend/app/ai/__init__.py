"""AI check: a second opinion on boxes, made by models that run on this machine (SAM 2.1 for box refinement, the trained YOLO as detector).

The heavy parts (torch, ultralytics) live in `sam.py` / `detector.py` and are only imported when a check runs, so the rest of the
backend, and all its tests, work without them. `geometry.py` and `verdict.py` are plain Python + numpy.
"""
