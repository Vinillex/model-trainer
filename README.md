# Model Trainer: IMX500 drone detector

Tools to build a drone-detection dataset (upload, scrape, generate), review and box the images in a web UI, train a
YOLO11n or YOLOv8n model on your own GPU, and package it as an `.rpk` for the Sony IMX500 (Raspberry Pi AI Camera).

**Start here: [HANDOVER.md](HANDOVER.md).** It explains how everything works, how to run it, and what has and hasn't been verified.

- `backend/` FastAPI API, dataset building and the training-run state machine
- `frontend/` Next.js review and training UI
- `agents/` the GPU worker (train and export) and the Raspberry Pi worker (package the `.rpk`)
- `handover-data/` a snapshot of the current project so a new machine doesn't have to rebuild the dataset

## About the images in `handover-data/`

Early development data: 2 uploaded images and 50 images collected from public web pages (product pages and news sites) with
an image search, used to build and test the dataset pipeline. The original page of each scraped image is recorded in
`handover-data/db.json`. Copyright in those images belongs to their owners. If you are a rights holder and want an image
removed, please open an issue.
