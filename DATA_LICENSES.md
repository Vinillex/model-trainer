# Third-party data and attribution

## Anti-UAV Dataset (Roboflow Universe)

| | |
|---|---|
| Title | Anti-UAV Dataset (version 1, "2025-11-01 12:45pm") |
| Author | Yogith (Roboflow workspace `yogith-nams8`) |
| Source | https://universe.roboflow.com/yogith-nams8/anti-uav-s8wri |
| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as stated on the dataset page |
| Retrieved | 2026-10-21 (YOLOv11 export, 19,849 images) |

```bibtex
@misc{ anti-uav-s8wri_dataset,
    title = { Anti-UAV Dataset },
    type = { Open Source Dataset },
    author = { Yogith },
    howpublished = { \url{ https://universe.roboflow.com/yogith-nams8/anti-uav-s8wri } },
    url = { https://universe.roboflow.com/yogith-nams8/anti-uav-s8wri },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2025 },
    month = { nov },
    note = { visited on 2026-10-21 },
}
```

**Changes made** (CC BY asks for changes to be indicated). Our importer (`backend/scripts/import_yolo.py`) selects a subset of the
frames: one copy of each source frame, at most 20 per visually similar scene, skipping frames with no boxes or with a box smaller
than 5 px. The single class `UAV` is relabelled `multirotor`. Images are used as exported (the uploader had already stretched
them to 640x640 and augmented the training split); they are not re-encoded. Each imported image keeps a record of its source,
author, licence and original filename in the database (`import_records`).

**Licence caveat.** The "CC BY 4.0" label is what the uploader put on Roboflow Universe; the dataset page has no description
and does not say where the frames came from. It looks like a re-upload of frames from video recordings, so the label may not
match the terms of the original source. Check the original source before redistributing these images or using them
commercially. For that reason the imported images are **not** committed to this repository (`handover-data/` excludes them by
default), only this attribution is.

## Images collected from web pages

The 50 scraped images in `handover-data/` were collected with an image search for early development. Their original pages are
in `handover-data/db.json`; copyright belongs to their owners. See the note in `README.md`.
