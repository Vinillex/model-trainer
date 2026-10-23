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
match the terms of the original source. Check the original source before reusing these images commercially.

**Where the images are.** At the owner's request the imported images are committed in this repository, under
`handover-data/storage/projects/1/raw/multirotor/imported/anti-uav/` (4,859 images; the bundle export normally leaves imported
images out, this one was exported with `--include-imported`). Their boxes and per-image provenance (dataset, author, licence, URL,
original filename) are in `handover-data/db.json` (`import_records`). The licence caveat above applies to these files.

## Object Detection dataset by cmy (Roboflow Universe)

| | |
|---|---|
| Title | Object Detection Dataset (version 1, "2025-10-11 11:19am") |
| Author | cmy (Roboflow workspace `cmy-rfaq1`) |
| Source | https://universe.roboflow.com/cmy-rfaq1/object-detection-gtwuo |
| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as stated on the dataset page |
| Retrieved | 2026-10-21 (YOLOv11 export, 9,455 images) |
| Imported as | dataset `cmy-uav` |

```bibtex
@misc{ object-detection-gtwuo_dataset,
    title = { Object Detection Dataset },
    type = { Open Source Dataset },
    author = { cmy },
    howpublished = { \url{ https://universe.roboflow.com/cmy-rfaq1/object-detection-gtwuo } },
    url = { https://universe.roboflow.com/cmy-rfaq1/object-detection-gtwuo },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2025 },
    month = { oct },
    note = { visited on 2026-10-21 },
}
```

**Changes made.** The same importer as for Anti-UAV selected a subset: one copy of each source frame (5,200 distinct frames in
the 9,455 files), at most 20 per visually similar scene, skipping frames with a box smaller than 5 px. That leaves 3,033 images.
The dataset's two classes, `UAV` and an unnamed `0`, are both relabelled `multirotor`. Images are used as exported (stretched to
640x640 by the uploader), not re-encoded. Each image keeps a record of its source, author, licence and original filename in the
database (`import_records`).

**Caveats.** The dataset page has no description and does not say where the frames came from, so the "CC BY 4.0" label may not
match the terms of the original source. In a spot check many boxes sat beside the drone instead of on it; the images are
reviewable per dataset in the app's gallery and review screen, and should stay out of training versions (untick `cmy-uav` in "Data
to include") until that review is done.

**Where the images are.** In the local database and storage only. `handover-data/` has not been re-exported since this import.

## Images collected from web pages

The 50 scraped images in `handover-data/` were collected with an image search for early development. Their original pages are
in `handover-data/db.json`; copyright belongs to their owners. See the note in `README.md`.
