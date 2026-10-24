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

## uav-annotated (Roboflow Universe)

| | |
|---|---|
| Title | uav-annotated Dataset |
| Author | GKh (Roboflow workspace `gkh`) |
| Source | https://universe.roboflow.com/gkh/uav-annotated |
| Licence | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) as stated on the dataset page |
| Retrieved | 2026-10-24 (YOLOv11 export of a fork of the live dataset, 3,520 images; the published versions v1 to v16 have a single class, so the fork was needed to get `copter` and `fixed_wing` apart; only Auto-Orient was applied, no resize, no augmentation) |
| Imported as | dataset `uav-annotated` |

```bibtex
@misc{ uav-annotated_dataset,
    title = { uav-annotated Dataset },
    type = { Open Source Dataset },
    author = { GKh },
    howpublished = { \url{ https://universe.roboflow.com/gkh/uav-annotated } },
    url = { https://universe.roboflow.com/gkh/uav-annotated },
    journal = { Roboflow Universe },
    publisher = { Roboflow },
    year = { 2025 },
    month = { dec },
    note = { visited on 2026-10-24 },
}
```

**Changes made.** The same importer as for Anti-UAV selected a subset: one copy of each source frame (3,335 distinct frames), dropping
646 frames with no boxes, at most 20 per visually similar scene, skipping frames with a box smaller than 5 px. That left 2,183
frames, of which 2,180 were added (3 were already in the project, matched by checksum). `copter` was relabelled `multirotor` (1,700
images) and `fixed_wing` became `fixed-wing` (480 images). Each image keeps a record of its source, author, licence and original
filename in the database (`import_records`).

**Caveats.** The dataset page says its images come from two GitHub datasets ([UAV2UAV](https://github.com/hapless19/UAV2UAV-dataset)
and [DroneDetectionDataset](https://github.com/Maciullo/DroneDetectionDataset)) plus images generated with DALL-E and Stable
Diffusion, so the "CC BY 4.0" label may not match the terms of the original sources. Some frames show third-party watermarks or
captions (video stills, a drone retailer's product pictures). Many fixed-wing frames come from one fisheye UAV-to-UAV scene. The
images were marked reviewed on import without a check of the boxes, so look at them in the app before relying on them.

**Where the images are.** In the local database and storage only, as for every dataset since 2026-10-24: `handover-data/` is in
`.gitignore` and nothing from this dataset is committed.
