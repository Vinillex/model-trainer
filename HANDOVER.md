# Handover: IMX500 drone-detector trainer

You are picking this project up cold, on a machine with an NVIDIA GPU. This file is everything the previous session
knew. Read it top to bottom once, then follow **section 3** (the first-hour checklist) before changing any code.

The person you are working for is **Vinit**. They build and own this; you are continuing, not starting over. They prefer
that you act on clear requests instead of asking, report plainly (including what you did *not* test), and tell them about
anything that costs money, exposes a port, or touches their data before you do it.

## 1. What this is

A tool to build a dataset and train an **object-detection** model that identifies **drone types** (multirotor,
fixed-wing, shahed-class, and others), producing a **`.rpk`** file that runs on a **Sony IMX500** (the Raspberry Pi AI
Camera). Images come from three sources: **upload**, **scraping** (SerpAPI Google Images) and **AI generation**
(not built yet). The user reviews and boxes images in a built-in web UI, freezes a *dataset version*, trains in the
cloud/own GPU, exports for the IMX500 and packages the `.rpk` on a Raspberry Pi.

Decisions already made (don't re-litigate without a reason):

| Decision | Detail |
|---|---|
| Compute | **Rented RunPod GPUs** (decided 2026-10-22: the user has no GPU access). The Training page's **Train on RunPod…** button (price dialog, live log, Stop; `app/api/runpod.py`) or `python -m scripts.runpod launch --yes` rents one for the length of a run and deletes the pod afterwards; see `agents/RUNPOD.md`. The launch path has run on four real pods (1-epoch tests, $0.45 in all, see the status below); the IMX export has still never finished. Earlier plan, still supported by `agents/README.md`: local GPUs (RTX 4060 Ti 8 GB and RTX 3060 12 GB in Windows PCs). |
| Model | **YOLO11n / YOLOv8n only.** Ultralytics' IMX export supports nothing else, and the sensor has about 8 MB of model memory. Input up to 640x640. |
| Labeling | Built-in review UI (box editor), not CVAT or Label Studio. |
| Auth | **None.** Single user, trusted network. The only protection is a loopback guard plus a worker token (section 4.6). |
| Generation | v1 = copy-paste composites (SAM-cut drones pasted on sky backgrounds, exact boxes). **v2 = self-hosted FLUX** (the user wants this later). The job framework is generic so FLUX is just a new job type. |
| Test split | Real images only. Generated images are train-only so synthetic data never measures the model. |

## 2. Repo map

```
backend/            FastAPI + SQLAlchemy 2 (SQLite by default). Python.
  app/
    main.py         app, LoopbackOnly middleware, routers; lifespan = create_all + fail stale jobs + start in-process job worker
    config.py       Settings (env prefix MT_, read from backend/.env)
    models.py       all tables
    db.py storage.py   sessions; LocalStorage (default) and S3Storage (written, UNTESTED)
    schemas.py      pydantic request/response models
    api/            projects, images, annotations, jobs (scrape + quota), datasets, training (UI side), workers (agent side)
    scrape.py       SerpAPI client, image download (SSRF-guarded), the scrape job, dedupe layers
    imagehash.py    64-bit dHash for near-duplicate detection
    jobs.py         tiny DB-backed queue + in-process worker thread (scrape and ai_check jobs)
    datasets.py     dataset versions: grouped/stratified train-val-test split, stats, warnings
    yolo_export.py  frozen manifest -> YOLO zip (no DB access, reusable)
    training.py     training-run state machine: stages, leases, claim/heartbeat/complete (section 4.5)
    security.py     LoopbackOnly middleware + worker-token check
    pi_display.py      what the Pi's screen should show (request + the agent's reports, one JSON file); api/pi_display.py is its page-facing side
    api/runpod.py      the UI's RunPod buttons: price check, confirmed launch as a detached process, log, stop, delete a stray pod
    runpod_client.py   RunPod: balance and GPU prices (GraphQL), start/list/delete pods (REST, https://rest.runpod.io/v1)
    runpod_pod.py   what a training pod is: pinned image, GPU preference list, the POST /pods body
    ai/             the AI check (section 4.8): geometry.py, verdict.py, check.py (the rules), service.py (glue, model cache, auto-label), job.py (batch),
                    registry.py (which models a run can pick, uploaded weights),
                    sam.py + detector.py (the heavy models, imported only when a check runs)
  scripts/
    runpod.py       the launcher: status / gpus / launch (dry run unless --yes) / stop; tunnel, pod, watch, always delete
    bundle.py       export/import the whole database + files as a portable bundle
    ai_spike.py     calibration for the AI check: SAM on jittered/loose/wrong boxes, by drone size (re-run it before changing thresholds)
    seed_synthetic.py  make a synthetic project so the pipeline can be tested with no real data
  tests/            518 tests (pytest)
agents/             what runs on the GPU and the Pi: gpu_agent.py, pi_agent.py, common.py, README.md, runpod_start.sh (the pod's
                    start command), RUNPOD.md
frontend/           Next.js 15 (app router, TypeScript, Tailwind 3). Talks to the API at NEXT_PUBLIC_API_URL (default http://localhost:8000)
  app/projects/[id]/(main)/   the project's sections behind a left drawer (components/Drawer.tsx): / (Overview), images, add (Upload | Scrape),
                    ai, datasets, training; layout.tsx provides the project once (lib/projectContext.tsx: project, stats, refreshKey,
                    imagesChanged(), classesChanged()). app/projects/[id]/review is outside the group: full screen, no drawer.
handover-data/      the data bundle (db.json + files + checksums). Restore with `python -m scripts.bundle import`.
HANDOVER.md         this file
```

## 3. First-hour checklist (do these in order)

**Environment.** Run *everything* (backend, frontend, GPU agent) inside **WSL2 Ubuntu 24.04**, and clone into the Linux
filesystem (`~/model_trainer`), **not** under `/mnt/c` (slow, and file permissions break). Why WSL2: Ultralytics' IMX
export is Linux-only. If this machine is already native Linux, ignore the WSL notes.

1. **GPU visible?** `nvidia-smi` inside Ubuntu must list the card. It uses the Windows driver; never install an NVIDIA
   driver inside WSL.
2. **Networking mode.** Put this in `%UserProfile%\.wslconfig` on Windows, then `wsl --shutdown`:
   ```
   [wsl2]
   networkingMode=mirrored
   ```
   Why: the backend's `LoopbackOnly` guard only lets *loopback* clients use the normal API. In WSL2's default NAT mode a
   Windows browser may arrive from a non-loopback address and get 403 everywhere. Mirrored mode also lets the Raspberry
   Pi reach the backend at the PC's LAN address. In mirrored mode Windows firewall rules, including the Hyper-V firewall
   for WSL, must allow inbound TCP 8000 for the Pi. **I never tested any of this**; see Microsoft's WSL networking docs.
3. **Clone and set up the backend.**
   ```
   git clone https://github.com/Vinillex/model-trainer.git ~/model_trainer && cd ~/model_trainer/backend    # public: no login needed
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
   The backend was developed and tested on **Python 3.13**. Ubuntu 24.04 ships 3.12: try it, and if anything fails get
   3.13 with `uv` (`pip install uv && uv venv --python 3.13 .venv`). This venv is **separate** from the GPU venv in step 8.
4. **Secrets.** Copy `backend/.env.example` to `backend/.env`. **Do not copy the Mac's `.env`.**
   - `MT_WORKER_TOKEN`: generate a fresh one: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Required for any worker.
   - `MT_SERPAPI_KEY`: only needed for scraping. Ask Vinit. The free plan is 250 searches/month **shared** with their Mac.
   - `MT_RUNPOD_API_KEY`: only on the machine that runs `scripts.runpod` (the Mac). Leave it empty here.
   Never print or log these values. `.env` is gitignored; keep it that way.
5. **Run the tests first.** `python -m pytest -q` from `backend/` should say **230 passed**. If not, stop and tell Vinit.
   (The tests are isolated from real data by `tests/conftest.py`; see gotchas.)
6. **Restore the data.** From `backend/`: `python -m scripts.bundle import`. It needs an **empty** database and prints
   counts only. Expect: project `drone-detector` with 13 classes, 52 images (6 reviewed), 8 box rows, 57 remembered scrape URLs.
7. **Start the app and look at it.**
   ```
   cd backend && uvicorn app.main:app --port 8000          # no --reload; restart after code changes
   cd frontend && npm ci && npm run dev                    # Node 20 LTS recommended (tested 20.11.1; Next 15 needs >= 18.18)
   # npm may warn that eslint-visitor-keys wants Node >= 20.19: harmless, lint still runs.
   ```
   Open `http://localhost:3000/projects/1` from the Windows browser. You should see the 52 images. If requests fail with
   403, it is the networking issue from step 2.
8. **GPU environment and the IMX export test.** Follow `agents/README.md` section 1, steps 3-5. In short: Java 17+, a
   **Python 3.11** venv via `uv` with `torch>=2.9,<2.12`, then run the stock export once:
   `python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='imx', data='coco8.yaml')"`.
   **This is the single biggest unknown in the project** (see section 6). If it fails, fix it before anything else.
9. **Prove the pipeline on synthetic data** (the real data is too thin to train on, section 5):
   ```
   cd backend && python -m scripts.seed_synthetic http://127.0.0.1:8000 --start-run --epochs 3 --export-imx
   # in the GPU venv, same machine:
   export MT_SERVER=http://127.0.0.1:8000 MT_WORKER_TOKEN=<the token from .env>
   python agents/gpu_agent.py --name gpu-local --device 0 --once
   ```
   Watch the run in the UI ("Train & export"). Success = stage `awaiting_package` with `best.pt`, `packerOut.zip` and
   `labels.txt` downloadable. That is the real export working.
10. **The Raspberry Pi** (`agents/README.md` section 2): the Training page's **Copy Pi setup command** button copies a one-time
    command; pasted on the Pi it installs `pi_agent.py` as a systemd service (`mt-pi-agent`, starts at boot). After that a finished
    run is packaged by itself whenever the Pi is on. Success = the run reaches `done` and `network.rpk` downloads.
    Then load it on the Pi with Ultralytics' Sony IMX500 guide.

Then tell Vinit what worked. Steps 8-10 are the first time those parts ever run.

## 4. How it works

### 4.1 Data model (`models.py`)
`projects` -> `classes` (`is_negative` = hard-negative category such as birds or balloons; negative classes get no boxes,
their images count as background; `index` = creation-order position among the positive classes, `None` for negatives; each dataset version then numbers its positives 0..n-1 in that order, so use a model's own `labels.txt`) ->
`images` (`source` uploaded|scraped|generated, `review_status` pending|reviewed|rejected, `storage_key`, `sha256`,
`group_key`) -> `annotations` (normalized YOLO boxes cx,cy,w,h; `status` auto|auto_exact|reviewed|rejected).
Also `image_hashes`, `scrape_records` (provenance: query, original URL, page URL), `scraped_urls` (every URL ever tried),
`search_log`, `jobs`, `dataset_versions`, `training_runs`, `upload_records` (an uploaded image's folder, see below). Images live on disk under
`backend/data/storage/projects/<id>/raw/<class>/<source>/<sha256>.<ext>` (the per-class `uploaded/scraped/generated`
layout the user asked for); the DB is `backend/data/model_trainer.db`.

**Filing and the `mixed` folder.** The class chosen at scrape/upload time is only a first guess (one search returns many kinds
of drone). When an image is saved as **reviewed**, it is re-filed by what its boxes show (`app/refile.py`): one box class means it
moves to that class's folder (file and `class_id`); two or more classes mean the `mixed` folder; no boxes, pending or rejected
means no move. Editing a reviewed image re-evaluates it, so it can move back out of `mixed`. `mixed` is a **reserved class name**
auto-created per project (`index` is `None`; `DetClass.is_mixed` / `is_trainable`): it is never a label, never in `data.yaml`,
and nothing is scraped, uploaded or boxed into it by hand. The move is the last step and never raises: the boxes are committed
first, then the file is copied, the row switched, and the old copy deleted, so a failed move leaves the image where it was.
`POST /projects/{id}/images/refile?dry_run=false` applies the rule to images reviewed before it existed (dry run by default).

**Images page extras.** Active filters show as removable chips with "Clear filters" and a "Review these" button (the queue those filters
describe). Cards have a checkbox (always visible below 768 px, on hover above it): the bar offers **Reject** (inline confirm, `POST
/projects/{id}/images/reject`, up to 500, all or nothing, ids of another project refuse the whole request; it goes through the same
`apply_annotations()` as the editor's save, so a rejected image leaves "AI: needs a look" and "AI drew the boxes" exactly as a single Reject
does; boxes stay as they are) and **Run AI on these** (hands the ids to the AI page through `sessionStorage`, `image_ids` in the job; the mode's
rules still apply, an empty list is refused, never "everything"). There is **no bulk accept**: accepting means a person looked, and it would
turn unchecked boxes into training data. The selection is cleared whenever a filter changes and pruned to the images on screen after every reload.
There is no "move to folder": moving existing images between folders is not built.
**Trash.** Rejected images live in the **Trash** section (`TrashPanel.tsx`, drawer item with a count, a row in the Overview places table). They are
kept out of the Images page and the review queue: `imageQuery()` sends `exclude_rejected` unless a review status is chosen on purpose (an explicit
`review_status=rejected` still finds them), and dataset versions already exclude them. Places on the Overview count images without the trash.
Select cards, then **Restore** (`POST /projects/{id}/images/restore`: back to "not reviewed", boxes as they were, and the AI's doubts return because
the decision recorded at the reject is forgotten) or **Delete forever** (`POST .../images/delete`, inline confirm; rows first, files after, cascades
included). Both take only rejected images of the project, all or nothing, up to 500. A frozen dataset version reads its image files whenever it is
downloaded, so images that a version contains (matched on id and SHA-256: SQLite reuses ids) are refused with a 409 naming the version; the panel then
offers "Delete anyway" (`allow_breaking_versions`). The single Delete on a gallery card is unchanged and unprotected, as before (a version then reports
its missing files). Opening a card reviews the trash; accepting an image there takes it out of the rejected queue.

**Classes and what uses them.** The Overview lists the drone classes and the hard negatives as two tables (negative images arrive already reviewed,
so that table has no "To review" column). Remove is always visible and greyed out while the class is in use (an image filed under it, or any
box of that class); beside it "show N boxes" opens the review queue on exactly the images that have such a box (`GET /images?box_class_id=`,
wherever they are filed: an image in `mixed` has boxes of several classes) and "show images" opens the gallery on the class. The 409 message says
how many images are filed under the class and how many boxes on how many images.
**Overview.** `PlacesTable.tsx` (from `GET /projects/{id}/folders`, which now also returns `ai_ok`, `ai_flagged`, `ai_unchecked` per place): images,
reviewed %, AI ok %, needs a look, not checked, with "Review N" and "Open". The AI counts use `ai_condition()` in `app/ai/filters.py`, the one
SQL rule behind every AI filter (the gallery list, these counts and the drawer's "needs a look" badge); `Image.ai_status` is the same rule in
Python, and a test compares the counts with what the gallery filter lists. The drawer polls every 10 s while the tab is visible for that badge,
a running AI run and a running training run.
**Upload hint.** Picking a folder starts the upload at once, and a negative class files images as already reviewed background, so a folder whose
name and class disagree (`lib/uploadHints.ts`, a name heuristic) waits for "Upload anyway" / Cancel before anything is sent.

**Upload folders.** Uploading a folder files its images under `raw/<class>/uploaded/<folder>/` (`POST /projects/{id}/classes/{class_id}/images?folder=NAME`,
`relative_paths` = one form field per file saying where it sat inside the folder, kept in `upload_records.original_path`, never used as a path). The
folder name is cleaned by `clean_folder()` (A-Z a-z 0-9 . - _, others become `-`, no leading dot, at most 80; a name with nothing usable is a 422, not a
guess); nested subfolders all go into the top folder. `refile` keeps the folder when a reviewed image changes class (like an imported dataset's name).
`GET /projects/{id}/folders` lists every class/source/(dataset or folder) with review counts; `GET .../images?folder=` and the AI batch's `folder`
param select one. The UI: Choose folder / drop a folder (`lib/dropFiles.ts` walks `webkitGetAsEntry`, 100 entries per read), non-images skipped,
then the gallery opens on `folder:<name>` (scope value in `lib/galleryFilters.ts`) with a one-line notice. The same picture is never filed twice,
even into another folder (sha256 per project).

**Removing a class** (`DELETE /projects/{id}/classes/{class_id}`, the Remove button in the class table) is allowed only when
nothing uses it: no images, no boxes drawn with it on any image, and no queued/running scrape job aimed at it. Dataset
versions are self-contained snapshots, so existing ones are unaffected. New positive classes get `index = max + 1`, never a
count, because a removal leaves a gap and a count would reuse a live class's index.

There are **no migrations**. `create_all` makes missing *tables* on startup; a new *column* on an existing table needs a
manual `ALTER TABLE`. Alembic is a known to-do. `scripts/bundle.py` tolerates schema drift (section 4.7).

### 4.2 Review UI
**The review screen never scrolls** (`ReviewScreen.tsx`): it is a fixed full-window layer (`fixed inset-0`, the page behind is locked while it is
open), the image on the left and every control in a panel on the right (two columns from 768 px; stacked below that). `BoxEditor` fills the
space its parent gives it (measured with a ResizeObserver; at zoom 100% the whole image fits, enlarged at most 2x) instead of a share of the
window height. Windows under 700 px tall get a denser panel (the `short:` Tailwind variant in `tailwind.config.ts`); only the list of AI
suggestions scrolls, inside its own box, and the panel itself scrolls only as a last resort rather than clipping a button. Measured on a wide, a
tall and a panorama image at 1440x900, 1280x720, 1024x640, 900x560, 800x500 and a 400x800 phone: no page scroll, every button on screen.
`?` opens the shortcuts list (`ShortcutsHelp.tsx`); while it is open the screen ignores every other key (Enter would otherwise accept the image
behind it). The A (AI check) and S (snap box) keys are wired in the same handler.
`/projects/<id>/review` (the queue opens from the Images page's "Review these", the Overview table's "Review N", a card, or the stats table) shows the queue the **Images page's gallery filters** select (class, source, one imported dataset or upload folder,
manual check, AI check, review status). It has no filters of its own: the gallery keeps them in `sessionStorage` (`lib/galleryFilters.ts`)
and the review screen reads them; both build the image query with the one `imageQuery()` there (a filter added to only one of them once
made the review queue ignore the AI filter). The gallery reloads on return to its tab (focus / visibility) so it shows the current list; the stats table's "To review" link sets them to that class + not reviewed. Opening an
**imported** image there records a **manual check** (`manual_checks` table, `POST /images/{id}/manual-check`), shown as a tag
in the gallery and filterable per imported dataset; it is how you tell which imported images a person has really looked at.
Hotkeys (1-9 pick class, Enter accept, X reject, arrows, Delete, +/- zoom, 0 fit; scrolling over the image zooms at the cursor
(no key needed; a trackpad pinch does too) and holding the middle button while dragging pans it; a draw drag under 4 screen
pixels is a click, so zoom in to draw tiny boxes), box
drawing/moving/resizing (`BoxEditor`, pointer-capture, normalized corners), autosaves on navigation. Saving boxes replaces
all boxes of an image (`PUT /images/{id}/annotations`) and sets its review status.

### 4.3 Scraping (`scrape.py`, `ScrapePanel.tsx`)
SerpAPI `engine=google_images`, 100 images per page, page index `ijn` 0-99. Only *successful* searches count against
quota; an identical search within an hour is cached and free; a real search can take up to ~80 s and is still charged
(client timeout is 180 s). Four dedupe layers: (1) a URL ledger (`scraped_urls`): never re-download a URL already tried,
even if its image was deleted, except `download_failed` ones which are retried; (2) SHA-256 exact duplicates; (3) 64-bit
dHash near-duplicates (distance <= 4 by default; featureless sky images can over-merge); (4) minimum side in pixels.
Downloads are SSRF-guarded (http/https only, public IPs only, re-checked per redirect, size cap, content-type check).
The API key never appears in errors or logs. Jobs run on the in-process worker thread and can be cancelled per image.

### 4.4 Dataset versions (`datasets.py`, `yolo_export.py`)
A version freezes the **reviewed** images and their boxes (only `reviewed`/`auto_exact` boxes) into a `manifest.json` in
storage; later edits or deletions never change it. Split is **grouped** (images from the same source page/video stay
together, so near-duplicates can't leak from train into val), **stratified by the class of each image's boxes** (most common box class; the class an image was *filed under* is only a folder, so a mixed search filed under one class still gets every class into val/test; background images use their filing class), deterministic by seed, and
generated images are train-only. Warnings are attached (empty val/test, class with no boxes, fewer than 20 boxes).
Download gives a standard Ultralytics folder: `data.yaml`, `images/{train,val,test}`, `labels/...`.
**Trap:** `data.yaml` says `path: .`, which Ultralytics resolves against *its own datasets directory*, not the yaml's
folder. `agents/gpu_agent.py` rewrites it to the absolute extract path; do the same if you ever train by hand.
Because reviewed images can move folders after a version is frozen, the download resolves each manifest item to its image's
*current* storage key by image id **and SHA-256** (`datasets.resolve_manifest`; the SHA check matters because SQLite reuses the
id of a deleted image). Never rely on a manifest's recorded `key` alone.
**Choosing the data** (`Selection` in `datasets.py`): a version can use any mix of `uploaded`, `scraped`, `generated`, and, separately, each *imported dataset by name* (`DatasetCreate.sources`, `.imported_datasets`; `None` = all, `[]` = none). The UI's "Data to include" picker sends both lists and re-reads `GET .../datasets/preview?sources=a,b&imported=x,y` live; the version records the choice (`config.sources`, `config.imported_datasets`) and what it used (`stats.by_source`, e.g. `imported:anti-uav`). Hard negatives follow their own source. Mind the balance: one imported dataset can be 95% of a version, so the picker exists to control it.
A dataset version used by a training run can't be deleted. If images are deleted after a version was made, its zip fails
with a clear message (create a new version).

### 4.5 Training pipeline (`training.py`, `api/workers.py`, `agents/`)
```
UI "Train & export" -> POST /projects/{id}/training-runs   (refused if the version has no val images or no train boxes)
   queued -> training -> exporting -> awaiting_package -> packaging -> done     (or failed / cancelled)
            GPU agent (Linux/WSL2)                   Raspberry Pi agent
```
- **Workers pull** over HTTP, so no inbound ports are needed on the PC or the Pi. They authenticate with the
  `X-Worker-Token` header and name themselves in `X-Worker-Name`.
- **Lease model.** Claiming a run gives the worker a 15-minute lease, renewed by a background heartbeat every 20 s. A
  silent worker loses its run (requeued; after 3 lost leases it fails). **Time when the backend itself wasn't answering
  never counts against a worker**: after a >2 minute gap in worker traffic, or at startup, all active leases restart
  (`mark_served`). Without this, a Mac that slept would hand healthy runs to whichever agent called first.
- Cancel: the heartbeat response carries it; the GPU agent sets Ultralytics' `trainer.stop` (verified to end training at
  the next epoch). Uploads/completion retry with backoff for hours, so a backend restart doesn't lose a long run.
- **Not a `Job`.** `training_runs` is deliberately separate from `jobs`, because the in-process worker claims *any*
  queued Job and startup fails every running Job.
- The GPU agent checks **Linux and Java first** (fail in seconds, not after hours), trains, exports with
  `format="imx"`, and uploads `best.pt`, `packerOut.zip`, `labels.txt`. The Pi agent runs
  `imx500-package -i packerOut.zip -o out` and uploads `network.rpk`. A run with "Export for IMX500" off stops after training.
- `scripts/seed_synthetic.py` exists so all of this can be tested with no real data.

### 4.6 Security model (`security.py`)
`LoopbackOnly` middleware: only requests from the machine itself may use the normal API; any other client gets 403 on
everything except `/workers/*`, where it must present `MT_WORKER_TOKEN` (unset = worker API disabled, 503). The backend
binds `127.0.0.1` by default. **If you bind `0.0.0.0` so the Pi can connect, tell Vinit.** (Since 2026-10-24 Vinit runs it on `0.0.0.0`
so the always-on Pi worker can reach it: `uvicorn app.main:app --host 0.0.0.0 --port 8000`; other machines still get 403 outside `/workers/*`.) Still no user auth: anyone who
can run a browser *on the machine* can do anything.

A request that carries a proxy or tunnel header (`X-Forwarded-For`, `Forwarded`, `X-Real-IP`, `CF-Connecting-IP`, `CF-Ray`, ...)
counts as **not local even when it comes from 127.0.0.1**: a tunnel client such as cloudflared (used for RunPod) connects from
loopback, and without this the whole API would be open to the internet through it. Only the headers' presence is checked, never
their values. Keep any new proxy header in `PROXY_HEADERS`. `scripts/runpod.py` also tests this through the live tunnel before
it starts a pod. `GET /workers/agent` (token required) serves the three agent files so a fresh pod runs the code on the Mac.

### 4.6b Importing public datasets (`scripts/import_yolo.py`)
`python -m scripts.import_yolo import <folder> --dataset anti-uav --map UAV=multirotor --author "..." [--dry-run --sample-sheet out.jpg]`
imports a YOLO-format export (Roboflow layout: `<split>/images`, `<split>/labels`, `data.yaml`) as **reviewed** images with boxes,
source `imported`, stored under `raw/<class>/imported/<dataset>/` (one folder per imported dataset, e.g. `raw/multirotor/imported/anti-uav/`; the dataset name is a folder-safe slug), with an `ImportRecord` (dataset, original name and split, URL, licence,
author). It collapses augmented copies to one image per source frame, drops frames with no boxes, an unmapped class, or any box
under `--min-box-px` (5), groups frames into **scenes** with a 12x12 colour-thumbnail descriptor (flip- and brightness-robust;
a plain grayscale dHash chains unrelated skies into one giant group, measured on Anti-UAV), keeps at most `--max-per-scene` (20),
clean frames first, and uses the scene as `group_key` so dataset versions never split a scene. The dataset's own train/valid/test
split is ignored: **Anti-UAV's leaks** (84% of its valid images have a near-identical train image, because video frames were split
at random; its filenames restart per split, so they say nothing about which video a frame is from).
It backs up the live SQLite database first (online backup API, to `backend/data/backups/`), commits every 50 images, and a re-run
adds nothing (SHA dedupe + deterministic selection). `rollback --dataset <name>` removes the images, rows, orphan files and the empty folder.
The importer needs `numpy` (pinned). **Order matters:** the running backend must already know the `imported` source (restart it
after pulling this code) *before* the first import, or every endpoint that loads those images fails.
Imported images are **left out of the data bundle by default** (`bundle export --include-imported` to include them): they can be
thousands of files and belong to someone else (see `DATA_LICENSES.md`). Dataset versions that contain them only download on a
machine that has them. **The committed `handover-data/` was exported with `--include-imported`**, so a fresh clone gets Anti-UAV
too; when you re-export for a push, decide again whether to pass the flag.

### 4.7 The data bundle (`scripts/bundle.py`, `handover-data/`)
`export` dumps **every table** to `db.json` (primary-key order, no timestamps, so diffs are stable and re-export is
repeatable), copies every stored file under `storage/`, and writes `files.json` with SHA-256s. `import` verifies every
checksum *before* touching anything (catches CRLF conversion or a lost file), refuses a non-empty database, restores
files, then inserts rows with their **original ids** so dataset-version manifests stay valid. SQLite only. Tables/columns
the current schema doesn't know are skipped and counted. **It is a snapshot, not a sync**: whichever machine runs the backend owns the data. Don't review on two machines at once.

**Refreshing the data** (Vinit will keep creating data after this handover):
1. On the machine that owns the data: `cd backend && python -m scripts.bundle export`, then commit and push `handover-data/`.
   Images are stored under their SHA-256, so a re-export only adds the new files.
2. On another machine, the importer needs an **empty** database. Stop the backend, `mv backend/data backend/data.old`,
   `git pull`, `python -m scripts.bundle import`. That **replaces** that machine's database: anything only it has (reviews,
   `training_runs` and their metrics) is gone from the DB unless you exported it first. If a GPU machine holds runs worth
   keeping, export from there *before* replacing, and don't let two machines add data independently.
3. There is **no merge import** (matching images by SHA-256 and remapping ids, since dataset manifests reference image ids).
   Build one if two machines ever need to add data in parallel.
4. Moved images (re-filed after review) show up in git as renames of files under `handover-data/storage/`.
5. `export` copies *every* file under storage, including training artifacts (`projects/<id>/training/<run>/`: `best.pt`
   about 5 MB, `packerOut.zip`, the `.rpk`). Delete runs you don't need in the UI (that removes their files) before
   exporting, or the weights go into the public history.
6. **Size.** 21 MB for 52 images (the 12 MB one is a single PNG). Plain git is fine up to roughly a thousand images; GitHub
   rejects any file over 100 MB, warns over 50 MB, and recommends keeping a repo under about 1 GB. Beyond that, move images
   out of git (a bucket, rsync, or Git LFS, whose free bandwidth is small) and commit only `db.json`.

### 4.8 AI check and AI labelling (`app/ai/`, `components/AiLabelling.tsx`, the review screen)
A second opinion on an image's boxes, made on this Mac, shown next to the manual check. **It only suggests.** Nothing it does edits a
box or a review status: results live in their own table (`ai_checks`, one row per image), never in `annotations`, because the review
screen loads every annotation as a real box and saves them all back as `reviewed`. Accepting a suggestion edits the boxes in the
editor (unsaved) like drawing would; Save / Accept & next persists them and `put_annotations` then marks the check **stale**
(`ai_checks.stale`, set when the box set's fingerprint no longer matches the one that was checked).

- **AI labelling page** (`/projects/<id>/ai`, `AiLabelling.tsx`; one run = what + models + images). *What*: `mode` **check** (the opinion described
  here) or **autolabel**. *Models*: any detector (`GET /ai/models`: every finished run's `best.pt`, newest first with its mAP50, plus `.pt`
  files in `backend/data/ai/models/custom/`; id `run:5` or `file:name.pt`, default = the setting / newest run) and a SAM size (`t s b l`;
  `get_models(db, detector, sam)` keeps the last 2 loaded sets, keyed by both, so switching needs no restart; a running job keeps the set it
  started with; `set_models` fakes win in tests). *Images*: a class, one source / imported dataset / upload folder, "only not reviewed", a
  limit; `POST /projects/{id}/jobs/ai-check/preview` counts what the run would take (same selection code) for the estimate. Bad choices
  (unknown detector id, a run with no model) are refused when the job is created (422), not after it was queued.
- **Auto-label** (`service.autolabel_image`). Only images that are **pending and have no boxes** (a reviewed image with none is a deliberate
  background picture; nobody's boxes are replaced; checked again right before writing, because the person may be editing in review). The
  detector's class **names** map to the project's trainable classes (never indexes: dataset versions renumber them); a detection of any other
  name is left out and counted (`unmapped`), and a model sharing no class name with the project fails the job before the first image.
  Detections under `min_score` (default 0.5) are ignored, the same drone seen as two classes keeps the surer box, SAM tightens each box
  (the detector's own box when SAM's is not believable), boxes are clipped to the image. Boxes are written as `AnnotationStatus.auto`
  (+ confidence) on an image that stays `pending`; an earlier AI check of that image is marked stale. **They cannot reach training**:
  dataset versions keep only `reviewed`/`auto_exact` boxes (`KEPT_BOX_STATUSES`) on `reviewed` images. A save in review (any edit, Accept or
  Reject) rewrites every box as `reviewed`; navigating without editing saves nothing. `ImageOut.ai_labelled` = some box is still `auto`;
  gallery tag, filter `ai=autolabelled` ("AI drew the boxes (not confirmed)"), a note in the review screen; Accept takes the image out of
  that queue.
- **Uploaded `.pt` files** (`POST /ai/models/upload`, `registry.save_upload`). A YOLO `.pt` is a pickle: **loading one runs code**. So an
  upload is only stored (cleaned name, `.pt`, starts with the zip magic, 300 MB cap, written to a temporary name then renamed, never
  overwrites) and is opened only when a run picks it. The endpoint requires the header `X-MT-Upload: 1`: a multipart POST from another web
  page is a "simple" request that needs no CORS preflight, but a custom header forces one, and the allow-list only names this app's own
  origin. No pickle scanner (a heuristic would give false comfort): the UI says to upload only weights you trust.
- **Models.** *Detector* = `best.pt` of the newest finished training run (copied once to `backend/data/ai/models/detector-run<N>.pt`;
  `MT_AI_DETECTOR_WEIGHTS` overrides). *SAM 2.1 small* through Ultralytics (`sam2.1_s.pt`, 92 MB, downloaded on first use from
  Ultralytics' official GitHub assets into `backend/data/ai/models/`; `MT_AI_SAM_WEIGHTS` picks `_t/_s/_b/_l`). Both run on the
  Mac's GPU (`mps`). A call the GPU refuses is redone on the CPU for that call only; five failures in a row move everything to the
  CPU for good (`device.py`). `GET /ai/status` and every batch job's progress report the live device, the number of fallbacks and the
  last reason, so a slow batch can be explained. Speed: 0.5 s per image in a batch (first project-wide run: 954 images in 7.6 minutes;
  scraped images with many drones are slower because SAM runs once per box), so the ~7,300 reviewed images take about an hour.
- **Why not RF-DETR.** It has no drone class (COCO), so it needs a GPU fine-tune on our data (money, not done). The detector is
  one small class (`detector.py`, `.detect(image_rgb) -> [Detection]`): a fine-tuned RF-DETR drops in by writing that one class.
- **What SAM can and cannot do** (`scripts/ai_spike.py`, 14 images per size bin, anti-uav, sam2.1_s): given a box it segments
  whatever fills the box, so it **polishes** a sloppy box (better than the drawn box 77% of the time, median IoU 0.53 -> 0.65;
  exact labels: 0 of 97 under IoU 0.6 against SAM's tight box, 0.82-0.94 median at every size, even under 8 px, because each box is
  cropped with context and upscaled) but it **cannot fix a loose box** (a 2x box comes back as about the same box) and it **cannot tell
  a right box from a wrong place** (SAM's score: exact 0.87, wrong place 0.63; it finds nothing in 41 of 98 wrong places). So the
  verdict about a label comes from the detector; SAM only tightens suggestions and says whether there is any compact object in a box.
- **Verdicts** (`check.py`, `Policy`, `POLICY_VERSION`): `ok` (a detection overlaps IoU >= 0.6 and SAM's tight box is within IoU 0.45 of the label),
  `adjust` (the detector finds that drone but the box is loose / off / partly on it: suggests SAM's tight box; a detection sitting
  inside a bigger label counts as that label's drone), `suspect` (no detection at the box: either the model sees a drone elsewhere
  and suggests it, or finds none, and says whether SAM sees an object there), `too_small` (nothing confirms it and it is under
  16 px on the 640 px input: the detector finds only 48% / 69% / 82% of drones under 8 / 8-12 / 12-16 px, so "wrong" would be
  mostly false alarms), `missing` (a detection >= 0.5 with no label: a possible unlabelled drone, with SAM's box), `none`. An image's
  verdict is the worst of its items (suspect > missing > adjust > too_small > ok).
  **Rules version:** `POLICY_VERSION` is stored in every check's `model` ("... · rules v2"); bump it when a rule or threshold changes and the
  batch job redoes older checks by itself. History: v1 had `tight_iou` 0.55, taken from exact anti-uav labels (SAM's box agreed with
  all 97 of them). The first project-wide run showed human-drawn labels are looser: 23 of 52 `adjust` flags came from SAM alone while the
  detector agreed with the label, and 83% of those sat at IoU 0.45-0.55 (ordinary looseness, not an error), so v2 uses 0.45.
- **API** (all loopback-only like the rest of the UI API): `GET /ai/status`, `POST|GET /images/{id}/ai-check`,
  `POST /images/{id}/ai-box` (click-to-box: SAM's tight box for a click and/or a rough box; stores nothing),
  `POST /projects/{id}/jobs/ai-check` (batch: the gallery's class/source/dataset/review filters, `limit`, `force`; skips images whose
  boxes and models are unchanged, so it resumes after a restart; hard-negative images are left out), `GET .../jobs?type=ai_check`,
  and `GET .../images?ai=unchecked|stale|ok|flagged|dismissed|too_small`. `ImageOut.ai_status` is the verdict, `stale`, `dismissed`, or null.
- **Dismissed** (so reviewed images leave "AI: needs a look"). Accept & next and Reject in the review screen send `ai_dismiss: true` with the
  save (`PUT /images/{id}/annotations`); where the image has a **flagged** check (suspect / missing / adjust) the server records
  `ai_dismissals(image_id, fingerprint)` with the fingerprint **of the boxes saved**. A current flagged check whose fingerprint equals
  that one is `dismissed` (`AiCheck.dismissed`, `Image.ai_status`, and the SQL in `list_images`: the same rule written twice, tested to
  agree). So a forced re-run or new rules never bring a reviewed image back, whatever they say about those boxes; it is listed again
  only when its boxes change *without* a new decision and the new boxes are flagged. Nothing is recorded for an image the AI had not
  flagged (a later check may raise doubts the person never saw), nor on plain Save or Next. `DELETE /images/{id}/ai-dismiss` undoes it
  ("Put it back in the list" on the review screen); filter `AI: dismissed` finds them. It is its own table because there are no
  migrations: a column on `ai_checks` would not exist in the live database.
- **UI.** Review screen: `A` runs the check (first call loads the models, 10-60 s), suggestions are dashed boxes on the image plus a
  list with "Use this box" / "Add box" / Dismiss / Accept all; `S` snaps the selected box to the object; **Alt/Option + click** on a drone
  draws a SAM box around it. Gallery: an "AI check" filter and a tag on every card. Project page: "AI check" panel (batch job).
- **Install.** `cd backend && .venv/bin/pip install -r requirements-ai.txt` (ultralytics, which brings torch), then restart the backend.
  Without it the backend, its tests and every other feature work as before; the AI endpoints answer 503 with that instruction, and a
  test checks that importing the app loads neither torch nor ultralytics.
- **First real results** (project-wide batch, rules v2, all 7,306 reviewed images in ~45 minutes, 0 failures, no GPU fallbacks):

  | dataset | images | ok | adjust | suspect | missing | too small |
  |---|---|---|---|---|---|---|
  | anti-uav | 4,859 | 4,330 (89%) | 123 | 70 | 41 | 295 |
  | cmy-uav | 2,095 | 1,381 (66%) | 256 | 216 | 20 | 222 |
  | scraped | 350 | 214 | 4 | 7 | 83 | 2 (+40 with no boxes and nothing found) |

  **anti-uav** is optimistic (the detector trained on it). **cmy-uav** (never trained on) agrees with the earlier finding that its boxes are
  displaced: of its labels on drones >= 64 px on the 640 input, **51% are doubtful** (28% at 32-64 px, 11% at 16-32, 3% under 16), and
  on a contact sheet the label box sits on empty sky / trees / a wall while the detector's box sits on the drone (68 of the 82 large
  doubtful labels come with a suggested box elsewhere); the rest are real drones the detector misses (a delivery drone, a hexacopter).
  Overall about a third of cmy-uav's labels are flagged (suspect or adjust or too small to judge). **Do not trust a small contiguous
  sample**: the first 200 cmy-uav images (ids 5229-5391, probably a few scenes) looked 87% ok and gave the wrong impression.
  **Scraped images** carry many unlabelled drones (270 `missing` items in 350 images: swarms, product shots and pending images with no
  boxes; the 12 inspected by eye were all real) and a few loose boxes. Large objects sometimes get an odd suggestion (a long fixed-wing
  UAV came back with a near-full-image box), which is why every suggestion is optional. Nothing was accepted or saved from any of this.
- **Operating notes.** The job worker is one thread: a long AI batch makes scrape jobs wait behind it. The detector is chosen once per
  backend run (a newer finished training run is used only after a restart). If the backend restarts mid-batch the job ends as
  "Interrupted by a server restart"; start it again (`POST /projects/1/jobs/ai-check {"limit": 20000}` or the AI panel): finished
  images are skipped, so it resumes. Restart by port: `lsof -ti tcp:8000 | xargs kill`, then `cd backend && source .venv/bin/activate &&
  uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- **Not exercised:** the S3 storage path with AI checks, a non-Mac machine (CUDA/CPU only through the same code), the `_t/_b/_l`
  SAM sizes, and tiny-drone behaviour of the detector on images it was trained on (run 5 saw anti-uav, so its agreement there is
  optimistic).

## 5. State of the data (honest)

*The numbers below describe the first snapshot. See "Data added after the handover snapshot" just before section 6 for the current state.*

- Project 1 `drone-detector`, 13 classes: positives `multirotor`, `fixed-wing`, `shahed`, `vtol`, `quad-copter`,
  `flying-wing`; hard negatives `birds`, `aircraft`, `helicopters`, `balloons`, `empty-sky`, `drones-in-fog`, `smoke`.
- **52 images, all in `multirotor`**: 2 uploaded + 50 scraped (one real SerpAPI search, "DJI Quadcopter", 50 of 100
  results kept). **Only 6 are reviewed, carrying 6 boxes.** 46 are pending; 2 of those already have a box drawn but were never marked
  reviewed, so a dataset version would leave them out until they are. (The database holds 8 box rows in total.) No other class has any image.
- **There are 0 dataset versions and nothing is trainable yet.** A run needs a version with validation images and boxes;
  six boxed images of one class won't do. Don't create a dataset version from them.
- Many scraped images are product shots (white background, packaging). Expect to reject a good share.
- SerpAPI quota: 2 of 250 searches used this month (as of the handover).
- **This repo is public, and the 50 scraped images are in it, by Vinit's informed choice.** They came from commercial and
  editorial pages (DJI, Amazon, eBay, Walmart, Wired, The Verge, Space.com, Popular Mechanics and others) and remain the
  property of their owners; each image's source page is in `scrape_records` / `handover-data/db.json`. If a rights holder
  asks for removal, tell Vinit: deleting the file is not enough, because git history keeps it (removing it needs a history
  rewrite) and forks may exist. Think before committing *new* scraped images: each export adds more third-party images to a
  public history.

### Data added after the handover snapshot (2026-10-21)
The snapshot above is what `handover-data/` holds. On the owner's machine the project has since grown: more scraped images (hundreds
reviewed, boxes started for `fixed-wing`, `shahed`, `vtol`, `interceptor`) and the **Anti-UAV** dataset imported as
`imported/anti-uav` (4,859 reviewed `multirotor` images across 2,065 scenes, ~190 MB; CC BY 4.0, see `DATA_LICENSES.md`).
The committed `handover-data/` was re-exported with `--include-imported` after that import, so it holds the *later* state too: a
fresh clone plus `python -m scripts.bundle import` restores Anti-UAV with no re-download (the raw Roboflow export, 826 MB with
augmented copies, is deliberately not in git). If you re-download it anyway (YOLOv11, version 1), the importer reproduces the same selection.

## 6. What has and hasn't been verified

Verified on the previous machine (a Mac mini M2, CPU only):
- 230 backend tests including mutation checks (deliberately breaking the code to confirm tests fail) on the state
  machine, LAN guard, leases, agents, retries and the bundle, and (2026-10-22) on the RunPod launcher's pod clean-up, the tunnel
  security check, the pod script's exit trap and the proxy-header guard.
- RunPod, read-only and against the live API: balance, GPU prices, `GET /pods`, and the dry-run launcher. cloudflared is installed
  (Homebrew) and a real quick tunnel was checked 3 times: only `/workers/*` answers through it, the rest gives 403, the agent bundle
  downloads. The Mac's own DNS often can't resolve a new trycloudflare.com name, so `scripts/runpod.py` falls back to Cloudflare's
  resolver for its self-test. **First real run (2026-10-22, run 1, 1 epoch, A4000, ~$0.04):** pod creation, the start script, the
  agent claiming the run through the tunnel, dataset download, training (mAP50 0.116 after one epoch) and the artifact upload
  worked, and the pod was deleted. The IMX export failed on a matplotlib version swap (fixed in `runpod_start.sh`, untested on a pod).
  Run 2 (~$0.05): setup with the export tools passed, then the dataset download arrived cut short ("not a zip file"); the agent
  now checks the received size, retries, and reports what arrived. Pod 3 (run 3, ~$0.09): the launcher deleted the pod in the middle of the
  export after one failed look at RunPod (a blip in the Mac's internet); it now tolerates 10 minutes of that. Pod 4 (run 3 again,
  ~$0.26, the pod's whole hour): trained, then the upload of `best.pt` was cut off 13 times in a row (heartbeats on the same tunnel
  were fine, and the same upload had worked twice before) until the pod's time limit killed it; cause unknown, candidates in
  `agents/RUNPOD.md`. Since then: the agent gives up on an unreachable backend after 15 minutes on a pod (`MT_PATIENCE`), still runs
  the IMX export if only the `best.pt` upload failed and says so in the run's error, and the launcher reports an unfinished run
  instead of "ended by itself". **Pod 5 (run 4, ~$0.07) passed end to end:** dataset from the R2 bucket, training, `best.pt` upload, and the **IMX export**
  (`packerOut.zip` + `labels.txt`); the pod deleted itself. **Not verified:** `imx500-package` on the Pi and the box format. Four pods on three runs cost $0.45; balance $14.55 of $15.00. Run 3 is dead (stage `training`, lease expired about 06:36 IST;
  the next launch releases it) and its `best_pt` is from pod 3.
- Real Ultralytics training and mid-training cancel on CPU, with correct metrics reaching the backend.
- The whole UI end to end in a browser on throwaway data: queue, live progress, stages, cancel, delete, downloads.
- Bundle export -> import round trip, row for row and byte for byte.

**Never run** (this is why you're here):
1. Ultralytics `format="imx"` for real (Linux only). Read from the Ultralytics 8.4.183 source: output folder
   `<weights>_imx_model/` containing `packerOut.zip` and `labels.txt`; `export()` returns that path; it auto-installs
   Java via apt if missing (sudo), and pip-installs `model-compression-toolkit>=2.4.1`, `edge-mdt-cl<1.1.0`,
   `edge-mdt-tpc>=1.2.0`, `pydantic<2.12`, `imx500-converter[pt]>=3.17.3`; `nms` is forced on; calibration images come
   from the **val** split of `data.yaml`. Ultralytics declares the IMX env as **Python 3.11 with torch >=2.9,<2.12**.
2. `imx500-package` (Raspberry Pi only, apt `imx500-tools`) and loading `network.rpk` on the camera.
3. Networking from WSL2 and from the Pi to the backend.
4. Training on a real CUDA GPU. (CPU was the only thing available.)
5. Backend on Python 3.12, and anything under a native Windows Python.

## 7. Roadmap (not built)

In rough priority order:
1. **Prove the toolchain** (section 3, steps 8-10).
2. **Auto-labeling** (the biggest time saver; the AI check, section 4.8, already proposes boxes for images with none, from the
   trained detector + SAM; still to do: an open-vocabulary detector for classes the YOLO cannot see yet): propose boxes with (Grounding DINO or OWLv2),
   optionally refine with SAM 2, store as `annotations` with `status=auto` and a confidence, and let the user correct in
   the review UI. The local 3060 12 GB is plenty. Implement as a new job type or a worker kind, like the training agents.
3. **Composite generator** (v1 generation): SAM-cut drones from reviewed images pasted onto sky/background images at random
   scale (biased to tiny), blur, brightness and noise, written to `.../generated/` as `auto_exact` boxes.
4. Real data: review the 46 pending, scrape more per class (mind the quota), upload own footage frames, add yt-dlp frame extraction.
5. Real training run, then an on-device check on the Pi (FPS and accuracy on a test clip).
6. **v2: FLUX generation** (the user asked for this explicitly). The 3060 12 GB can run a quantized FLUX slowly; RunPod is the
   alternative for volume.
7. A merge-capable data import (section 4.7), then smaller items: Alembic migrations, gallery thumbnails, a CLIP relevance filter for scraped images, store SerpAPI
   `license_details_url` (present on ~21% of results) and download-failure reasons, hash thumbnails *before* downloading to
   catch the same photo at different URLs, a "release this run" call so Ctrl-C on an agent frees its run immediately,
   training resume after a lost worker, S3 storage testing, docker-compose.

## 8. Gotchas (each cost the previous session time)

- **Never run tests against real data.** `tests/conftest.py` sets `MT_DATABASE_URL`, storage, and blank API keys *before*
  importing the app. A real uvicorn server runs the app's lifespan (creates tables, fails running jobs, starts the job
  worker), which once nearly ran against the real database. Keep that guard and use `lifespan="off"` in test servers.
- **Restart servers by port** after code changes (uvicorn runs without `--reload`): `lsof -ti tcp:8000 | xargs kill`.
- **Never run `next build` while `next dev` is running**: they share `.next` and the browser gets ChunkLoadError. Use
  `npx tsc --noEmit` and `npx next lint` for checks.
- **Next.js is pinned to 15.1.12** (exact). Every 15.1.x up to 15.1.10 is flagged by npm as vulnerable (CVE-2025-66478), and GitHub
  would raise an alert. Don't downgrade; if you upgrade, stay on a non-deprecated version (`npm view next@<v> deprecated` is empty).
- Server timestamps are **UTC with no zone suffix**; `new Date(x)` reads them as local time. Use `parseUtc()` from `lib/api.ts`.
- Tests use a **file-backed** SQLite in `tmp_path`, not `:memory:` + `StaticPool`: threads shared one connection and a cancel
  test flaked about 1 run in 8.
- This Starlette's `TestClient` has no `client=` argument; `tests/helpers.from_ip` wraps the app to fake the peer address.
- The real scrape job costs SerpAPI credits. Tests use fake transports; never point a test at the real API.
- If you drive the UI with browser automation: set checkboxes with real clicks, not a form-fill tool (React's `onChange`
  doesn't fire), and each browser-tool call takes ~10-15 s, so test timers need long durations.
- macOS `sed -i` needs `-i ''`; zsh doesn't word-split `$VAR` (use a Python loop). Not an issue on Ubuntu/bash.
- Windows/WSL: `.gitattributes` forces LF and marks images binary. If the bundle import says "corrupted", a checkout
  rewrote bytes; re-clone with `git config core.autocrlf false`.
- Don't let two backends (or two machines) edit the same project. The bundle is one-way.
- **Mutation-testing by rewriting a source file and rerunning pytest can leave stale bytecode**: a pyc records the source's mtime in
  whole seconds and its size, so a mutation and its restoration made in the same second with the same size keep the mutated pyc. Run the
  mutated tests with `python -B` / `PYTHONDONTWRITEBYTECODE=1` and delete `__pycache__` afterwards (this once left two tests failing
  against restored source).
- macOS clears `/tmp` on reboot (and the Claude Code scratchpad lives there): keep anything you need in `backend/data/` (git-ignored) or the repo.

## 9. Rules for working here

- **Never echo secrets** (`.env` values, tokens, the worker token). Read them from the file or environment inside a script;
  don't `cat .env`. If a tool prints one by accident, say so.
- Ask before anything that **costs money** (RunPod, SerpAPI searches beyond a test), **opens a network port**, or
  **deletes/overwrites Vinit's data**. Prefer throwaway data (`scripts/seed_synthetic.py`, a temp `MT_DATABASE_URL`) for experiments.
- Put the work in **small, separately tested commits** and mutation-check new tests (break the code, see the test fail).
- Report what you did *not* run. The previous session's biggest risk was claiming a path worked that nothing had exercised.
