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
| Compute | **Local GPUs** (RTX 4060 Ti 8 GB and RTX 3060 12 GB, both in Windows PCs). **RunPod is not on the training path.** An optional API key setting exists (`MT_RUNPOD_API_KEY`, read-only account check only) and is unused. Keep RunPod as an option for FLUX or large auto-labeling jobs only. |
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
    jobs.py         tiny DB-backed queue + in-process worker thread (scrape jobs only)
    datasets.py     dataset versions: grouped/stratified train-val-test split, stats, warnings
    yolo_export.py  frozen manifest -> YOLO zip (no DB access, reusable)
    training.py     training-run state machine: stages, leases, claim/heartbeat/complete (section 4.5)
    security.py     LoopbackOnly middleware + worker-token check
    runpod_client.py   read-only RunPod balance query
  scripts/
    bundle.py       export/import the whole database + files as a portable bundle
    seed_synthetic.py  make a synthetic project so the pipeline can be tested with no real data
  tests/            119 tests (pytest)
agents/             what runs on the GPU PC and the Pi: gpu_agent.py, pi_agent.py, common.py, README.md
frontend/           Next.js 15 (app router, TypeScript, Tailwind 3). Talks to the API at NEXT_PUBLIC_API_URL (default http://localhost:8000)
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
   git clone <repo-url> ~/model_trainer && cd ~/model_trainer/backend
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
   The backend was developed and tested on **Python 3.13**. Ubuntu 24.04 ships 3.12: try it, and if anything fails get
   3.13 with `uv` (`pip install uv && uv venv --python 3.13 .venv`). This venv is **separate** from the GPU venv in step 8.
4. **Secrets.** Copy `backend/.env.example` to `backend/.env`. **Do not copy the Mac's `.env`.**
   - `MT_WORKER_TOKEN`: generate a fresh one: `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Required for any worker.
   - `MT_SERPAPI_KEY`: only needed for scraping. Ask Vinit. The free plan is 250 searches/month **shared** with their Mac.
   - `MT_RUNPOD_API_KEY`: leave empty.
   Never print or log these values. `.env` is gitignored; keep it that way.
5. **Run the tests first.** `python -m pytest -q` from `backend/` should say **119 passed**. If not, stop and tell Vinit.
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
10. **The Raspberry Pi** (`agents/README.md` section 2): `sudo apt install imx500-tools`, copy `common.py` and
    `pi_agent.py`, point it at this machine's LAN address. Success = the run reaches `done` and `network.rpk` downloads.
    Then load it on the Pi with Ultralytics' Sony IMX500 guide.

Then tell Vinit what worked. Steps 8-10 are the first time those parts ever run.

## 4. How it works

### 4.1 Data model (`models.py`)
`projects` -> `classes` (`is_negative` = hard-negative category such as birds or balloons; negative classes get no boxes,
their images count as background; `index` = creation-order position among the positive classes, `None` for negatives; each dataset version then numbers its positives 0..n-1 in that order, so use a model's own `labels.txt`) ->
`images` (`source` uploaded|scraped|generated, `review_status` pending|reviewed|rejected, `storage_key`, `sha256`,
`group_key`) -> `annotations` (normalized YOLO boxes cx,cy,w,h; `status` auto|auto_exact|reviewed|rejected).
Also `image_hashes`, `scrape_records` (provenance: query, original URL, page URL), `scraped_urls` (every URL ever tried),
`search_log`, `jobs`, `dataset_versions`, `training_runs`. Images live on disk under
`backend/data/storage/projects/<id>/raw/<class>/<source>/<sha256>.<ext>` (the per-class `uploaded/scraped/generated`
layout the user asked for); the DB is `backend/data/model_trainer.db`.

There are **no migrations**. `create_all` makes missing *tables* on startup; a new *column* on an existing table needs a
manual `ALTER TABLE`. Alembic is a known to-do. `scripts/bundle.py` tolerates schema drift (section 4.7).

### 4.2 Review UI
`/projects/<id>/review?...` queue with filters, hotkeys (1-9 pick class, Enter accept, X reject, arrows, Delete), box
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
together, so near-duplicates can't leak from train into val), **stratified per class**, deterministic by seed, and
generated images are train-only. Warnings are attached (empty val/test, class with no boxes, fewer than 20 boxes).
Download gives a standard Ultralytics folder: `data.yaml`, `images/{train,val,test}`, `labels/...`.
**Trap:** `data.yaml` says `path: .`, which Ultralytics resolves against *its own datasets directory*, not the yaml's
folder. `agents/gpu_agent.py` rewrites it to the absolute extract path; do the same if you ever train by hand.
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
binds `127.0.0.1` by default. **If you bind `0.0.0.0` so the Pi can connect, tell Vinit.** Still no user auth: anyone who
can run a browser *on the machine* can do anything.

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
4. `export` copies *every* file under storage, including training artifacts (`projects/<id>/training/<run>/`: `best.pt`
   about 5 MB, `packerOut.zip`, the `.rpk`). Delete runs you don't need in the UI (that removes their files) before
   exporting, or the weights go into the public history.
5. **Size.** 21 MB for 52 images (the 12 MB one is a single PNG). Plain git is fine up to roughly a thousand images; GitHub
   rejects any file over 100 MB, warns over 50 MB, and recommends keeping a repo under about 1 GB. Beyond that, move images
   out of git (a bucket, rsync, or Git LFS, whose free bandwidth is small) and commit only `db.json`.

## 5. State of the data (honest)

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

## 6. What has and hasn't been verified

Verified on the previous machine (a Mac mini M2, CPU only):
- 119 backend tests including mutation checks (deliberately breaking the code to confirm tests fail) on the state
  machine, LAN guard, leases, agents, retries and the bundle.
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
2. **Auto-labeling** (the biggest time saver): propose boxes with an open-vocabulary detector (Grounding DINO or OWLv2),
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

## 9. Rules for working here

- **Never echo secrets** (`.env` values, tokens, the worker token). Read them from the file or environment inside a script;
  don't `cat .env`. If a tool prints one by accident, say so.
- Ask before anything that **costs money** (RunPod, SerpAPI searches beyond a test), **opens a network port**, or
  **deletes/overwrites Vinit's data**. Prefer throwaway data (`scripts/seed_synthetic.py`, a temp `MT_DATABASE_URL`) for experiments.
- Put the work in **small, separately tested commits** and mutation-check new tests (break the code, see the test fail).
- Report what you did *not* run. The previous session's biggest risk was claiming a path worked that nothing had exercised.
