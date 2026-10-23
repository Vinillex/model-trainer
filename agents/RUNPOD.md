# Training on RunPod (no GPU of your own needed)

`python -m scripts.runpod launch` rents one RunPod GPU for the length of a training run, lets the usual GPU agent
(`gpu_agent.py`) train and export on it, and then deletes the pod. You only pay while the pod exists: at $0.25 to $0.50 an hour, a
run of one to two hours costs well under a dollar (how long a run takes is not measured yet; the first run will tell). The
Raspberry Pi still does the final `.rpk` packaging (`agents/README.md` section 2),
unless you turn **Export for IMX500** off and only want `best.pt`.

```
your Mac                                                RunPod
 backend (127.0.0.1:8000)  <-- cloudflared tunnel <--   pod: gpu_agent.py --once
 python -m scripts.runpod launch --yes                    1. fetches the agent code from the backend
   starts the tunnel, checks it, starts the pod,          2. downloads the dataset, trains, exports for the IMX500
   watches the run, then deletes the pod                  3. uploads best.pt / packerOut.zip / labels.txt, deletes itself
```

> **First real run (2026-10-22, 1 epoch, RTX A4000, about $0.04, 10 minutes):** the tunnel check, pod creation, the pod's setup
> (about 5 minutes), the agent claiming the run, the 364 MB dataset download, training, uploading `best.pt`, and the pod being
> deleted afterwards all worked. The **IMX500 export failed**: Ultralytics installed its export tools in the middle of the export,
> and model-compression-toolkit (which needs matplotlib < 3.10) swapped matplotlib under the running program. The pod script
> now installs those tools during setup and checks that they import, and Ultralytics is pinned to the version whose package
> list that is. **The export itself has still not succeeded anywhere**, so the next run is the real test. Also learned: asking RunPod
> for hosts with CUDA 12.8 or newer left no stock at all, so the pod now picks its torch build from whatever driver the host has.

> **Second run (run 2, same day, about $0.05):** the pod's setup, now including the export tools and their import check, got
> through and the agent claimed the run. It then failed because the downloaded dataset "is not a zip file": the 364 MB transfer
> through the tunnel was most likely cut short, and the agent had accepted whatever arrived as the whole file. The agent now
> compares the received size with the announced one (a short download is fetched again), retries a bad zip up to 3 times, and a
> final failure says how many bytes arrived and what they started with. Also: RunPod sometimes has no stock for the cheap GPUs
> (it said so on 2 of 4 attempts); a failed request costs nothing, and the launcher can simply be run again.

## What it costs and what protects your money

- Live prices (Secure cloud, per hour, 2026-10-22): RTX A4000 $0.25, A4500 $0.25, A5000 $0.27, RTX 4000 Ada $0.28, RTX 3090 $0.50.
  `python -m scripts.runpod gpus` shows today's. The pod gets the first GPU in that list that has stock.
- **A hard limit**: `--max-hours` (default 3, at most 12). The pod stops itself at the limit (inside the pod, so even if your Mac
  dies) and the launcher stops it too. The most a run can cost is that many hours at the dearest GPU it may pick.
- `launch` refuses to start unless your **balance covers that worst case**: $2.67 for the default (3 h at the RTX 4090 fallback, $0.89/h;
  with only cheap GPUs in `--gpu` it is under $1). Add **$5** and leave auto-reload off; then the most you can ever lose is $5.
- `launch` without `--yes` is a **dry run**: it checks everything and prints what would happen. `--yes` is the only thing that spends money.
- The pod **deletes itself** when its work ends, fails, or hits the limit; the launcher then checks and deletes it again if needed; and
  `python -m scripts.runpod stop --all` removes every pod this project started (names start with `mt-`; it never touches other pods).
- No run waiting means no pod: `launch` needs a queued run, so an idle pod can't happen by accident.

## What leaves your Mac

- The **images of the dataset version** being trained (from the queued run), through a Cloudflare tunnel to RunPod. Cloudflare and
  RunPod can therefore see them. The repo is public already, but your reviewed images are yours, and `cmy-uav` is ticked by default in
  "Data to include": untick whatever shouldn't go.
- The **worker token** (`MT_WORKER_TOKEN`) as a setting of the pod. It only opens `/workers/*` on your backend, and only while the tunnel is up.
  Nothing else: not your RunPod key, not your SerpAPI key, not the database.
- The tunnel is open **only while `launch` runs**. Through it the internet can reach nothing but `/workers/*`, and the launcher
  tests that before it starts a pod: if `/health` or `/projects` answer through the tunnel it stops and starts nothing.

## One-time setup

1. **Add credit** at https://console.runpod.io/user/billing ($5 is plenty). Only you can do this.
2. **Install the tunnel tool**: `brew install cloudflared`. No Cloudflare account is needed.
3. **Restart the backend** so it runs the current security code (it has no `--reload`):
   `cd backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000`. Keep it on 127.0.0.1: the tunnel runs on this Mac,
   so the backend doesn't need to listen on the network.
4. Check: `cd backend && source .venv/bin/activate && python -m scripts.runpod status`. It should show the key, token,
   cloudflared, your balance, and no pods.

## Every run

1. In the UI, create a dataset version (untick datasets you don't want) and start a run in **Train & export**. For the first run
   use **epochs 1**.
2. `python -m scripts.runpod launch` (a dry run). Read the plan: run, GPUs, worst-case cost, what leaves the Mac.
3. `python -m scripts.runpod launch --yes`. It starts the tunnel, checks it, starts the pod and prints progress until the run
   leaves the GPU stage. Keep the window open and the Mac awake (it also runs `caffeinate`). **Ctrl-C ends the pod.**
4. The pod needs a few minutes to install its tools (Python 3.11, torch 2.9.1 for CUDA 12.8, Java, Ultralytics) before it picks the
   run up. Progress appears in the UI exactly as with a local GPU. When the run reaches `awaiting_package`, the Pi takes over.

Options: `--gpu ID` (repeat; see `gpus`), `--cloud community` (cheaper, less steady), `--max-hours N`, `--image TAG`,
`--server URL` (use a tunnel you started yourself instead of cloudflared).

## Optional: a dataset bucket (Cloudflare R2), so pods don't download through your Mac

Your Mac uploads at only about 0.15-0.35 MB/s, and without a bucket every pod downloads the whole dataset through it (the full v1
dataset: 20-40 paid minutes). With a bucket, the zip is uploaded **once per dataset version** and **before any pod exists** (free),
and each pod downloads it from Cloudflare in seconds through a signed link that opens only that file and expires after 6 hours.
The bucket stays private. If anything is wrong with the bucket's copy, the pod falls back to the tunnel by itself.

1. In the Cloudflare dashboard: **R2 Object Storage**, enable it (free up to 10 GB; Cloudflare asks for a payment method), and
   create a bucket, e.g. `model-trainer-datasets` (location: Automatic).
2. **R2 > Manage API tokens > Create API token**: permission **Object Read & Write**, applied to **that bucket only**.
3. Copy into `backend/.env` yourself (shown only once by Cloudflare; never paste them into a chat):
   ```
   MT_DATASET_BUCKET=model-trainer-datasets
   MT_DATASET_BUCKET_ENDPOINT=https://<account id>.r2.cloudflarestorage.com
   MT_DATASET_BUCKET_ACCESS_KEY=<Access Key ID>
   MT_DATASET_BUCKET_SECRET_KEY=<Secret Access Key>
   ```
4. Restart the backend. `python -m scripts.runpod status` then shows the bucket.
5. Optional, ahead of time (for example overnight): `python -m scripts.runpod upload-dataset` uploads the zip of the next waiting
   run's dataset (or `upload-dataset <version id>`). Otherwise `launch --yes` does it first, before the tunnel and the pod.
   Deleting a dataset version in the UI deletes its copy too.

## What the real pods showed (2026-10-22: four pods, three runs, 1 epoch each, $0.45 in all)

- **Works:** the tunnel and its self-test, pod creation, the pod script (about 5 minutes of setup), the agent claiming the run, the
  364 MB dataset download, training (1 epoch: mAP50 about 0.11), uploading `best.pt` (pods 1 and 3), deleting the pod, closing the tunnel,
  and the pod's own hard time limit (pod 4 was stopped by it, before the launcher's limit plus grace).
- **Pod 1 (run 1):** the IMX export crashed with a matplotlib error. The export tools are now installed before any Python runs (`MT_PREP_IMX`).
- **Pod 2 (run 2):** the dataset download was cut short (a bad zip). Downloads are now length-checked and retried.
- **Pod 3 (run 3):** a blip in the Mac's internet made the launcher delete the pod in the middle of the export. It now keeps watching for 10 minutes.
- **Pod 4 (run 3 again, about $0.26, the pod's whole hour):** trained fine, then the upload of `best.pt` (5.5 MB) was cut off 13 times in a
  row, about 2 minutes apart (the agent's 120 s socket timeout plus its backoff; the backend logged `ClientDisconnect` each time), until
  the pod's one-hour limit killed it. The stream stopped completely, it was not just slow. Heartbeats and the 364 MB download worked on the
  same host and tunnel the whole time, and the same upload had worked on pods 1 and 3, so what changed is the host (pod 4 appeared in the
  backend log as 213.173.98.71, pod 3 as 69.30.85.173, pods 1 and 2 as 185.226.15x.x). **Cause not known.** Candidates, most likely
  first: that host's outbound path for large packets; cloudflared's default QUIC transport on the Mac side (`cloudflared tunnel --protocol
  http2` is an untested option); the Mac's own connection. Chunked, resumable uploads would make this much sturdier if it repeats.
  Run 3 was left at stage `training` with no error because nothing ever called `fail`: the retry patience was 6 hours.
- **Fixed after pod 4:** the agent's patience is now `--patience` / `MT_PATIENCE` (6 h by default; **15 minutes on a pod**). If the upload of
  `best.pt` can't get through, the agent still runs the IMX export, then fails the run saying whether the export worked, so one more pod
  answers the export question even if uploads stall again. The progress line shows "uploading the trained weights". The launcher no longer
  says "ended by itself" and exits 0 for a pod that vanished with its run unfinished: it says the run did not finish, exits 1, prints what
  RunPod reports about the pod (`gone`, `exited`, ...), and says when its delete found the pod already gone.
- **Checked from the Mac afterwards (free, no pod):** a 5.5 MB PUT through a quick tunnel to a throwaway local server failed 3 of 8
  times with cloudflared's default QUIC (the server received 0 bytes, as with pod 4) and 0 of 8 with `--protocol http2`, so the
  launcher's tunnel now uses http2 (the security self-test passes over it). The Mac's own **upload** speed measured only about
  0.15-0.35 MB/s (download about 10 MB/s). Pods download the dataset through that upload link, so the full v1 dataset (364 MB) takes
  roughly 20-40 minutes of paid pod time; uploads from the pod to the Mac use the fast direction.
- **Pod setup now also checks**, before any training: Ultralytics' own requirement check of the export tools (without installing),
  `imxconv-pt` in the venv, and Java 17 or newer. Any of these missing would otherwise only show after training.
- **Next test:** run 4 `export-smoke-test` (1 epoch, IMX export on) on dataset version `smoke-scraped` (229 reviewed scraped images,
  62 MB, val 23 images), with `--max-hours 0.75` (worst case $0.21). Expected: about 5 min setup, 3-5 min download, a minute of training,
  5-10 min export. Run 3 (dead) was cancelled so the launcher can't pick it first.
- **Pod 5 (run 4 `export-smoke-test`, 2026-10-22 12:36-12:52, about $0.07): everything worked, including the IMX export.**
  Setup 7 min (the new export-tool checks passed), dataset from the R2 bucket in seconds, 1 epoch in under a minute, `best.pt`
  uploaded through the http2 tunnel, IMX export about 8 min, `packerOut.zip` (2.2 MB: sdpsA/C.bin, cfgA/C.bin, l2staticData.bin,
  dnnParams.xml, manifest.json) and `labels.txt` (5 classes) uploaded; the run is `awaiting_package`. The pod **deleted itself**
  (RunPod no longer knew it when the launcher tried). Still open: `imx500-package` on the Pi, and the box format on the camera.
- **Pi, same day (Raspberry Pi 5, Debian 13, imx500-tools preinstalled):** `pi_agent.py --once` packaged run 4 in 3 s
  (`network.rpk`, 3.1 MB); run 4 is **done**. On the camera: the network loads in 13-20 s per program start (about 70 s the
  very first time), input 640x640, outputs `(1,300,4)` boxes, `(1,300)` scores, `(1,300)` classes, `(1,1)` count (NMS on the
  chip). **Boxes are in pixels of the 640x640 input, not normalized** (values up to about 667), so `bbox_normalization=False`;
  Ultralytics writes them as x1,y1,x2,y2 (`bbox_order="xy"`), still to be confirmed with an object at a known place. The camera
  ran at 30 fps but network output came on 112 of 451 frames (about 7.5 inferences/s) although the chip reports about 30 ms of
  DNN time per inference; worth looking into before relying on a frame rate.
- **Pods 6 and 7 (run 5 `v1-100ep`, 2026-10-22 evening, about $0.17 in all): both stuck in the PyTorch install and given up on.**
  Both got through the system packages and fetching the agent, then sat in "installing Python 3.11 and torch" with no error for 13+
  minutes (pod 6 on host 213.173.99.32; pod 7 reported its steps and was in the torch install from 20:45 until the launcher's 20:58
  limit). Healthy pods do that step in 3-6 minutes. Pod 7's driver said CUDA 13, so the script picked the never-tried `cu130` build.
  Not this Mac's internet: those downloads go from the data centre straight to PyTorch's server. Cause unknown (slow route from that
  host, or a hung connection). **Fixes:** the pod now caps torch at `cu128` (any CUDA 13 driver runs it), times a 40 MB range of the real
  wheel first and gives up in under a minute if the route is slow (`SLOW DOWNLOAD`), runs each install under a time limit
  (`MT_INSTALL_SECONDS`, 8 min) with one retry, and reports every step to the backend (`POST /workers/log`, file per worker in
  `backend/data/worker_logs/`), which the launcher prints as `pod: ...` lines. The launcher checks from the Mac that the wheels,
  the uv installer and PyPI answer before spending anything, gives up on a pod that has reported but then goes quiet for 12 minutes
  (40 minutes at most while it keeps reporting; 20 minutes if it never reports), and says how to retry. Rerun the same command after
  `SLOW DOWNLOAD`: a new pod gets another host.
- **Run 5 `v1-100ep` (2026-10-22 evening, pod 8, 100 epochs, full v1 from R2): trained fine, then the pod was killed (exit 137, no message)
  13 minutes into the IMX export** after 55 minutes (about $0.28; setup 5 min thanks to the fixes above, epochs 16-32 s each, last-epoch
  val mAP50 0.603). `best.pt` had been uploaded before, so nothing was lost. Cause not proven: the export calibrated on all 520 validation
  images (one at a time) where the smoke test used 23; likely memory. The pod now exports on at most 150 images (`calibration_fraction`)
  and reports its memory every 2 minutes plus the cgroup `oom_kill` counter and "exit 137 = killed" when it dies.
  **The export also works on the Mac** (no GPU, no pod, free): the converter is pure Java and Ultralytics only asserts `LINUX`; setting
  `ultralytics.engine.exporter.LINUX = True` in a Python 3.11 venv with torch 2.9.1 + the same export tools did the whole export in
  7.5 minutes, 150 calibration images, 2.8 GB of memory. The result was attached to run 5 through the worker API and the Pi packaged it
  (`network.rpk`, 3.3 MB). Run 5's best.pt measured on the Mac CPU: **test split (523 images, never trained on) P 0.84, R 0.48,
  mAP50 0.651, mAP50-95 0.531; multirotor alone AP50 0.929** (532 boxes). The other four classes have 1-6 boxes per split, so their
  numbers mean nothing yet.
- **Seen before pod 5 only as failures:** a finished IMX export and a pod deleting itself (both confirmed by pod 5). The old launcher printed
  "pod deleted" even when the pod was already gone, so pod 4 can't settle it; the next run's output will. The launcher cleans up either
  way; if a pod is left, `python -m scripts.runpod stop --all`.
- While a run is on, the RunPod console's Pods page shows one pod named `mt-MMDD-HHMMSS`. When the launcher says "pod deleted", the page
  must show none. The pod's Logs (lines start with `[mt-pod`) are lost when the pod is deleted.
- If the export fails but training works, untick **Export for IMX500** to get `best.pt` anyway.

## If something looks wrong

| You see | Likely cause / what to do |
|---|---|
| `Your RunPod balance is $0.00...` | Add credit (step 1). |
| `SECURITY: ... the internet can reach the whole API` | The backend is running old code. Restart it (step 3). Nothing was started. |
| `cloudflared isn't installed` | `brew install cloudflared`. |
| `The tunnel doesn't work: ConnectError` | Seen on the real Mac: its ISP DNS often never learns a brand-new `*.trycloudflare.com` name. The launcher then asks Cloudflare's resolver (1.1.1.1) directly, so this should be rare; if it still happens, run it again (each try gets a new name). The pod itself uses RunPod's DNS and is not affected. |
| `RunPod refused to start the pod: ...` | RunPod's message is shown (no stock, a field it rejects). Try `--cloud community` or another `--gpu`; tell me the message if it is about a field. |
| Pod never picks the run up | Look at the pod's Logs. Usual suspects: the CUDA/driver combination (`torch cannot see a GPU`), a package download failing, the tunnel address no longer working. |
| `POD ... IS STILL THERE AND CHARGING` | Delete it in the console now, or run `python -m scripts.runpod stop --all`. |

## Still unconfirmed

- RunPod gives every pod `RUNPOD_POD_ID` and `RUNPOD_API_KEY` (a key for that pod), which the pod uses to delete itself. If not,
  the launcher on the Mac deletes the pod, and the hard limit inside the pod still stops the work. (Confirmed so far: `POST /pods`
  takes the spec, the `runpod/base` image and the `bash -c` start command; torch's CUDA build is picked from the host's driver.)
- The Sony export tools install and import on the pod, but a whole export has not finished yet.
