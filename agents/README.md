# Workers: GPU PC and Raspberry Pi

The backend (on the Mac) never trains anything itself. Two small agents pull work from it over your home network:

```
Mac: backend + UI ──────────────── queue of training runs
        ▲                 ▲
        │ polls           │ polls
GPU PC (Windows + WSL2)   Raspberry Pi + AI Camera
gpu_agent.py              pi_agent.py
 1. download dataset zip   1. download packerOut.zip
 2. train YOLO11n          2. imx500-package → network.rpk
 3. export for the IMX500  3. upload network.rpk
 4. upload best.pt + packerOut.zip + labels.txt
```

Agents only make outgoing requests, so the PC and the Pi need no open ports. A worker that goes silent for 15 minutes
loses its run, which goes back in the queue (after 3 lost leases it fails). Time when the *backend* wasn't answering
(the Mac slept, you restarted it) never counts against a worker: the agents keep retrying and carry on, and their
15 minutes start again once the backend is back.

> **Not verified yet:** the real IMX export and `imx500-package` have not been run, only the plumbing around them.
> Step 1.5 below tests the export on its own before you rely on it.

## 0. On the Mac (backend)

1. The token is already in `backend/.env` as `MT_WORKER_TOKEN`. Show it on your own screen when you need to copy it:
   `grep MT_WORKER_TOKEN backend/.env`
2. Make the backend reachable from other machines on your network:
   ```
   cd backend && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   macOS may ask whether Python may accept incoming connections: choose Allow.
   Other machines can only reach `/workers/*`, and only with the token. Everything else answers 403 unless the
   request comes from the Mac itself.
3. Find the Mac's address: `ipconfig getifaddr en0` (Wi-Fi; try `en1` if empty). Give the Mac a fixed address
   (DHCP reservation) in your router, otherwise the workers lose it when the address changes. `.local` names often
   don't resolve inside WSL2, so use the IP.

## 1. GPU PC (Windows)

Both cards are on Windows. The IMX export needs Linux, so the agent runs inside **WSL2 Ubuntu**; WSL2 sees the NVIDIA
GPU through the normal Windows driver.

1. In PowerShell (as administrator): `wsl --install -d Ubuntu-24.04`, then reboot and create the Linux user.
2. Install a current NVIDIA driver **on Windows**. Do not install any NVIDIA driver inside Ubuntu.
   In Ubuntu, `nvidia-smi` must list your card. If it doesn't, nothing below will use the GPU.
3. In Ubuntu:
   ```
   sudo apt update && sudo apt install -y python3-venv curl openjdk-17-jre-headless
   java -version        # must say 17 or newer
   ```
4. Copy this `agents/` folder into Ubuntu, for example by pasting it in File Explorer at
   `\\wsl.localhost\Ubuntu-24.04\home\<your-user>\agents`. The IMX export is only set up for **Python 3.11**
   (Ubuntu 24.04 ships 3.12), so make a 3.11 environment with `uv`, which fetches that Python for you:
   ```
   cd ~/agents
   python3 -m venv ~/tools && ~/tools/bin/pip install uv
   ~/tools/bin/uv venv --python 3.11 ~/venv
   source ~/venv/bin/activate
   ~/tools/bin/uv pip install -r requirements-gpu.txt
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"     # expect True
   ```
5. **Test the export on its own first.** On first use it pip-installs the Sony conversion tools
   (model-compression-toolkit, imx500-converter) into this venv, so it needs internet and takes several minutes:
   ```
   python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='imx', data='coco8.yaml')"
   ls yolo11n_imx_model        # should contain packerOut.zip and labels.txt
   ```
   If this fails, stop and fix it (Python version, Java, missing packages) before running the agent.
6. Check the Mac is reachable (use the IP from step 0.3 and the token from step 0.1):
   ```
   export MT_SERVER=http://192.168.1.20:8000
   export MT_WORKER_TOKEN=<token>
   curl -s -H "X-Worker-Token: $MT_WORKER_TOKEN" $MT_SERVER/workers/ping     # {"ok":true}
   ```
   If it hangs, the Mac's firewall is blocking it, or WSL2's network can't reach your LAN. Adding
   `networkingMode=mirrored` under `[wsl2]` in `%UserProfile%\.wslconfig` (then `wsl --shutdown`) usually fixes the latter.
7. Start the agent (leave it running; `tmux` keeps it alive after you close the terminal):
   ```
   python gpu_agent.py --name gpu-3060 --device 0
   ```
   With the 3060 and the 4060 Ti in different PCs, run one agent in each with different `--name`s. With both in the
   same PC, run two agents with `--device 0` and `--device 1`.

## 2. Raspberry Pi

1. `sudo apt update && sudo apt full-upgrade && sudo apt install imx500-tools`
2. Copy two files from this folder to the Pi, from the Mac:
   `scp agents/common.py agents/pi_agent.py <user>@<pi-address>:~/agent/`
3. On the Pi (standard library only, nothing to pip install):
   ```
   cd ~/agent
   export MT_SERVER=http://192.168.1.20:8000
   export MT_WORKER_TOKEN=<token>
   python3 pi_agent.py --name pi-camera
   ```

## 3. Training

1. Review images and create a dataset version in the UI. It needs at least one validation image and some boxes.
2. In the **Training** section, pick the version and start a run. Progress, metrics and the stage show up live.
3. When the run is `done`, download `network.rpk` and `labels.txt` and follow Ultralytics' Sony IMX500 guide to run
   them on the Pi.

A run with *Export for IMX500* turned off stops after training and gives you `best.pt` only, with no Pi involved.

## Options

| | |
|---|---|
| `--once` | do one run (or none, if nothing is waiting) and exit |
| `--poll 5` | seconds between checks for new runs |
| `--heartbeat 20` | seconds between lease renewals |
| `--workdir work` | where datasets and weights live during a run |
| `--device auto` | (GPU agent) `0`, `1`, `cpu`; auto picks CUDA if there is one |

Stopping an agent with Ctrl-C doesn't tell the backend; the run is requeued once its 15-minute lease runs out.
Cancel it in the UI instead for an immediate stop.
