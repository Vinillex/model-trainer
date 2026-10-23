"""What a training pod looks like: image, GPUs and start command. Pure functions; nothing here talks to RunPod."""

from pathlib import Path

# Every pod this project starts is named like this, so `scripts.runpod stop --all` can never touch anyone else's pods.
POD_PREFIX = "mt-"

# A small Ubuntu image from RunPod: the pod installs its own Python 3.11 and the torch build that fits its driver (see agents/runpod_start.sh), so
# a big preloaded PyTorch image would only slow the start. Tag is pinned; the launcher checks it still exists on Docker Hub.
IMAGE = "runpod/base:1.4.2-rc.175-ubuntu2204"

# Cheapest first; the pod gets the first one that has stock. Any of these has plenty of memory for a nano YOLO model.
DEFAULT_GPUS = ("NVIDIA RTX A4000", "NVIDIA RTX A4500", "NVIDIA RTX A5000", "NVIDIA RTX 4000 Ada Generation",
                "NVIDIA GeForce RTX 3090", "NVIDIA L4", "NVIDIA GeForce RTX 4090")

START_SCRIPT = Path(__file__).resolve().parents[2] / "agents" / "runpod_start.sh"


def start_script() -> str:
    return START_SCRIPT.read_text()


def build_pod_spec(*, name: str, server_url: str, worker_token: str, gpu_ids, cloud: str = "SECURE", max_hours: float = 3.0,
                   image: str = IMAGE, script: str | None = None, export_imx: bool = True) -> dict:
    """The body for POST /pods. `name` must start with POD_PREFIX."""
    if not name.startswith(POD_PREFIX):
        raise ValueError(f"pod names start with '{POD_PREFIX}'")
    return {
        "name": name,
        "imageName": image,
        "computeType": "GPU",
        "gpuCount": 1,
        "gpuTypeIds": list(gpu_ids),
        "gpuTypePriority": "custom",  # try them in the order given, cheapest first
        "cloudType": cloud,
        # No allowedCudaVersions: asking for CUDA 12.8+ hosts left RunPod with no instances at all (2026-10-22). The start script
        # installs the torch build that fits whatever driver the host has.
        "interruptible": False,  # a spot pod can be taken away in the middle of a run
        "minVCPUPerGPU": 8,  # the default of 2 leaves Ultralytics' image loading starved
        "minRAMPerGPU": 16,
        "containerDiskInGb": 40,
        "ports": [],  # nothing is exposed: the pod only makes outgoing requests
        "env": {
            "MT_SERVER": server_url,
            "MT_WORKER_TOKEN": worker_token,
            "MT_WORKER_NAME": "runpod-" + name,
            "MT_MAX_SECONDS": str(int(max_hours * 3600)),
            "MT_PATIENCE": "900",  # a rented GPU bills while the agent waits for a backend it can't reach: give up after 15 minutes
            "MT_PREP_IMX": "1" if export_imx else "0",  # install the IMX export tools during setup, only if the run will export
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",  # a plain image doesn't say it needs the GPU for compute
        },
        "dockerEntrypoint": ["bash", "-c"],
        "dockerStartCmd": [script if script is not None else start_script()],
    }
