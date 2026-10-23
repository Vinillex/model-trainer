# Start command of a RunPod training pod. The launcher (backend/scripts/runpod.py) passes this whole file inline, as
# `bash -c`, so the clean-up below exists before anything that can fail (an install, a download) has run.
#
# From the launcher: MT_SERVER, MT_WORKER_TOKEN, MT_WORKER_NAME, MT_MAX_SECONDS (a hard limit on all of this) and
# MT_PREP_IMX (1 = the run exports for the IMX500, so install the export tools first).
# From RunPod itself: RUNPOD_POD_ID and RUNPOD_API_KEY, used only to delete this pod.
set -u

# Each line also goes to the backend (best effort, 5 s at most): the launcher on the Mac prints it, so a pod that stalls while
# installing says where, and the lines outlive the pod. Nothing here may stop the script.
say() {
  local line="[mt-pod $(date -u +%H:%M:%S)] $*"
  echo "$line"
  if [ -n "${MT_SERVER:-}" ] && command -v curl >/dev/null 2>&1; then
    curl -fsS -m 5 -X POST -H "X-Worker-Token: ${MT_WORKER_TOKEN:-}" -H "X-Worker-Name: ${MT_WORKER_NAME:-pod}" \
      -H "Content-Type: text/plain" --data-binary "$line" "$MT_SERVER/workers/log" >/dev/null 2>&1 || true
  fi
}

# Delete this pod through whichever route exists. RunPod bills a pod for as long as it exists, so this is the point of the
# script: it runs on every way out, success, failure or time limit.
delete_self() {
  local id="${RUNPOD_POD_ID:-}" key="${RUNPOD_API_KEY:-}" api="${MT_RUNPOD_REST:-https://rest.runpod.io/v1}"
  if [ -z "$id" ] || [ -z "$key" ]; then
    say "RUNPOD_POD_ID or RUNPOD_API_KEY is not set here"
    return 1
  fi
  if command -v runpodctl >/dev/null 2>&1 && runpodctl remove pod "$id"; then return 0; fi
  if command -v curl >/dev/null 2>&1 && curl -fsS -m 30 -X DELETE -H "Authorization: Bearer $key" "$api/pods/$id"; then return 0; fi
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import os, sys, urllib.request as u
u.urlopen(u.Request(sys.argv[2] + "/pods/" + sys.argv[1], method="DELETE",
                    headers={"Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]}), timeout=30)' "$id" "$api"; then return 0; fi
  return 1
}

# This pod's memory, from its cgroup (v2). A process killed for using too much memory leaves nothing else behind: the 13-minute
# export of run 5 ended with exit code 137 and no message, and this pod's logs are gone once it is deleted.
cg="${MT_CGROUP:-/sys/fs/cgroup}"
mem_report() {
  local now max
  now=$(cat "$cg/memory.current" 2>/dev/null) || return 0
  max=$(cat "$cg/memory.max" 2>/dev/null || echo max)
  [ "$max" = max ] && max="no limit" || max="$((max / 1048576)) MB"
  say "memory: $((now / 1048576)) MB used of $max"
}

end_pod() {
  local rc=$?
  trap - EXIT
  say "stopped with exit code $rc; deleting this pod"
  if [ "$rc" -eq 137 ]; then say "exit 137 means the process was killed (SIGKILL), most often by the memory limit"; fi
  mem_report
  [ -r "$cg/memory.events" ] && say "memory events: $(tr '\n' ' ' < "$cg/memory.events")"  # oom_kill N > 0 settles it
  if ! delete_self; then
    say "COULD NOT DELETE THIS POD. The launcher on the Mac deletes it when it notices; otherwise delete it in the RunPod console (Pods) or run: python -m scripts.runpod stop --all --yes"
  fi
  exit "$rc"
}
trap end_pod EXIT

# Which PyTorch wheel build fits this host. The "CUDA Version" printed by nvidia-smi is the newest CUDA the driver supports, and a
# driver runs any older CUDA build, so a CUDA 13 host takes cu128 too: that is the build every pod that worked used. (A pod that
# picked cu130 for its CUDA 13 driver sat in the install for 13 minutes and was given up on; cu130 was never seen working.) A
# cu126 build also runs on drivers for CUDA 12.0 to 12.5 (CUDA 12 minor-version compatibility, driver 525 or newer); below
# CUDA 12 torch 2.9 cannot run. Messages go to stderr because the answer is captured.
cuda_index() {
  local v major minor
  v=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
  if [ -z "$v" ]; then
    say "cannot read the driver's CUDA version from nvidia-smi" >&2
    return 1
  fi
  major=${v%%.*}
  minor=${v#*.}
  if [ "$major" -ge 13 ] || { [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; }; then echo cu128
  elif [ "$major" -eq 12 ]; then echo cu126
  else
    say "this host's driver only supports CUDA $v, and torch 2.9 needs CUDA 12 or newer" >&2
    return 1
  fi
}

mem_logger() {
  while :; do mem_report; sleep "${MT_MEM_EVERY:-120}" >/dev/null 2>&1; done  # the sleep must not keep the script's output open after the logger is killed
}

main() {
  set -euo pipefail
  local root="${MT_ROOT:-/opt}"  # only the tests change this
  export DEBIAN_FRONTEND=noninteractive
  say "installing system packages"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends curl ca-certificates tar openjdk-17-jre-headless libgl1 libglib2.0-0
  mkdir -p "$root/mt"
  cd "$root/mt"

  say "fetching the agent from the backend"
  curl -fsS -m 120 --retry 3 -H "X-Worker-Token: $MT_WORKER_TOKEN" "$MT_SERVER/workers/agent" | tar xz

  # Ultralytics' IMX export is only set up for Python 3.11 and torch >= 2.9, < 2.12 (see requirements-gpu.txt), so build that
  # environment rather than trusting what the image ships, with the torch build that matches this host's driver.
  local cuda py
  cuda=$(cuda_index)
  check_download_speed "$cuda"
  say "installing uv"
  curl -LsSf https://astral.sh/uv/0.8.0/install.sh | env UV_UNMANAGED_INSTALL="$root/uv" sh
  export PATH="$root/uv:$PATH"
  export UV_HTTP_TIMEOUT=60  # a read that stalls for a minute is an error to retry, not something to wait on
  py="$root/venv/bin/python"
  install_step "installing Python 3.11" uv venv --python 3.11 --seed "$root/venv"
  install_step "installing torch 2.9.1 ($cuda), about 3 GB" \
    uv pip install --python "$py" torch==2.9.1 torchvision==0.24.1 --index-url "https://download.pytorch.org/whl/$cuda"
  install_step "installing ultralytics" uv pip install --python "$py" -r requirements-gpu.txt

  if [ "${MT_PREP_IMX:-1}" = "1" ]; then
    # Ultralytics installs these itself the first time it exports, but into the program that is already running and has already
    # loaded matplotlib. model-compression-toolkit needs matplotlib < 3.10, so that swap left matplotlib half old and half new
    # and the export crashed (the first real run, 2026-10-22). Installing them here, before any program has loaded anything,
    # in one resolve with torch held where it is, avoids changing packages under a running program. The list is what
    # ultralytics' IMX export asks for (ultralytics/utils/export/imx.py), so requirements-gpu.txt pins that ultralytics version.
    install_step "installing the IMX500 export tools" \
      uv pip install --python "$py" torch==2.9.1 torchvision==0.24.1 "model-compression-toolkit>=2.4.1" "edge-mdt-cl<1.1.0" \
      "edge-mdt-tpc>=1.2.0" "pydantic<2.12" "imx500-converter[pt]>=3.17.3"
    # Fail here, minutes in, rather than after training: everything the export imports, Ultralytics' own check of the same list
    # (install=False: if it is unhappy it would otherwise pip-install in the middle of the export again), the converter binary,
    # and Java 17+ (without it Ultralytics apt-installs Java mid-export).
    "$py" -c "import matplotlib.figure, model_compression_toolkit, onnx, edgemdt_tpc
from ultralytics.utils.checks import check_requirements
assert check_requirements(('model-compression-toolkit>=2.4.1', 'edge-mdt-cl<1.1.0', 'edge-mdt-tpc>=1.2.0', 'pydantic<2.12', 'imx500-converter[pt]>=3.17.3'), install=False), 'Ultralytics does not accept the installed export tools'
print('IMX500 export tools import cleanly')"
    say "checking the export tools"
    test -x "$root/venv/bin/imxconv-pt" || { say "imxconv-pt is missing from the venv"; exit 1; }
    java --version | head -1 | grep -Eq '(openjdk|java) (1[7-9]|[2-9][0-9])' || { say "Java 17 or newer is missing"; exit 1; }
  fi

  say "checking the GPU"
  "$py" -c "import torch; assert torch.cuda.is_available(), 'torch cannot see a GPU'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

  say "starting the agent"
  mem_logger &
  logger=$!  # not local: the EXIT trap runs after this function has returned
  trap 'kill "$logger" 2>/dev/null || true' EXIT
  "$py" gpu_agent.py --once --name "$MT_WORKER_NAME" --device 0 --workdir "$root/mt/work"
}
# Is this host's route to PyTorch's download server any good? Fetch the first 40 MB of the wheel about to be installed and time it.
# The installs below move 3 GB from the same place; two pods in a row sat in them for 13+ minutes, and this finds out in under a
# minute, so a bad host is given up on (and a new pod tried) instead of billing while it crawls. Healthy pods did this step in
# 3-6 minutes, which is 10 MB/s or more; the bar is far below that.
check_download_speed() {
  local cuda=$1 url bytes secs mbps
  url="https://download.pytorch.org/whl/$cuda/torch-2.9.1%2B$cuda-cp311-cp311-manylinux_2_28_x86_64.whl"
  secs=$(date +%s)
  bytes=$(curl -fsS -m "${MT_SPEED_TEST_SECONDS:-45}" -r 0-41943039 -o /dev/null -w '%{size_download}' "$url" 2>/dev/null || true)
  bytes=${bytes:-0}  # curl prints what it got even when the time ran out
  secs=$(( $(date +%s) - secs )); [ "$secs" -lt 1 ] && secs=1
  mbps=$(( bytes / secs / 1048576 ))
  say "PyTorch download speed test: $((bytes / 1048576)) MB in ${secs} s (about ${mbps} MB/s)"
  if [ "$bytes" -lt 41943040 ] || [ "$mbps" -lt "${MT_MIN_MBPS:-2}" ]; then
    say "SLOW DOWNLOAD: this host reaches PyTorch's server too slowly to install (need 40 MB at 2 MB/s or better). Giving up so a new pod can be tried."
    return 1
  fi
}

# Run an install with a time limit, and once more if it ran out of time. A stalled download otherwise never ends.
install_step() {
  local what=$1; shift
  local limit="${MT_INSTALL_SECONDS:-480}" try
  for try in 1 2; do
    say "$what (try $try of 2, up to $((limit / 60)) min)"
    if timeout -k 10 "$limit" "$@"; then return 0; fi
    say "$what did not finish"
  done
  say "SLOW DOWNLOAD: $what failed twice. Giving up so a new pod can be tried."
  return 1
}
export -f say cuda_index check_download_speed install_step mem_report mem_logger main
export cg

# The hard limit: if anything hangs (a tunnel that went away, a stuck install), this still ends and the pod is deleted.
timeout -k 30 "${MT_MAX_SECONDS:-10800}" bash -c main
exit $?
