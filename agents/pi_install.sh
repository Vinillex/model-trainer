#!/usr/bin/env bash
# Installs the Pi worker as a service that starts at boot and packages training runs by itself.
# Run on the Raspberry Pi, with pi_agent.py and common.py next to this script:
#   MT_SERVER=http://<the Mac>:8000 MT_WORKER_TOKEN=<token> bash pi_install.sh
# The Training page's "Copy Pi setup command" button gives a one-liner that does all of it. Safe to run again (it updates the address and token).
set -euo pipefail

: "${MT_SERVER:?MT_SERVER is not set}"
: "${MT_WORKER_TOKEN:?MT_WORKER_TOKEN is not set}"
dir="$(cd "$(dirname "$0")" && pwd)"
[ -f "$dir/pi_agent.py" ] && [ -f "$dir/common.py" ] && [ -f "$dir/pi_display.py" ] && [ -f "$dir/pi_viewer.py" ] || { echo "pi_agent.py, common.py, pi_display.py and pi_viewer.py must sit next to this script" >&2; exit 1; }
command -v imx500-package >/dev/null || { echo "imx500-package is missing: run  sudo apt install imx500-tools  and try again" >&2; exit 1; }

name="${MT_WORKER_NAME:-pi-camera}"
unit=mt-pi-agent

# the token lives in a file only this user can read, never in the unit file or on a command line
(umask 077 && printf 'MT_SERVER=%s\nMT_WORKER_TOKEN=%s\n' "$MT_SERVER" "$MT_WORKER_TOKEN" > "$dir/pi_agent.env")

sudo tee "/etc/systemd/system/$unit.service" > /dev/null <<EOF
[Unit]
Description=Model trainer Pi worker (packages .rpk files)
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$dir
EnvironmentFile=$dir/pi_agent.env
ExecStart=/usr/bin/python3 $dir/pi_agent.py --name $name --keep-trying --display
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$unit" > /dev/null 2>&1
sudo systemctl restart "$unit"  # also picks up a new address or token
sleep 2
if systemctl is-active --quiet "$unit"; then
  echo "Pi worker '$name' is running and will start at every boot."
else
  echo "The service did not start. Look at:  journalctl -u $unit -n 30" >&2
  exit 1
fi
