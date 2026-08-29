#!/usr/bin/env bash
# One-time installer: makes start.sh (and everything it launches — PaddleOCR,
# pipeline-service, Next.js, watchdog.sh) survive an Oracle VM reboot, not
# just an SSH/laptop disconnect.
#
# start.sh already backgrounds its 3 services with `nohup`, and watchdog.sh
# (also launched by start.sh) already restarts any of them if they crash —
# so closing your laptop or losing your SSH connection was already safe.
# What was NOT covered: the VM itself rebooting (maintenance, an OOM event,
# `sudo reboot`) — nothing was set up to run start.sh again automatically
# when the machine comes back up. This installs a systemd unit that does
# exactly that, on top of the existing start.sh + watchdog.sh setup (no
# duplicate supervision — systemd just makes sure start.sh runs once at
# boot and again if the whole stack somehow exits; watchdog.sh still does
# the fine-grained per-service crash restarts).
#
# Usage: sudo bash install-systemd.sh
# Uninstall: sudo systemctl disable --now manhwa-recap-studio
#            sudo rm /etc/systemd/system/manhwa-recap-studio.service

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this with sudo: sudo bash install-systemd.sh" >&2
    exit 1
fi

# The user who should own the running processes — whoever invoked sudo,
# falling back to the owner of this checkout if run as raw root.
RUN_USER="${SUDO_USER:-$(stat -c '%U' "$(dirname "$(readlink -f "$0")")")}"
PROJECT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

if [ ! -f "$PROJECT_DIR/start.sh" ]; then
    echo "start.sh not found next to this script — run it from the project root." >&2
    exit 1
fi

UNIT_FILE="/etc/systemd/system/manhwa-recap-studio.service"

echo "Installing systemd unit for user '$RUN_USER', project dir '$PROJECT_DIR'..."

cat > "$UNIT_FILE" << EOF
[Unit]
Description=Manhwa Recap Studio (PaddleOCR + pipeline-service + Next.js, via start.sh)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$PROJECT_DIR
# start.sh launches all 3 services + watchdog.sh in the background and then
# exits; the "exec tail -f /dev/null" keeps THIS unit's main process alive
# so systemd has something to supervise and restart on failure. If start.sh
# itself fails (e.g. a dependency install error), the && short-circuits and
# this process exits non-zero, which Restart=on-failure below picks up.
ExecStart=/bin/bash -c 'bash start.sh && exec tail -f /dev/null'
ExecStop=/bin/bash -c 'pkill -f "watchdog.sh" || true; fuser -k 3000/tcp 2>/dev/null || true; fuser -k 3002/tcp 2>/dev/null || true; pkill -f "pipeline-service" || true; pkill -f "index.ts" || true; pkill -f "uvicorn main:app" || true'
Restart=on-failure
RestartSec=15
# Give start.sh room: it installs system packages / downloads models on a
# cold first run, which can take a few minutes.
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable manhwa-recap-studio
systemctl restart manhwa-recap-studio

echo ""
echo "✅ Installed and started. It will now also start automatically on every VM reboot."
echo ""
echo "  Status:       sudo systemctl status manhwa-recap-studio"
echo "  Logs:         sudo journalctl -u manhwa-recap-studio -f"
echo "  Service logs: tail -f $PROJECT_DIR/logs/pipeline.log $PROJECT_DIR/logs/paddleocr.log $PROJECT_DIR/logs/nextjs.log $PROJECT_DIR/logs/watchdog.log"
echo "  Restart:      sudo systemctl restart manhwa-recap-studio"
echo "  Stop:         sudo systemctl stop manhwa-recap-studio"
echo ""
echo "  Do NOT also run 'bash start.sh' by hand anymore — systemd owns it now."
echo "  To change code and pick it up: git pull, then sudo systemctl restart manhwa-recap-studio"
echo "  (start.sh's own stale-build check will rebuild Next.js automatically when needed)."
