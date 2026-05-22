#!/usr/bin/env bash
# Re-launch the publisher whenever it exits. The publisher itself catches
# transient network errors; this loop covers hard kills (container 503,
# kernel restart, OOM).
cd /notebooks/Anemon/sidepanel_api
while true; do
  python3 -u publisher.py --interval 30 >> state/publisher.log 2>&1
  echo "[watchdog] $(date '+%F %T') publisher exited, restarting in 5s..." >> state/publisher.log
  sleep 5
done
