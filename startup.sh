#!/usr/bin/env bash
set -euo pipefail

# Install CLI tools used in earlier repo setup.
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex

# Optional Anthropic-compatible endpoint configuration.
# Export ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN before running if needed.

# Install basic dependencies.
pip install numpy torch torchvision

# Mamba2 stack: causal-conv1d must be installed BEFORE mamba-ssm 2.x.
# Both shipped as prebuilt wheels (cu122, torch2.1, cp311, abiFALSE).
pip install --upgrade \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

pip install --upgrade \
  https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.1cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Install remaining Python requirements.
pip install -r requirements.txt

# IS_SANDBOX=1 claude --dangerously-skip-permissions
# Export Jupyter token for jlab CLI auto-connect
echo "$JUPYTER_TOKEN" > /notebooks/.jlab-token
echo "jlab: token saved to /notebooks/.jlab-token"

# Sidepanel publisher: auto-restart watchdog. Container restarts kill all
# processes; running jlab setup again re-installs this watchdog cleanly.
SIDEPANEL_DIR="/notebooks/Anemon/sidepanel_api"
if [ -d "$SIDEPANEL_DIR" ]; then
  pkill -f publisher_watchdog.sh 2>/dev/null || true
  pkill -f 'publisher.py' 2>/dev/null || true
  cat > "$SIDEPANEL_DIR/publisher_watchdog.sh" <<'WATCHDOG_EOF'
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
WATCHDOG_EOF
  chmod +x "$SIDEPANEL_DIR/publisher_watchdog.sh"
  mkdir -p "$SIDEPANEL_DIR/state"
  nohup bash "$SIDEPANEL_DIR/publisher_watchdog.sh" \
    > "$SIDEPANEL_DIR/state/watchdog.log" 2>&1 &
  disown
  echo "sidepanel publisher watchdog started (pid $!)"
fi
