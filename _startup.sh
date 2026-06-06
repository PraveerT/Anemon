#!/usr/bin/env bash
# Lean recovery startup (POSIX-safe; NO 'set -e/-o pipefail' so it always runs to
# completion). On every `jlab setup` after a container restart it:
#   1) re-saves the jlab token
#   2) restarts the sidepanel publisher watchdog
#   3) auto-resumes the active training run from its latest checkpoint
# Heavy pip installs dropped: torch/numpy are in the base image (resume works without them).

echo "${JUPYTER_TOKEN:-}" > /notebooks/.jlab-token 2>/dev/null || true
echo "jlab: token saved"

# --- sidepanel publisher watchdog ---
SP=/notebooks/Manta/sidepanel_api
if [ -d "$SP" ]; then
  pkill -f publisher.py 2>/dev/null || true
  mkdir -p "$SP/state"
  # setsid -> own session, survives this script's exit and any child cleanup.
  setsid bash -c 'cd /notebooks/Manta/sidepanel_api; while true; do python3 -u publisher.py --interval 30 >> state/publisher.log 2>&1; sleep 5; done' < /dev/null > "$SP/state/watchdog.log" 2>&1 &
  disown 2>/dev/null || true
  echo "[startup] publisher watchdog restarted (setsid)"
fi

# --- auto-resume the active training run ---
EXP=/notebooks/Manta/experiments
if [ -f "$EXP/.active_run" ]; then
  RUN=$(cat "$EXP/.active_run")
  if pgrep -f 'main.py --config' >/dev/null 2>&1; then
    echo "[startup] training already running; no resume"
  elif [ "$RUN" = "cyclezoo" ]; then
    ( cd "$EXP" && DENSE=0 GPU_RESIDENT=1 DETERMINISTIC=0 nohup bash run_cyclezoo.sh > work_dir/cyclezoo_driver.out 2>&1 & )
    echo "[startup] auto-resumed cyclezoo chain"
  elif [ "$RUN" = "logitnorm" ]; then
    ( cd "$EXP" && DENSE=0 GPU_RESIDENT=1 DETERMINISTIC=0 nohup bash run_logitnorm.sh > work_dir/logitnorm_driver.out 2>&1 & )
    echo "[startup] auto-resumed logitnorm sweep"
  elif [ "$RUN" = "skewscratch" ]; then
    ( cd "$EXP" && DENSE=0 GPU_RESIDENT=1 nohup bash run_skewscratch.sh > work_dir/skewscratch_driver.out 2>&1 & )
    echo "[startup] auto-resumed skew-scratch seed sweep (deterministic)"
  elif [ "$RUN" = "pgcnet_bdnq" ] || [ "$RUN" = "pgcnet_bdnq_vec" ] || [ "$RUN" = "pgcnet_skewlean" ] || [ "$RUN" = "pgcnet_gridt2s" ]; then
    ( cd "$EXP" && python _resume_run.py "$RUN"; DENSE=0 GPU_RESIDENT=1 DETERMINISTIC=0 nohup python main.py --config "${RUN}_resume.yaml" > "work_dir/${RUN}.out" 2>&1 & )
    echo "[startup] auto-resumed bdnq ($RUN, fast)"
  elif [ "$RUN" = "seeds_p32" ]; then
    ( cd "$EXP" && DENSE=0 GPU_RESIDENT=1 nohup bash run_seeds_p32.sh > work_dir/seeds_p32.out 2>&1 & )
    echo "[startup] auto-resumed seed sweep: $RUN"
  elif [ "$RUN" = "skew9191" ]; then
    ( cd /notebooks/wt9191/experiments && nohup bash run_skew9191.sh > work_dir/skew9191.out 2>&1 & )
    echo "[startup] auto-resumed skew9191 worktree run"
  elif [ "$RUN" = "quat9191" ]; then
    ( cd /notebooks/wt9191/experiments && python _resume_run.py cn_xxl_quat_s2; nohup python -u main.py --config cn_xxl_quat_s2_resume.yaml > work_dir/cn_xxl_quat_s2.out 2>&1 & )
    echo "[startup] auto-resumed quat9191 worktree run s2 (non-det)"
  elif [ "$RUN" = "quatzero" ]; then
    ( cd /notebooks/wt9191/experiments && nohup bash run_quatzero.sh > work_dir/quatzero_driver.out 2>&1 & )
    echo "[startup] auto-resumed quatzero control (non-det)"
  elif [ "$RUN" = "noaux" ]; then
    ( cd /notebooks/wt9191/experiments && nohup bash run_noaux.sh > work_dir/noaux_driver.out 2>&1 & )
    echo "[startup] auto-resumed noaux baseline (non-det)"
  elif [ "$RUN" = "lean_seeds" ]; then
    ( cd "$EXP" && DENSE=0 GPU_RESIDENT=1 nohup bash run_lean_seeds.sh > work_dir/lean_seeds.out 2>&1 & )
    echo "[startup] auto-resumed lean seed sweep"
  else
    ( cd "$EXP" && python _resume_run.py "$RUN" && DENSE=0 GPU_RESIDENT=1 nohup python main.py --config "${RUN}_resume.yaml" > "work_dir/${RUN}.out" 2>&1 & )
    echo "[startup] auto-resumed run: $RUN"
  fi
fi
echo "[startup] done"
