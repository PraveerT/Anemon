# Anemon - jlab CLI

## Remote Machine Access

This project includes `jlab`, a CLI for interacting with a remote JupyterLab GPU server (Paperspace Gradient).

## ALWAYS use the REST API — the kernel is unreliable (IMPORTANT)

The Paperspace container is flaky: it 503-bounces, hangs, and dies mid-run. Default to REST and never block on the kernel:

- **Reads / status / file transfer → REST only:** `jlab ls`, `jlab cat`, `jlab download`, `jlab upload`. These hit the Jupyter REST API and work even when the kernel is dead.
- **Never block on `jlab exec` for output.** Launch every long job DETACHED and poll the result file over REST:
  ```bash
  jlab exec --cwd <dir> "nohup python -u main.py --config X.yaml > run.log 2>&1 & echo LAUNCHED \$!"
  # then, kernel-independent:
  jlab cat <work_dir>/log.txt
  ```
  Read the clean recoder log (`work_dir/.../log.txt`) — NOT the raw stdout `.log`, whose tqdm bars contain Unicode (`▏`) that crashes the Windows console codec.
- **On 503 / 500 / Service Unavailable / hang → run `jlab setup` immediately** (re-provisions the container). A 503 is usually a container bounce that kills nohup jobs: after recovery, check `ps`/checkpoints and relaunch any killed training. The box comes back on a NEW URL; the persistent `/notebooks` filesystem (code, logs, checkpoints) survives, running processes do not.
- The fetcher (sidepanel publisher) dies on every bounce — restart it after recovery.

## Session Workflow (IMPORTANT)

Always start a session first. This keeps a kernel alive so every command is fast (~1s instead of ~5s):

```bash
# Start session (do this FIRST, once per conversation)
jlab session start

# Change session working directory (persists across exec calls)
jlab session cd /notebooks/Manta

# Check session / Stop when done
jlab session status
jlab session stop
```

## Batch Commands (IMPORTANT - reduces tool calls)

**Always batch multiple commands into a single `jlab exec` call:**

```bash
# GOOD — one tool call for 3 commands:
jlab exec "ls /notebooks/Manta" "cat README.md" "wc -l *.py"

# BAD — 3 separate tool calls:
jlab exec "ls /notebooks/Manta"
jlab exec "cat README.md"
jlab exec "wc -l *.py"
```

Same for `jlab run` (Python code):
```bash
jlab run "import torch" "print(torch.cuda.is_available())" "print(torch.__version__)"
```

## Commands

```bash
# Shell commands (batch multiple for efficiency)
jlab exec "cmd1" "cmd2" "cmd3"
jlab exec --cwd /notebooks/Manta "python train.py"

# Browse remote files (REST API, no kernel needed)
jlab ls [path]
jlab cat path/to/file

# Find files
jlab find "*.py" --path /notebooks/Manta

# Transfer files
jlab download Anemon/model.py
jlab upload model.py Anemon/model.py

# Run Python code (batch multiple)
jlab run "code1" "code2" "code3"

# Run a notebook
jlab nb run Anemon/experiment.ipynb

# Server info
jlab status
jlab kernels
```

## Remote Machine

- GPU server on Paperspace Gradient
- Projects in `/notebooks/`: Anemon, REQNN, paper, research, viz-qcc
- Python 3.11, PyTorch 2.1.1+cu121, NumPy, CUDA
- Config: `~/.jlab/config.json`, Session: `~/.jlab/session.json`

## Notes

- `jlab shell` is interactive-only (needs TTY) — use `jlab exec` instead
- `jlab ls` and `jlab cat` use REST API (always fast, no kernel)
- `jlab exec`, `jlab run`, `jlab find` use kernel (fast with active session)
- Always `jlab session start` at the beginning of work
