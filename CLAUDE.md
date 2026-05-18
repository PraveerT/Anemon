# Anemon - jlab CLI

## Remote Machine Access

This project includes `jlab`, a CLI for interacting with a remote JupyterLab GPU server (Paperspace Gradient).

## Session Workflow (IMPORTANT)

Always start a session first. This keeps a kernel alive so every command is fast (~1s instead of ~5s):

```bash
# Start session (do this FIRST, once per conversation)
jlab session start

# Change session working directory (persists across exec calls)
jlab session cd /notebooks/Anemon

# Check session / Stop when done
jlab session status
jlab session stop
```

## Batch Commands (IMPORTANT - reduces tool calls)

**Always batch multiple commands into a single `jlab exec` call:**

```bash
# GOOD — one tool call for 3 commands:
jlab exec "ls /notebooks/Anemon" "cat README.md" "wc -l *.py"

# BAD — 3 separate tool calls:
jlab exec "ls /notebooks/Anemon"
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
jlab exec --cwd /notebooks/Anemon "python train.py"

# Browse remote files (REST API, no kernel needed)
jlab ls [path]
jlab cat path/to/file

# Find files
jlab find "*.py" --path /notebooks/Anemon

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
