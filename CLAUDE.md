# Anemon - jlab CLI

## Remote Machine Access

This project includes `jlab`, a CLI for interacting with a remote JupyterLab GPU server (Paperspace Gradient).

## Session Workflow (IMPORTANT)

Always start a session first. This keeps a kernel alive so every command is fast (~1s instead of ~5s):

```bash
# Start session (do this FIRST, once per conversation)
jlab session start

# Now all commands reuse the same kernel — fast!
jlab exec "ls /notebooks"
jlab exec "nvidia-smi"
jlab exec "cd /notebooks/PMamba && python train.py"

# Change session working directory (persists across exec calls)
jlab session cd /notebooks/PMamba
jlab exec "ls"          # runs in /notebooks/PMamba
jlab exec "python train.py"  # still in /notebooks/PMamba

# Check session
jlab session status

# Stop when done
jlab session stop
```

## Commands

```bash
# Shell commands on remote
jlab exec "command here"
jlab exec --cwd /notebooks/PMamba "python train.py"

# Browse remote files (uses REST API, no kernel needed)
jlab ls                          # list root
jlab ls PMamba                   # list subdirectory
jlab cat PMamba/train.py         # view file with syntax highlighting

# Find files
jlab find "*.py"                 # find Python files
jlab find "motion.py"            # find specific file
jlab find "*.ipynb" --path /notebooks/PMamba

# Transfer files
jlab download PMamba/model.py    # download to local
jlab upload model.py PMamba/model.py  # upload to remote

# Run Python code directly on remote kernel
jlab run "import torch; print(torch.cuda.is_available())"

# Run a notebook
jlab nb run PMamba/experiment.ipynb

# Server info
jlab status
jlab kernels
```

## Remote Machine

- GPU server on Paperspace Gradient
- Projects in `/notebooks/`: PMamba, REQNN, paper, research, viz-qcc
- Python 3.11, PyTorch 2.1.1+cu121, NumPy, CUDA
- Config: `~/.jlab/config.json`, Session: `~/.jlab/session.json`

## Notes

- `jlab shell` is interactive-only (needs TTY) — use `jlab exec` instead
- `jlab ls` and `jlab cat` use REST API (always fast, no kernel)
- `jlab exec`, `jlab run`, `jlab find` use kernel (fast with active session)
- Always `jlab session start` at the beginning of work
