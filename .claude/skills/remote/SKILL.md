---
name: remote
description: Connect to the remote GPU server and start a persistent session for running commands
allowed-tools: Bash
argument-hint: "[command]"
---

# Remote GPU Server Session

Auto-connect to the remote Paperspace Gradient GPU server and execute commands.

## Setup (auto-connect)

Run this first — it handles everything (starts notebook if stopped, connects, starts session, runs startup.sh):

!`jlab setup 2>&1`

## If arguments were provided

Run this command on the remote: `jlab exec "$ARGUMENTS"`

## CRITICAL: Batch commands to minimize tool calls

**Always combine multiple commands into a single `jlab exec` call:**

```bash
# GOOD — one tool call:
jlab exec "ls /notebooks/PMamba" "cat README.md" "head -20 train.py"

# BAD — three tool calls:
jlab exec "ls /notebooks/PMamba"
jlab exec "cat README.md"
jlab exec "head -20 train.py"
```

Same for Python: `jlab run "import torch" "print(torch.cuda.is_available())"`

## Available commands

```bash
jlab setup                             # auto-connect (starts notebook if needed)
jlab start                             # start the notebook (Free-A6000)
jlab stop                              # stop the notebook
jlab exec "cmd1" "cmd2" "cmd3"         # batch shell commands (single connection)
jlab run "code1" "code2"               # batch Python code (single connection)
jlab session cd /notebooks/PMamba      # change working directory
jlab find "filename"                   # find files by name
jlab ls [path]                         # list remote files (fast, REST)
jlab cat path/to/file                  # view file contents (fast, REST)
jlab upload local remote               # upload file
jlab download remote [local]           # download file
jlab nb run notebook.ipynb             # run notebook
```

## Remote machine info

- GPU: NVIDIA RTX A6000 (49 GB VRAM), Free-A6000 tier
- Projects in `/notebooks/`: PMamba, REQNN, paper, research, viz-qcc
- Python 3.11, PyTorch 2.1.1+cu121, CUDA 12.4
- `jlab exec` and `jlab run` reuse the session kernel (fast, ~1s per command)
- `jlab ls` and `jlab cat` use REST API (always fast)

## Important

- Always use `jlab exec` for shell commands, NOT `jlab shell`
- **Batch commands** — put multiple commands in one `jlab exec` call to save time
- Session persists across calls — `session cd` changes cwd for all subsequent `exec` calls
- Use `jlab stop` when completely done to free the GPU
