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

**Always combine as many commands as possible into a single `jlab exec` call.
Think ahead — gather all the info you need in ONE call, not iteratively.**

```bash
# GOOD — one tool call for everything:
jlab exec "ls experiments/" "cat experiments/config.yaml" "grep -r 'accuracy' experiments/work_dir/*/log.txt | tail -20"

# BAD — three separate tool calls:
jlab exec "ls experiments/"
jlab exec "cat experiments/config.yaml"
jlab exec "grep -r 'accuracy' experiments/work_dir/*/log.txt | tail -20"
```

Same for Python: `jlab run "import torch" "print(torch.cuda.is_available())" "print(torch.__version__)"`

## Available commands

```bash
jlab setup                             # auto-connect (starts notebook if needed)
jlab start                             # start the notebook (Free-A6000)
jlab stop                              # stop the notebook
jlab exec "cmd1" "cmd2" "cmd3"         # batch shell commands (USE THIS FOR EVERYTHING)
jlab run "code1" "code2"               # batch Python code
jlab session cd /notebooks/PMamba      # change working directory
jlab find "filename"                   # find files by name
jlab upload local remote               # upload file
jlab download remote [local]           # download file
jlab nb run notebook.ipynb             # run notebook
```

## AVOID using jlab ls / jlab cat

`jlab ls` and `jlab cat` use REST API with paths relative to /notebooks/ root,
NOT the session cwd. They often fail. **Use `jlab exec` instead:**

```bash
# GOOD:
jlab exec "ls" "cat config.yaml"       # uses session cwd

# BAD (path confusion):
jlab cat config.yaml                   # looks for /notebooks/config.yaml, not /notebooks/PMamba/config.yaml
```

## Remote machine info

- GPU: NVIDIA RTX A6000 (49 GB VRAM), Free-A6000 tier
- Session cwd: /notebooks/PMamba (after setup)
- Projects in `/notebooks/`: PMamba, REQNN, paper, research, viz-qcc
- Python 3.11, PyTorch 2.1.1+cu121, CUDA 12.4
- `jlab exec` and `jlab run` reuse the session kernel (fast, ~1s per command)

## Important

- **Use `jlab exec` for everything** — file reading, shell commands, searching
- **Batch aggressively** — put as many commands as possible in one `jlab exec` call
- **Think ahead** — anticipate what info you'll need and fetch it all at once
- Session persists across calls — `session cd` changes cwd for all subsequent `exec` calls
- Use `jlab stop` when completely done to free the GPU
