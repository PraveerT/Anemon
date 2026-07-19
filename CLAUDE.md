# Anemon - jlab CLI

This repository contains only `jlab`, a Python CLI for interacting with remote
JupyterLab servers.

## Development

```bash
python -m pip install -e .
jlab --help
python -m compileall -q src/jlab
```

The package uses a `src` layout. CLI commands live in `src/jlab/cli.py`, REST
operations in `client.py`, kernel communication in `kernel.py`, terminal support
in `terminal.py`, and configuration plus Paperspace helpers in `config.py`.

## Command behavior

- Prefer REST-backed commands (`ls`, `cat`, `upload`, `write`, and `download`)
  for file operations.
- Kernel-backed commands (`exec`, `run`, `find`, `repl`, and `shell`) require a
  live Jupyter kernel.
- Use `jlab session start` when several kernel-backed commands need to share a
  working directory.
- `jlab setup`, `start`, and `stop` are Paperspace-specific helpers.
