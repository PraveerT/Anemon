# Anemon JupyterLab CLI

Anemon's existing `jlab` commands continue to work unchanged. The `qwen` group adds a durable terminal chat whose model and inference process run on the remote Paperspace GPU while prompts and replies travel through the existing authenticated JupyterLab kernel connection.

## Qwen remote chat

The configured default is `cyankiwi/Qwen3.8-27B-AWQ-INT4`, an open 4-bit
quantization of the official `Qwen/Qwen3.8-27B`. The requested official model
does exist, but its 55.6 GB BF16 snapshot is larger than the RTX A6000's 48 GB
VRAM, and runtime quantization is not stable within this notebook's 44 GB system
RAM. The configured AWQ snapshot is about 21 GB and runs the same 27B-class
Qwen3.8 architecture with FP16 compute. The model ID remains explicit and
overridable.

The live A6000 host exposes a CUDA 12.4-capable NVIDIA 550.144.03 driver. vLLM 0.28 currently requires a Torch/CUDA 13 stack that is not compatible with that driver, and FP8 is not a suitable A6000 path. The remote service instead uses an isolated persistent virtual environment with PyTorch 2.6.0 CUDA 12.4 wheels, Transformers 5.15.x, Accelerate, bitsandbytes, and compressed-tensors. Nothing is installed into the remote global Python environment. The inference HTTP listener binds only to `127.0.0.1` on the remote host; it is never exposed publicly.

Provision once:

```text
jlab qwen setup
```

The first run downloads roughly 21 GB and installs a separate Torch environment, so it can take several minutes or longer depending on the notebook and network. Setup is idempotent: completed dependencies and model snapshots are reused. To select another compatible repository or change context length:

```text
jlab qwen setup --model cyankiwi/Qwen3.8-27B-AWQ-INT4 --max-model-len 16384
```

Send one prompt or start an interactive loop:

```text
jlab qwen chat "Explain this algorithm"
jlab qwen chat --chat research
jlab qwen chat --chat research --temperature 0.3 --top-p 0.9 --max-tokens 1500
```

Terminal chat uses Qwen's documented non-thinking template so replies contain the
answer rather than exposing a reasoning preamble. Conversation context is still
preserved through the durable transcript.

`--chat` names a durable conversation. Every request carries the complete stored transcript, and a turn is saved only after a full assistant response succeeds. This lets a conversation continue after the bridge kernel or the approximately six-hour Paperspace notebook lease ends.

On the first prompt after a shutdown, Anemon checks the saved Jupyter endpoint, uses the saved Paperspace credential to reconnect to a running notebook or start the stopped notebook, waits for JupyterLab, creates a fresh Qwen-specific bridge kernel, and lazily restarts the local-only inference service. It does not stop or restart an already-running Paperspace notebook. The first reply after a cold start can be slow while the 27B model loads.

Inspect configuration/service state and read a transcript without starting inference:

```text
jlab qwen status
jlab qwen history --chat research
jlab qwen history --chat research --json
```

Local Qwen configuration, its dedicated kernel record, and authoritative transcripts live under `~/.jlab/qwen`. This is separate from the normal `~/.jlab/session.json` used by `jlab session`. The persistent remote environment, model/cache, service log, PID, and transcript mirror live under `/notebooks/.anemon-qwen`.
