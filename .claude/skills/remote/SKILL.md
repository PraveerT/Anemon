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
- Python 3.11, PyTorch 2.1.1+cu121, CUDA 12.4
- `jlab exec` and `jlab run` reuse the session kernel (fast, ~1s per command)

## PMamba project structure (session cwd)

```
/notebooks/PMamba/
  startup.sh                          # run by jlab setup
  requirements.txt
  dataset/
    nvidia_dataset_split.py           # data loading/splitting
    nvidia_process.py                 # preprocessing
    utils.py
  experiments/
    main.py                           # training entry point
    oracle_analysis.py                # oracle + fusion analysis
    fusion.yaml                       # fusion config
    quaternion.yaml                   # quaternion branch config
    pointlstm.yaml                    # PMamba config
    nvidia_dataloader.py              # dataloader
    nvidia_dataset_stats.npy          # precomputed stats
    models/
      motion.py                       # PMamba motion model
      reqnn_motion.py                 # quaternion motion model
      fusion.py                       # fusion model
      op.py                           # operators
    utils/
      optimizer.py, parameters.py, pts_transform.py, record.py, tools.py, ...
    work_dir/
      pmamba_branch/                  # trained PMamba (best: 89.83%)
      quaternion_branch/              # trained quaternion (best: 80.29%)
      fusion_branch/                  # trained fusion (best: 90.25%)
  external/REQNN/                     # reference REQNN implementation
  third_party/qecnetworks/           # quaternion equivariant networks
```

Other projects in `/notebooks/`: REQNN, paper, research, viz-qcc

## Important

- **Use `jlab exec` for everything** — file reading, shell commands, searching
- **Batch aggressively** — put as many commands as possible in one `jlab exec` call
- **Think ahead** — anticipate what info you'll need and fetch it all at once
- Session persists across calls — `session cd` changes cwd for all subsequent `exec` calls
- Use `jlab stop` when completely done to free the GPU

---

## Quaternion experiment naming convention

Every quaternion run is identified by a single compact string that fully encodes its lineage and recipe. **Use this format consistently** when describing runs, queueing experiments, or comparing results.

### Format

```
<preload>_<corr>_<aux>_<segments>_<pts>_<epochs>
```

For preloaded runs the parent's full description is nested in `[...]` recursively, e.g. `P[P[Sc_..._e140]E135_..._e140]E125_..._e140` (depth 2).

### Tokens

| Token | Meaning |
|---|---|
| `Sc` | from scratch (random init) |
| `P[<parent>]E<n>` | preloaded from `<parent>` at epoch `n` |
| `Nc` | `return_correspondence: false` (no correspondence data loaded) |
| `Co` | `return_correspondence: true` (correspondence data loaded) |
| `none` | `qcc_weight: 0` (no auxiliary loss) |
| `gc<w>` | grounded_cycle aux, weight (e.g. `gc01`=0.1, `gc02`=0.2, `gc002`=0.02) |
| `pr<w>` | prediction aux, weight |
| `co<w>` | contrastive aux, weight |
| `pr<w>_gc<w>` | stacked aux: prediction + grounded_cycle (any combination of variants stacks similarly) |
| `-D` suffix on aux | deep_mlp cycle module variant |
| `-XF` suffix on aux | transformer cycle module variant |
| `N<n>` | `num_cycle_segments` (only meaningful when grounded_cycle is in the aux stack) |
| `pts<a>-<b>` | dynamic pts_size schedule from `a` to `b` (e.g. `pts48-256`) |
| `pts<n>` | static pts_size = `n` (e.g. `pts256`) |
| `e<n>` | `num_epoch` |

### How to derive `pts<...>` from a config

```yaml
dynamic_pts_size: true
pts_size: 96            # nominal init; ramp goes 48 -> 128 -> 256
                        # -> token: pts48-256

# vs

dynamic_pts_size: false
pts_size: 256
                        # -> token: pts256
```

When you see `dynamic_pts_size: true` in a yaml, the canonical token is `pts48-256` regardless of `pts_size` field (which is just the initial value before the ramp kicks in).

### Examples

| Run | Compact name |
|---|---|
| nocorr_v4 (from scratch, no aux, 250 ep, dynamic pts) | `Sc_Nc_none_pts48-256_e250` |
| v6k (from scratch, gc01 N=3, 140 ep) | `Sc_Nc_gc01_N3_pts48-256_e140` |
| v6p (from scratch, prediction, corr, 140 ep) | `Sc_Co_pr01_N3_pts48-256_e140` |
| v6r (preload v6p ep135, same recipe) | `P[Sc_Co_pr01_N3_pts48-256_e140]E135_Co_pr01_N3_pts48-256_e140` |
| v6s (preload v6r ep125) | `P[P[Sc_Co_pr01_N3_pts48-256_e140]E135_Co_pr01_N3_pts48-256_e140]E125_Co_pr01_N3_pts48-256_e140` |
| v7a (stacked aux, N=3) | `Sc_Co_pr01_gc02_N3_pts48-256_e140` |
| v7b (stacked aux, N=24) | `Sc_Co_pr01_gc02_N24_pts48-256_e140` |
| v6L (preload corr_fixed_finetune, recipe switch) | `P[P[Sc_Nc_none_pts48-256_e250]E110_Co_pr01_N3_pts48-256_e140]E130_Nc_gc01_N3_pts48-256_e140` |

### What is a "family"?

> **A family is defined by its depth-0 model code.** Two runs belong to the same family if and only if their depth-0 root has the same compact name. The family identifier IS the depth-0 token string.

Examples:
- Family `Sc_Co_pr01_N3_pts48-256_e140` contains v6p, v6r, v6s — every run whose chain starts from the same v6p depth-0 root, regardless of what happens at depth 1 or 2.
- Family `Sc_Nc_none_pts48-256_e250` contains nocorr_v4 (depth 0), v6b (depth 1), corr_fixed_finetune (depth 1), v6L (depth 2 via corr_fixed) — all chains rooted in nocorr_v4.

When proposing a new experiment, **always state the family it belongs to** (i.e. the depth-0 root) and the chain depth, so the lineage is unambiguous.

---

## Naming-discipline checklist when starting any new quaternion run

1. **Decode the compact name** before launching: every token has to map cleanly to a config field.
2. **Look up `dynamic_pts_size` and `pts_size`** in the config and write the `pts<a>-<b>` or `pts<n>` token explicitly. Don't assume.
3. **State the family** (= depth-0 root) and the depth of the new run.
4. **Verify each token corresponds to actual code behavior** — e.g. `N<n>` only matters for grounded_cycle variants; the prediction loss iterates over frames and ignores N. Don't add N to a recipe where it has no effect.
5. **For stacked aux** (e.g. `pr01_gc02`), confirm the model file actually supports list-valued `qcc_variant` / `qcc_weight` (Change A in `reqnn_motion.py`).
