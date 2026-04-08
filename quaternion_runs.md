# Quaternion experiment runs — naming + full table

## Naming convention

```
Format:  <preload>_<corr>_<aux>_<segments>_<epochs>
```

For nested preloads, the parent's full description goes inside `[...]` recursively.

| Token | Meaning |
|---|---|
| `Sc` | from scratch (random init) |
| `P[<parent>]E<n>` | preloaded from `<parent>` at epoch `n` |
| `Nc` | `return_correspondence: false` (no correspondence data loaded) |
| `Co` | `return_correspondence: true` (correspondence data loaded) |
| `none` | `qcc_weight: 0` (no auxiliary loss) |
| `gc<w>` | grounded_cycle aux, qcc_weight (e.g. `gc01` = 0.1, `gc02` = 0.2, `gc002` = 0.02) |
| `pr<w>` | prediction aux, qcc_weight |
| `co<w>` | contrastive aux, qcc_weight |
| `-D` | suffix on aux: deep_mlp cycle module variant |
| `-XF` | suffix on aux: transformer cycle module variant |
| `N<n>` | num_cycle_segments (only meaningful for grounded_cycle) |
| `e<n>` | num_epoch |

### Preload-source short codes (used inside `P[...]`)

`?` indicates a preload source whose original is unknown (e.g. circular self-reference in the work_dir's saved config).

For depth ≥ 1, the parent is encoded as its own full compact name. The deepest experiments end up looking like:

```
P[P[Sc_Nc_none_e250]E110_Co_pr01_N3_e140]E130_Nc_gc01_N3_e140
```

which decodes as:
- v6L,
- preloaded at epoch 130 from corr_fixed_finetune,
- which itself was preloaded at epoch 110 from nocorr_v4,
- which was trained from scratch.

---

## Depth 0 — from scratch

| Compact name | Original work_dir | Peak |
|---|---|---|
| `Sc_Nc_none_e250` | quaternion_nocorr_v4 | **77.59%** |
| `Sc_Nc_gc01_N3_e140` | quaternion_fair_baseline_v6k | 76.14% |
| `Sc_Nc_gc02_N3_e140` | quaternion_qcc_strong_v6g | 76.35% |
| `Sc_Nc_gc01-D_N3_e140` | quaternion_cycle_deep_v6h3 | 75.31% |
| `Sc_Nc_gc01-D_N6_e140` | quaternion_cycle_n6_v6i | 76.35% |
| `Sc_Nc_gc01-D_N9_e140` | quaternion_cycle_n9_v6j | 76.76% |
| `Sc_Nc_gc01-XF_N3_e140` | quaternion_cycle_xfmr_v6h | 40.04% (collapsed at ep2) |
| `Sc_Nc_gc002-XF_N3_e140` | quaternion_cycle_xfmr_v6h2 | 34.44% (collapsed) |
| `Sc_Nc_none-XF_N3_e30` | quaternion_cycle_xfmr_zero_v6h0 | 55.39% (sanity, 30 ep) |
| `Sc_Co_pr01_N3_e140` | quaternion_prediction_scratch_v6p | 76.14% |
| `Sc_Co_none_e140` | quaternion_prediction_off_scratch_v6q | 75.93% |

## Depth 1 — single preload

| Compact name | Original work_dir | Peak |
|---|---|---|
| `P[Sc_Nc_none_e250]E110_Nc_none_e140` | quaternion_rigidity_control_v6b | 78.01% |
| `P[Sc_Nc_none_e250]E110_Nc_none_e140 +noRig` | quaternion_no_rigidity_v6a | 78.22% |
| `P[Sc_Nc_none_e250]E110_Co_pr01_N3_e140` | quaternion_corr_fixed_finetune | **79.88%** |
| `P[Sc_Nc_none_e250]E110_Co_pr01_N3_e140 +decouple` | quaternion_decoupled_v6 | 73.44% (broken decouple_sampling) |
| `P[Sc_Nc_none_e250]E110_Nc_gc01-D_N12_e140` | quaternion_cycle_n12_pretrained_v6n | 78.01% (incomplete, killed ~ep131) |
| `P[?]E110_Nc_gc01_N3_e140` | quaternion_branch | **80.29%** |

> `quaternion_branch`'s parent is unknown — its config self-references its own work_dir's `epoch110_model.pt`, a circular reference. The original source is lost.

## Depth 2 — preload of a preload

| Compact name | Original work_dir | Peak |
|---|---|---|
| `P[P[?]E110_Nc_gc01_N3_e140]E112_Nc_none_e40` | quaternion_finetune_pure_v6d | 79.05% |
| `P[P[?]E110_Nc_gc01_N3_e140]E112_Nc_gc01_N3_e40` | quaternion_grounded_finetune_v6f | 79.25% |
| `P[P[?]E110_Nc_gc01_N3_e140]E112_Co_co03_N3_e60` | quaternion_finetune_best_v6c | 79.46% |
| `P[P[Sc_Nc_none_e250]E110_Co_pr01_N3_e140]E130_Nc_gc01_N3_e140` | quaternion_second_best_v6L | **81.33%** |
| `P[Sc_Co_pr01_N3_e140]E135_Co_pr01_N3_e140` | quaternion_v6p_warmrestart_v6r | running (peak so far 78.42%) |

---

## Lineage trees

### Chain A — corr_fixed → v6L (where the wins are)
```
Sc_Nc_none_e250                                                        77.59%
   └─E110─→ P[Sc_Nc_none_e250]E110_Co_pr01_N3_e140                     79.88%
              └─E130─→ P[P[Sc_Nc_none_e250]E110_..]E130_Nc_gc01_N3_e140  81.33%
```

### Chain B — quaternion_branch (unknown root)
```
?                                                                          ?
   └─E110─→ P[?]E110_Nc_gc01_N3_e140                                    80.29%
              └─E112─→ {v6c: 79.46%, v6d: 79.05%, v6f: 79.25%}        ALL DROPPED
```

### Chain C — v6r (clean test, in progress)
```
Sc_Co_pr01_N3_e140                                                     76.14%
   └─E135─→ P[Sc_Co_pr01_N3_e140]E135_Co_pr01_N3_e140             ≥ 78.42% so far
```

---

## Apples-to-apples groupings

**A. From-scratch, 140-ep cluster (~75-77%)**
- All variants of aux loss + segment count + cycle module type land in the 75.3 — 76.8 % band, within seed-noise of each other
- Only nocorr_v4 (250 ep) clearly exceeds this band at 77.59%

**B. Preloaded from `Sc_Nc_none_e250` (nocorr_v4) at ep110 (~66% mid-training)**
- Cluster 78.0 — 79.9 %; corr_fixed_finetune is the high water mark here

**C. Preloaded from `P[?]E110_Nc_gc01_N3_e140` (quaternion_branch) at ep112 (peak 80.29%)**
- All 3 finetunes (v6c/v6d/v6f) **dropped** ~1% from the source. Restarting from a peak hurts.

**D. Preloaded from `P[Sc_Nc_none_e250]E110_Co_pr01_N3_e140` (corr_fixed_finetune) at ep130 (peak 79.88%)**
- v6L = 81.33% (single-epoch peak; saved checkpoints near it 79-80%)
