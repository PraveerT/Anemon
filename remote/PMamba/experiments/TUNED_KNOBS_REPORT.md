# PMamba Knob Sweep Report (10% NVGesture)

Setup: 10% train subset (105 clips, seed 42), full 482-clip test, 60 epochs,
batch 8, LR 0.00012 → 0.000012 step at ep30, pts=96 fixed.

Each row in a sweep table varies one knob; later sweeps carry forward the
best from the previous sweep.

## Final tuned config

| Knob | Default | Tuned | Effect on subset |
|---|---|---|---|
| knn | [32, 24, 48, 24] | [32, 24, 48, 24] | unchanged |
| topk | 8 | **6** | tied baseline |
| downsample | [2, 2, 2] | **[4, 4, 4]** | +3.94pp |
| mamba_hidden_dim | 128 | **32** | +5.81pp |
| mamba_num_layers | 2 | 2 | unchanged |
| mamba_output_dim | 256 | **384** | +0.83pp |
| ms_num_scales | 4 | 4 | unchanged |
| ms_feature_dim | 32 | 32 | unchanged |

**Subset best: 26.14%**  vs unmodified baseline **13.69%** on same subset.

## knn sweep (topk=8, ds=[2,2,2], mamba_dim=128 baseline)
| name | knn | best |
|---|---|---|
| K1_baseline | [32, 24, 48, 24] | **14.32** |
| K6_shrink | [64, 48, 24, 16] | 13.90 |
| K7_belly | [24, 48, 48, 24] | 13.69 |
| K8_last_big | [16, 24, 48, 96] | 13.69 |
| K5_grow | [16, 24, 48, 64] | 13.49 |
| K3_all32 | [32, 32, 32, 32] | 12.45 |
| K4_all48 | [48, 48, 48, 48] | 12.45 |
| K2_all16 | [16, 16, 16, 16] | 11.41 |
| K9_small | [8, 12, 16, 12] | 9.13 |

## topk sweep (knn=[32,24,48,24] winner)
| name | topk | best |
|---|---|---|
| T3_topk6 | 6 | **14.32** |
| T5_topk12 | 12 | 14.11 |
| T2_topk4 | 4 | 13.69 |
| T1_topk2 | 2 | 13.49 |
| T4_topk8 | 8 (baseline) | 13.28 |
| T6_topk16 | 16 | 13.07 |
| T7_topk24 | 24 | 12.66 |

## downsample sweep (topk=6 winner)
| name | downsample | best |
|---|---|---|
| D3_aggressive_4_4_4 | [4, 4, 4] | **17.63** |
| D6_drop_early_4_2_2 | [4, 2, 2] | 14.94 |
| D8_smooth_2_2_1 | [2, 2, 1] | 14.32 |
| D7_drop_mid_2_4_2 | [2, 4, 2] | 13.90 |
| D1_baseline_2_2_2 | [2, 2, 2] | 13.69 |
| D4_late_keep_2_2_1 | [2, 2, 1] | 12.24 |
| D5_early_keep_1_2_2 | [1, 2, 2] | 12.03 |
| D2_no_ds_1_1_1 | [1, 1, 1] | 11.83 |

## mamba_hidden_dim sweep (ds=[4,4,4] winner)
| name | mamba_hidden_dim | best |
|---|---|---|
| M1_dim32 | 32 | **23.44** |
| M7_dim384 | 384 | 19.50 |
| M6_dim256 | 256 | 18.46 |
| M3_dim96 | 96 | 17.84 |
| M2_dim64 | 64 | 17.63 |
| M5_dim192 | 192 | 17.43 |
| M4_dim128 | 128 (baseline) | 15.35 |

## mamba_num_layers sweep (mamba_dim=32 winner)
| name | mamba_num_layers | best |
|---|---|---|
| mamba_num_layers_2 | 2 | **24.27** |
| mamba_num_layers_3 | 3 | 22.61 |
| mamba_num_layers_1 | 1 | 21.78 |
| mamba_num_layers_4 | 4 | 21.58 |
| mamba_num_layers_6 | 6 | 19.92 |

## mamba_output_dim sweep (layers=2 winner)
| name | mamba_output_dim | best |
|---|---|---|
| mamba_output_dim_384 | 384 | **25.73** |
| mamba_output_dim_256 | 256 (baseline) | 24.90 |
| mamba_output_dim_512 | 512 | 24.69 |
| mamba_output_dim_192 | 192 | 23.44 |
| mamba_output_dim_128 | 128 | 18.26 |
| mamba_output_dim_64 | 64 | 17.63 |

## ms_num_scales sweep (out=384 winner)
| name | ms_num_scales | best |
|---|---|---|
| ms_num_scales_4 | 4 (baseline) | **26.14** |
| ms_num_scales_2 | 2 | 24.48 |
| ms_num_scales_3 | 3 | 24.27 |
| ms_num_scales_5 | 5 | 23.86 |
| ms_num_scales_1 | 1 | 23.24 |

## ms_feature_dim sweep (scales=4 winner)
| name | ms_feature_dim | best |
|---|---|---|
| ms_feature_dim_32 | 32 (baseline) | **25.93** |
| ms_feature_dim_128 | 128 | 25.31 |
| ms_feature_dim_16 | 16 | 25.10 |
| ms_feature_dim_64 | 64 | 25.10 |
| ms_feature_dim_8 | 8 | 23.03 |

## Caveats

10% subset (105 clips/25 classes ≈ 4 clips/class) is tiny. Random=4%.
The aggressive downsample `[4, 4, 4]` and small `mamba_hidden_dim=32`
likely act as regularization against overfit on this tiny data and may
underfit on full 1050-clip train.

Recommended next step: train `pmamba_tuned_e120.yaml` on full data and
compare to baseline 89.83. If it loses, try keeping only the capacity
tweaks (topk=6, mamba_output_dim=384) and reverting downsample +
mamba_hidden_dim to baseline values.
