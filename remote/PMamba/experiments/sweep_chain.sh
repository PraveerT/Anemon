#!/bin/bash
# Run mamba_num_layers, mamba_output_dim, ms_num_scales, ms_feature_dim sweeps
# sequentially. Carries forward each winner via env vars.

set -e
cd /notebooks/PMamba/experiments

# Apply patches once
python patch_more_knobs.py

# Prior winners (set by earlier sweeps)
export TOPK_WINNER="${TOPK_WINNER:-6}"
export DS_WINNER="${DS_WINNER:-4,4,4}"
export MAMBA_DIM_WINNER="${MAMBA_DIM_WINNER:-32}"
export MAMBA_LAYERS_WINNER=2
export MAMBA_OUT_WINNER=256
export MS_SCALES_WINNER=4
export MS_FEATURE_WINNER=32

extract_winner() {
    # parses first line of summary csv (sorted by best desc) and prints val column
    awk -F',' 'NR==2 {print $2}' "$1"
}

echo "============================================="
echo "Sweep 1/4: mamba_num_layers (deps topk=$TOPK_WINNER, ds=$DS_WINNER, mamba_dim=$MAMBA_DIM_WINNER)"
echo "============================================="
python -u generic_sweep_10pct.py mamba_num_layers
export MAMBA_LAYERS_WINNER=$(extract_winner /tmp/mamba_num_layers_sweep_summary.txt)
echo "WINNER mamba_num_layers=$MAMBA_LAYERS_WINNER"

echo "============================================="
echo "Sweep 2/4: mamba_output_dim (carries layers=$MAMBA_LAYERS_WINNER)"
echo "============================================="
python -u generic_sweep_10pct.py mamba_output_dim
export MAMBA_OUT_WINNER=$(extract_winner /tmp/mamba_output_dim_sweep_summary.txt)
echo "WINNER mamba_output_dim=$MAMBA_OUT_WINNER"

echo "============================================="
echo "Sweep 3/4: ms_num_scales (carries out=$MAMBA_OUT_WINNER)"
echo "============================================="
python -u generic_sweep_10pct.py ms_num_scales
export MS_SCALES_WINNER=$(extract_winner /tmp/ms_num_scales_sweep_summary.txt)
echo "WINNER ms_num_scales=$MS_SCALES_WINNER"

echo "============================================="
echo "Sweep 4/4: ms_feature_dim (carries scales=$MS_SCALES_WINNER)"
echo "============================================="
python -u generic_sweep_10pct.py ms_feature_dim
export MS_FEATURE_WINNER=$(extract_winner /tmp/ms_feature_dim_sweep_summary.txt)
echo "WINNER ms_feature_dim=$MS_FEATURE_WINNER"

echo "============================================="
echo "Final winners:"
echo "  topk=$TOPK_WINNER"
echo "  downsample=$DS_WINNER"
echo "  mamba_hidden_dim=$MAMBA_DIM_WINNER"
echo "  mamba_num_layers=$MAMBA_LAYERS_WINNER"
echo "  mamba_output_dim=$MAMBA_OUT_WINNER"
echo "  ms_num_scales=$MS_SCALES_WINNER"
echo "  ms_feature_dim=$MS_FEATURE_WINNER"
echo "============================================="
