#!/bin/bash

# === Paths ===
SAMPLESHEET=/mnt/lustre/groups/nahnsen/nahpo775/workdir/projects/tcrpha_pred/t2pmhc/data/t2pmhc_train.tsv
GRAPHS=/mnt/lustre/groups/nahnsen/nahpo775/workdir/projects/tcrpha_pred/t2pmhc/data/graphs/gat_train_graphs.pt


# === Step 1: Create graphs (run once before the sweep) ===
# Graphs are built with a 10 Å contact-map threshold (hardcoded in t2pmhc).
# Uncomment and run this block once; comment it out again before launching the sweep.
#
# t2pmhc create-t2pmhc-graphs \
#     --mode t2pmhc-gat \
#     --samplesheet $SAMPLESHEET \
#     --training-mode \
#     --out $GRAPHS


# === Step 2: Run hyperparameter sweep ===
SWEEP_ID=""      # Set to existing sweep ID to resume; leave empty to start a new sweep
WANDB_PROJECT="" # Set to your W&B project name

if [ -n "$SWEEP_ID" ]; then
    python gat_hyperparam_search.py \
        --resume_sweep $SWEEP_ID \
        --max_runs 30 \
        --graphs $GRAPHS \
        --wandb_project $WANDB_PROJECT
else
    python gat_hyperparam_search.py \
        --max_runs 30 \
        --graphs $GRAPHS \
        --wandb_project $WANDB_PROJECT
fi
