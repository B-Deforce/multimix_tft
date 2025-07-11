#!/bin/bash
# run from root of the repo
set -e

# parameters
config_path="configs/bdb_306090.yml"
model="MultiMix_bdb_306090" # in lightning_logs
model_version=0 # version of the model in lightning logs
checkpoint_name="MultiMix_MultiMix_bdb_306090-epoch=00-val_mse_loss1=0.20.ckpt"
task="regression"

# paths, automatically created based on parameters
checkpoint_path="lightning_logs/MultiMix_$model/version_$model_version/checkpoints/$checkpoint_name"
results_dir="results/$model/"

python run_eval.py \
    --config_path "$config_path" \
    --ckpt_path "$checkpoint_path" \
    --results_dir "$results_dir" \
    --task "$task" \
    #--return_x true \