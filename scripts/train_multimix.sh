#!/bin/bash
# run from the root of the project
set -e

config_path="configs/bdb_306090.yml"

python main.py \
    --config_path "$config_path" \
    --task "regression" \
    --ckpt_name "MultiMix_bdb_306090" \
