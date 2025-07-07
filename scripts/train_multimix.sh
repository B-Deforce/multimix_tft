#!/bin/bash
# run from the root of the project
set -e

config_path="configs/config_103050_10_4_classification.yml"

python main.py \
    --config_path "$config_path" \
    --task "classification" \
