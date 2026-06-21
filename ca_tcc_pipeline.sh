#!/usr/bin/env bash
set -euo pipefail
# Adapted replacement for the author's 1%-only pipeline.
python experiments/run_experiments.py \
  --methods ts_tcc ca_tcc \
  --label_ratios 1 5 10 \
  --seeds 0 \
  --device auto
