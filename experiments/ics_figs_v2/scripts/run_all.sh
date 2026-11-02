#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
ICS_FIG_PY=/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python
ICS_FIG_ROOT=experiments/ics_figs_v2
mkdir -p "$ICS_FIG_ROOT/logs"
"$ICS_FIG_PY" -B -u "$ICS_FIG_ROOT/scripts/drift_figures.py" > "$ICS_FIG_ROOT/logs/drift.log" 2>&1
"$ICS_FIG_PY" -B -u "$ICS_FIG_ROOT/scripts/probe_diagnostic.py" > "$ICS_FIG_ROOT/logs/probe_numerical_diagnostic.log" 2>&1
for model in F HF; do
  "$ICS_FIG_PY" -B -u "$ICS_FIG_ROOT/scripts/quadrant_analysis.py" --model "$model" > "$ICS_FIG_ROOT/logs/quadrant_$model.log" 2>&1
done
"$ICS_FIG_PY" -B -u "$ICS_FIG_ROOT/scripts/build_report.py" > "$ICS_FIG_ROOT/logs/report.log" 2>&1
"$ICS_FIG_PY" -B -u "$ICS_FIG_ROOT/scripts/validate.py" > "$ICS_FIG_ROOT/logs/validation.log" 2>&1
