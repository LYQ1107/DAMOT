#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
HF_NATIVE_PY=/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python
HF_NATIVE_OUT=experiments/ics_figs_v3
mkdir -p "$HF_NATIVE_OUT/logs"
"$HF_NATIVE_PY" -B -u "$HF_NATIVE_OUT/scripts/hf_native.py" > "$HF_NATIVE_OUT/logs/hf_native.log" 2>&1
"$HF_NATIVE_PY" -B -u "$HF_NATIVE_OUT/scripts/figures.py" > "$HF_NATIVE_OUT/logs/figures.log" 2>&1
"$HF_NATIVE_PY" -B -u "$HF_NATIVE_OUT/scripts/report.py" > "$HF_NATIVE_OUT/logs/report.log" 2>&1
"$HF_NATIVE_PY" -B -u "$HF_NATIVE_OUT/scripts/validate.py" > "$HF_NATIVE_OUT/logs/validation.log" 2>&1
