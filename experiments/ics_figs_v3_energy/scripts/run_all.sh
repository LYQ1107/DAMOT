#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
HF_ENERGY_PY=/data1/liuyeqiang/damot_rebuttal_20261026/env/bin/python
HF_ENERGY_OUT=experiments/ics_figs_v3_energy
mkdir -p "$HF_ENERGY_OUT/logs"
"$HF_ENERGY_PY" -B -u "$HF_ENERGY_OUT/scripts/drift_energy.py" > "$HF_ENERGY_OUT/logs/drift_energy.log" 2>&1
"$HF_ENERGY_PY" -B -u "$HF_ENERGY_OUT/scripts/validate.py" > "$HF_ENERGY_OUT/logs/validation.log" 2>&1
