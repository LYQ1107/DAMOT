#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT=/data1/liuyeqiang/damot_rebuttal_20261026
PY="$RUN_ROOT/env/bin/python"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$RUN_ROOT/code"
python3 "$RUN_ROOT/scripts/wait_existing_controllers.py"
if [[ ! -f "$RUN_ROOT/manifests/datasets.json" ]]; then
  python3 "$RUN_ROOT/scripts/prepare_inputs.py" inventory
fi
if [[ ! -f "$RUN_ROOT/cache/weights/F_stage4_e3.pth" ]]; then
  python3 "$RUN_ROOT/scripts/prepare_inputs.py" weights
fi
for domain in clear_holdout_45v night; do
  if [[ ! -f "$RUN_ROOT/manifests/${domain}_decode_validation.json" ]]; then
    "$PY" "$RUN_ROOT/scripts/bulk_cache_images.py" "$domain"
  fi
done
python3 "$RUN_ROOT/scripts/collect_exp1.py"
python3 "$RUN_ROOT/scripts/run_accuracy_queue.py"
CUDA_VISIBLE_DEVICES='' "$PY" "$RUN_ROOT/scripts/all_frame_hota.py"
export CUDA_VISIBLE_DEVICES=9
if [[ ! -f "$RUN_ROOT/results/exp4/reconstructed_history.pkl" ]]; then
  "$PY" "$RUN_ROOT/scripts/profile_damot.py" rebuild
fi
if [[ ! -f "$RUN_ROOT/results/exp4/adaptation_complete.json" ]]; then
  "$PY" "$RUN_ROOT/scripts/profile_damot.py" throughput
fi
for n in 0 1 2 3; do
  if [[ ! -f "$RUN_ROOT/results/exp4/module_time_history${n}.csv" ]]; then
    "$PY" "$RUN_ROOT/scripts/profile_damot.py" modules --history "$n"
  fi
done
if [[ ! -f "$RUN_ROOT/results/exp4/inference_complete.json" ]]; then
  "$PY" "$RUN_ROOT/scripts/profile_damot.py" inference
fi
python3 "$RUN_ROOT/scripts/aggregate_results.py" all
CUDA_VISIBLE_DEVICES='' "$PY" "$RUN_ROOT/scripts/validate_delivery.py"
