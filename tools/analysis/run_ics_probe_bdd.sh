#!/usr/bin/env bash
# ICS subspace probe on BDD100K (frozen checkpoints, no training).
# Usage (from the code root that contains darth/ and tools/):
#   RUN=/data1/liuyeqiang/damot_rebuttal_20261026 OUT=$RUN/results/ics_probe \
#   GPU=0 bash tools/analysis/run_ics_probe_bdd.sh [selftest|dryrun|extract|stages|analyze|all]
set -euo pipefail
[[ -f tools/analysis/ics_probe_extract.py ]] || { echo "run from code root" >&2; exit 1; }

RUN=${RUN:-/data1/liuyeqiang/damot_rebuttal_20261026}
PY=${PY:-$RUN/env/bin/python}
OUT=${OUT:-$RUN/results/ics_probe}
GPU=${GPU:-0}
WEIGHTS=${WEIGHTS:-$RUN/cache/weights}
ANN=${ANN:-$RUN/data/annotations}
IMG=${IMG:-$RUN/data/views}
MAIN=${MAIN:-F_stage4_e5}
DOMAINS=(clear_holdout_45v overcast rainy dawn_dusk night)
STAGE_LABELS=(F_stage1_e5 F_stage2_e5 F_stage3_e5 F_stage4_e5)
STAGE_DOMAINS=(overcast rainy dawn_dusk night)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export MKL_THREADING_LAYER=${MKL_THREADING_LAYER:-GNU}
mkdir -p "$OUT/logs"

domain_args(){ for d in "${DOMAINS[@]}"; do printf -- "--domain %s %s/%s.json %s/%s/ " "$d" "$ANN" "$d" "$IMG" "$d"; done; }

step=${1:-all}

if [[ $step == selftest || $step == all ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" tools/analysis/ics_probe_selftest.py --out "$OUT/selftest" \
    2>&1 | tee "$OUT/logs/selftest.log"
  grep -q FAIL "$OUT/logs/selftest.log" && { echo "SELFTEST FAILED" >&2; exit 1; }
fi

if [[ $step == dryrun || $step == all ]]; then
  "$PY" tools/analysis/ics_probe_extract.py --checkpoint "$WEIGHTS/$MAIN.pth" \
    --label "$MAIN" $(domain_args) --out-dir "$OUT/feats" --dry-run \
    2>&1 | tee "$OUT/logs/dryrun.log"
fi

if [[ $step == extract || $step == all ]]; then
  # main checkpoint: all views + dose-response views
  CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/analysis/ics_probe_extract.py \
    --checkpoint "$WEIGHTS/$MAIN.pth" --label "$MAIN" $(domain_args) \
    --views clean,fap,photo,student --dose-frames-per-video 3 \
    --out-dir "$OUT/feats" 2>&1 | tee "$OUT/logs/extract_$MAIN.log"
fi

if [[ $step == stages || $step == all ]]; then
  # earlier stages: only clean + FAP, no dose views (for P5)
  for lab in "${STAGE_LABELS[@]}"; do
    [[ $lab == "$MAIN" ]] && continue
    CUDA_VISIBLE_DEVICES=$GPU "$PY" tools/analysis/ics_probe_extract.py \
      --checkpoint "$WEIGHTS/$lab.pth" --label "$lab" $(domain_args) \
      --views clean,fap --dose-frames-per-video 0 \
      --out-dir "$OUT/feats" 2>&1 | tee "$OUT/logs/extract_$lab.log"
  done
fi

if [[ $step == analyze || $step == all ]]; then
  CUDA_VISIBLE_DEVICES= "$PY" tools/analysis/ics_probe_analyze.py \
    --feat-root "$OUT/feats" --label "$MAIN" --domains "${DOMAINS[@]}" \
    --ref-domain clear_holdout_45v \
    --stage-labels "${STAGE_LABELS[@]}" --stage-domains "${STAGE_DOMAINS[@]}" \
    --classes 1 --montage --out "$OUT/analysis_$MAIN" \
    2>&1 | tee "$OUT/logs/analyze_$MAIN.log"
fi
echo "[ok] $step -> $OUT"
