#!/usr/bin/env bash
# Waits for pipeline_final.sh to finish, then re-measures latency with the
# corrected protocol (device-resident frame; see scripts/measure_latency.py)
# on an otherwise idle GPU, and pushes. Safe to launch while the pipeline runs:
#   git pull && nohup bash scripts/after_pipeline.sh > runs/after_pipeline.log 2>&1 &
#   tail -3 runs/after_pipeline.log
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
LOG=runs/pipeline_final.log
echo "[$(date '+%F %T')] waiting for 'STEP DONE' in $LOG"
until grep -q "STEP DONE" "$LOG" 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] pipeline finished; GPU state:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
J=runs/p2c_bcrs_joint_v2
git pull -q || echo "WARN: git pull failed (continuing on current code)"
python scripts/measure_latency.py --ckpt $J/detector_best.pth --scorer-ckpt $J/scorer_best.pth \
  --out results/latency/latency_val.json \
  && bash scripts/push_results.sh latency || echo "FAILED latency"
echo "[$(date '+%F %T')] DONE"
