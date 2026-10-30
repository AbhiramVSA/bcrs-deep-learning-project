#!/usr/bin/env bash
# Unattended recipe-v2 pipeline (~9-10 h). Each step pushes its results.
#   nohup bash scripts/pipeline_v2.sh > runs/pipeline_v2.log 2>&1 &
#   tail -f runs/pipeline_v2.log        # progress;  grep STEP runs/pipeline_v2.log
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs
push() { bash scripts/push_results.sh "$1" || echo "WARN: push of $1 failed (continuing)"; }
step() { echo; echo "=== STEP $1 : $(date '+%F %T') ==="; }
FUSE="--overlap 64 --coarse-full-min-side 128"

step "0 push s1_v2";            push p1_uniform_s1_v2
step "1 train s4_v2";           python -m bcrs.train --config configs/p1_uniform_s4_v2.yaml || { echo "FAILED s4_v2"; exit 1; }; push p1_uniform_s4_v2
step "2 train s2_v2";           python -m bcrs.train --config configs/p1_uniform_s2_v2.yaml || { echo "FAILED s2_v2"; exit 1; }; push p1_uniform_s2_v2
step "3 train scorer on s4_v2"; python -m bcrs.train_bcrs --config configs/p2b_scorer_s4v2.yaml || { echo "FAILED scorer"; exit 1; }; push p2b_scorer_s4v2
step "4 sweep v2 tile512"
python scripts/sweep_curve.py --ckpt runs/p1_uniform_s1_v2/best.pth --coarse-ckpt runs/p1_uniform_s4_v2/best.pth \
  --scorer-ckpt runs/p2b_scorer_s4v2/scorer_best.pth --whole-image $FUSE --out results/p3_sweep_v2_t512/curve_val.json \
  && push p3_sweep_v2_t512 || echo "FAILED sweep t512 (continuing)"
step "5 sweep v2 tile320"
python scripts/sweep_curve.py --ckpt runs/p1_uniform_s1_v2/best.pth --coarse-ckpt runs/p1_uniform_s4_v2/best.pth \
  --tile 320 --overlap 32 --coarse-full-min-side 64 --out results/p3_sweep_v2_t320/curve_val.json \
  && push p3_sweep_v2_t320 || echo "FAILED sweep t320 (continuing)"
step "6 joint BCRS training";   python -m bcrs.train_joint --config configs/p2c_bcrs_joint_v2.yaml || { echo "FAILED joint"; exit 1; }; push p2c_bcrs_joint_v2
step "7 sweep joint model (shared weights)"
python scripts/sweep_curve.py --ckpt runs/p2c_bcrs_joint_v2/detector_best.pth \
  --scorer-ckpt runs/p2c_bcrs_joint_v2/scorer_best.pth --whole-image $FUSE --out results/p3_sweep_joint_v2/curve_val.json \
  && push p3_sweep_joint_v2 || echo "FAILED joint sweep"
step "DONE"
