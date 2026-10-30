#!/usr/bin/env bash
# Everything the paper still needs, unattended (~17 h). Each step pushes.
#   nohup bash scripts/pipeline_final.sh > runs/pipeline_final.log 2>&1 &
#   grep -E "STEP|FAILED|WARN" runs/pipeline_final.log; tail -2 runs/pipeline_final.log
# Order: cheap eval-only ablations first, then latency (needs a quiet GPU,
# so it runs before the long trainings), then the four trainings + sweeps.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs
push() { bash scripts/push_results.sh "$1" || echo "WARN: push of $1 failed (continuing)"; }
step() { echo; echo "=== STEP $1 : $(date '+%F %T') ==="; }
J=runs/p2c_bcrs_joint_v2
FUSE="--overlap 64 --coarse-full-min-side 128"
sweep() {  # name detector scorer [extra args]
  python scripts/sweep_curve.py --ckpt "$2" --scorer-ckpt "$3" $FUSE "${@:4}" --out "results/$1/curve_val.json" \
    && push "$1" || echo "FAILED sweep $1 (continuing)"
}
train_joint() {  # config-name
  python -m bcrs.train_joint --config "configs/$1.yaml" || { echo "FAILED $1"; return 1; }; push "$1"
}

step "1 ablation (i): budget-independent ranking (FiLM input fixed) — eval only"
sweep abl_film_fixed_budget $J/detector_best.pth $J/scorer_best.pth --budget-fixed 0.15 --rankings learned

step "2 latency, batch 1, 100 images (run while GPU is otherwise idle)"
python scripts/measure_latency.py --ckpt $J/detector_best.pth --scorer-ckpt $J/scorer_best.pth \
  --out results/latency/latency_val.json && push latency || echo "FAILED latency (continuing)"

step "3 ablation (ii): objectness targets";   train_joint abl_targets_objectness && sweep abl_targets_objectness_sweep runs/abl_targets_objectness/detector_best.pth runs/abl_targets_objectness/scorer_best.pth --rankings learned
step "4 ablation (iii): gumbel top-k";        train_joint abl_grad_gumbel        && sweep abl_grad_gumbel_sweep        runs/abl_grad_gumbel/detector_best.pth        runs/abl_grad_gumbel/scorer_best.pth        --rankings learned
step "5 ablation (iii): no selection grad";   train_joint abl_grad_none          && sweep abl_grad_none_sweep          runs/abl_grad_none/detector_best.pth          runs/abl_grad_none/scorer_best.pth          --rankings learned
step "6 seed 1";  train_joint p2c_bcrs_joint_v2_seed1 && sweep p3_sweep_joint_v2_seed1 runs/p2c_bcrs_joint_v2_seed1/detector_best.pth runs/p2c_bcrs_joint_v2_seed1/scorer_best.pth
step "7 seed 2";  train_joint p2c_bcrs_joint_v2_seed2 && sweep p3_sweep_joint_v2_seed2 runs/p2c_bcrs_joint_v2_seed2/detector_best.pth runs/p2c_bcrs_joint_v2_seed2/scorer_best.pth
step "DONE"
