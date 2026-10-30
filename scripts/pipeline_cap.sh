#!/usr/bin/env bash
# Post-audit experiments, unattended. Waits for scripts/after_pipeline.sh to
# finish (it owns the GPU for the latency measurement), then:
#   A. Candidate-cap study (audit F8). torchvision FCOS keeps at most 1000
#      pre-NMS candidates per FPN level PER IMAGE. A whole VisDrone frame with
#      hundreds of objects can exceed that at P3, while every 512 tile gets its
#      own 1000, so the cap alone could explain why exhaustive tiling with an
#      unchanged detector beats whole-image inference (19.6 vs 15.8). Evaluate
#      whole-image s=1 at caps 1000 / 4000 / 16000 (eval only, minutes), and
#      the frozen two-pass sweep at 16000.
#   B. If the cap moves whole-image AP by more than 0.5 points, re-evaluate
#      EVERY number in the paper at cap 16000 so one post-processing setting is
#      used throughout: uniform s4/s2/s1, joint sweep (with whole), ablation
#      sweeps, FiLM fixed-budget sweep, seed sweeps. Eval only, ~4-5 h.
#   C. Matched-epochs control (audit F41): continue the native detector for 12
#      epochs on whole images (configs/p1_uniform_s1_v2_cont12.yaml), ~1.5 h,
#      then evaluate it at both caps.
# Launch:
#   git pull && nohup bash scripts/pipeline_cap.sh > runs/pipeline_cap.log 2>&1 &
#   grep -E "STEP|FAILED|WARN|DECISION" runs/pipeline_cap.log; tail -2 runs/pipeline_cap.log
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p runs results/cap_study
push() { bash scripts/push_results.sh "$1" || echo "WARN: push of $1 failed (continuing)"; }
step() { echo; echo "=== STEP $1 : $(date '+%F %T') ==="; }

echo "[$(date '+%F %T')] waiting for after_pipeline.sh DONE"
until grep -q "DONE" runs/after_pipeline.log 2>/dev/null; do sleep 300; done
git pull -q || echo "WARN: git pull failed (continuing on current code)"

S1=runs/p1_uniform_s1_v2/best.pth; S2=runs/p1_uniform_s2_v2/best.pth; S4=runs/p1_uniform_s4_v2/best.pth
SC=runs/p2b_scorer_s4v2/scorer_best.pth
J=runs/p2c_bcrs_joint_v2
FUSE="--overlap 64 --coarse-full-min-side 128"
CAP=16000

step "A1 whole-image s=1 at caps 1000 / 4000 / 16000"
for c in 1000 4000 16000; do
  python scripts/eval_ckpt.py --ckpt $S1 --topk-candidates $c --out results/cap_study/whole_s1_cap${c}.json \
    || echo "FAILED whole_s1_cap${c}"
done
push cap_study

step "A2 frozen two-pass sweep at cap $CAP (compare with results/p3_sweep_v2_t512)"
python scripts/sweep_curve.py --ckpt $S1 --coarse-ckpt $S4 --scorer-ckpt $SC $FUSE --rankings oracle,random,learned \
  --whole-image --topk-candidates $CAP --out results/cap_study/frozen_t512_cap${CAP}.json || echo "FAILED frozen sweep"
push cap_study

DELTA=$(python - <<'EOF'
import json
def ap(p):
    d = json.load(open(p)); m = d.get("metrics", d.get("final", d)); return float(m["AP"])
try:
    print(f"{ap('results/cap_study/whole_s1_cap16000.json') - ap('results/cap_study/whole_s1_cap1000.json'):.4f}")
except Exception as e:
    print("nan"); import sys; print("decision error:", e, file=sys.stderr)
EOF
)
echo "DECISION: whole-image AP(cap 16000) - AP(cap 1000) = $DELTA"
REEVAL=$(python -c "d='$DELTA'; print(1 if d!='nan' and abs(float(d))>0.005 else 0)")
echo "DECISION: re-evaluate everything at cap $CAP = $REEVAL"

if [ "$REEVAL" = "1" ]; then
  step "B1 uniform s4 / s2 at cap $CAP"
  python scripts/eval_ckpt.py --ckpt $S4 --topk-candidates $CAP --out results/cap_study/whole_s4_cap${CAP}.json || echo "FAILED s4"
  python scripts/eval_ckpt.py --ckpt $S2 --topk-candidates $CAP --out results/cap_study/whole_s2_cap${CAP}.json || echo "FAILED s2"
  push cap_study
  step "B2 joint sweep at cap $CAP (all rankings + whole)"
  python scripts/sweep_curve.py --ckpt $J/detector_best.pth --scorer-ckpt $J/scorer_best.pth $FUSE \
    --rankings oracle,random,learned --whole-image --topk-candidates $CAP \
    --out results/p3_sweep_joint_v2_cap16k/curve_val.json || echo "FAILED joint sweep"
  push p3_sweep_joint_v2_cap16k
  step "B3 ablation + seed sweeps at cap $CAP (learned ranking)"
  for r in abl_targets_objectness abl_grad_gumbel abl_grad_none p2c_bcrs_joint_v2_seed1 p2c_bcrs_joint_v2_seed2; do
    [ -f runs/$r/detector_best.pth ] || { echo "WARN: runs/$r missing, skipped"; continue; }
    python scripts/sweep_curve.py --ckpt runs/$r/detector_best.pth --scorer-ckpt runs/$r/scorer_best.pth $FUSE \
      --rankings learned --topk-candidates $CAP --out results/${r}_sweep_cap16k/curve_val.json || echo "FAILED sweep $r"
    push ${r}_sweep_cap16k
  done
  python scripts/sweep_curve.py --ckpt $J/detector_best.pth --scorer-ckpt $J/scorer_best.pth $FUSE \
    --rankings learned --budget-fixed 0.15 --topk-candidates $CAP \
    --out results/abl_film_fixed_budget_cap16k/curve_val.json || echo "FAILED film sweep"
  push abl_film_fixed_budget_cap16k
fi

step "C matched-epochs control: native detector + 12 epochs on whole images"
python -m bcrs.train --config configs/p1_uniform_s1_v2_cont12.yaml || echo "FAILED cont12 training"
push p1_uniform_s1_v2_cont12
for c in 1000 $CAP; do
  python scripts/eval_ckpt.py --ckpt runs/p1_uniform_s1_v2_cont12/best.pth --topk-candidates $c \
    --out results/cap_study/whole_s1_cont12_cap${c}.json || echo "FAILED cont12 eval cap $c"
done
push cap_study
step "DONE"
