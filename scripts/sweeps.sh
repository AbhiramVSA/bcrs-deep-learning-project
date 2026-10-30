#!/usr/bin/env bash
# Named sweep presets (long argument lists get truncated when pasted).
#   bash scripts/sweeps.sh fusion_v2     # wider band + size-aware fusion vs whole-image (~6 min)
#   bash scripts/sweeps.sh tile320       # 320 px tiles, oracle/random/learned curves (~40 min)
#   bash scripts/sweeps.sh full512       # the original full sweep (512 px tiles)
# Each preset pushes its results when done.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
FINE=runs/p1_uniform_s1/best.pth
COARSE=runs/p1_uniform_s4/best.pth
SCORER=runs/p2b_scorer_s4coarse/scorer_best.pth
case "${1:?preset name required: fusion_v2 | tile320 | full512}" in
  fusion_v2)
    python scripts/sweep_curve.py --ckpt "$FINE" --coarse-ckpt "$COARSE" --k-list 0,3,6,15 --whole-image \
      --overlap 64 --coarse-full-min-side 128 --out results/p2a_fusion_diag/curve_ov64_sizeaware.json
    bash scripts/push_results.sh p2a_fusion_diag ;;
  tile320)
    python scripts/sweep_curve.py --ckpt "$FINE" --coarse-ckpt "$COARSE" --scorer-ckpt "$SCORER" \
      --tile 320 --overlap 32 --out results/p2a_tile320_sweep/curve_val.json
    bash scripts/push_results.sh p2a_tile320_sweep ;;
  full512)
    python scripts/sweep_curve.py --ckpt "$FINE" --coarse-ckpt "$COARSE" --scorer-ckpt "$SCORER" \
      --out results/p2a_oracle_sweep/curve_val.json
    bash scripts/push_results.sh p2a_oracle_sweep ;;
  *) echo "unknown preset: $1"; exit 1 ;;
esac
