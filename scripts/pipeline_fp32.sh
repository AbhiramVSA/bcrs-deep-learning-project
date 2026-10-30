#!/usr/bin/env bash
# fp32 box-decode correction (found 2026-10-05): every remaining experiment,
# unattended. Plan and rationale: docs/PLAN_FP32.md. Each step pushes its results.
#
# Stages, run in the order given by STAGES (default "A B D C E"):
#   A  eval only (~6 h). Re-evaluate every v2 checkpoint with the fp32 decoder:
#      uniform s4/s2/s1 (+ s1 in legacy bf16 mode as a reproduction check),
#      frozen sweeps g=512 and g=320, joint sweep, seed sweeps, ablation sweeps.
#      Then scripts/compare_fp32.py -> results/fp32eval_compare (old vs new).
#   B  retrain the main chain as recipe v3 = v2 + fp32 decoder (~17 h):
#      uniform s4/s2/s1, scorer, frozen sweeps, joint model + sweep, and the
#      matched-epochs control (v3). With RETRAIN=never, B runs only the v2
#      matched-epochs control under v2 conditions.
#   D  latency on the newest joint model (~1.5 h): GPU batch 1, GPU throughput
#      at batch 1 and 8, CPU at 4 and 16 threads (compute-bound reference).
#   C  seeds 1 and 2 of the v3 joint model + sweeps (~12 h).
#   E  the three v3 ablation trainings + sweeps, and the fixed-budget sweep (~13 h).
#
# Launch on the server:
#   git pull && nohup bash scripts/pipeline_fp32.sh > runs/pipeline_fp32.log 2>&1 &
# Options (environment variables, all optional):
#   STAGES="A B D C E"   RETRAIN=always|auto|never (default always; auto follows
#   compare_fp32.py)   NEED_TRAIN_MB=20000 NEED_EVAL_MB=8000 (free GPU memory to
#   wait for before a step)   MAX_WAIT_MIN=240   REDO=1 (do not skip finished steps)
#   DRY=1 (print the commands, run nothing):
#     DRY=1 bash scripts/pipeline_fp32.sh | grep -E "STEP|RUN|SKIP|FAILED|DECISION"
# Progress:
#   grep -E "^=== |FAILED|WARN|DECISION|WAIT" runs/pipeline_fp32.log | tail -25
#   tail -c 400 runs/logs_fp32/$(ls -t runs/logs_fp32 | head -1)
# Finished when the log ends with "=== PIPELINE_FP32_FINISHED".
#
# The body is one function, so bash parses the whole file before it runs:
# push_results.sh does a git pull after every step, and an update of this file
# must not change the instance that runs.

main() {
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
mkdir -p runs/logs_fp32

STAGES="${STAGES:-A B D C E}"
RETRAIN="${RETRAIN:-always}"
NEED_TRAIN_MB="${NEED_TRAIN_MB:-20000}"
NEED_EVAL_MB="${NEED_EVAL_MB:-8000}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-240}"
REDO="${REDO:-0}"
DRY="${DRY:-0}"
if (( ! DRY )); then
  exec 9> runs/pipeline_fp32.lock
  flock -n 9 || { echo "another pipeline_fp32.sh holds runs/pipeline_fp32.lock; exit"; exit 1; }
fi
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"; GPU_ID="${GPU_ID%%,*}"
# shared 40 GB GPU: the 2026-09-23 control run died with 358 MB reserved but
# unallocated next to a 310 MB request; expandable segments reduce that waste
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FUSE=(--overlap 64 --coarse-full-min-side 128)
V2S1=runs/p1_uniform_s1_v2/best.pth; V2S4=runs/p1_uniform_s4_v2/best.pth
V2SC=runs/p2b_scorer_s4v2/scorer_best.pth; V2J=runs/p2c_bcrs_joint_v2
V3S1=runs/p1_uniform_s1_v3/best.pth; V3S4=runs/p1_uniform_s4_v3/best.pth
V3SC=runs/p2b_scorer_s4v3/scorer_best.pth; V3J=runs/p2c_bcrs_joint_v3

step() { echo; echo "=== STEP $1 : $(date '+%F %T') ==="; }
push() { (( DRY )) && return 0; bash scripts/push_results.sh "$1" > "runs/logs_fp32/push_$1.log" 2>&1 \
           && echo "pushed results/$1" || echo "WARN: push of $1 failed (see runs/logs_fp32/push_$1.log)"; }

gpu_free_mb() {
  nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}
wait_gpu() {  # block until the shared GPU has $1 MB free; after MAX_WAIT_MIN minutes start anyway
  local need=$1 waited=0 free
  (( need > 0 && ! DRY )) || return 0
  while :; do
    free=$(gpu_free_mb)
    [[ "$free" =~ ^[0-9]+$ ]] || { echo "WARN: cannot read free GPU memory; not waiting"; return 0; }
    (( free >= need )) && return 0
    (( waited % 30 == 0 )) && echo "[$(date '+%F %T')] WAIT: GPU has ${free} MB free, step needs ${need} MB"
    (( waited >= MAX_WAIT_MIN )) && { echo "WARN: waited ${waited} min for GPU memory; starting anyway"; return 0; }
    sleep 60; waited=$((waited + 1))
  done
}
run() {  # label need_mb cmd... : step log in runs/logs_fp32/<label>.log; up to 3 tries on CUDA OOM
  local label=$1 need=$2; shift 2
  local log="runs/logs_fp32/${label}.log" attempt rc
  for attempt in 1 2 3; do
    wait_gpu "$need"
    echo "[$(date '+%F %T')] RUN ${label} (try ${attempt}): $*"
    (( DRY )) && return 0
    "$@" > "$log" 2>&1; rc=$?
    if (( rc == 0 )); then echo "[$(date '+%F %T')] OK ${label}"; return 0; fi
    if grep -qE "out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED" "$log"; then
      cp "$log" "runs/logs_fp32/${label}.oom${attempt}.log"
      echo "WARN: ${label} ran out of GPU memory (try ${attempt}); next try in 10 min"
      sleep 600; continue
    fi
    echo "FAILED ${label} (exit ${rc}). Last lines of ${log}:"
    tail -c 3000 "$log" | tr '\r' '\n' | tail -6 | sed 's/^/    /'
    return 1
  done
  echo "FAILED ${label}: out of GPU memory 3 times"; return 1
}
need_files() {  # label file... : fail the step (not the pipeline) when an input is missing
  local label=$1 f; shift
  for f in "$@"; do [[ -f "$f" ]] || { echo "FAILED ${label}: input ${f} missing"; return 1; }; done
}
train() {  # module config-name   (module: train | train_bcrs | train_joint)
  local mod=$1 name=$2
  if [[ -f "results/${name}/metrics.json" && "$REDO" != 1 ]]; then echo "SKIP ${name}: already trained"; return 0; fi
  run "$name" "$NEED_TRAIN_MB" python -m "bcrs.${mod}" --config "configs/${name}.yaml" || return 1
  push "$name"
}
sweep() {  # result-name sweep_curve.py-args...   (needs --ckpt; --scorer-ckpt/--coarse-ckpt optional)
  local name=$1; shift
  if [[ -f "results/${name}/curve_val.json" && "$REDO" != 1 ]]; then echo "SKIP ${name}: results exist"; return 0; fi
  local a ckpts=() prev=""
  for a in "$@"; do [[ "$prev" == --ckpt || "$prev" == --scorer-ckpt || "$prev" == --coarse-ckpt ]] && ckpts+=("$a"); prev=$a; done
  (( DRY )) || need_files "$name" "${ckpts[@]}" || return 1
  run "$name" "$NEED_EVAL_MB" python scripts/sweep_curve.py "$@" --out "results/${name}/curve_val.json" || return 1
  push "$name"
}
evalc() {  # out-json ckpt [eval_ckpt.py args]   (caller pushes the directory)
  local out=$1 ckpt=$2; shift 2
  local label; label="eval_$(basename "$(dirname "$out")")_$(basename "$out" .json)"
  if [[ -f "$out" && "$REDO" != 1 ]]; then echo "SKIP ${out}: exists"; return 0; fi
  (( DRY )) || need_files "$label" "$ckpt" || return 1
  run "$label" "$NEED_EVAL_MB" python scripts/eval_ckpt.py --ckpt "$ckpt" "$@" --out "$out"
}

stage_A() {
  step "A1 uniform v2 checkpoints with the fp32 decoder (+ s=1 in legacy bf16 mode: must reproduce 15.88 AP)"
  local s i a r
  for s in 4 2 1; do evalc "results/fp32eval_uniform/s${s}.json" "runs/p1_uniform_s${s}_v2/best.pth"; done
  evalc results/fp32eval_uniform/s1_legacy_bf16.json "$V2S1" --legacy-bf16-decode
  push fp32eval_uniform
  step "A2 frozen-detector sweep g=512, fp32 (all rankings + whole image)"
  sweep p3_sweep_v2_t512_fp32 --ckpt "$V2S1" --coarse-ckpt "$V2S4" --scorer-ckpt "$V2SC" "${FUSE[@]}" --whole-image
  step "A3 joint model sweep, fp32 (all rankings + whole image)"
  sweep p3_sweep_joint_v2_fp32 --ckpt "$V2J/detector_best.pth" --scorer-ckpt "$V2J/scorer_best.pth" "${FUSE[@]}" --whole-image
  for i in 1 2; do
    step "A4.$i seed $i sweep, fp32 (+ whole image, which the v2 seed sweeps lacked)"
    r=runs/p2c_bcrs_joint_v2_seed$i
    sweep "p3_sweep_joint_v2_seed${i}_fp32" --ckpt "$r/detector_best.pth" --scorer-ckpt "$r/scorer_best.pth" "${FUSE[@]}" --whole-image
  done
  step "A5 ablation sweeps, fp32 (learned ranking)"
  sweep abl_film_fixed_budget_fp32 --ckpt "$V2J/detector_best.pth" --scorer-ckpt "$V2J/scorer_best.pth" "${FUSE[@]}" \
    --budget-fixed 0.15 --rankings learned
  for a in abl_targets_objectness abl_grad_gumbel abl_grad_none; do
    sweep "${a}_sweep_fp32" --ckpt "runs/$a/detector_best.pth" --scorer-ckpt "runs/$a/scorer_best.pth" "${FUSE[@]}" --rankings learned
  done
  step "A6 frozen-detector sweep g=320, fp32 (oracle + random)"
  sweep p3_sweep_v2_t320_fp32 --ckpt "$V2S1" --coarse-ckpt "$V2S4" --tile 320 --overlap 32 --coarse-full-min-side 64
  step "A7 old (bf16) vs new (fp32) comparison"
  (( DRY )) && { echo "RUN compare_fp32.py"; return 0; }
  python scripts/compare_fp32.py > runs/logs_fp32/compare_fp32.log 2>&1 || echo "FAILED compare_fp32.py"
  grep -E "DECISION|missing|AP " runs/logs_fp32/compare_fp32.log | sed 's/^/    /'
  push fp32eval_compare
}

decide_retrain() {  # sets V3=1 (train recipe v3) or V3=0
  local s
  case "$RETRAIN" in
    always) V3=1; echo "DECISION: RETRAIN=always: stage B trains recipe v3" ;;
    never)  V3=0; echo "DECISION: RETRAIN=never: no v3 training; stage B runs the v2 matched-epochs control only" ;;
    auto)
      s=$(python -c "import json; print(json.load(open('results/fp32eval_compare/compare.json'))['decision']['retrain_suggested'])" 2>/dev/null)
      if [[ "$s" == "False" ]]; then V3=0; else V3=1; fi
      echo "DECISION: RETRAIN=auto, compare_fp32 retrain_suggested=${s:-unknown} -> V3=${V3}" ;;
    *) echo "FAILED: RETRAIN must be always, auto or never (is '$RETRAIN')"; V3=0 ;;
  esac
}

stage_B() {
  if (( ! V3 )); then
    step "B (v2) matched-epochs control under v2 conditions (bf16 decode in training), evaluated in fp32"
    train train p1_uniform_s1_v2_cont12 \
      && evalc results/fp32eval_uniform/s1_cont12.json runs/p1_uniform_s1_v2_cont12/best.pth \
      && push fp32eval_uniform
    return 0
  fi
  local s
  step "B1 uniform s=4, recipe v3 (coarse detector)"
  train train p1_uniform_s4_v3 || { echo "FAILED stage B stopped: no v3 coarse detector"; return 1; }
  step "B2 uniform s=2, recipe v3"
  train train p1_uniform_s2_v3 || echo "WARN: s=2 baseline failed; stage B continues (only a curve point)"
  step "B3 uniform s=1, recipe v3 (fine-detector init)"
  train train p1_uniform_s1_v3 || { echo "FAILED stage B stopped: no v3 native detector"; return 1; }
  step "B4 uniform v3 checkpoints with eval_ckpt.py (same protocol as A1)"
  for s in 4 2 1; do evalc "results/v3eval_uniform/s${s}.json" "runs/p1_uniform_s${s}_v3/best.pth"; done
  push v3eval_uniform
  step "B5 tile scorer on the frozen v3 s=4 detector"
  train train_bcrs p2b_scorer_s4v3 || { echo "FAILED stage B stopped: no v3 scorer"; return 1; }
  step "B6 frozen-detector sweep g=512, v3"
  sweep p3_sweep_v3_t512 --ckpt "$V3S1" --coarse-ckpt "$V3S4" --scorer-ckpt "$V3SC" "${FUSE[@]}" --whole-image
  step "B7 joint BCRS training, v3"
  train train_joint p2c_bcrs_joint_v3 || { echo "FAILED stage B stopped: joint v3 training"; return 1; }
  step "B8 joint v3 sweep (all rankings + whole image)"
  sweep p3_sweep_joint_v3 --ckpt "$V3J/detector_best.pth" --scorer-ckpt "$V3J/scorer_best.pth" "${FUSE[@]}" --whole-image
  step "B9 matched-epochs control, v3: native detector + 12 epochs on whole images"
  train train p1_uniform_s1_v3_cont12 \
    && evalc results/v3eval_uniform/s1_cont12.json runs/p1_uniform_s1_v3_cont12/best.pth \
    && push v3eval_uniform
  step "B10 frozen-detector sweep g=320, v3"
  sweep p3_sweep_v3_t320 --ckpt "$V3S1" --coarse-ckpt "$V3S4" --tile 320 --overlap 32 --coarse-full-min-side 64
}

stage_D() {
  local J L t
  if [[ -f "$V3J/detector_best.pth" ]]; then J=$V3J; L=latency_v3; else J=$V2J; L=latency_v2_fp32; fi
  (( DRY )) || need_files "stage D" "$J/detector_best.pth" "$J/scorer_best.pth" || return 1
  step "D1 GPU latency, batch 1, 100 images ($J)"
  run "${L}_gpu_b1" "$NEED_EVAL_MB" python scripts/measure_latency.py --ckpt "$J/detector_best.pth" \
    --scorer-ckpt "$J/scorer_best.pth" --out "results/$L/gpu_b1.json" && push "$L"
  step "D2 GPU throughput, batch 1 and 8 (frames of one size, all tiles of a batch in one call)"
  for t in 1 8; do
    run "${L}_gpu_throughput_b$t" "$NEED_EVAL_MB" python scripts/measure_throughput.py --ckpt "$J/detector_best.pth" \
      --scorer-ckpt "$J/scorer_best.pth" --batch "$t" --n-images 96 --out "results/$L/gpu_throughput_b$t.json" && push "$L"
  done
  step "D3 CPU latency, fp32, 40 images (compute-bound reference; load average recorded)"
  for t in 4 16; do
    (( t <= $(nproc) )) || { echo "WARN: only $(nproc) CPUs; skip ${t} threads"; continue; }
    run "${L}_cpu_t$t" 0 python scripts/measure_latency.py --device cpu --threads "$t" --n-images 40 --warmup 3 \
      --ckpt "$J/detector_best.pth" --scorer-ckpt "$J/scorer_best.pth" --out "results/$L/cpu_t$t.json" && push "$L"
  done
}

stage_C() {
  local i r
  (( DRY )) || need_files "stage C" "$V3S1" "$V3SC" || return 1
  for i in 1 2; do
    step "C$i joint v3, seed $i (+ sweep with whole image)"
    r=p2c_bcrs_joint_v3_seed$i
    train train_joint "$r" \
      && sweep "p3_sweep_joint_v3_seed$i" --ckpt "runs/$r/detector_best.pth" --scorer-ckpt "runs/$r/scorer_best.pth" \
           "${FUSE[@]}" --whole-image
  done
}

stage_E() {
  local a
  (( DRY )) || need_files "stage E" "$V3S1" "$V3SC" || return 1
  step "E1 ablation (i): budget input fixed at B(3), joint v3 (eval only)"
  sweep abl_film_fixed_budget_v3 --ckpt "$V3J/detector_best.pth" --scorer-ckpt "$V3J/scorer_best.pth" "${FUSE[@]}" \
    --budget-fixed 0.15 --rankings learned
  for a in abl_targets_objectness abl_grad_gumbel abl_grad_none; do
    step "E ablation ${a}_v3 (training + sweep)"
    train train_joint "${a}_v3" \
      && sweep "${a}_v3_sweep" --ckpt "runs/${a}_v3/detector_best.pth" --scorer-ckpt "runs/${a}_v3/scorer_best.pth" \
           "${FUSE[@]}" --rankings learned
  done
}

echo "pipeline_fp32 start $(date '+%F %T') commit $(git rev-parse --short HEAD) stages='${STAGES}' RETRAIN=${RETRAIN}"
python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
nvidia-smi -i "$GPU_ID" --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo "CPUs: $(nproc)   disk free: $(df -h --output=avail . | tail -1 | tr -d ' ')"
# the fix must be active before any GPU hour is spent (works without pytest installed)
python - > runs/logs_fp32/test_decode.log 2>&1 <<'PYEOF'
import importlib.util
spec = importlib.util.spec_from_file_location("t", "tests/test_box_decode_dtype.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
for name in sorted(n for n in dir(t) if n.startswith("test_")):
    getattr(t, name)(); print("passed", name)
PYEOF
if (( $? == 0 )); then echo "fp32 decode tests passed"
else echo "FAILED fp32 decode tests (runs/logs_fp32/test_decode.log); stop"; exit 1; fi

V3=1
decided=0
for stg in $STAGES; do
  case "$stg" in
    A) stage_A ;;
    B) decide_retrain; decided=1; stage_B ;;
    C|E) (( decided )) || decide_retrain; decided=1
         if (( V3 )); then "stage_$stg"; else echo "SKIP stage $stg: no v3 models (RETRAIN=$RETRAIN)"; fi ;;
    D) stage_D ;;
    *) echo "WARN: unknown stage '$stg'" ;;
  esac
done
echo
echo "failed steps: $(grep -c '^FAILED' runs/pipeline_fp32.log 2>/dev/null) (list: grep ^FAILED runs/pipeline_fp32.log)"
echo "=== PIPELINE_FP32_FINISHED : $(date '+%F %T') ==="
}

main "$@"
exit $?
