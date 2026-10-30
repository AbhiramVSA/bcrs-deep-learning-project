#!/usr/bin/env bash
# Robust training launcher: activates the venv itself, keeps startup output,
# and verifies the process survived startup instead of failing silently.
# Usage: bash scripts/launch.sh configs/<run>.yaml [module]
#   module defaults to bcrs.train (detector training);
#   use bcrs.train_bcrs for the scorer / BCRS stages.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${1:?usage: launch.sh configs/<run>.yaml [module]}"
MODULE="${2:-bcrs.train}"
test -f "$CFG" || { echo "no such config: $CFG"; exit 1; }
RUN="$(basename "${CFG%.yaml}")"
source .venv/bin/activate
mkdir -p runs
LOG="runs/${RUN}.nohup.out"
nohup python -m "$MODULE" --config "$CFG" > "$LOG" 2>&1 &
PID=$!
echo "launched ${RUN} pid=${PID} (startup output: ${LOG})"
echo "waiting 30s to verify startup (NFS imports are slow) ..."
sleep 30
if ps -p "${PID}" > /dev/null; then
  echo "ALIVE. recent output:"
  tail -3 "$LOG" || true
  echo "follow training with: tail -f results/${RUN}/train_log.txt"
else
  echo "PROCESS DIED DURING STARTUP — full output:"
  cat "$LOG"
  exit 1
fi
