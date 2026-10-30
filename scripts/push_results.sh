#!/usr/bin/env bash
# Commit and push the metrics of one run from the A100 server.
# Usage: bash scripts/push_results.sh <RUN_NAME>
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${1:?usage: push_results.sh <RUN_NAME>}"

test -d "results/${RUN}" || { echo "results/${RUN} does not exist"; exit 1; }
find "results/${RUN}" -maxdepth 1 \( -name '*.json' -o -name '*.txt' -o -name '*.md' \) -print0 |
  xargs -0 -r git add
if git diff --cached --quiet; then
  echo "nothing new to commit for ${RUN}"
  exit 0
fi
git commit -m "results: ${RUN}"
git pull --rebase
git push
echo "pushed results/${RUN}"
