# BCRS: Budget-Conditioned Region Selection for Efficient Small-Object Detection

Research code for the paper *Budget-Conditioned Region Selection for Efficient
Small-Object Detection in High-Resolution Imagery* (proposal: `initial-resources/main.pdf`).

BCRS is a two-pass detector: a coarse pass on a downsampled image produces
provisional detections and per-tile scores; a budget-conditioned differentiable
top-k head selects tiles for native-resolution re-detection under a compute
budget B given **at inference time**; fine and coarse detections are fused.
One trained model traces the full accuracy-computation curve.

## Layout
- `PLAN.md` - master research plan (methodology, metrics, experiment matrix)
- `docs/A100_SETUP.md` - commands to run on the A100 training server
- `docs/DATASETS.md` - dataset acquisition and preprocessing
- `src/bcrs/` - model, training, evaluation code
- `configs/` - experiment configs (one YAML per run)
- `scripts/` - setup, download, launch, and table-generation scripts
- `results/` - committed metrics JSONs + curves (weights stay on the server)
- `paper/` - LaTeX sources

## Quick start (A100 server)
```bash
git clone <this-repo> && cd small-obj-detection
bash scripts/setup_server.sh          # venv + deps
bash scripts/download_visdrone.sh     # ~1.8 GB into data/visdrone/
```
See `docs/A100_SETUP.md` for the full workflow.
