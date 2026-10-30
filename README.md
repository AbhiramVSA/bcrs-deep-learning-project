# Budget-Constrained Region Selection (BCRS)

BCRS is a two-pass FCOS model for small-object detection in aerial images. It scores image tiles and runs native-resolution detection on the top-k tiles.

The paper studies this method on VisDrone-DET validation data. It reports that learned ranking improves over random selection at the same tile budget. It also reports lower accuracy and slower GPU latency than dense native-resolution inference. See [the paper](paper/main.pdf) and its [LaTeX source](paper/main.tex).

## Submission status

The source and paper files are ready. The full submission ZIP includes smoke-test weights for the BCRS command. Those weights do not reproduce the paper results. The final v3 detector and scorer checkpoints are missing. The Google Form fields and file-size limits remain unknown because the form returned HTTP 401 during inspection.

## Repository contents

- `src/bcrs/` contains the detector, scorer, tile selection, training, and evaluation code.
- `configs/` contains experiment settings.
- `scripts/` contains training, evaluation, figure, table, and image prediction commands.
- `results/` contains archived metrics and training logs. Some hardware process names and paths are redacted.
- `paper/` contains the paper PDF, LaTeX source, tables, figures, and references.

The BCRS image command is `scripts/predict.py`. It requires a trained BCRS detector checkpoint and scorer checkpoint. The released source snapshot does not contain the final v3 checkpoints. The command rejects a missing checkpoint and does not create model weights.

## Install

Install a PyTorch and torchvision build for your system. For CPU-only use, run:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Predict one image

The full submission ZIP has smoke-test checkpoints at the paths below. They show the command flow and do not reproduce the paper results. Then run:

```bash
python scripts/predict.py \
  --image path/to/image.jpg \
  --detector-ckpt models/smoke/detector_best.pth \
  --scorer-ckpt models/smoke/scorer_best.pth \
  --k 3 \
  --output-dir predictions
```

The command writes `predictions/annotated.png` and `predictions/predictions.json`. Set `--k 0` to run the coarse pass without native-resolution tiles. The command uses CPU by default. Add `--device cuda` when CUDA is available.

The command uses the experiment's size-aware fusion rule. It keeps coarse boxes at full weight when their longer side is at least two times the tile overlap.

## Paper source

Upload `paper/` as a ZIP project to Overleaf. The source uses `main.tex`, `references.bib`, and the `generated/` and `figures/` folders. The authors are Abhiram Adabala (23BCE8643) and Dheeraj Karukola (24BCA7891) of VIT-AP University.

## Data and results

The experiment code uses VisDrone-DET. The dataset is not included. Download instructions and the archive source appear in [docs/DATASETS.md](docs/DATASETS.md). The paper discloses limits in checkpoint provenance and experiment reproduction.

The earlier demo used ESOD code and weights. This repository's prediction command uses BCRS code. The ESOD demo and its weights are not part of this repository.
