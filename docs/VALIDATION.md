# Submission checks

Date: 2026-10-30. These checks apply to the submission source and the recovered final seed-0 BCRS v3 checkpoints. We did not repeat the complete paper experiments.

- The CPU regression suite completed with 55 passes and two CUDA-only skips. This includes the reserved-background-label regression and a synthetic dataset fixture.
- The test environment used Python 3.12.13, torch 2.6.0+cpu, torchvision 0.21.0+cpu, and pycocotools 2.0.11.
- Strict detector and scorer loads completed with `weights_only=True`. Both record epoch 11 and identical joint configs. The model and head use float32 box decoding.
- Image inference with the recovered final detector and scorer completed at k=3 and selected three tiles. The command wrote an annotated image and valid detection JSON.
- Remote and local checkpoint hashes agree. Run config, metrics, and log files match the archived copies. Four model source files also match the current training project. See `models/PROVENANCE.json`.
- A fresh pdfLaTeX/BibTeX build produced the ten-page paper with both student names, roll IDs, and VIT-AP University. The log contains no overfull boxes, undefined references, undefined citations, or rerun requests. All fonts are embedded.
- The public export omits private Git history, credentials, environments, datasets, the ESOD demo, unrelated local files, and agent-operation files.
- The timing JSON redaction changes process names and paths only. All numerical metrics remain unchanged.

Run the CPU checks from the project root:

```sh
python -m pip install -e .
PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests
```

Build the paper on a TeX system with IEEEtran:

```sh
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The Google Form requires sign-in. Its required fields and upload limits remain unknown. The archive does not include the VisDrone dataset or the original server environment. The packaged model is seed 0, not the three-seed average.
