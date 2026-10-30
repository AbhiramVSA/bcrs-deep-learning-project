# Submission checks

Date: 2026-10-30. These checks apply to the submission source and the available smoke checkpoints. They do not verify the archived paper accuracy.

- The CPU regression suite completed with 54 passes and two CUDA-only skips.
- The test environment used Python 3.12.13, torch 2.6.0+cpu, torchvision 0.21.0+cpu, and pycocotools 2.0.11.
- Image inference with the joint smoke detector and scorer completed at k=3 and k=0. The k=3 run selected three tiles. The k=0 run selected no tiles.
- Strict checkpoint loads used `weights_only=True`. The detector uses float32 box decoding and post-processing.
- A fresh pdfLaTeX/BibTeX build produced the ten-page paper. The log contains no overfull boxes, undefined references, undefined citations, or rerun requests. All fonts are embedded.
- The public export omits Git history, credentials, environments, datasets, the ESOD demo, unrelated local files, and agent-operation files.
- The timing JSON redaction changes process names and paths only. The numerical metrics remain unchanged.

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

The final v3 checkpoints and the original experiment environment are unavailable. The Google Form requires sign-in. Its fields and upload limits remain unknown. Team identifiers and institution details require the student's own form entries. The paper retains the known author name and email and omits an unknown affiliation.
