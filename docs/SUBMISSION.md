# Submission files

This repository contains the BCRS source, experiment configurations, archived results, paper PDF, and LaTeX source. The PDF file is `paper/main.pdf`. The Overleaf files are in `paper/`.

To import the paper, create a new Overleaf project from a ZIP that contains `main.tex`, `references.bib`, `generated/`, and `figures/` at its root. The files in `paper/` form that source set. Keep the PDF as a separate file for direct upload.

The source includes `scripts/predict.py` for BCRS inference. The Git repository and source ZIP do not contain model weights. The [full submission ZIP](https://github.com/AbhiramVSA/bcrs-deep-learning-project/releases/tag/submission-2026-10-30) contains `models/v3/detector_best.pth` and `models/v3/scorer_best.pth`.

These files are the seed-0 checkpoints from run `p2c_bcrs_joint_v3`, epoch 11. The archived result for this pair is 19.84 AP at three tiles. This is one seed value. The paper reports a three-seed mean of 19.92 AP. This package does not claim a new evaluation.

The detector checkpoint SHA-256 is `a2889927fcca22547f5c5a428dcc93307abf75acc408cae29d559bbb00499265`. The scorer checkpoint SHA-256 is `6b185a7651ee334de68188a90333e3f859df776795c35d0d0a8850f3b1eb2e05`.

The paper lists Abhiram Adabala (23BCE8643) and Dheeraj Karukola (24BCA7891) of VIT-AP University. It does not list a department or location.

The Google Form link returned HTTP 401 during read-only inspection. The form fields, file-size limits, and required documents are therefore unknown. Check the form and its size limits before upload. Do not use this note as proof that the form accepts these files.

The latency result files include GPU process inventories from the experiment host. This repository replaces each process name and path with `<redacted-process>`. It retains each process ID, process count, memory value, and all experiment metrics. The paper's reported measurements do not use the process names or paths.

The initial project Git revision was `208ce2b9cbd9ae58afd9bca313c46a4026184e99`. The release snapshot includes the final paper and research files from the local working tree. It does not import the original Git history.
