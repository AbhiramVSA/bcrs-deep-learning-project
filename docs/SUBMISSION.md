# Submission files

This repository contains the BCRS source, experiment configurations, archived results, paper PDF, and LaTeX source. The PDF file is `paper/main.pdf`. The Overleaf files are in `paper/`.

To import the paper, create a new Overleaf project from a ZIP that contains `main.tex`, `references.bib`, `generated/`, and `figures/` at its root. The files in `paper/` form that source set. Keep the PDF as a separate file for direct upload.

The source includes `scripts/predict.py` for BCRS inference. The final v3 detector and scorer checkpoints were not present in the local project artifacts. This source package therefore does not provide the model weights needed for BCRS inference. The available local checkpoints are smoke-test weights, trained on a small image subset. They do not support the paper's reported results.

The manuscript still contains author affiliation fields. Replace those fields with the approved institution and department before final paper submission.

The Google Form link returned HTTP 401 during read-only inspection. The form fields, file-size limits, and required documents are therefore unknown. Check the form and its size limits before upload. Do not use this note as proof that the form accepts these files.

The latency result files include GPU process inventories from the experiment host. This repository replaces each process name and path with `<redacted-process>`. It retains each process ID, process count, memory value, and all experiment metrics. The paper's reported measurements do not use the process names or paths.

The initial project Git revision was `208ce2b9cbd9ae58afd9bca313c46a4026184e99`. The release snapshot includes the final paper and research files from the local working tree. It does not import the original Git history.
