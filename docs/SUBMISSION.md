# Submission files

This repository contains the BCRS source, experiment configurations, archived results, paper PDF, and LaTeX source. The PDF file is `paper/main.pdf`. The Overleaf files are in `paper/`.

To import the paper, create a new Overleaf project from a ZIP that contains `main.tex`, `references.bib`, `generated/`, and `figures/` at its root. The files in `paper/` form that source set. Keep the PDF as a separate file for direct upload.

The source includes `scripts/predict.py` for BCRS inference. The full submission ZIP includes smoke-test weights at `models/smoke/`. These weights show command flow only. They do not support the paper's reported results. The final v3 detector and scorer checkpoints were not present in the local project artifacts.

The paper lists Abhiram Adabala (23BCE8643) and Dheeraj Karukola (24BCA7891) of VIT-AP University. It does not list a department or location.

The Google Form link returned HTTP 401 during read-only inspection. The form fields, file-size limits, and required documents are therefore unknown. Check the form and its size limits before upload. Do not use this note as proof that the form accepts these files.

The latency result files include GPU process inventories from the experiment host. This repository replaces each process name and path with `<redacted-process>`. It retains each process ID, process count, memory value, and all experiment metrics. The paper's reported measurements do not use the process names or paths.

The full submission ZIP includes smoke-test weights at `models/smoke/detector_best.pth` and `models/smoke/scorer_best.pth`. These weights support a command-flow check only. They do not reproduce the paper results. The final v3 weights are missing.

The initial project Git revision was `208ce2b9cbd9ae58afd9bca313c46a4026184e99`. The release snapshot includes the final paper and research files from the local working tree. It does not import the original Git history.
