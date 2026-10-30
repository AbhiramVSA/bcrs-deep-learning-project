# Model files

The full project ZIP contains `smoke/detector_best.pth` and `smoke/scorer_best.pth`. The source ZIP contains this inventory only.

These joint BCRS checkpoints come from `p2c_joint_smoke`, epoch 1. The run used at most 12 training images and eight validation images. It used the validation split for both roles. These weights are for a functional check only. They do not reproduce the paper metrics.

The final v3 checkpoints are missing from the available project files. The archive does not replace them with random weights or ESOD weights.

Download the full project archive from the [submission release](https://github.com/AbhiramVSA/bcrs-deep-learning-project/releases/tag/submission-2026-10-30). Verify its checksum before extraction. See `CHECKPOINTS.json` for each model file hash.
