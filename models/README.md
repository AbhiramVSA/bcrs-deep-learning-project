# Model files

The full project ZIP contains `v3/detector_best.pth` and `v3/scorer_best.pth`. The source ZIP contains the model inventory only. Download the full project ZIP from the [submission release](https://github.com/AbhiramVSA/bcrs-deep-learning-project/releases/tag/submission-2026-10-30).

These are the final seed-0 joint BCRS v3 checkpoints from `p2c_bcrs_joint_v3`. Both files record epoch 11, the final epoch of the 12-epoch joint run. The saved detector and scorer configs agree. The detector uses FCOS ResNet-18 with frozen batch normalization. The coarse scale is four, the tile size is 512, the overlap is 64, and the budget divisor is 20.

The archived seed-0 three-tile AP is 19.84. The paper's 19.92 AP is the mean of three seeds. This pair represents seed 0 only. We did not repeat the complete accuracy or latency experiments for this release.

We recovered the original files from the training project. Remote and local SHA-256 values agree. The recovered run config, metrics, and training log match the local archives byte for byte. Current source hashes for the detector, scorer, two-pass inference, and tiles also agree. These checks establish file identity at recovery. They do not recreate the historical server environment.

Verify the hashes in `CHECKPOINTS.json` before use. The loader uses `weights_only=True` and strict state loads. See `PROVENANCE.json` for the source and run metadata.
