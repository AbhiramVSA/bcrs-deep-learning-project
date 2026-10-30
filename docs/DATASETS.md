# Datasets

## VisDrone-DET 2019 (primary, committed)

- 6,471 train / 548 val / 1,610 test-dev images, 10 categories, up to 2000×1500 px.
- Official source: https://github.com/VisDrone/VisDrone-Dataset (Google Drive links).
- Scriptable mirror (verified live 2026-09-02): Ultralytics release assets —
  used by `scripts/download_visdrone.sh`:
  - https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip
  - https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip
  - https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip
- Backup mirror: HuggingFace `Voxel51/VisDrone2019-DET`.

Annotation format (per-image `.txt`, one object per line):
`<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>`

Categories: 0=ignored-region, 1=pedestrian, 2=people, 3=bicycle, 4=car, 5=van,
6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor, 11=others.

Protocol decisions (fixed for the whole project):
- Evaluate classes 1–10 (standard VisDrone-DET protocol).
- Ignored regions (cat 0) and class 11: excluded from loss and evaluation;
  detections falling inside ignored regions are not counted as false positives.
- COCO-format conversion via `scripts/prepare_visdrone.py`; eval uses
  pycocotools with maxDets=500.

## UAVDT-M (stretch: cross-dataset transfer)

- ~40k annotated frames, 3 vehicle classes, altitude/weather variation.
- Zenodo: https://zenodo.org/records/14575517 (direct download, scriptable).
- Official page: https://sites.google.com/view/grli-uavdt

## TinyPerson (stretch)

- 1,610 images, 72,651 boxes, maritime persons frequently <20 px.
- Official: https://github.com/ucas-vg/PointTinyBenchmark/tree/TinyBenchmark
  (Google Drive / Baidu links).
