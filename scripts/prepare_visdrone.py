"""Convert VisDrone-DET annotation txt files to COCO-format JSON.

Usage:
    python scripts/prepare_visdrone.py data/visdrone/VisDrone2019-DET-val
    python scripts/prepare_visdrone.py data/visdrone/VisDrone2019-DET-train

Writes <split_dir>/coco.json.

Protocol (fixed for the whole project, see docs/DATASETS.md):
- categories 1..10 with score flag 1 kept as COCO categories (pedestrian..motor);
- category 0 (ignored regions) AND any category-1..10 box with score flag 0
  (the official VisDrone "ignore this box" marker) are stored per-image under
  image["ignored_regions"] (x,y,w,h) — used at eval time to filter detections,
  never trained on or evaluated against;
- category 11 (others) dropped entirely.
"""
import json
import sys
from pathlib import Path

from PIL import Image

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]  # VisDrone category ids 1..10


def convert_split(split_dir: Path) -> dict:
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    images, annotations = [], []
    ann_id = 1
    n_ignored, n_others, n_score0 = 0, 0, 0

    img_files = sorted(img_dir.glob("*.jpg"))
    assert img_files, f"no images in {img_dir}"

    for img_id, img_path in enumerate(img_files, start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        ignored_regions = []
        ann_path = ann_dir / (img_path.stem + ".txt")
        if ann_path.exists():
            for line in ann_path.read_text().strip().splitlines():
                parts = line.strip().rstrip(",").split(",")
                if len(parts) < 6:
                    continue
                x, y, bw, bh = (int(v) for v in parts[:4])
                score, cat = int(parts[4]), int(parts[5])
                if bw <= 0 or bh <= 0:
                    continue
                if cat == 0:
                    ignored_regions.append([x, y, bw, bh])
                    n_ignored += 1
                    continue
                if cat == 11:
                    n_others += 1
                    continue
                if not (1 <= cat <= 10):
                    continue
                if score == 0:  # official per-box ignore flag
                    ignored_regions.append([x, y, bw, bh])
                    n_score0 += 1
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
            "ignored_regions": ignored_regions,
        })

    coco = {
        "info": {"description": f"VisDrone-DET {split_dir.name} (BCRS protocol)"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name} for i, name in enumerate(CLASSES)
        ],
    }
    print(f"{split_dir.name}: {len(images)} images, {len(annotations)} boxes, "
          f"{n_ignored} ignored-region entries, {n_score0} score-0 boxes ignored, "
          f"{n_others} 'others' dropped")
    return coco


def main() -> None:
    split_dir = Path(sys.argv[1])
    assert split_dir.is_dir(), f"not a directory: {split_dir}"
    coco = convert_split(split_dir)
    out = split_dir / "coco.json"
    out.write_text(json.dumps(coco))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
