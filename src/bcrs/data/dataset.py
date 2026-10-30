"""VisDrone detection dataset (COCO-format JSON produced by scripts/prepare_visdrone.py).

Resolution policy: ALL resizing happens here (downsample factor `s`), never
inside the model — the model's internal transform is an identity-resize
passthrough (see bcrs.models.detector). Boxes/targets are in resized coords;
evaluation maps predictions back to native coords via the per-image meta.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class VisDroneDataset(Dataset):
    """Returns (image_tensor [3,H,W] float in [0,1], target dict).

    target: boxes (xyxy, resized coords), labels (VisDrone cat ids 1..10),
            image_id, orig_size (w,h), input_size (w,h).
    """

    def __init__(
        self,
        split_dir: str | Path,
        downsample: float = 1.0,
        train: bool = False,
        hflip_p: float = 0.5,
        max_images: int | None = None,
        seed: int = 0,
        scale_jitter: tuple[float, float] | None = None,
    ) -> None:
        """scale_jitter=(lo, hi): training-only random resize factor f~U(lo,hi)
        applied on top of `downsample` (effective factor downsample/f), so the
        detector sees each object at a range of scales (SNIP/YOLO-style
        multi-scale training). Boxes follow the same factor."""
        self.split_dir = Path(split_dir)
        self.img_dir = self.split_dir / "images"
        self.downsample = float(downsample)
        self.train = train
        self.hflip_p = hflip_p if train else 0.0
        self.scale_jitter = tuple(scale_jitter) if (train and scale_jitter) else None

        coco = json.loads((self.split_dir / "coco.json").read_text())
        self.images = coco["images"]
        if max_images is not None:
            rng = random.Random(seed)
            self.images = sorted(self.images, key=lambda im: im["id"])
            rng.shuffle(self.images)
            self.images = self.images[:max_images]
        self.anns_by_img: dict[int, list] = {im["id"]: [] for im in self.images}
        keep = set(self.anns_by_img)
        for ann in coco["annotations"]:
            if ann["image_id"] in keep:
                self.anns_by_img[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        img = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        ow, oh = img.size
        factor = self.downsample
        if self.scale_jitter is not None:
            factor = factor / random.uniform(*self.scale_jitter)
        if factor != 1.0:
            nw, nh = max(1, round(ow / factor)), max(1, round(oh / factor))
            img = img.resize((nw, nh), Image.BILINEAR)
        else:
            nw, nh = ow, oh
        sx, sy = nw / ow, nh / oh

        boxes, labels = [], []
        for ann in self.anns_by_img[info["id"]]:
            x, y, w, h = ann["bbox"]
            x0, y0 = x * sx, y * sy
            x1, y1 = (x + w) * sx, (y + h) * sy
            if x1 - x0 >= 1.0 and y1 - y0 >= 1.0:
                boxes.append([x0, y0, x1, y1])
                labels.append(ann["category_id"])

        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        if self.hflip_p > 0 and random.random() < self.hflip_p:
            tensor = torch.flip(tensor, dims=[2])
            if boxes_t.numel():
                x0 = boxes_t[:, 0].clone()
                boxes_t[:, 0] = nw - boxes_t[:, 2]
                boxes_t[:, 2] = nw - x0

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor(info["id"]),
            "orig_size": torch.tensor([ow, oh]),
            "input_size": torch.tensor([nw, nh]),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
