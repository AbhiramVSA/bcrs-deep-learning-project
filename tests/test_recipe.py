"""Tests for the base-recipe upgrades: scale-jitter augmentation and weight EMA."""
import json
import math
import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from bcrs.data import VisDroneDataset
from bcrs.models import build_detector
from bcrs.train import ModelEMA

@pytest.fixture
def val_dir(tmp_path: Path) -> Path:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (200, 160), color=(30, 60, 90)).save(image_dir / "sample.png")
    (tmp_path / "coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "sample.png", "width": 200, "height": 160}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [20, 20, 100, 80], "area": 8000, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }))
    return tmp_path


def test_scale_jitter_changes_size_and_keeps_boxes_consistent(val_dir):
    torch.manual_seed(0)
    random.seed(0)
    ds = VisDroneDataset(val_dir, downsample=2.0, train=True, hflip_p=0.0, scale_jitter=(0.7, 1.4), max_images=1)
    base = VisDroneDataset(val_dir, downsample=2.0, train=False, max_images=1)
    img_b, t_b = base[0]
    sizes = set()
    for _ in range(6):
        img, t = ds[0]
        sizes.add(tuple(img.shape[-2:]))
        nw, nh = t["input_size"].tolist()
        assert img.shape[-1] == nw and img.shape[-2] == nh
        # boxes stay inside the resized image and scale with it
        if t["boxes"].numel():
            assert (t["boxes"][:, 2] <= nw + 1).all() and (t["boxes"][:, 3] <= nh + 1).all()
            ratio = nw / t_b["input_size"][0].item()
            # the largest box width should scale by ~ratio (same annotation set modulo the min-size filter)
            assert math.isclose(t["boxes"][:, 2].max().item() / t_b["boxes"][:, 2].max().item(), ratio, rel_tol=0.05)
    assert len(sizes) > 1  # jitter actually varied the size


def test_no_jitter_in_eval_mode(val_dir):
    ds = VisDroneDataset(val_dir, downsample=2.0, train=False, scale_jitter=(0.7, 1.4), max_images=1)
    assert ds.scale_jitter is None
    a, _ = ds[0]
    b, _ = ds[0]
    assert a.shape == b.shape


def test_ema_tracks_model_and_saves_loadable_state():
    model = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=True)
    ema = ModelEMA(model, decay=0.5, tau=1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    # after one update with decay ramp d = 0.5*(1-e^-1) ~ 0.316: ema = d*old + (1-d)*new
    p_model = next(model.parameters())
    p_ema = next(ema.ema.parameters())
    d = 0.5 * (1 - math.exp(-1))
    expected = d * (p_model - 1.0) + (1 - d) * p_model
    assert torch.allclose(p_ema, expected, atol=1e-5)
    # EMA state dict loads strictly into a fresh detector of the same config
    fresh = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=True)
    fresh.load_state_dict(ema.ema.state_dict())
