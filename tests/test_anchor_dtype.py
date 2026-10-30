"""Regression test for the s=1 NaN (root cause: anchors built in bf16).

torchvision builds the anchor grid in the feature-map dtype. Under bf16
autocast, coordinates above 1024 have 8-px spacing, so stride-8 anchor edges
(8k +/- 4) round onto multiples of 8 and some anchors collapse to zero size;
the box coder then divides by anchor size and the centerness target goes NaN.
The detector must build anchors in fp32 regardless of autocast.
"""
import torch
from torchvision.models.detection.image_list import ImageList

from bcrs.models import build_detector

LEVEL_SIZES = [8, 16, 32, 64, 128]


def _anchors_for(model, h, w, feat_dtype):
    strides = [8, 16, 32, 64, 128]
    feats = [torch.zeros(1, 256, h // s, w // s, dtype=feat_dtype) for s in strides]
    images = ImageList(torch.zeros(1, 3, h, w), [(h, w)])
    return model.anchor_generator(images, feats)[0]


def test_anchors_are_fp32_and_well_formed_even_with_bf16_features():
    model = build_detector("fcos_r18", pretrained_backbone=False)
    anchors = _anchors_for(model, 1056, 1408, torch.bfloat16)
    assert anchors.dtype == torch.float32
    widths = (anchors[:, 2] - anchors[:, 0]).unique().tolist()
    heights = (anchors[:, 3] - anchors[:, 1]).unique().tolist()
    assert sorted(widths) == LEVEL_SIZES, widths
    assert sorted(heights) == LEVEL_SIZES, heights
    # every anchor must have positive size (no division-by-zero in encode)
    assert ((anchors[:, 2] - anchors[:, 0]) > 0).all()


def test_anchors_identical_regardless_of_feature_dtype():
    model = build_detector("fcos_r18", pretrained_backbone=False)
    a32 = _anchors_for(model, 1056, 1408, torch.float32)
    a16 = _anchors_for(model, 1056, 1408, torch.bfloat16)
    assert torch.equal(a32, a16)
