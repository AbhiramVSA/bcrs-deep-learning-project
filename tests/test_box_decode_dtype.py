"""Regression test for bf16 box decoding (found 2026-10-05).

torchvision's BoxLinearCoder.decode casts the anchors to the regression
output's dtype. Under bf16 autocast every decoded coordinate is bf16: x above
1024 lands on an 8 px grid, which costs small objects their IoU on whole
native-resolution frames but not on 512 px tiles. The same decode feeds the
GIoU loss in training. The detector must decode and post-process in fp32.

The tests feed synthetic bf16 head outputs straight into post-processing, so
they exercise the exact code path without a (slow, on CPU) bf16 backbone pass.
"""
import torch
from torchvision.models.detection.image_list import ImageList

from bcrs.models import build_detector
from bcrs.models.detector import Float32BoxCoder

H, W = 384, 1408  # wide enough that many boxes land at x > 1024 (bf16 spacing 8 px)
STRIDES = [8, 16, 32, 64, 128]


def _postprocess(legacy: bool) -> dict:
    torch.manual_seed(0)
    model = build_detector("fcos_r18", pretrained_backbone=False, score_thresh=0.0,
                           legacy_bf16_decode=legacy).eval()
    feats = [torch.zeros(1, 256, H // s, W // s) for s in STRIDES]
    anchors = model.anchor_generator(ImageList(torch.zeros(1, 3, H, W), [(H, W)]), feats)
    n_per_level = [f.shape[2] * f.shape[3] for f in feats]
    n = sum(n_per_level)
    head = {  # what the head emits under bf16 autocast
        "cls_logits": torch.randn(1, n, 11).bfloat16(),
        "bbox_regression": torch.rand(1, n, 4).bfloat16() * 2,
        "bbox_ctrness": torch.randn(1, n, 1).bfloat16(),
    }
    split_head = {k: list(v.split(n_per_level, dim=1)) for k, v in head.items()}
    split_anchors = [list(a.split(n_per_level)) for a in anchors]
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        return model.postprocess_detections(split_head, split_anchors, [(H, W)])[0]


def _far_right_x(det: dict) -> torch.Tensor:
    xs = det["boxes"][:, [0, 2]].flatten().float()
    return xs[xs > 1040]


def test_fp32_postprocess_gives_fp32_boxes_and_scores_off_the_bf16_grid():
    det = _postprocess(legacy=False)
    assert det["boxes"].dtype == torch.float32
    assert det["scores"].dtype == torch.float32
    xs = _far_right_x(det)
    assert xs.numel() > 50
    on_grid = ((xs % 8) == 0).float().mean().item()
    assert on_grid < 0.2, f"{on_grid:.2f} of x > 1040 still on the bf16 8-px grid"


def test_legacy_mode_reproduces_the_bf16_quantization():
    # proves the test above can detect the defect
    det = _postprocess(legacy=True)
    assert det["boxes"].dtype == torch.bfloat16
    xs = _far_right_x(det)
    assert xs.numel() > 50
    assert ((xs % 8) == 0).all()


def test_training_loss_decodes_in_fp32():
    model = build_detector("fcos_r18", pretrained_backbone=False)
    assert isinstance(model.head.box_coder, Float32BoxCoder)
    assert isinstance(model.box_coder, Float32BoxCoder)
    rel = torch.rand(10, 4, dtype=torch.bfloat16)
    anchors = torch.tensor([[1200.0, 100.0, 1208.0, 108.0]]).repeat(10, 1)
    out = model.head.box_coder.decode(rel, anchors)
    assert out.dtype == torch.float32
    # an fp32 decode keeps sub-pixel offsets that bf16 would round to 8 px
    assert ((out[:, [0, 2]] % 8) != 0).any()


def test_legacy_flag_does_not_change_weights_or_state_dict_keys():
    a = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=True)
    b = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=True, legacy_bf16_decode=True)
    assert a.state_dict().keys() == b.state_dict().keys()
    b.load_state_dict(a.state_dict(), strict=True)
