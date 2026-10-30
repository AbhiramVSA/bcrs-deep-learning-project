"""Detector builders: FCOS-style anchor-free detector on ResNet-FPN (P3-P7).

Convention: num_classes = 11 (index 0 unused background, 1..10 = VisDrone
category ids) so dataset labels and COCO category ids coincide — mirrors the
torchvision COCO convention and removes an id-mapping failure mode.

The model's internal GeneralizedRCNNTransform is replaced by an identity-resize
passthrough: all resolution decisions live in the data pipeline / BCRS logic.
"""
from __future__ import annotations

import torch
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from torchvision.models.detection._utils import BoxLinearCoder
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.fcos import FCOS
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

NUM_CLASSES = 11  # 0 = unused background, 1..10 = VisDrone categories


class IdentityResizeTransform(GeneralizedRCNNTransform):
    """Normalize + batch (pad to size_divisible) but never resize."""

    def resize(self, image, target=None):
        return image, target


class Float32AnchorGenerator(AnchorGenerator):
    """Build the anchor grid in fp32 regardless of the feature-map dtype.

    torchvision derives the anchor dtype from the feature maps, so under bf16
    autocast the grid is built in bf16. bf16 keeps 8 mantissa bits: above
    coordinate 1024 its spacing is 8 px, so stride-8 anchor edges (8k +/- 4)
    round onto multiples of 8 and ~10% of anchors collapse to zero size. The
    box coder divides by anchor size (normalize_by_size) -> inf/NaN regression
    targets -> NaN centerness loss from the first iteration. Observed on
    VisDrone at native resolution (coords > 1024); s=2 (max coord 1000) and
    s=4 were unaffected. See tests/test_anchor_dtype.py.
    """

    def forward(self, image_list, feature_maps):
        # AnchorGenerator.forward reads only shape[-2:], dtype and device of the
        # feature maps, so empty fp32 stand-ins avoid copying anything.
        stand_ins = [f.new_empty((0, *f.shape[-2:]), dtype=torch.float32) for f in feature_maps]
        return super().forward(image_list, stand_ins)


class Float32BoxCoder(BoxLinearCoder):
    """Encode and decode boxes in fp32 regardless of the head-output dtype.

    torchvision's BoxLinearCoder.decode casts the anchors to the dtype of the
    regression output. Under bf16 autocast that output is bf16, so every
    decoded coordinate is rounded to bf16: multiples of 8 px above 1024,
    multiples of 4 px in [512, 1024), multiples of 2 px in [256, 512). A
    whole 1360 px frame is therefore localized 2-4x more coarsely than a
    512 px tile, which biases every whole-image vs tile comparison toward
    tiles. The same decode feeds the GIoU regression loss in
    FCOSHead.compute_loss, so training at native resolution was affected too.
    Measured 2026-10-05: 100% of x > 1024 coordinates on the 8 px grid.
    See tests/test_box_decode_dtype.py.
    """

    def encode(self, reference_boxes, proposals):
        return super().encode(reference_boxes.float(), proposals.float())

    def decode(self, rel_codes, boxes):
        return super().decode(rel_codes.float(), boxes.float())


class FCOSFp32Post(FCOS):
    """FCOS whose post-processing (scores, box decode, clipping, NMS) runs in
    fp32 outside autocast. Under bf16 autocast torchvision computes the
    detection scores as bf16 too (sigmoid and sqrt keep the input dtype), which
    quantizes them to about three significant digits and creates ties in the
    score ranking that COCO AP is computed from."""

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        head_outputs = {k: [t.float() for t in v] for k, v in head_outputs.items()}
        anchors = [[a.float() for a in per_image] for per_image in anchors]
        device_type = head_outputs["cls_logits"][0].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            return super().postprocess_detections(head_outputs, anchors, image_shapes)


def build_detector(
    arch: str = "fcos_r18",
    pretrained_backbone: bool = True,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.6,
    detections_per_img: int = 500,
    topk_candidates: int = 1000,
    frozen_bn: bool = False,
    legacy_bf16_decode: bool = False,
) -> FCOS:
    """frozen_bn: replace backbone BatchNorm with FrozenBatchNorm2d (ImageNet
    statistics kept fixed). Standard for detection fine-tuning — trainable BN
    with the small per-GPU batches that high-resolution inputs force (4 images
    at native VisDrone resolution) learns noisy statistics.

    legacy_bf16_decode: reproduce the pre-2026-10-05 behaviour (stock
    torchvision box coder and post-processing, bf16 under autocast). Only for
    reproducing old numbers; every new run and evaluation uses fp32."""
    norm_layer = FrozenBatchNorm2d if frozen_bn else torch.nn.BatchNorm2d
    if arch == "fcos_r18":
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
                       norm_layer=norm_layer)
    elif arch == "fcos_r50":
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None,
                       norm_layer=norm_layer)
    else:
        raise ValueError(f"unknown arch: {arch}")

    backbone = _resnet_fpn_extractor(
        net,
        trainable_layers=5,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    # same geometry as torchvision's FCOS default (one square anchor per
    # location, sizes 8..128 over P3..P7), built in fp32 under any autocast
    anchor_generator = Float32AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)), aspect_ratios=((1.0,),) * 5,
    )
    model = (FCOS if legacy_bf16_decode else FCOSFp32Post)(
        backbone,
        num_classes=NUM_CLASSES,
        anchor_generator=anchor_generator,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
        topk_candidates=topk_candidates,
    )
    if not legacy_bf16_decode:
        # the model's coder decodes at inference, the head's feeds the GIoU loss
        model.box_coder = Float32BoxCoder(normalize_by_size=True)
        model.head.box_coder = Float32BoxCoder(normalize_by_size=True)
    # Identity resize: keep torchvision's normalization stats & batching.
    model.transform = IdentityResizeTransform(
        min_size=800,  # unused (resize is identity) but required by ctor
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        size_divisible=32,
    )
    return model


def build_detector_from_ckpt(ckpt: dict, **overrides) -> FCOS:
    """Rebuild the exact architecture a checkpoint was trained with (arch and
    frozen_bn come from its saved config; BN and FrozenBN checkpoints have
    different state-dict keys) and load the weights strictly."""
    cfg = ckpt["cfg"]
    kwargs = {"arch": cfg["arch"], "frozen_bn": bool(cfg.get("frozen_bn", False)),
              "pretrained_backbone": False}
    kwargs.update(overrides)
    model = build_detector(**kwargs)
    model.load_state_dict(ckpt["model"])
    return model


@torch.no_grad()
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
