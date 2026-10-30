"""Model-level pieces for joint BCRS training (P2c).

Everything here reuses torchvision's own FCOS matcher/loss (via
detector.compute_loss) so no loss code is re-implemented:

- fcos_train_outputs: one batched train-mode pass returning the batch loss
  dict, FPN features, AND post-processed detections (for self-derived targets).
- tile_targets: GT restricted to a tile (center ownership), tile-local coords.
- per_tile_fine_losses: batched backbone/head over all selected tiles, then a
  per-tile compute_loss call on each tile's slice -> one loss dict per tile
  (normalized by that tile's own foreground count).
- recovered_counts: how many coarse-missed GT in a tile the fine pass found.
"""
from __future__ import annotations

import torch
import torchvision
from torch import Tensor

from bcrs.tiles import boxes_in_tile


def _head_and_anchors(detector, images: list[Tensor]):
    image_list, _ = detector.transform(images, None)
    feats = detector.backbone(image_list.tensors)
    feats = list(feats.values()) if isinstance(feats, dict) else [feats]
    head_outputs = detector.head(feats)
    anchors = detector.anchor_generator(image_list, feats)
    num_anchors_per_level = [x.size(2) * x.size(3) for x in feats]
    return image_list, feats, head_outputs, anchors, num_anchors_per_level


@torch.no_grad()
def _postprocess(detector, head_outputs, anchors, num_anchors_per_level, image_list, original_sizes):
    split_head_outputs = {k: list(v.split(num_anchors_per_level, dim=1)) for k, v in head_outputs.items()}
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
    dets = detector.postprocess_detections(split_head_outputs, split_anchors, image_list.image_sizes)
    return detector.transform.postprocess(dets, image_list.image_sizes, original_sizes)


def fcos_train_outputs(detector, images: list[Tensor], targets: list[dict]):
    """Train-mode FCOS pass: (loss_dict, feats, detections). Detections are
    post-processed under no_grad from the same head outputs (no second
    forward). Requires detector.training == True."""
    assert detector.training
    original_sizes = [img.shape[-2:] for img in images]
    image_list, feats, head_outputs, anchors, napl = _head_and_anchors(detector, images)
    losses = detector.compute_loss(targets, head_outputs, anchors, napl)
    detections = _postprocess(detector, {k: v.detach() for k, v in head_outputs.items()},
                              anchors, napl, image_list, original_sizes)
    return losses, feats, detections


def tile_targets(boxes: Tensor, labels: Tensor, tile: Tensor, min_size: float = 1.0) -> dict:
    """GT owned by the tile (center inside), clipped to it, in tile coords."""
    if boxes.numel() == 0:
        return {"boxes": boxes.new_zeros((0, 4)), "labels": labels.new_zeros((0,))}
    m = boxes_in_tile(boxes, tile)
    b = boxes[m].clone()
    b[:, [0, 2]] -= tile[0]
    b[:, [1, 3]] -= tile[1]
    tw, th = float(tile[2] - tile[0]), float(tile[3] - tile[1])
    b[:, 0::2] = b[:, 0::2].clamp(0.0, tw)
    b[:, 1::2] = b[:, 1::2].clamp(0.0, th)
    keep = ((b[:, 2] - b[:, 0]) >= min_size) & ((b[:, 3] - b[:, 1]) >= min_size)
    return {"boxes": b[keep], "labels": labels[m][keep]}


def per_tile_fine_losses(detector, tile_images: list[Tensor], tile_tgts: list[dict]):
    """One batched backbone/head pass over all tiles, then torchvision's own
    compute_loss per tile. Returns (list of per-tile loss dicts, per-tile
    detections in TILE coords [no_grad], feats)."""
    assert detector.training
    original_sizes = [img.shape[-2:] for img in tile_images]
    image_list, feats, head_outputs, anchors, napl = _head_and_anchors(detector, tile_images)
    losses = []
    for t in range(len(tile_images)):
        ho = {k: v[t:t + 1] for k, v in head_outputs.items()}
        losses.append(detector.compute_loss([tile_tgts[t]], ho, [anchors[t]], napl))
    dets = _postprocess(detector, {k: v.detach() for k, v in head_outputs.items()},
                        anchors, napl, image_list, original_sizes)
    return losses, dets, feats


def recovered_counts(missed_boxes: Tensor, missed_labels: Tensor, tile: Tensor, fine_det: dict,
                     iou_thr: float = 0.5, score_thr: float = 0.3) -> int:
    """Number of coarse-missed GT owned by `tile` that the fine pass detects
    (fine_det in tile coords; missed_* in frame coords)."""
    if missed_boxes.numel() == 0:
        return 0
    m = boxes_in_tile(missed_boxes, tile)
    if not m.any():
        return 0
    gt = missed_boxes[m].clone()
    gt[:, [0, 2]] -= tile[0]
    gt[:, [1, 3]] -= tile[1]
    gl = missed_labels[m]
    keep = fine_det["scores"] >= score_thr
    if not keep.any():
        return 0
    iou = torchvision.ops.box_iou(gt, fine_det["boxes"][keep].to(gt.device))
    same = gl[:, None] == fine_det["labels"][keep].to(gl.device)[None, :]
    return int(((iou >= iou_thr) & same).any(dim=1).sum())
