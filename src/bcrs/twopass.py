"""Two-pass (coarse + tiled fine) inference machinery for BCRS.

All boxes returned by these functions are in NATIVE frame coordinates.
The expensive work (one coarse pass + one fine pass over every tile) is done
once per image; fused detection sets for any (ranking, k) are then assembled
from the cached per-tile outputs — this is what makes full-curve sweeps cheap.
"""
from __future__ import annotations

import contextlib

import torch
import torchvision
from torch import Tensor
from torch.nn import functional as F

from bcrs.tiles import assign_boxes_to_tiles, crop_tile, tile_boxes_to_frame, truncated_at_tile_border


def downsample_image(image: Tensor, s: float) -> Tensor:
    """[3,H,W] in [0,1] -> bilinear+antialias downsample by factor s
    (antialias matches the PIL resize used at training time closely)."""
    if s == 1.0:
        return image
    _, h, w = image.shape
    nh, nw = max(1, round(h / s)), max(1, round(w / s))
    return F.interpolate(image[None], size=(nh, nw), mode="bilinear",
                         antialias=True, align_corners=False)[0]


def fcos_forward_with_features(detector, images: list[Tensor], targets: list[dict] | None = None):
    """torchvision FCOS forward, re-expressed to also return the FPN feature
    maps (needed by the tile scorer). Mirrors FCOS.forward exactly:
    train mode + targets -> (loss_dict, features); eval -> (detections, features).
    Features are the backbone's FPN levels in order P3..P7."""
    original_sizes = [img.shape[-2:] for img in images]
    image_list, targets = detector.transform(images, targets)
    feats = detector.backbone(image_list.tensors)
    feats = list(feats.values()) if isinstance(feats, dict) else [feats]
    head_outputs = detector.head(feats)
    anchors = detector.anchor_generator(image_list, feats)
    num_anchors_per_level = [x.size(2) * x.size(3) for x in feats]

    if detector.training:
        assert targets is not None
        losses = detector.compute_loss(targets, head_outputs, anchors, num_anchors_per_level)
        return losses, feats

    split_head_outputs = {
        k: list(v.split(num_anchors_per_level, dim=1)) for k, v in head_outputs.items()
    }
    split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]
    detections = detector.postprocess_detections(split_head_outputs, split_anchors, image_list.image_sizes)
    detections = detector.transform.postprocess(detections, image_list.image_sizes, original_sizes)
    return detections, feats


@torch.no_grad()
def _detect_batched(detector, images: list[Tensor], device, amp_dtype, chunk: int = 8) -> list[dict]:
    """Run the detector over a list of [3,h,w] tensors in chunks; outputs on CPU."""
    detector.eval()
    outs: list[dict] = []
    for i in range(0, len(images), chunk):
        batch = [im.to(device, non_blocking=True) for im in images[i:i + chunk]]
        ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype is not None else contextlib.nullcontext())
        with ctx:
            res = detector(batch)
        outs.extend({k: v.float().cpu() for k, v in r.items()} for r in res)
    return outs


@torch.no_grad()
def coarse_pass(detector, image: Tensor, s: float, device, amp_dtype=None,
                return_features: bool = False):
    """Detect on the s-downsampled image; boxes mapped back to native coords.
    With return_features=True also returns the FPN feature list (P3..P7, on
    device) for the tile scorer."""
    small = downsample_image(image, s)
    detector.eval()
    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
           if amp_dtype is not None else contextlib.nullcontext())
    with ctx:
        dets, feats = fcos_forward_with_features(detector, [small.to(device, non_blocking=True)])
    out = {k: v.float().cpu() for k, v in dets[0].items()}
    _, h, w = image.shape
    _, sh, sw = small.shape
    out["boxes"][:, [0, 2]] *= w / sw
    out["boxes"][:, [1, 3]] *= h / sh
    if return_features:
        return out, feats
    return out


@torch.no_grad()
def tile_pass(detector, image: Tensor, tiles: Tensor, device, amp_dtype=None,
              chunk: int = 8) -> list[dict]:
    """Detect on every native-resolution tile crop. Returns one dict per tile:
    boxes (frame coords), scores, labels, truncated (mask of boxes touching a
    non-image tile border — candidates for suppression at fusion)."""
    _, h, w = image.shape
    crops = [crop_tile(image, t) for t in tiles]
    outs = _detect_batched(detector, crops, device, amp_dtype, chunk=chunk)
    results = []
    for t, out in zip(tiles, outs):
        trunc = truncated_at_tile_border(out["boxes"], t, img_w=w, img_h=h)
        results.append({
            "boxes": tile_boxes_to_frame(out["boxes"], t),
            "scores": out["scores"],
            "labels": out["labels"],
            "truncated": trunc,
        })
    return results


def unmatched_gt_mask(gt_boxes: Tensor, gt_labels: Tensor, det: dict,
                      iou_thr: float = 0.5, score_thr: float = 0.3) -> Tensor:
    """Bool mask over GT boxes NOT matched by any detection of the same class
    with score >= score_thr and IoU >= iou_thr (proposal section 4.5)."""
    g = gt_boxes.shape[0]
    if g == 0:
        return torch.zeros(0, dtype=torch.bool)
    keep = det["scores"] >= score_thr
    if not keep.any():
        return torch.ones(g, dtype=torch.bool)
    d_boxes, d_labels = det["boxes"][keep], det["labels"][keep]
    iou = torchvision.ops.box_iou(gt_boxes, d_boxes)              # [G, D]
    same_class = gt_labels[:, None] == d_labels[None, :]
    matched = ((iou >= iou_thr) & same_class).any(dim=1)
    return ~matched


def oracle_tile_scores(gt_boxes: Tensor, gt_labels: Tensor, coarse: dict,
                       tiles: Tensor, iou_thr: float = 0.5,
                       score_thr: float = 0.3) -> Tensor:
    """Per-tile count of coarse-missed GT objects, each counted once via
    unique tile ownership. This is the selection target BCRS trains toward and
    the upper bound any scorer can achieve."""
    n = tiles.shape[0]
    miss = unmatched_gt_mask(gt_boxes, gt_labels, coarse, iou_thr, score_thr)
    if not miss.any():
        return torch.zeros(n)
    owners = assign_boxes_to_tiles(gt_boxes[miss], tiles)
    return torch.bincount(owners, minlength=n).float()


def fuse(coarse: dict, tile_outs: list[dict], selected: Tensor, tiles: Tensor,
         coarse_in_tile_weight: float = 0.5, nms_thr: float = 0.6,
         max_dets: int = 500, suppress_truncated: bool = True,
         coarse_full_min_side: float | None = None) -> dict:
    """Fuse coarse detections with fine detections from the selected tiles
    (proposal section 4.4).

    - fine boxes truncated at a non-image tile border are dropped (the coarse
      pass or a neighboring tile covers that object better);
    - coarse boxes whose center falls inside any selected tile are
      down-weighted (the fine pass is authoritative there) — EXCEPT, when
      coarse_full_min_side is set, boxes whose longer side is >= that value:
      objects too large for a tile to see whole are the coarse pass's job
      (measured: without this, tiling loses ~2 AP on medium/large objects
      through clipped boxes while AP50 is unchanged);
    - class-aware NMS over the union, top max_dets kept.
    selected: LongTensor of tile indices (possibly empty -> coarse only).
    """
    boxes = [coarse["boxes"]]
    scores = [coarse["scores"].clone()]
    labels = [coarse["labels"]]

    if selected.numel() and coarse["boxes"].numel():
        cb = coarse["boxes"]
        cx = (cb[:, 0] + cb[:, 2]) / 2
        cy = (cb[:, 1] + cb[:, 3]) / 2
        in_sel = torch.zeros(cb.shape[0], dtype=torch.bool)
        for i in selected.tolist():
            t = tiles[i]
            in_sel |= (cx >= t[0]) & (cx < t[2]) & (cy >= t[1]) & (cy < t[3])
        if coarse_full_min_side is not None:
            side = torch.maximum(cb[:, 2] - cb[:, 0], cb[:, 3] - cb[:, 1])
            in_sel &= side < coarse_full_min_side
        scores[0][in_sel] *= coarse_in_tile_weight

    for i in selected.tolist():
        out = tile_outs[i]
        keep = ~out["truncated"] if suppress_truncated else torch.ones_like(out["truncated"])
        boxes.append(out["boxes"][keep])
        scores.append(out["scores"][keep])
        labels.append(out["labels"][keep])

    boxes_c = torch.cat(boxes)
    scores_c = torch.cat(scores)
    labels_c = torch.cat(labels)
    if boxes_c.numel() == 0:
        return {"boxes": boxes_c, "scores": scores_c, "labels": labels_c}
    keep = torchvision.ops.batched_nms(boxes_c, scores_c, labels_c, nms_thr)
    keep = keep[:max_dets]
    return {"boxes": boxes_c[keep], "scores": scores_c[keep], "labels": labels_c[keep]}
