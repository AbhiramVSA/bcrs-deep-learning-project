"""Single evaluation entry point for ALL methods (PLAN.md section 6, rule 2).

COCO-style AP via pycocotools with maxDets=[10,100,500]; detections falling
mostly inside VisDrone ignored regions are removed before scoring; predictions
are mapped from input (resized) coords back to native coords.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = [10, 100, 500]

METRIC_NAMES = [
    "AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l",
    f"AR@{MAX_DETS[0]}", f"AR@{MAX_DETS[1]}", f"AR@{MAX_DETS[2]}",
    "AR_s", "AR_m", "AR_l",
]


def _filter_ignored(dets: list[dict], ignored_regions_by_img: dict[int, list], thr: float = 0.5) -> list[dict]:
    """Drop detections whose intersection-over-detection-area with any ignored
    region exceeds thr (approximates the official VisDrone toolkit protocol)."""
    kept = []
    for d in dets:
        regions = ignored_regions_by_img.get(d["image_id"], [])
        x, y, w, h = d["bbox"]
        area = max(w * h, 1e-6)
        drop = False
        for rx, ry, rw, rh in regions:
            iw = min(x + w, rx + rw) - max(x, rx)
            ih = min(y + h, ry + rh) - max(y, ry)
            if iw > 0 and ih > 0 and (iw * ih) / area > thr:
                drop = True
                break
        if not drop:
            kept.append(d)
    return kept


@torch.no_grad()
def collect_predictions(model, dataloader, device, amp_dtype=None) -> tuple[list[dict], list[int]]:
    """Run the model over a dataloader. Returns (COCO-format detections in
    NATIVE image coordinates, evaluated image ids)."""
    model.eval()
    dets: list[dict] = []
    img_ids: list[int] = []
    for images, targets in dataloader:
        images = [im.to(device, non_blocking=True) for im in images]
        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None else contextlib.nullcontext()
        )
        with ctx:
            outputs = model(images)
        for out, tgt in zip(outputs, targets):
            ow, oh = tgt["orig_size"].tolist()
            nw, nh = tgt["input_size"].tolist()
            sx, sy = ow / nw, oh / nh
            img_id = int(tgt["image_id"])
            img_ids.append(img_id)
            boxes = out["boxes"].float().cpu().numpy()
            scores = out["scores"].float().cpu().numpy()
            labels = out["labels"].cpu().numpy()
            for (x0, y0, x1, y1), s, lab in zip(boxes, scores, labels):
                if lab < 1:  # unused background channel of the 11-class head
                    continue
                dets.append({
                    "image_id": img_id,
                    "category_id": int(lab),
                    "bbox": [float(x0 * sx), float(y0 * sy),
                             float((x1 - x0) * sx), float((y1 - y0) * sy)],
                    "score": float(s),
                })
    return dets, img_ids


def coco_metrics(gt_json: str | Path, dets: list[dict], img_ids: list[int] | None = None) -> dict:
    """Evaluate detections (native coords) against a prepared coco.json.
    img_ids restricts evaluation to a subset of images (e.g. smoke runs)."""
    gt_raw = json.loads(Path(gt_json).read_text())
    ignored = {im["id"]: im.get("ignored_regions", []) for im in gt_raw["images"]}
    dets = _filter_ignored(dets, ignored)

    # build the COCO object from the already-parsed dict (no second file parse)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt_raw
        coco_gt.createIndex()
    if not dets:
        return {name: 0.0 for name in METRIC_NAMES} | {"n_dets": 0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.maxDets = list(MAX_DETS)
    if img_ids is not None:
        ev.params.imgIds = sorted(set(img_ids))
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats = [float(s) for s in np.asarray(ev.stats).ravel()]
    out = dict(zip(METRIC_NAMES, stats))
    out["n_dets"] = len(dets)
    return out


def evaluate_model(model, dataloader, device, gt_json, amp_dtype=None) -> dict:
    dets, img_ids = collect_predictions(model, dataloader, device, amp_dtype)
    return coco_metrics(gt_json, dets, img_ids=img_ids)
