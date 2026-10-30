"""Evaluate a saved checkpoint with the current evaluation protocol.

    python scripts/eval_ckpt.py --ckpt runs/p1_uniform_s4/best.pth \
        [--split-dir data/visdrone/VisDrone2019-DET-val] [--downsample 4.0] \
        [--out results/p1_uniform_s4/reeval_val.json]

Config (arch, downsample, batch size) defaults to what the checkpoint was
trained with; flags override. Used to re-score old runs under the single
shared protocol (PLAN.md section 6 rule 2) and for test-dev freezes.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bcrs.data import VisDroneDataset, collate_fn
from bcrs.evaluate import evaluate_model
from bcrs.models.detector import build_detector_from_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split-dir", default=None)
    ap.add_argument("--downsample", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--topk-candidates", type=int, default=1000,
                    help="pre-NMS candidates kept per FPN level (dense scenes may need more)")
    ap.add_argument("--detections-per-img", type=int, default=500)
    ap.add_argument("--legacy-bf16-decode", action="store_true",
                    help="reproduce pre-2026-10-05 numbers (bf16 box decode and scores under autocast)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    split_dir = Path(args.split_dir or cfg["val_dir"])
    downsample = args.downsample if args.downsample is not None else float(cfg["downsample"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detector_from_ckpt(ckpt, topk_candidates=args.topk_candidates,
                                     detections_per_img=args.detections_per_img,
                                     legacy_bf16_decode=args.legacy_bf16_decode).to(device)
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None

    ds = VisDroneDataset(split_dir, downsample=downsample, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    metrics = evaluate_model(model, dl, device, split_dir / "coco.json", amp_dtype=amp_dtype)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    out = {
        "ckpt": str(args.ckpt),
        "ckpt_epoch": ckpt.get("epoch"),
        "eval_commit": commit,
        "split_dir": str(split_dir),
        "downsample": downsample,
        "topk_candidates": args.topk_candidates,
        "detections_per_img": args.detections_per_img,
        "box_decode": "bf16-legacy" if args.legacy_bf16_decode else "fp32",
        "metrics": metrics,
    }
    print(json.dumps(metrics, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
