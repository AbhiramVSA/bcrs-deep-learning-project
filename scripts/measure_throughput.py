"""GPU throughput of dense inference vs BCRS with B frames per call.

    python scripts/measure_throughput.py --ckpt runs/p2c_bcrs_joint_v3/detector_best.pth \
        --scorer-ckpt runs/p2c_bcrs_joint_v3/scorer_best.pth --batch 8 \
        --out results/latency_v3/gpu_throughput_b8.json

Why. At batch 1 a ResNet-18 FCOS on an A100 is bound by kernel launches and
post-processing: every input size takes about 30 ms (results/latency), so the
operation count of the paper's cost axis never reaches the clock. This script
measures the regime where computation dominates. Frames of one size (the most
common one in the split, 1360x765 on VisDrone val) are processed B per call;
for BCRS the B coarse passes are one call and all B*k selected tiles are one
batched call (chunks of --tile-chunk). Frames are device-resident before
timing; detections are moved to the host at the end of every mode, because
fusion runs on the host.

Modes: dense_s{1,2,4} (full detector incl. post-processing), bcrs_k{K} (coarse
pass, scorer, top-k, batched fine pass, fusion), and fwd_* variants with
backbone + head only (no post-processing), which isolate network compute.
Reports median ms per frame over batches after warm-up, and frames per second.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path

import torch

from bcrs.data import VisDroneDataset
from bcrs.models.detector import build_detector_from_ckpt
from bcrs.tiles import crop_tile, tile_boxes_to_frame, tile_grid, truncated_at_tile_border
from bcrs.twopass import downsample_image, fcos_forward_with_features, fuse


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def gpu_snapshot() -> dict:
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,clocks.sm",
             "--format=csv,noheader"], text=True, timeout=10).strip()
        p = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, timeout=10).strip()
        return {"query_gpu": q, "compute_apps": [ln for ln in p.splitlines() if ln.strip()]}
    except Exception as e:
        return {"error": str(e)}


def forward_only(det, imgs, amp):
    with torch.autocast(device_type=imgs[0].device.type, dtype=amp, enabled=amp is not None):
        il, _ = det.transform(imgs, None)
        feats = det.backbone(il.tensors)
        feats = list(feats.values()) if isinstance(feats, dict) else [feats]
        return det.head(feats)


def detect(det, imgs, amp, chunk):
    outs = []
    with torch.autocast(device_type=imgs[0].device.type, dtype=amp, enabled=amp is not None):
        for i in range(0, len(imgs), chunk):
            outs.extend(det(imgs[i:i + chunk]))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scorer-ckpt", required=True)
    ap.add_argument("--split-dir", default="data/visdrone/VisDrone2019-DET-val")
    ap.add_argument("--n-images", type=int, default=96, help="frames of the most common size to use")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tile-chunk", type=int, default=32)
    ap.add_argument("--k-list", default="1,2,3,4,6")
    ap.add_argument("--s", type=float, default=4.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3, help="untimed batches per mode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" else None
    torch.backends.cudnn.benchmark = True  # fixed shapes per mode; warm-up absorbs the autotuning
    snap_before = gpu_snapshot()

    det = build_detector_from_ckpt(torch.load(args.ckpt, map_location="cpu", weights_only=False)).to(device).eval()
    from bcrs.train_bcrs import build_scorer
    sck = torch.load(args.scorer_ckpt, map_location="cpu", weights_only=False)
    scorer = build_scorer(sck["cfg"]).to(device).eval()
    scorer.load_state_dict(sck["scorer"])
    bnk = float(sck["cfg"].get("budget_norm_k") or 20)
    ks = [int(x) for x in args.k_list.split(",")]

    ds = VisDroneDataset(args.split_dir, downsample=1.0, train=False)
    # most common frame size in the split (1360x765 on VisDrone val: 408 of 548 frames)
    h, w = Counter((int(i["height"]), int(i["width"])) for i in ds.images).most_common(1)[0][0]
    idxs = [i for i, im in enumerate(ds.images) if (int(im["height"]), int(im["width"])) == (h, w)]
    frames = []
    for idx in idxs[: args.n_images]:
        img, _ = ds[idx]
        assert tuple(img.shape[-2:]) == (h, w), (img.shape, h, w)
        frames.append(img.to(device))
    nb = len(frames) // args.batch
    frames = frames[: nb * args.batch]
    batches = [frames[i * args.batch:(i + 1) * args.batch] for i in range(nb)]
    tiles = tile_grid(w, h, args.tile, args.overlap)
    print(f"{len(frames)} frames of {w}x{h}, {nb} batches of {args.batch}, {len(tiles)} tiles per frame", flush=True)

    def run_mode(fn) -> dict:
        times = []
        for i, b in enumerate(batches):
            sync(device)
            t0 = time.perf_counter()
            fn(b)
            sync(device)
            if i >= args.warmup:
                times.append((time.perf_counter() - t0) * 1000.0 / len(b))
        med = statistics.median(times)
        return {"ms_per_frame_median": round(med, 3), "ms_per_frame_mean": round(statistics.fmean(times), 3),
                "fps": round(1000.0 / med, 2), "n_batches": len(times)}

    def dense(s):
        def fn(b):
            imgs = [downsample_image(f, s) for f in b]
            outs = detect(det, imgs, amp, chunk=args.batch)
            return [{k: v.float().cpu() for k, v in o.items()} for o in outs]
        return fn

    def dense_fwd(s):
        return lambda b: forward_only(det, [downsample_image(f, s) for f in b], amp)

    def bcrs(k):
        kk = min(k, len(tiles))

        def fn(b):
            smalls = [downsample_image(f, args.s) for f in b]
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                cdets, feats = fcos_forward_with_features(det, smalls)
            budget = torch.full((len(b),), kk / bnk, device=device)
            scores = scorer(feats[-1], budget, [tiles] * len(b), args.s * 128.0)
            sel = [torch.argsort(sc.float(), descending=True)[:kk].cpu() for sc in scores]
            crops = [crop_tile(f, tiles[i]) for f, si in zip(b, sel) for i in si.tolist()]
            touts = detect(det, crops, amp, chunk=args.tile_chunk)
            fused = []
            for j, (f, si) in enumerate(zip(b, sel)):
                c = {k2: v.float().cpu() for k2, v in cdets[j].items()}
                sh, sw = smalls[j].shape[-2:]
                c["boxes"][:, [0, 2]] *= w / sw
                c["boxes"][:, [1, 3]] *= h / sh
                per_tile = []
                for n, i in enumerate(si.tolist()):
                    o = {k2: v.float().cpu() for k2, v in touts[j * kk + n].items()}
                    t = tiles[i]
                    per_tile.append({"boxes": tile_boxes_to_frame(o["boxes"], t), "scores": o["scores"],
                                     "labels": o["labels"],
                                     "truncated": truncated_at_tile_border(o["boxes"], t, img_w=w, img_h=h)})
                fused.append(fuse(c, per_tile, torch.arange(len(si)), tiles[si], coarse_full_min_side=128.0))
            return fused
        return fn

    def bcrs_fwd(k):
        kk = min(k, len(tiles))
        sel = torch.arange(kk)  # which tiles does not change the compute

        def fn(b):
            forward_only(det, [downsample_image(f, args.s) for f in b], amp)
            crops = [crop_tile(f, tiles[i]) for f in b for i in sel.tolist()]
            for i in range(0, len(crops), args.tile_chunk):
                forward_only(det, crops[i:i + args.tile_chunk], amp)
        return fn

    modes: dict[str, dict] = {}
    with torch.no_grad():
        for s in (1.0, 2.0, 4.0):
            modes[f"dense_s{int(s)}"] = run_mode(dense(s))
            modes[f"fwd_dense_s{int(s)}"] = run_mode(dense_fwd(s))
        for k in ks:
            modes[f"bcrs_k{k}"] = run_mode(bcrs(k))
            modes[f"fwd_bcrs_k{k}"] = run_mode(bcrs_fwd(k))
            print(f"k={k} done", flush=True)

    out = {"gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "torch": torch.__version__, "batch": args.batch, "frame_hw": [h, w], "n_frames": len(frames),
           "tiles_per_frame": len(tiles), "geometry": vars(args),
           "protocol": "frames device-resident; B frames per call; BCRS tiles of all B frames batched; "
                       "cudnn.benchmark on with untimed warm-up batches; detections moved to host; "
                       "median ms per frame over batches",
           "gpu_state_before": snap_before, "gpu_state_after": gpu_snapshot(), "modes": modes}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    for k, v in modes.items():
        print(f"{k:<14} {v['ms_per_frame_median']:8.2f} ms/frame   {v['fps']:7.1f} frames/s")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
