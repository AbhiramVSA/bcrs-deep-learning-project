"""Measured wall-clock latency (batch 1) of BCRS at each tile budget vs uniform
resolution, on the GPU this runs on. Complements the FLOP-based cost axis.

    python scripts/measure_latency.py --ckpt runs/p2c_bcrs_joint_v2/detector_best.pth \
        --scorer-ckpt runs/p2c_bcrs_joint_v2/scorer_best.pth \
        --out results/latency/latency_val.json

Protocol. The decoded frame is moved to the device ONCE before any timed region
(a deployed pipeline hands the detector a device-resident frame; the host-to-
device copy is reported separately as a component). Downsampling and tile
cropping run on the device. For each of N images (fixed order): time the coarse
pass, the scorer, the fine pass at k in K_LIST (selected tiles only, batched),
fusion, and whole-image inference at s in {1,2,4}. Also times backbone+head
alone ("forward-only", no post-processing) for s=1, s=4 and one 512 tile, to
separate network compute from post-processing and launch overhead. Reports
median and p95 in ms after warm-up, with CUDA synchronisation around every
timed region. Records the GPU's other processes at start: run when nothing
else is on the GPU, a shared GPU inflates every number.

--device cpu --threads N times the same pipeline on the CPU in fp32 (no
autocast). There the operation count sets the time, so it is the reference for
the paper's cost axis. Other jobs on a shared CPU inflate the times; the load
average before and after is recorded.

The first version of this script timed the CPU-side antialiased resize inside
the coarse pass, which made s=2 and s=4 slower than s=1 on an A100 (53 ms vs
28 ms). That number is kept as the component `cpu_resize_s4` for reference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from bcrs.data import VisDroneDataset
from bcrs.models.detector import build_detector_from_ckpt
from bcrs.tiles import crop_tile, tile_grid
from bcrs.twopass import coarse_pass, downsample_image, fuse, tile_pass


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1000.0


def summarize(xs: list[float]) -> dict:
    xs = sorted(xs)
    i95 = min(len(xs) - 1, math.ceil(0.95 * (len(xs) - 1)))
    return {"median_ms": round(statistics.median(xs), 3), "p95_ms": round(xs[i95], 3),
            "mean_ms": round(statistics.fmean(xs), 3), "n": len(xs)}


def gpu_snapshot() -> dict:
    """Other processes on the GPU and its utilisation right now (nvidia-smi)."""
    snap: dict = {}
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,clocks.sm,clocks.max.sm",
             "--format=csv,noheader"], text=True, timeout=10).strip()
        snap["query_gpu"] = q
        p = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True, timeout=10).strip()
        snap["compute_apps"] = [ln for ln in p.splitlines() if ln.strip()]
        snap["n_other_processes"] = len(snap["compute_apps"])  # includes this process once it has a context
    except Exception as e:  # nvidia-smi absent (CPU run) or unreadable
        snap["error"] = str(e)
    return snap


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _loadavg() -> list[float] | None:
    """1/5/15-min load average: other jobs on a shared CPU inflate CPU timings."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except Exception:
        return None


def forward_only(det, img: torch.Tensor, amp):
    """backbone + head, no anchors / decoding / NMS."""
    ctx = torch.autocast(device_type=img.device.type, dtype=amp) if amp is not None else torch.autocast("cpu", enabled=False)
    with ctx:
        image_list, _ = det.transform([img], None)
        feats = det.backbone(image_list.tensors)
        feats = list(feats.values()) if isinstance(feats, dict) else [feats]
        return det.head(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scorer-ckpt", required=True)
    ap.add_argument("--split-dir", default="data/visdrone/VisDrone2019-DET-val")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--k-list", default="1,2,3,4,6")
    ap.add_argument("--s", type=float, default=4.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="cpu = compute-bound reference (fp32, no autocast); auto = cuda if available")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (set for --device cpu)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    amp = torch.bfloat16 if device.type == "cuda" else None
    torch.backends.cudnn.benchmark = False  # variable shapes: autotuning per shape would dominate
    snap_before = gpu_snapshot()
    load_before = _loadavg()
    det = build_detector_from_ckpt(torch.load(args.ckpt, map_location="cpu", weights_only=False)).to(device).eval()
    from bcrs.train_bcrs import build_scorer
    sck = torch.load(args.scorer_ckpt, map_location="cpu", weights_only=False)
    scorer = build_scorer(sck["cfg"]).to(device).eval()
    scorer.load_state_dict(sck["scorer"])
    bnk = float(sck["cfg"].get("budget_norm_k") or 20)
    ks = [int(x) for x in args.k_list.split(",")]

    ds = VisDroneDataset(args.split_dir, downsample=1.0, train=False, max_images=args.n_images + args.warmup, seed=0)
    rec = {f"bcrs_k{k}": [] for k in ks}
    rec.update({"coarse_only": [], "scorer": [], "uniform_s1": [], "uniform_s2": [], "uniform_s4": []})
    parts = {f"fine_k{k}": [] for k in ks}
    parts.update({"fuse": [], "h2d_copy": [], "gpu_resize_s4": [], "cpu_resize_s4": [],
                  "forward_only_s1": [], "forward_only_s4": [], "forward_only_tile": []})
    shapes = []

    with torch.no_grad():
        for idx in range(len(ds)):
            image_cpu, _ = ds[idx]
            warm = idx < args.warmup
            _, h, w = image_cpu.shape
            # reference: what the CPU-side antialiased resize alone costs (not part of any pipeline number)
            _, t_cpu_rs = timed(lambda: downsample_image(image_cpu, args.s), device)
            image, t_h2d = timed(lambda: image_cpu.to(device, non_blocking=False), device)
            tiles = tile_grid(w, h, args.tile, args.overlap)
            n = len(tiles)
            # untimed spin-up: the GPU idles (and down-clocks) while the CPU decodes
            # the next JPEG; without this the first timed region per image pays the
            # clock ramp and the ordering of regions biases the comparison
            forward_only(det, downsample_image(image, args.s), amp)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _, t_rs = timed(lambda: downsample_image(image, args.s), device)
            (coarse, feats), t_coarse = timed(lambda: coarse_pass(det, image, args.s, device, amp, return_features=True), device)
            for k in ks:
                kk = min(k, n)
                b = torch.tensor([kk / bnk], device=device)
                sc, t_sc = timed(lambda: scorer(feats[-1], b, [tiles], args.s * 128.0)[0].float().cpu(), device)
                sel = tiles[torch.argsort(sc, descending=True)[:kk]]
                outs, t_fine = timed(lambda: tile_pass(det, image, sel, device, amp, chunk=8), device)
                _, t_fuse = timed(lambda: fuse(coarse, outs, torch.arange(len(sel)), sel, coarse_full_min_side=128.0), device)
                if not warm:
                    rec[f"bcrs_k{k}"].append(t_coarse + t_sc + t_fine + t_fuse)
                    parts[f"fine_k{k}"].append(t_fine)
                    if k == ks[0]:
                        rec["scorer"].append(t_sc)
                        parts["fuse"].append(t_fuse)
            for s in (1.0, 2.0, 4.0):
                _, t_u = timed(lambda: coarse_pass(det, image, s, device, amp), device)
                if not warm:
                    rec[f"uniform_s{int(s)}"].append(t_u)
            small = downsample_image(image, args.s)
            _, t_f1 = timed(lambda: forward_only(det, image, amp), device)
            _, t_f4 = timed(lambda: forward_only(det, small, amp), device)
            _, t_ft = timed(lambda: forward_only(det, crop_tile(image, tiles[0]), amp), device)
            if not warm:
                rec["coarse_only"].append(t_coarse)
                parts["h2d_copy"].append(t_h2d)
                parts["gpu_resize_s4"].append(t_rs)
                parts["cpu_resize_s4"].append(t_cpu_rs)
                parts["forward_only_s1"].append(t_f1)
                parts["forward_only_s4"].append(t_f4)
                parts["forward_only_tile"].append(t_ft)
                shapes.append([h, w, n])
            if (idx + 1) % 25 == 0:
                print(f"[{idx + 1}/{len(ds)}]", flush=True)

    out = {"gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "device": device.type, "threads": torch.get_num_threads() if device.type == "cpu" else None,
           "cpu_model": _cpu_model(),
           "torch": torch.__version__, "n_images": args.n_images, "geometry": vars(args),
           "protocol": ("frame device-resident before timing; resize and crops on device; bf16 autocast; "
                        "cudnn.benchmark off; untimed spin-up forward per image; CUDA-synchronised regions; "
                        "median/p95 over images after warm-up") if device.type == "cuda" else
                       (f"CPU, fp32 (no autocast), {torch.get_num_threads()} torch threads; untimed spin-up "
                        "forward per image; median/p95 over images after warm-up"),
           "loadavg_before": load_before, "loadavg_after": _loadavg(),
           "gpu_state_before": snap_before, "gpu_state_after": gpu_snapshot(),
           "mean_frame_hw_tiles": [round(statistics.fmean(x), 1) for x in zip(*shapes)] if shapes else None,
           "latency": {k: summarize(v) for k, v in rec.items() if v},
           "components": {k: summarize(v) for k, v in parts.items() if v}}
    try:
        out["commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("gpu state before:", snap_before)
    for k, v in out["latency"].items():
        print(f"{k:<16} median {v['median_ms']:7.1f} ms   p95 {v['p95_ms']:7.1f} ms")
    print("components:")
    for k, v in out["components"].items():
        print(f"  {k:<16} median {v['median_ms']:7.1f} ms")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
