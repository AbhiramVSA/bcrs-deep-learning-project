"""P2a/P3 sweep: accuracy-vs-k curves for tile-selection rankings.

    python scripts/sweep_curve.py --ckpt runs/p1_uniform_s1/best.pth \
        --out results/p2a_oracle_sweep/curve_val.json \
        [--split-dir data/...-val] [--s 4] [--tile 512] [--overlap 32] \
        [--rankings oracle,random] [--max-images N] [--k-list 0,1,2,...]

Per image: ONE coarse pass + ONE fine pass over every tile; fused detection
sets for each (ranking, k) are assembled from the cached tile outputs, so the
whole curve costs a single GPU sweep. Curves are evaluated with the shared
COCO protocol and invariant-checked (oracle >= random; more tiles >= fewer for
oracle) — violations WARN loudly but do not crash, so partial diagnostics
still land in the JSON.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch

from bcrs.data import VisDroneDataset
from bcrs.evaluate import coco_metrics
from bcrs.models.detector import build_detector_from_ckpt
from bcrs.tiles import tile_grid
from bcrs.twopass import coarse_pass, fuse, oracle_tile_scores, tile_pass


def dets_to_coco(det: dict, image_id: int) -> list[dict]:
    out = []
    for (x0, y0, x1, y1), s, lab in zip(det["boxes"].tolist(),
                                        det["scores"].tolist(),
                                        det["labels"].tolist()):
        if lab < 1:
            continue
        out.append({"image_id": image_id, "category_id": int(lab),
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="detector for the FINE pass (native-res tiles)")
    ap.add_argument("--coarse-ckpt", default=None,
                    help="detector for the COARSE pass (default: same as --ckpt)")
    ap.add_argument("--scorer-ckpt", default=None,
                    help="trained TileScorer checkpoint -> adds the 'learned' ranking")
    ap.add_argument("--split-dir", default="data/visdrone/VisDrone2019-DET-val")
    ap.add_argument("--s", type=float, default=4.0)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--rankings", default="oracle,random")
    ap.add_argument("--k-list", default=None, help="comma ints; default 0..K_max")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    # fusion diagnostics (PLAN.md: tiled-all must not lose to whole-image native)
    ap.add_argument("--keep-truncated", action="store_true",
                    help="do NOT drop fine boxes touching interior tile borders")
    ap.add_argument("--coarse-in-tile-weight", type=float, default=0.5,
                    help="score multiplier for coarse boxes inside selected tiles")
    ap.add_argument("--coarse-full-min-side", type=float, default=None,
                    help="coarse boxes with longer side >= this keep full weight (size-aware fusion); "
                         "a natural value is 2*overlap (the band a tile is guaranteed to see whole)")
    ap.add_argument("--budget-fixed", type=float, default=None,
                    help="ablation (i): feed the scorer this constant budget input for every k, so the "
                         "ranking is budget-independent and the top-k is a truncation of one list")
    ap.add_argument("--whole-image", action="store_true",
                    help="also evaluate the fine detector on the whole native image (curve 'whole', cost 1.0)")
    ap.add_argument("--topk-candidates", type=int, default=1000,
                    help="pre-NMS candidates kept per FPN level, per image or tile (torchvision default 1000). "
                         "A whole 1360x765 frame with hundreds of objects can exceed 1000 at P3, while every tile "
                         "gets its own 1000, so the cap favours tiling; raise it to test that hypothesis")
    ap.add_argument("--detections-per-img", type=int, default=500)
    ap.add_argument("--legacy-bf16-decode", action="store_true",
                    help="reproduce pre-2026-10-05 numbers: stock torchvision box decode and scores in bf16 "
                         "under autocast (coordinates above 1024 rounded to 8 px)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
    post = dict(topk_candidates=args.topk_candidates, detections_per_img=args.detections_per_img,
                legacy_bf16_decode=args.legacy_bf16_decode)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    detector = build_detector_from_ckpt(ckpt, **post).to(device)
    if args.coarse_ckpt and args.coarse_ckpt != args.ckpt:
        cckpt = torch.load(args.coarse_ckpt, map_location="cpu", weights_only=False)
        coarse_detector = build_detector_from_ckpt(cckpt, **post).to(device)
    else:
        coarse_detector = detector

    scorer, budget_norm_k = None, None
    if args.scorer_ckpt:
        from bcrs.train_bcrs import build_scorer
        sckpt = torch.load(args.scorer_ckpt, map_location="cpu", weights_only=False)
        scorer = build_scorer(sckpt["cfg"]).to(device)
        scorer.load_state_dict(sckpt["scorer"])
        scorer.eval()
        budget_norm_k = sckpt["cfg"].get("budget_norm_k")

    ds = VisDroneDataset(args.split_dir, downsample=1.0, train=False,
                         max_images=args.max_images, seed=args.seed)
    ranking_names = args.rankings.split(",")
    if scorer is not None and "learned" not in ranking_names:
        ranking_names.append("learned")
    rng = random.Random(args.seed)

    # first pass over images: cache per-image coarse/tile outputs is too much
    # memory for the whole split, so we assemble per-(ranking,k) accumulators
    # inside the loop instead.
    acc: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    tiles_used: dict[int, int] = defaultdict(int)
    sel_recall: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    n_with_misses: dict[int, int] = defaultdict(int)
    img_ids: list[int] = []
    k_max_seen = 0
    t0 = time.time()

    for idx in range(len(ds)):
        image, target = ds[idx]
        img_id = int(target["image_id"])
        img_ids.append(img_id)
        _, h, w = image.shape
        tiles = tile_grid(w, h, g=args.tile, overlap=args.overlap)
        n = tiles.shape[0]
        k_max_seen = max(k_max_seen, n)

        coarse, feats = coarse_pass(coarse_detector, image, args.s, device, amp_dtype,
                                    return_features=True)
        tile_outs = tile_pass(detector, image, tiles, device, amp_dtype, chunk=args.chunk)
        if args.whole_image:
            whole = coarse_pass(detector, image, 1.0, device, amp_dtype)  # s=1: no resize
            acc["whole"][0].extend(dets_to_coco(whole, img_id))
        counts = oracle_tile_scores(target["boxes"], target["labels"], coarse,
                                    tiles, args.iou_thr, args.score_thr)

        # ranking -> callable(k) -> tile order (learned ranking may depend on k via FiLM)
        rankings: dict[str, object] = {}
        if "oracle" in ranking_names:
            o = torch.argsort(counts, descending=True)
            rankings["oracle"] = lambda k, o=o: o
        if "random" in ranking_names:
            perm = list(range(n))
            rng.shuffle(perm)
            r = torch.tensor(perm, dtype=torch.long)
            rankings["random"] = lambda k, r=r: r
        if scorer is not None:
            def learned_order(k, feats=feats, tiles=tiles, n=n):
                # joint-trained scorers are conditioned on k / budget_norm_k; P2b
                # scorers saw random budgets, so any consistent value works there
                bval = args.budget_fixed if args.budget_fixed is not None else min(k, n) / (budget_norm_k or max(1, n))
                b = torch.tensor([bval], device=device)
                with torch.no_grad():
                    sc = scorer(feats[-1], b, [tiles], args.s * 128.0)[0]
                return torch.argsort(sc.float().cpu(), descending=True)
            rankings["learned"] = learned_order

        k_values = ([int(x) for x in args.k_list.split(",")] if args.k_list
                    else list(range(0, 64)))  # trimmed to k_max_seen at the end
        total_missed = counts.sum().item()
        for name, order_fn in rankings.items():
            for k in k_values:
                kk = min(k, n)
                order = order_fn(k)
                fused = fuse(coarse, tile_outs, order[:kk], tiles,
                             coarse_in_tile_weight=args.coarse_in_tile_weight,
                             suppress_truncated=not args.keep_truncated,
                             coarse_full_min_side=args.coarse_full_min_side)
                acc[name][k].extend(dets_to_coco(fused, img_id))
                if name == ranking_names[0]:
                    tiles_used[k] += kk
                if total_missed > 0:
                    sel_recall[name][k] += counts[order[:kk]].sum().item() / total_missed
                    if name == ranking_names[0]:
                        n_with_misses[k] += 1

        if (idx + 1) % 25 == 0:
            rate = (idx + 1) / (time.time() - t0)
            print(f"[{idx + 1}/{len(ds)}] {rate:.2f} img/s", flush=True)

    gt_json = Path(args.split_dir) / "coco.json"
    # explicit --k-list is honored verbatim (k is clamped per image already);
    # the default dense range is trimmed to the largest tile count actually seen
    if args.k_list:
        k_values = sorted({int(x) for x in args.k_list.split(",")})
    else:
        k_values = sorted(k for k in acc[ranking_names[0]] if k <= k_max_seen)
    curves: dict[str, dict] = {}
    if "whole" in acc:
        m = coco_metrics(gt_json, acc["whole"][0], img_ids=img_ids)
        m["avg_tiles"] = 0.0
        m["sel_recall"] = 0.0
        curves["whole"] = {"0": m}
        print(f"whole-image native: AP={m['AP']:.4f} AP_s={m['AP_s']:.4f} AP_m={m['AP_m']:.4f} "
              f"AP_l={m['AP_l']:.4f} AR@500={m['AR@500']:.4f}")
    for name in acc:
        if name == "whole":
            continue
        curves[name] = {}
        for k in k_values:
            m = coco_metrics(gt_json, acc[name][k], img_ids=img_ids)
            m["avg_tiles"] = tiles_used[k] / len(ds)
            m["sel_recall"] = sel_recall[name][k] / max(1, n_with_misses[k])
            curves[name][str(k)] = m
            print(f"{name} k={k:>2}: AP={m['AP']:.4f} AP_s={m['AP_s']:.4f} "
                  f"AR@500={m['AR@500']:.4f} sel_recall={m['sel_recall']:.3f} "
                  f"avg_tiles={m['avg_tiles']:.1f}")

    warnings = []
    if "oracle" in curves and "random" in curves:
        for k in k_values:
            if k > 0 and curves["oracle"][str(k)]["AP"] + 1e-4 < curves["random"][str(k)]["AP"]:
                warnings.append(f"INVARIANT: oracle AP < random AP at k={k}")
    if "oracle" in curves and len(k_values) > 1:
        k0, kN = str(k_values[0]), str(k_values[-1])
        if curves["oracle"][kN]["AP"] + 1e-4 < curves["oracle"][k0]["AP"]:
            warnings.append("INVARIANT: oracle AP(k_max) < AP(k=0)")
    for wmsg in warnings:
        print(f"WARNING {wmsg}", flush=True)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    out = {
        "ckpt": args.ckpt, "coarse_ckpt": args.coarse_ckpt or args.ckpt,
        "scorer_ckpt": args.scorer_ckpt, "eval_commit": commit, "split_dir": args.split_dir,
        "geometry": {"s": args.s, "tile": args.tile, "overlap": args.overlap,
                     "k_max_seen": k_max_seen},
        "match": {"iou_thr": args.iou_thr, "score_thr": args.score_thr},
        "postprocess": {"topk_candidates": args.topk_candidates, "detections_per_img": args.detections_per_img,
                        "box_decode": "bf16-legacy" if args.legacy_bf16_decode else "fp32"},
        "budget_fixed": args.budget_fixed,
        "fusion": {"suppress_truncated": not args.keep_truncated,
                   "coarse_in_tile_weight": args.coarse_in_tile_weight,
                   "coarse_full_min_side": args.coarse_full_min_side},
        "n_images": len(ds), "curves": curves, "warnings": warnings,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
