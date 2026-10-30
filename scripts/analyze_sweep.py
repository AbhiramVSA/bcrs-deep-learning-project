"""Turn a sweep curve JSON into the paper's accuracy-vs-cost table.

    python scripts/analyze_sweep.py results/p2a_oracle_sweep/curve_val.json \
        [--uniform results/p1_uniform_s4/metrics.json results/p1_uniform_s2/metrics.json results/p1_uniform_s1/metrics.json]

Cost model (proposal section 4.6), in units of dense native-resolution cost
C_full for a frame of area A: coarse pass = 1/s^2; each tile = g^2 / A
(overlap ignored). We use the sweep's per-k average tile count and the split's
mean frame area, so cost is an average over the split.

Prints, per ranking and k: AP, AP_s/m/l, AR@500, selection recall, avg tiles,
cost; plus oracle-normalized regret of the learned ranking
(AP_oracle - AP_learned) / (AP_oracle - AP_random), and the area under the
AP-vs-cost curve (trapezoid) over the common cost range for each ranking.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def mean_frame_area(split_dir: str) -> float:
    coco = json.loads((Path(split_dir) / "coco.json").read_text())
    return sum(im["width"] * im["height"] for im in coco["images"]) / len(coco["images"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("curve_json")
    ap.add_argument("--uniform", nargs="*", default=[], help="P1 metrics.json files for reference rows")
    ap.add_argument("--split-dir", default=None, help="override split dir for frame-area estimate")
    args = ap.parse_args()

    d = json.loads(Path(args.curve_json).read_text())
    geo = d["geometry"]
    s, g = float(geo["s"]), int(geo["tile"])
    split = args.split_dir or d["split_dir"]
    try:
        area = mean_frame_area(split)
    except Exception:
        area = 2000 * 1500
        print(f"(could not read {split}/coco.json; assuming 2000x1500 frames)")
    c_coarse = 1.0 / (s * s)
    c_tile = g * g / area
    print(f"cost model: coarse={c_coarse:.3f} C_full, tile={c_tile:.3f} C_full (mean frame area {area:.0f} px^2, tile {g})")
    print(f"images: {d['n_images']} | eval commit {d['eval_commit']} | fine={d['ckpt']} coarse={d.get('coarse_ckpt')} scorer={d.get('scorer_ckpt')}")
    if d.get("warnings"):
        print("WARNINGS:", *d["warnings"], sep="\n  ")

    curves = d["curves"]
    if "fusion" in d:
        print(f"fusion: {d['fusion']}")
    if "whole" in curves:
        w = curves["whole"]["0"]
        print(f"whole-image native (fine detector, cost 1.000): AP={w['AP']:.4f} AP_s={w['AP_s']:.4f} "
              f"AP_m={w['AP_m']:.4f} AP_l={w['AP_l']:.4f} AR500={w['AR@500']:.4f}")
    ks = sorted({int(k) for n, c in curves.items() if n != "whole" for k in c})
    header = f"{'rank':<8}{'k':>3}{'cost':>7}{'AP':>8}{'AP50':>8}{'AP_s':>8}{'AP_m':>8}{'AP_l':>8}{'AR500':>8}{'selR':>7}{'tiles':>7}"
    print(header)
    cost_of = {}
    for name in ("oracle", "learned", "random"):
        if name not in curves:
            continue
        for k in ks:
            m = curves[name].get(str(k))
            if m is None:
                continue
            cost = c_coarse + m["avg_tiles"] * c_tile
            cost_of[(name, k)] = cost
            print(f"{name:<8}{k:>3}{cost:>7.3f}{m['AP']:>8.4f}{m['AP50']:>8.4f}{m['AP_s']:>8.4f}{m['AP_m']:>8.4f}"
                  f"{m['AP_l']:>8.4f}{m['AR@500']:>8.4f}{m.get('sel_recall', 0):>7.3f}{m['avg_tiles']:>7.2f}")
        print()

    if "oracle" in curves and "learned" in curves and "random" in curves:
        print("oracle-normalized regret of learned selection (0 = oracle, 1 = random):")
        for k in ks:
            o, l, r = (curves[n].get(str(k)) for n in ("oracle", "learned", "random"))
            if not (o and l and r) or k == 0:
                continue
            gap = o["AP"] - r["AP"]
            reg = (o["AP"] - l["AP"]) / gap if gap > 1e-6 else float("nan")
            print(f"  k={k:>2}: regret={reg:.3f}  (oracle {o['AP']:.4f} / learned {l['AP']:.4f} / random {r['AP']:.4f})")
        print()

    # AUC over the common cost range
    print("area under AP-vs-cost curve (trapezoid, over each ranking's own cost range):")
    for name in ("oracle", "learned", "random"):
        if name not in curves:
            continue
        pts = sorted((cost_of[(name, k)], curves[name][str(k)]["AP"]) for k in ks if (name, k) in cost_of)
        auc = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
        print(f"  {name:<8} cost {pts[0][0]:.3f}..{pts[-1][0]:.3f}  AUC={auc:.5f}  (mean AP over range {auc / max(pts[-1][0] - pts[0][0], 1e-9):.4f})")

    if args.uniform:
        print("\nuniform-resolution reference points (single pass, cost = 1/s^2):")
        for p in args.uniform:
            m = json.loads(Path(p).read_text())
            sd = float(m["config"]["downsample"])
            best = max(m["history"], key=lambda h: h["AP"])
            print(f"  s={sd:<4} cost={1 / (sd * sd):.3f}  AP={best['AP']:.4f} AP_s={best['AP_s']:.4f} "
                  f"AP_m={best['AP_m']:.4f} AP_l={best['AP_l']:.4f} AR500={best['AR@500']:.4f}  ({m['run_name']}, epoch {best['epoch']})")


if __name__ == "__main__":
    main()
