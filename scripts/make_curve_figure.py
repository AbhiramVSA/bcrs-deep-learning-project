"""Accuracy-vs-cost figure for the paper (vector PDF), from sweep JSON.

    python scripts/make_curve_figure.py results/p3_sweep_joint_v3/curve_val.json \
        --uniform results/v3eval_uniform/s4.json results/v3eval_uniform/s2.json \
                  results/v3eval_uniform/s1.json --out paper/figures/curve.pdf

--uniform takes eval_ckpt.py outputs or training metrics.json files.

Cost is in units of one dense native-resolution pass (C_full): coarse = 1/s^2,
each tile = g^2 / mean frame area. Identity is carried by color AND line style
AND marker shape, so the figure reads in grayscale and under CVD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
# pi-lens-ignore: reportMissingImports
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# categorical slots 1-3 of the reference palette (validated all-pairs, light mode)
SERIES = {
    "oracle":  {"color": "#1baf7a", "ls": "--", "marker": "^", "label": "Oracle selection"},
    "learned": {"color": "#2a78d6", "ls": "-", "marker": "o", "label": "BCRS (learned selection)"},
    "random":  {"color": "#eb6834", "ls": ":", "marker": "s", "label": "Random selection"},
}
TEXT, GRID = "#0b0b0b", "#dcdcd8"


def cost_axis(d: dict) -> tuple[float, float]:
    # pi-lens-ignore: unchecked-throwing-call-python
    coco = json.loads((Path(d["split_dir"]) / "coco.json").read_text())
    area = sum(i["width"] * i["height"] for i in coco["images"]) / len(coco["images"])
    return 1.0 / d["geometry"]["s"] ** 2, d["geometry"]["tile"] ** 2 / area


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("curve_json")
    ap.add_argument("--uniform", nargs="*", default=[])
    ap.add_argument("--metric", default="AP", help="AP or AP_s")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--out", default="paper/figures/curve.pdf")
    args = ap.parse_args()

    # pi-lens-ignore: unchecked-throwing-call-python
    d = json.loads(Path(args.curve_json).read_text())
    c0, ct = cost_axis(d)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.5))  # IEEE single column

    for name in ("random", "learned", "oracle"):
        if name not in d["curves"]:
            continue
        pts = sorted((c0 + m["avg_tiles"] * ct, 100 * m[args.metric], int(k))
                     for k, m in d["curves"][name].items() if int(k) <= args.kmax)
        s = SERIES[name]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=s["color"], ls=s["ls"],
                marker=s["marker"], ms=3.5, lw=1.4, label=s["label"], zorder=3)
        if name == "learned":
            for x, y, k in pts:
                if k in (1, 3):
                    ax.annotate(f"$k$={k}", (x, y), xytext=(6, -4) if k == 1 else (4, -9), textcoords="offset points",
                                fontsize=6.5, color=TEXT)

    for p in args.uniform:
        # pi-lens-ignore: unchecked-throwing-call-python
        m = json.loads(Path(p).read_text())
        if "metrics" in m:  # eval_ckpt.py output (results/fp32eval_uniform, results/v3eval_uniform)
            s, b = float(m["downsample"]), m["metrics"]
        else:  # training metrics.json: best training-time evaluation
            s, b = float(m["config"]["downsample"]), max(m["history"], key=lambda h: h["AP"])
        x, y = 1 / s ** 2, 100 * b[args.metric]
        ax.plot(x, y, marker="D", ms=5, color=TEXT, ls="none", zorder=4,
                label="Uniform resolution" if p == args.uniform[0] else None)
        ax.annotate(f"$s$={int(s)}", (x, y), xytext=(-4, 5) if s == 2 else (5, 3), textcoords="offset points",
                    ha="right" if s == 2 else "left", fontsize=6.5, color=TEXT)

    ax.set_xlabel(r"Area-cost proxy (dense native-resolution pass = 1)")
    ax.set_ylabel("AP$_s$ (%) @ 500/category/image" if args.metric == "AP_s" else "AP (%) @ 100/category/image")
    ax.set_xlim(0, None)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=TEXT, length=2)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.02))
    ax.set_xlim(0, 2.25)
    fig.tight_layout(pad=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(str(Path(args.out).with_suffix(".png")), dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
