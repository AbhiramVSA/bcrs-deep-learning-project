"""Generate seed-spread and measured-latency figures from corrected v3 records.

Run from the repository root after the v3 result files and validation annotations
are available:

    python scripts/make_extra_figures.py
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_tables import cost_terms

R = Path("results")
OUT = Path("paper/figures")
SEED_SWEEPS = (
    R / "p3_sweep_joint_v3/curve_val.json",
    R / "p3_sweep_joint_v3_seed1/curve_val.json",
    R / "p3_sweep_joint_v3_seed2/curve_val.json",
)
K_VALUES = (1, 2, 3, 4, 6, 8)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=250)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


def seed_figure() -> None:
    seeds = [read(path) for path in SEED_SWEEPS]
    dense = read(R / "v3eval_uniform/s1.json")["metrics"]
    control = read(R / "v3eval_uniform/s1_cont12.json")["metrics"]
    assert len(seeds) == 3
    assert all(d["n_images"] == seeds[0]["n_images"] for d in seeds)
    assert all(d["split_dir"] == seeds[0]["split_dir"] for d in seeds)
    assert all(d["geometry"] == seeds[0]["geometry"] for d in seeds)
    assert all(
        all(str(k) in d["curves"]["learned"] for k in K_VALUES)
        for d in seeds
    )
    assert dense["AP"] > 0 and control["AP"] > 0

    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharex=True)
    panels = (
        ("AP", "AP (%) @ 100/category/image", (0, 27)),
        ("AP_s", "AP$_s$ (%) @ 500/category/image", (0, 17)),
    )
    for ax, (metric, ylabel, ylim) in zip(axes, panels, strict=True):
        values = [
            [100 * d["curves"]["learned"][str(k)][metric] for d in seeds]
            for k in K_VALUES
        ]
        means = [statistics.fmean(v) for v in values]
        spread = [statistics.stdev(v) for v in values]
        ax.errorbar(
            K_VALUES, means, yerr=spread, color="#2468a2", marker="o",
            markersize=3.5, linewidth=1.3, capsize=2.5, label="BCRS mean ± sample SD",
            zorder=3,
        )
        ax.axhline(
            100 * dense[metric], color="#222222", linestyle="--", linewidth=1,
            label="Dense native (single run)",
        )
        ax.axhline(
            100 * control[metric], color="#777777", linestyle=":", linewidth=1.1,
            label="Dense +12 epochs (single run)",
        )
        ax.set_ylim(*ylim)
        ax.set_xlim(0.5, 8.5)
        ax.set_xticks(K_VALUES)
        ax.set_xlabel("Tile budget $k$")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dededb", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("Overall AP")
    axes[1].set_title("Small-object AP")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.12, 1, 1), pad=0.4)
    save(fig, "seed_variation")


def latency_figure() -> None:
    curve = read(R / "p3_sweep_joint_v3/curve_val.json")
    b1 = read(R / "latency_v3/gpu_b1.json")
    b8 = read(R / "latency_v3/gpu_throughput_b8.json")
    assert b1["n_images"] == 100
    assert b8["batch"] == 8 and b8["n_frames"] == 96
    assert b1["geometry"]["ckpt"] == b8["geometry"]["ckpt"]
    assert b1["geometry"]["scorer_ckpt"] == b8["geometry"]["scorer_ckpt"]
    assert b8["frame_hw"] == [765, 1360]
    assert all(b1["latency"][f"bcrs_k{k}"]["n"] == 100 for k in (1, 2, 3, 4, 6))
    assert all(
        b8["modes"][f"bcrs_k{k}"]["n_batches"] == 9
        for k in (1, 2, 3, 4, 6)
    )

    c0, ct = cost_terms(curve)
    dense_cost = [1 / s**2 for s in (4, 2, 1)]
    dense_b1 = [
        b1["latency"][f"uniform_s{s}"]["median_ms"] for s in (4, 2, 1)
    ]
    dense_b8 = [
        b8["modes"][f"dense_s{s}"]["ms_per_frame_median"] for s in (4, 2, 1)
    ]
    ks = (1, 2, 3, 4, 6)
    bcrs_cost = [
        c0 + curve["curves"]["learned"][str(k)]["avg_tiles"] * ct for k in ks
    ]
    bcrs_b1 = [b1["latency"][f"bcrs_k{k}"]["median_ms"] for k in ks]
    bcrs_b8 = [
        b8["modes"][f"bcrs_k{k}"]["ms_per_frame_median"] for k in ks
    ]

    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=False)
    panels = (
        (axes[0], dense_b1, bcrs_b1, "Batch one: component-sum median"),
        (axes[1], dense_b8, bcrs_b8, "Batch eight: throughput median"),
    )
    for ax, dense_ms, bcrs_ms, title in panels:
        ax.plot(
            dense_cost, dense_ms, color="#4c4c4c", linestyle="--", marker="D",
            markersize=3.5, linewidth=1.2, label="Dense resolution",
        )
        ax.plot(
            bcrs_cost, bcrs_ms, color="#b34d25", linestyle="-", marker="o",
            markersize=3.5, linewidth=1.3, label="BCRS tile budget",
        )
        ax.set_title(title)
        ax.set_xlabel(
            "Full-validation reference area proxy\n(dense native = 1)",
            fontsize=7,
        )
        ax.set_xlim(0, 2.0)
        ax.set_xticks((0, 0.5, 1.0, 1.5, 2.0))
        ax.grid(axis="y", color="#dededb", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Median GPU time (ms/frame)")
    axes[1].set_ylabel("Median GPU time (ms/frame)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.5)
    save(fig, "latency_cost")


if __name__ == "__main__":
    seed_figure()
    latency_figure()
