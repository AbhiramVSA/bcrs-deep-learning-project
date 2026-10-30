"""Generate the paper's LaTeX tables from results/*.json (no hand-typed numbers).

    python scripts/make_tables.py [--gen auto|v3|fp32|v2]   # writes paper/generated/*.tex

Result generations (docs/PLAN_FP32.md):
  v2    recipe-v2 models, boxes decoded in bf16 (superseded on 2026-10-05)
  fp32  the same v2 checkpoints evaluated with the fp32 box decoder (stage A)
  v3    models retrained with the fp32 box decoder (stages B, C, E)
auto takes the newest generation whose headline sweep exists. Every caption
names its generation. Missing inputs produce a visible PENDING row rather
than a silent gap.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

R = Path("results")
OUT = Path("paper/generated")

ABL_LABELS = ("Full model (perturbed top-$k$, self-derived targets, FiLM)", "(i) budget input fixed at evaluation",
              "(ii) object-count targets, no refinement", "(iii) Gumbel-softmax surrogate", "(iv) no gradient through selection")

GENS = {
    "v2": dict(
        note=" Boxes decoded in bf16 (superseded).",
        uniform_eval={}, uniform_train={s: f"p1_uniform_s{s}_v2/metrics.json" for s in "421"},
        control=None, scorer="p2b_scorer_s4v2/metrics.json", scorer_note="",
        curve=("p3_sweep_joint_v2", "p3_sweep_v2_t512"),
        seeds=("p3_sweep_joint_v2", "p3_sweep_joint_v2_seed1", "p3_sweep_joint_v2_seed2"),
        ablations=("p3_sweep_joint_v2", "abl_film_fixed_budget", "abl_targets_objectness_sweep",
                   "abl_grad_gumbel_sweep", "abl_grad_none_sweep"),
        latency=("latency",)),
    "fp32": dict(
        note=" Recipe-v2 checkpoints (trained with the bf16 box decode) evaluated with the fp32 decoder.",
        uniform_eval={s: f"fp32eval_uniform/s{s}.json" for s in "421"},
        uniform_train={},  # no fallback: the training-time numbers used the bf16 decode
        control="fp32eval_uniform/s1_cont12.json", scorer="p2b_scorer_s4v2/metrics.json",
        scorer_note=" Training-time evaluation with the bf16 box decode.",
        curve=("p3_sweep_joint_v2_fp32", "p3_sweep_v2_t512_fp32"),
        seeds=("p3_sweep_joint_v2_fp32", "p3_sweep_joint_v2_seed1_fp32", "p3_sweep_joint_v2_seed2_fp32"),
        ablations=("p3_sweep_joint_v2_fp32", "abl_film_fixed_budget_fp32", "abl_targets_objectness_sweep_fp32",
                   "abl_grad_gumbel_sweep_fp32", "abl_grad_none_sweep_fp32"),
        latency=("latency_v2_fp32", "latency")),
    "v3": dict(
        note=" Corrected v3 generation.",
        uniform_eval={s: f"v3eval_uniform/s{s}.json" for s in "421"},
        uniform_train={s: f"p1_uniform_s{s}_v3/metrics.json" for s in "421"},
        control="v3eval_uniform/s1_cont12.json", scorer="p2b_scorer_s4v3/metrics.json", scorer_note="",
        curve=("p3_sweep_joint_v3", "p3_sweep_v3_t512"),
        seeds=("p3_sweep_joint_v3", "p3_sweep_joint_v3_seed1", "p3_sweep_joint_v3_seed2"),
        ablations=("p3_sweep_joint_v3", "abl_film_fixed_budget_v3", "abl_targets_objectness_v3_sweep",
                   "abl_grad_gumbel_v3_sweep", "abl_grad_none_v3_sweep"),
        latency=("latency_v3",)),
}


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def tex(s: str) -> str:
    return s.replace("_", "\\_")


def best(m: dict) -> dict:
    return max(m["history"], key=lambda h: h["AP"])


def cost_terms(d: dict) -> tuple[float, float]:
    """(coarse cost, cost per tile) in units of one dense native pass, averaged over the split's frame sizes."""
    coco = json.loads((Path(d["split_dir"]) / "coco.json").read_text())
    area = sum(i["width"] * i["height"] for i in coco["images"]) / len(coco["images"])
    return 1 / d["geometry"]["s"] ** 2, d["geometry"]["tile"] ** 2 / area


def table(env: str, caption: str, label: str, cols: str, head: str, rows: list[str]) -> str:
    return (f"\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\n"
            f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n" + "\n".join(rows)
            + f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def uniform_table(g: dict) -> None:
    rows, sources = [], set()

    def row(name: str, cost: float, m: dict) -> str:
        return (f"{name} & {cost:.3f} & {pct(m['AP'])} & {pct(m['AP50'])} & {pct(m['AP_s'])} & {pct(m['AP_m'])} & "
                f"{pct(m['AP_l'])} & {pct(m['AR@500'])} \\\\")

    for s in ("4", "2", "1"):
        e = load(R / g["uniform_eval"][s]) if s in g["uniform_eval"] else None
        t = load(R / g["uniform_train"][s]) if s in g["uniform_train"] else None
        if e is not None:
            m, src = e["metrics"], "eval"
        elif t is not None:
            m, src = best(t), "train"
        else:
            rows.append(f"$s{{=}}{s}$ & \\multicolumn{{7}}{{c}}{{PENDING}} \\\\")
            continue
        sources.add(src)
        rows.append(row(f"$s{{=}}{s}$", 1.0 / float(s) ** 2, m))
    c = load(R / g["control"]) if g["control"] else None
    ctrl = ""
    if c is not None:
        rows.append(row("$s{=}1$, +12 epochs", 1.0, c["metrics"]))
        ctrl = (" The row ``+12 epochs'' continues the $s{=}1$ model for 12 epochs on whole images, so that it has "
                "the training length of the jointly trained BCRS model.")
    how = {frozenset({"eval"}): " Evaluated with \\texttt{eval\\_ckpt.py}.",
           frozenset({"train"}): " Best training-time evaluation."}.get(frozenset(sources), " Mixed sources." if sources else "")
    cap = ("Uniform-resolution baselines on VisDrone-DET val (FCOS, ResNet-18). Each row is a separately trained "
           "model with the same architecture and recipe, evaluated at its training resolution. Cost is relative to "
           "one dense native-resolution pass. AP uses 100 detections per category per image; "
           "AP$_{50}$, size AP and AR$_{500}$ use 500. Single seed." + how + ctrl + g["note"])
    (OUT / "uniform.tex").write_text(table(
        "table", cap, "tab:uniform", "lccccccc",
        "Input & Cost & AP & AP$_{50}$ & AP$_s$ & AP$_m$ & AP$_l$ & AR$_{500}$", rows), encoding="utf-8")


def selection_table(g: dict) -> None:
    m = load(R / g["scorer"])
    if m is None:
        (OUT / "selection.tex").write_text("% PENDING: no scorer metrics\n")
        return
    f = m["final"]
    ks = [k for k in (1, 2, 3, 5, 8) if f"learned_recall@{k}" in f]
    rows = [name.capitalize() + " & " + " & ".join(pct(f[f"{name}_recall@{k}"]) for k in ks) + " \\\\"
            for name in ("random", "learned", "oracle")]
    cap = ("Owner recall@$k$ (\\%): mean per-image fraction of coarse misses whose unique owner tile is "
           f"selected. VisDrone-DET val, {f['n_images_with_misses']} images with at least one miss (run "
           f"\\texttt{{{tex(m['run_name'])}}}, final-epoch scorer on the frozen $s{{=}}4$ detector, not the "
           "best checkpoint used downstream). Single seed; conditioning input $b=0.5$."
           + g["scorer_note"])
    (OUT / "selection.tex").write_text(table(
        "table", cap, "tab:selection", "l" + "c" * len(ks),
        "Ranking & " + " & ".join(f"$k{{=}}{k}$" for k in ks), rows), encoding="utf-8")


def curve_table(g: dict, have_seeds: bool) -> None:
    for cand in g["curve"]:
        d = load(R / cand / "curve_val.json")
        if d is not None:
            break
    else:
        (OUT / "curve.tex").write_text("% PENDING: no sweep\n")
        return
    c0, ct = cost_terms(d)
    names = [n for n in ("random", "learned", "oracle") if n in d["curves"]]
    ks = [k for k in (0, 1, 2, 3, 4, 6) if str(k) in d["curves"][names[0]]]
    rows = []
    for k in ks:
        tiles = d["curves"][names[0]][str(k)]["avg_tiles"]
        cells = " & ".join(f"{pct(d['curves'][n][str(k)]['AP'])} / {pct(d['curves'][n][str(k)]['AP_s'])}" for n in names)
        rows.append(f"{k} & {c0 + tiles * ct:.2f} & {cells} \\\\")
    whole = d["curves"].get("whole", {}).get("0")
    joint = "joint" in cand
    extra = (f" Whole-image native inference with the {'jointly trained' if joint else 'frozen native-resolution'} "
             f"detector: AP {pct(whole['AP'])}, AP$_s$ {pct(whole['AP_s'])} at cost 1.00." if whole else "")
    seeds = " Seed 0; Table~\\ref{tab:seeds} gives three seeds." if have_seeds else " Single seed."
    cap = ("AP / AP$_s$ (\\%) versus tile budget $k$ on VisDrone-DET val "
           f"(source: \\texttt{{{tex(cand)}}}). Cost is relative to one dense native-resolution pass, "
           "using mean selected tile count divided by mean image area (Eq.~\\eqref{eq:proxy}). "
           "Evaluator caps per category per image: AP 100, AP$_s$ 500." + extra + seeds + g["note"])
    (OUT / "curve.tex").write_text(table(
        "table", cap, "tab:curve", "cc" + "c" * len(names),
        "$k$ & Cost & " + " & ".join(n.capitalize() for n in names), rows), encoding="utf-8")


def auc(curve: dict, c0: float, ct: float, kmax: int | None = None) -> float:
    """Trapezoid area under AP-vs-cost over the whole sweep (all k in the JSON), the
    same range analyze_sweep.py reports, so the text and the table agree."""
    ks = [int(k) for k in curve if kmax is None or int(k) <= kmax]
    pts = sorted((c0 + curve[str(k)]["avg_tiles"] * ct, curve[str(k)]["AP"]) for k in ks)
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False))


def ablation_table(g: dict) -> None:
    ks = (1, 2, 3, 4, 6)
    rows, c0ct, ref, spread, cost_hi = [], None, None, 0.0, None
    for label, run in zip(ABL_LABELS, g["ablations"], strict=True):
        d = load(R / run / "curve_val.json")
        if d is None or "learned" not in d.get("curves", {}):
            rows.append(f"{label} & \\multicolumn{{{len(ks) + 3}}}{{c}}{{PENDING}} \\\\")
            continue
        if c0ct is None:
            c0ct = cost_terms(d)
        c = d["curves"]["learned"]
        if cost_hi is None:
            cost_hi = max(c0ct[0] + v["avg_tiles"] * c0ct[1] for v in c.values())
        aps = [c[str(k)]["AP"] for k in ks]
        if ref is None:
            ref = aps
        else:
            spread = max(spread, *(abs(a - b) for a, b in zip(aps, ref, strict=True)))
        cells = " & ".join(pct(a) for a in aps)
        rows.append(f"{label} & {cells} & {pct(c['3']['AP_s'])} & {pct(c['3']['sel_recall'])} & {auc(c, *c0ct):.3f} \\\\")
    rng = f"cost {c0ct[0]:.2f} to {cost_hi:.2f}" if c0ct and cost_hi is not None else "the full sweep"
    cap = ("Ablations on VisDrone-DET val. AP (\\%) at $k$ selected tiles, AP$_s$ and selection recall (\\%) at "
           f"$k{{=}}3$, and area under the AP-versus-cost curve over the full sweep ({rng} in units of one dense "
           "native pass). Evaluator caps per category per image: AP 100, AP$_s$ 500. "
           "Recall is the mean per-image unique-owner recall over images with misses. "
           "Rows (ii)--(iv) are separate 12-epoch joint trainings from the same initialization; "
           "row (i) re-evaluates the full model with the budget input held at $b{=}3/20$ for every $k$. One training "
           f"run (seed 0) per row; at the displayed budgets, the largest AP "
           f"difference from the full model is {100 * spread:.1f} points."
           + g["note"])
    (OUT / "ablation.tex").write_text(table(
        "table*", cap, "tab:ablation", "l" + "c" * (len(ks) + 3),
        "Variant & " + " & ".join(f"$k{{=}}{k}$" for k in ks) + " & AP$_s$@3 & Rec@3 & AUC", rows), encoding="utf-8")


def seeds_table(g: dict) -> bool:
    ds = []
    for s in g["seeds"]:
        d = load(R / s / "curve_val.json")
        if d is None or "learned" not in d["curves"]:
            (OUT / "seeds.tex").write_text(f"% PENDING: seed sweep missing or incomplete: {s}\n")
            return False
        ds.append(d)
    c0, ct = cost_terms(ds[0])
    rows = []

    def row(name: str, cost: str, vals: list[float]) -> str:
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return (f"{name} & {cost} & " + " & ".join(pct(v) for v in vals)
                + f" & {pct(statistics.fmean(vals))} $\\pm$ {100 * sd:.2f} \\\\")

    if all("whole" in d["curves"] for d in ds):
        rows.append(row("whole image", "1.00", [d["curves"]["whole"]["0"]["AP"] for d in ds]))
    for k in (1, 2, 3, 4, 6, 8):
        if all(str(k) in d["curves"]["learned"] for d in ds):
            tiles = statistics.fmean(d["curves"]["learned"][str(k)]["avg_tiles"] for d in ds)
            rows.append(row(f"$k{{=}}{k}$", f"{c0 + tiles * ct:.2f}", [d["curves"]["learned"][str(k)]["AP"] for d in ds]))
    cap = ("Seed variation of the jointly trained model: AP (\\%) with learned selection at $k$ tiles, and "
           "whole-image inference with the same detector, for three joint trainings that differ only in the seed. "
           "Mean $\\pm$ sample standard deviation over the three runs. "
           "AP uses 100 detections per category per image." + g["note"])
    (OUT / "seeds.tex").write_text(table(
        "table", cap, "tab:seeds", "lccccc", "Inference & Cost & Seed 0 & Seed 1 & Seed 2 & Mean", rows),
        encoding="utf-8")
    return True


def latency_table(g: dict) -> None:
    for name in g["latency"]:
        d = R / name
        b1 = load(d / "gpu_b1.json") or load(d / "latency_val.json")
        if b1 is not None:
            break
    else:
        (OUT / "latency.tex").write_text("% PENDING: no latency results\n")
        return
    b8, t4, t16 = load(d / "gpu_throughput_b8.json"), load(d / "cpu_t4.json"), load(d / "cpu_t16.json")
    curve = next((load(R / c / "curve_val.json") for c in g["curve"] if (R / c / "curve_val.json").exists()), None)
    if curve is None:
        raise FileNotFoundError("Latency cost requires a sweep and its validation coco.json")
    c0, ct = cost_terms(curve)
    cols = [("GPU b1", b1, "latency"), ("GPU b8", b8, "modes"), ("CPU 4 thr.", t4, "latency"), ("CPU 16 thr.", t16, "latency")]
    cols = [c for c in cols if c[1] is not None]

    def ms(src: dict, kind: str, dense: str, mode: str):
        if kind == "modes":
            v = src["modes"].get(mode)
            return None if v is None else v["ms_per_frame_median"]
        v = src["latency"].get(dense)
        return None if v is None else v["median_ms"]

    rows = []
    for s in (4, 2, 1):
        cells = [ms(src, kind, f"uniform_s{s}", f"dense_s{s}") for _, src, kind in cols]
        rows.append(f"Dense, $s{{=}}{s}$ & {1 / s ** 2:.2f} & " + " & ".join("--" if v is None else f"{v:.1f}" for v in cells) + " \\\\")
    for k in (1, 2, 3, 4, 6):
        cells = [ms(src, kind, f"bcrs_k{k}", f"bcrs_k{k}") for _, src, kind in cols]
        if all(v is None for v in cells):
            continue
        rows.append(f"BCRS, $k{{=}}{k}$ & {c0 + k * ct:.2f} & " + " & ".join("--" if v is None else f"{v:.1f}" for v in cells) + " \\\\")
    gpu = b1.get("gpu", "GPU")
    cpu = (t4 or t16 or {}).get("cpu_model")
    cap = (f"Median timings (ms/frame) on VisDrone-DET val subsets. All modes use joint detector weights, "
           f"not the separately trained dense accuracy baselines. GPU: {tex(gpu)}, bf16, shared. "
           "b1: median of per-image component sums, excluding top-$k$, initial host-to-device transfer "
           "and loading (100 timed images); scoring includes device-to-host transfer. "
           "b8: median batch time divided by eight (nine timed batches); fine tiles are chunked at 32."
           + (f" CPU: {tex(cpu)}, fp32, 40 timed images." if cpu else "")
           + " Cost is a pixel-area proxy, not measured operations. See Section~\\ref{sec:latency} for scope "
           "and mean-versus-median differences (source: \\texttt{" + tex(d.name) + "})."
           + g["note"])
    (OUT / "latency.tex").write_text(table(
        "table", cap, "tab:latency", "lc" + "c" * len(cols),
        "Mode & Cost & " + " & ".join(c[0] for c in cols), rows), encoding="utf-8")


def latency_spread_table(g: dict) -> None:
    for name in g["latency"]:
        d = load(R / name / "gpu_b1.json")
        if d is not None:
            break
    else:
        raise FileNotFoundError("Batch-one GPU timing summary requires gpu_b1.json")

    modes = (
        ("Dense native", "uniform_s1"),
        ("BCRS, $k{=}1$", "bcrs_k1"),
        ("BCRS, $k{=}3$", "bcrs_k3"),
        ("BCRS, $k{=}6$", "bcrs_k6"),
    )
    rows = []
    for label, key in modes:
        v = d["latency"][key]
        assert v["n"] == 100
        assert all(stat in v for stat in ("median_ms", "mean_ms", "p95_ms"))
        rows.append(
            f"{label} & {v['median_ms']:.1f} & {v['mean_ms']:.1f} & {v['p95_ms']:.1f} \\\\"
        )
    cap = (
        "Batch-one timing spread for the joint detector on VisDrone-DET val. "
        "Values are milliseconds per frame from per-image component sums. "
        "Each mode uses 100 timed images on a shared A100. "
        "The p95 describes this sample and is not a deadline guarantee. "
        "Source: \\texttt{" + tex(name) + "/gpu\\_b1.json}."
    )
    (OUT / "latency_spread.tex").write_text(table(
        "table", cap, "tab:latency_spread", "lccc",
        "Mode & Median & Mean & p95", rows), encoding="utf-8")


def budget_breakdown_table(g: dict) -> None:
    curve = next((load(R / name / "curve_val.json") for name in g["curve"]
                  if (R / name / "curve_val.json").exists()), None)
    dense = load(R / g["uniform_eval"]["1"])
    control = load(R / g["control"])
    if curve is None or dense is None or control is None:
        raise FileNotFoundError("The v3 k=3 breakdown requires its sweep and dense controls")
    fields = ("AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l")
    modes = (
        ("Random tiles, $k{=}3$", curve["curves"]["random"]["3"]),
        ("BCRS, $k{=}3$", curve["curves"]["learned"]["3"]),
        ("Miss-count oracle, $k{=}3$", curve["curves"]["oracle"]["3"]),
        ("Joint detector, whole image", curve["curves"]["whole"]["0"]),
        ("Dense native baseline", dense["metrics"]),
        ("Dense native, +12 epochs", control["metrics"]),
    )
    rows = []
    for label, metrics in modes:
        assert all(field in metrics for field in fields)
        rows.append(f"{label} & " + " & ".join(pct(metrics[field]) for field in fields) + " \\\\")
    cap = (
        "Metric breakdown at $k{=}3$ on VisDrone-DET val. "
        "The first four rows use seed-0 joint weights; the last two use separate v3 dense checkpoints. "
        "The oracle ranks tiles by missed-object count and is not an AP upper bound. "
        "All AP values are reported in percentage points (pp). "
        "AP uses 100 detections per category per image; AP$_{50}$, AP$_{75}$, and size AP use 500. "
        "Sources: \\texttt{p3\\_sweep\\_joint\\_v3} and \\texttt{v3eval\\_uniform}."
    )
    (OUT / "budget_breakdown.tex").write_text(table(
        "table*", cap, "tab:budget-breakdown", "lrrrrrr",
        "Mode & AP (pp) & AP$_{50}$ (pp) & AP$_{75}$ (pp) & AP$_s$ (pp) & AP$_m$ (pp) & AP$_l$ (pp)", rows),
        encoding="utf-8")


def frame_cost_table(g: dict) -> None:
    for name in g["curve"]:
        curve = load(R / name / "curve_val.json")
        if curve is not None:
            break
    else:
        raise FileNotFoundError("Frame cost table requires a validation sweep")
    coco_path = Path(curve["split_dir"]) / "coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    geom = curve["geometry"]
    tile, overlap, scale = int(geom["tile"]), int(geom["overlap"]), int(geom["s"])
    stride = tile - 2 * overlap
    assert stride > 0

    def count_starts(size: int) -> int:
        # Match tile_grid: regular starts plus the edge-clamped final start.
        return 1 if size <= tile else len(range(0, size - tile, stride)) + 1

    sizes = Counter((image["width"], image["height"]) for image in coco["images"])
    assert sum(sizes.values()) == curve["n_images"]
    rows = []
    for (width, height), count in sorted(sizes.items()):
        candidates = count_starts(width) * count_starts(height)
        costs = [
            scale ** -2 + min(k, candidates) * tile ** 2 / (width * height)
            for k in (1, 3, 6)
        ]
        rows.append(
            "$" + str(width) + "\\times" + str(height) + "$ & " + str(count)
            + f" & {candidates} & " + " & ".join(f"{cost:.2f}" for cost in costs) + " \\\\"
        )
    cap = (
        "Per-frame area-cost proxy by VisDrone validation image geometry. "
        "Candidate counts follow the configured tile grid, including its edge-clamped final tiles. "
        "Each cost is $s^{-2}+\\min(k,N)g^2/(HW)$ relative to one dense pass for that frame. "
        "The rows group all 548 annotated validation frames by width and height. "
        "Sources: \\texttt{p3\\_sweep\\_joint\\_v3} and its validation \\texttt{coco.json}."
    )
    (OUT / "frame_cost.tex").write_text(table(
        "table*", cap, "tab:frame-cost", "lccccc",
        "Frame ($W\\times H$) & Images & Tiles & Cost $k{=}1$ & Cost $k{=}3$ & Cost $k{=}6$",
        rows), encoding="utf-8")


def pick_gen(arg: str) -> str:
    if arg != "auto":
        return arg
    for gen in ("v3", "fp32"):
        if (R / GENS[gen]["curve"][0] / "curve_val.json").exists():
            return gen
    return "v2"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="auto", choices=["auto", *GENS])
    gen = pick_gen(ap.parse_args().gen)
    g = GENS[gen]
    OUT.mkdir(parents=True, exist_ok=True)
    have_seeds = seeds_table(g)
    uniform_table(g)
    selection_table(g)
    curve_table(g, have_seeds)
    ablation_table(g)
    latency_table(g)
    if gen == "v3":
        latency_spread_table(g)
        budget_breakdown_table(g)
        frame_cost_table(g)
    print(f"generation {gen}: wrote", *sorted(p.name for p in OUT.glob("*.tex")))
