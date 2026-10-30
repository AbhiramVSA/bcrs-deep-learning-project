"""Old (bf16 box decode) vs new (fp32 box decode) numbers for the same v2 checkpoints.

    python scripts/compare_fp32.py [--threshold 0.3] [--out-dir results/fp32eval_compare]

Reads the v2 results as they were committed (bf16 decode) and their fp32
re-evaluations from stage A of scripts/pipeline_fp32.sh, and writes
  compare.json  every pair of numbers (AP, AP_s; points = AP x 100)
  report.md     the same as tables, plus the derived gaps the paper argues from
and prints one DECISION line that the pipeline reads. Missing inputs are
listed, never silently skipped.

Decision rule (stage B, retraining). The re-evaluation sees only the
evaluation half of the defect; the GIoU training loss also used bf16 boxes,
mostly at native resolution. So the rule flags a retrain when evaluation alone
already moves a number the paper's claims rest on by --threshold points or
more: whole-image native AP, or the BCRS-minus-whole gap at k=3 (joint model),
or the exhaustive-tiling gain of the frozen detector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

R = Path("results")
KS = ("0", "1", "2", "3", "4", "6", "8")

SWEEPS = [  # (label, old dir, new dir)
    ("frozen detectors, g=512", "p3_sweep_v2_t512", "p3_sweep_v2_t512_fp32"),
    ("frozen detectors, g=320", "p3_sweep_v2_t320", "p3_sweep_v2_t320_fp32"),
    ("joint, seed 0", "p3_sweep_joint_v2", "p3_sweep_joint_v2_fp32"),
    ("joint, seed 1", "p3_sweep_joint_v2_seed1", "p3_sweep_joint_v2_seed1_fp32"),
    ("joint, seed 2", "p3_sweep_joint_v2_seed2", "p3_sweep_joint_v2_seed2_fp32"),
    ("abl (i) budget input fixed", "abl_film_fixed_budget", "abl_film_fixed_budget_fp32"),
    ("abl (ii) objectness targets", "abl_targets_objectness_sweep", "abl_targets_objectness_sweep_fp32"),
    ("abl (iii) Gumbel top-k", "abl_grad_gumbel_sweep", "abl_grad_gumbel_sweep_fp32"),
    ("abl (iii) no selection grad", "abl_grad_none_sweep", "abl_grad_none_sweep_fp32"),
]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def pts(x: float | None) -> float | None:
    return None if x is None else round(100.0 * x, 2)


def best_history(m: dict) -> dict:
    return max(m["history"], key=lambda h: h["AP"])


def fmt(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def delta(a, b):
    return None if a is None or b is None else round(b - a, 2)


def uniform(missing: list[str]) -> list[dict]:
    rows = []
    for s in ("4", "2", "1"):
        old_m = load(R / f"p1_uniform_s{s}_v2/metrics.json")
        new = load(R / f"fp32eval_uniform/s{s}.json")
        old = best_history(old_m) if old_m else None
        if old_m is None:
            missing.append(f"p1_uniform_s{s}_v2/metrics.json")
        if new is None:
            missing.append(f"fp32eval_uniform/s{s}.json")
        n = new["metrics"] if new else None
        rows.append({"input": f"s={s}", "old_source": "training-time eval (best epoch)",
                     "AP_old": pts(old["AP"]) if old else None, "AP_new": pts(n["AP"]) if n else None,
                     "APs_old": pts(old["AP_s"]) if old else None, "APs_new": pts(n["AP_s"]) if n else None})
    # the same checkpoint and script in both modes: isolates the decode from everything else
    leg = load(R / "fp32eval_uniform/s1_legacy_bf16.json")
    new = load(R / "fp32eval_uniform/s1.json")
    ref = load(R / "cap_study/whole_s1_cap1000.json")
    rows.append({"input": "s=1, eval_ckpt.py both modes", "old_source": "eval_ckpt.py --legacy-bf16-decode",
                 "AP_old": pts(leg["metrics"]["AP"]) if leg else None, "AP_new": pts(new["metrics"]["AP"]) if new else None,
                 "APs_old": pts(leg["metrics"]["AP_s"]) if leg else None,
                 "APs_new": pts(new["metrics"]["AP_s"]) if new else None,
                 "legacy_reproduces_committed": (None if not (leg and ref) else
                                                 abs(leg["metrics"]["AP"] - ref["metrics"]["AP"]) < 5e-4)})
    for r in rows:
        r["dAP"], r["dAPs"] = delta(r["AP_old"], r["AP_new"]), delta(r["APs_old"], r["APs_new"])
    return rows


def sweep_rows(missing: list[str]) -> tuple[list[dict], dict]:
    rows, curves = [], {}
    for label, old_dir, new_dir in SWEEPS:
        old, new = load(R / old_dir / "curve_val.json"), load(R / new_dir / "curve_val.json")
        if old is None:
            missing.append(f"{old_dir}/curve_val.json")
        if new is None:
            missing.append(f"{new_dir}/curve_val.json")
        if old is None or new is None:
            continue
        curves[new_dir] = (old, new)
        for name in ("whole", "learned", "oracle", "random"):
            if name not in old["curves"] or name not in new["curves"]:
                continue
            ks = ["0"] if name == "whole" else [k for k in KS if k in old["curves"][name]]
            kmax = str(max(int(k) for k in old["curves"][name]))
            if name != "whole" and kmax not in ks:
                ks.append(kmax)
            for k in ks:
                o, n = old["curves"][name].get(k), new["curves"][name].get(k)
                if o is None or n is None:
                    continue
                rows.append({"sweep": label, "dir": new_dir, "ranking": name, "k": int(k),
                             "AP_old": pts(o["AP"]), "AP_new": pts(n["AP"]), "dAP": delta(pts(o["AP"]), pts(n["AP"])),
                             "APs_old": pts(o["AP_s"]), "APs_new": pts(n["AP_s"]),
                             "dAPs": delta(pts(o["AP_s"]), pts(n["AP_s"]))})
    return rows, curves


def gaps(curves: dict) -> list[dict]:
    """The comparisons the paper's claims rest on, before and after the fix."""
    out = []

    def ap(d, name, k):
        c = d["curves"].get(name, {})
        return pts(c[k]["AP"]) if k in c else None

    for key in ("p3_sweep_joint_v2_fp32", "p3_sweep_joint_v2_seed1_fp32", "p3_sweep_joint_v2_seed2_fp32"):
        if key not in curves:
            continue
        old, new = curves[key]
        for k in ("1", "3"):
            g_old = delta(ap(old, "whole", "0"), ap(old, "learned", k))
            g_new = delta(ap(new, "whole", "0"), ap(new, "learned", k))
            out.append({"what": f"BCRS learned k={k} minus whole image ({key})", "old": g_old, "new": g_new,
                        "change": delta(g_old, g_new)})
    if "p3_sweep_v2_t512_fp32" in curves:
        old, new = curves["p3_sweep_v2_t512_fp32"]
        kmax = str(max(int(k) for k in new["curves"]["oracle"]))
        g_old = delta(ap(old, "whole", "0"), ap(old, "oracle", kmax))
        g_new = delta(ap(new, "whole", "0"), ap(new, "oracle", kmax))
        out.append({"what": f"exhaustive tiling (k={kmax}) minus whole image, frozen detectors", "old": g_old,
                    "new": g_new, "change": delta(g_old, g_new)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.3, help="AP points that count as a change")
    ap.add_argument("--out-dir", default="results/fp32eval_compare")
    args = ap.parse_args()

    missing: list[str] = []
    uni = uniform(missing)
    rows, curves = sweep_rows(missing)
    gp = gaps(curves)

    checks = [("whole-image native AP (eval_ckpt, same checkpoint)", uni[-1]["dAP"])]
    checks += [(g["what"], g["change"]) for g in gp
               if ("k=3" in g["what"] and "seed" not in g["what"]) or "exhaustive" in g["what"]]
    moved = [(w, v) for w, v in checks if v is not None and abs(v) >= args.threshold]
    known = [v for _, v in checks if v is not None]
    decision = {"threshold_points": args.threshold, "checks": [{"what": w, "change": v} for w, v in checks],
                "retrain_suggested": bool(moved) if known else None,
                "reason": ("; ".join(f"{w}: {v:+.2f}" for w, v in moved) if moved else
                           "no check moved by the threshold" if known else "inputs missing")}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(
        {"uniform": uni, "sweeps": rows, "gaps": gp, "decision": decision, "missing": missing}, indent=2))

    md = ["# bf16 vs fp32 box decode: same v2 checkpoints", "",
          "Units: AP points (AP x 100). Old: the committed v2 numbers (bf16 decode). New: fp32 decode.", "",
          "## Uniform baselines", "", "| Input | Old source | AP old | AP new | dAP | AP_s old | AP_s new | dAP_s |",
          "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['input']} | {r['old_source']} | {fmt(r['AP_old'])} | {fmt(r['AP_new'])} | {fmt(r['dAP'])} | "
           f"{fmt(r['APs_old'])} | {fmt(r['APs_new'])} | {fmt(r['dAPs'])} |" for r in uni]
    rep = uni[-1].get("legacy_reproduces_committed")
    md += ["", f"Legacy mode reproduces the committed whole-image s=1 number (results/cap_study): {rep}.", "",
           "## Gaps that the claims use", "", "| Comparison | Old | New | Change |", "|---|---|---|---|"]
    md += [f"| {g['what']} | {fmt(g['old'])} | {fmt(g['new'])} | {fmt(g['change'])} |" for g in gp]
    md += ["", "## Sweeps", "", "| Sweep | Ranking | k | AP old | AP new | dAP | AP_s old | AP_s new | dAP_s |",
           "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['sweep']} | {r['ranking']} | {r['k']} | {fmt(r['AP_old'])} | {fmt(r['AP_new'])} | {fmt(r['dAP'])} | "
           f"{fmt(r['APs_old'])} | {fmt(r['APs_new'])} | {fmt(r['dAPs'])} |" for r in rows]
    md += ["", "## Decision on retraining", "", f"Threshold: {args.threshold} AP points.", ""]
    md += [f"- {c['what']}: change {fmt(c['change'])}" for c in decision["checks"]]
    md += ["", f"Retraining suggested: {decision['retrain_suggested']} ({decision['reason']}).", "",
           "The evaluation sees only half of the defect. The GIoU training loss also used bf16 boxes, "
           "mostly at native resolution, so a small change here does not show that the trained weights are correct."]
    if missing:
        md += ["", "## Missing inputs", ""] + [f"- results/{m}" for m in missing]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for r in uni:
        print(f"{r['input']:<32} AP {fmt(r['AP_old'])} -> {fmt(r['AP_new'])}  AP_s {fmt(r['APs_old'])} -> {fmt(r['APs_new'])}")
    for g in gp:
        print(f"{g['what']:<75} {fmt(g['old'])} -> {fmt(g['new'])}")
    if missing:
        print("missing:", ", ".join(missing))
    print(f"DECISION: retrain_suggested={decision['retrain_suggested']} ({decision['reason']})")
    print("wrote", out_dir / "compare.json", out_dir / "report.md")


if __name__ == "__main__":
    main()
