"""Capture current A100 evidence; never train or overwrite archived results.

From the updated repository, using its EXISTING CUDA environment:
    .venv/bin/python scripts/verify_a100.py --reevaluate --trust-local-checkpoints

--collect-only captures files/hashes without CUDA, tests or checkpoint loading.
Re-evaluation is fresh evidence, not authentication of the old environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_TESTS = {"test_scorer_accepts_bf16_features_outside_autocast",
             "test_scorer_runs_on_cuda_with_cpu_tiles"}
# Explicit generations: do not substitute another generation if an input is absent.
SOURCES = [
    *(f"v3eval_uniform/s{s}.json" for s in (4, 2, 1)),
    "v3eval_uniform/s1_cont12.json",
    *(f"{s}/curve_val.json" for s in (
        "p3_sweep_joint_v3", "p3_sweep_v3_t512", "p3_sweep_joint_v3_seed1",
        "p3_sweep_joint_v3_seed2", "abl_film_fixed_budget_v3",
        "abl_targets_objectness_v3_sweep", "abl_grad_gumbel_v3_sweep", "abl_grad_none_v3_sweep")),
    "fp32eval_uniform/s1.json", "fp32eval_uniform/s1_legacy_bf16.json",
    "p3_sweep_joint_v2/curve_val.json", "p3_sweep_joint_v2_fp32/curve_val.json",
]


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def local_path(name: str) -> Path:
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {name}")
    return p


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def file_hashes(paths) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def command_for(d: dict, output: Path, reconstructed_post: dict | None = None) -> list[str]:
    """Recorded flags where present; explicit reconstructed defaults elsewhere."""
    cmd = [sys.executable, "scripts/sweep_curve.py" if "curves" in d else "scripts/eval_ckpt.py",
           "--ckpt", str(local_path(d["ckpt"])), "--split-dir", str(local_path(d["split_dir"])),
           "--out", str(output)]
    post = d.get("postprocess", reconstructed_post if reconstructed_post is not None else d)
    for key in ("topk_candidates", "detections_per_img"):
        cmd += ["--" + key.replace("_", "-"), str(post[key])]
    if post["box_decode"] == "bf16-legacy":
        cmd.append("--legacy-bf16-decode")
    elif post["box_decode"] != "fp32":
        raise ValueError(f"Unknown decode mode: {post['box_decode']}")
    if "curves" not in d:
        return cmd + ["--downsample", str(d["downsample"]), "--batch-size", "8", "--num-workers", "8"]
    for key in ("coarse_ckpt", "scorer_ckpt"):
        if d.get(key):
            cmd += ["--" + key.replace("_", "-"), str(local_path(d[key]))]
    for key in ("s", "tile", "overlap"):
        cmd += ["--" + key, str(d["geometry"][key])]
    for key in ("iou_thr", "score_thr"):
        cmd += ["--" + key.replace("_", "-"), str(d["match"][key])]
    for key in ("coarse_in_tile_weight", "coarse_full_min_side"):
        if d["fusion"][key] is not None:
            cmd += ["--" + key.replace("_", "-"), str(d["fusion"][key])]
    if not d["fusion"]["suppress_truncated"]:
        cmd.append("--keep-truncated")
    if d.get("budget_fixed") is not None:
        cmd += ["--budget-fixed", str(d["budget_fixed"])]
    rankings = [k for k in d["curves"] if k != "whole"]
    ks = sorted({int(k) for ranking in rankings for k in d["curves"][ranking]})
    cmd += ["--rankings", ",".join(rankings), "--k-list", ",".join(map(str, ks)),
            "--seed", "0", "--chunk", "8"]
    if "whole" in d["curves"]:
        cmd.append("--whole-image")
    return cmd


def metric_comparison(old: dict, new: dict, tolerance: float) -> dict:
    """Tolerance is numerical agreement in AP/AR fractions, not significance."""
    rows = []
    def visit(a, b, prefix):
        for key, value in a.items():
            label = f"{prefix}/{key}"
            if isinstance(value, dict):
                visit(value, b.get(key, {}), label)
            elif key.startswith(("AP", "AR")) or key in ("sel_recall", "avg_tiles"):
                actual = b.get(key)
                ok = isinstance(actual, (int, float)) and math.isfinite(actual)
                delta = actual - value if ok else None
                rows.append({"metric": label, "archived": value, "fresh": actual,
                             "delta": delta, "within_tolerance": delta is not None and abs(delta) <= tolerance})
    visit(old.get("curves", old.get("metrics", {})), new.get("curves", new.get("metrics", {})), "")
    return {"tolerance_fraction": tolerance, "all_within_tolerance": bool(rows) and all(
        r["within_tolerance"] for r in rows), "metrics": rows}


def check_junit(path: Path) -> dict:
    # Only parse our own pytest output, and reject DTD/entity declarations.
    payload = path.read_bytes()
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("Unexpected DTD/entity in local JUnit report")
    root = ET.fromstring(payload)  # noqa: S314 -- local pytest output; DTD/entities rejected above
    cases = list(root.iter("testcase"))
    seen = {c.attrib["name"] for c in cases}
    skipped = [c.attrib["name"] for c in cases if c.find("skipped") is not None]
    failed = [c.attrib["name"] for c in cases if c.find("failure") is not None or c.find("error") is not None]
    return {"tests": len(cases), "skipped": skipped, "failed": failed,
            "missing_gpu_tests": sorted(GPU_TESTS - seen),
            "passed_without_skips": bool(cases) and not skipped and not failed and seen.issuperset(GPU_TESTS)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, help="new, nonexistent directory; default: unique runs/a100-verification-*")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--reevaluate", action="store_true", help="full archived splits/curves, no training; may take hours")
    ap.add_argument("--trust-local-checkpoints", action="store_true", help="permit existing evaluator's pickle loading; trusted own weights ONLY")
    ap.add_argument("--metric-tolerance", type=float, default=1e-4, help="fraction units (default 0.01 AP point); not a statistical test")
    args = ap.parse_args()
    if args.collect_only and args.reevaluate:
        ap.error("--collect-only and --reevaluate are mutually exclusive")
    if args.reevaluate and not args.trust_local_checkpoints:
        ap.error("--reevaluate requires --trust-local-checkpoints: evaluators use torch.load(weights_only=False)")
    if not math.isfinite(args.metric_tolerance) or args.metric_tolerance < 0:
        ap.error("--metric-tolerance must be finite and nonnegative")
    os.chdir(ROOT)
    if args.out:
        out = args.out.resolve()
        if (out.is_relative_to(ROOT) and not out.is_relative_to(ROOT / "runs")) or out == ROOT / "runs":
            ap.error("Evidence output must be a new runs/ subdirectory or outside the repository")
        if Path(str(out) + ".tar.gz").exists():
            ap.error("Output bundle already exists; select a new output directory")
        out.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "runs").mkdir(exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="a100-verification-", dir=ROOT / "runs"))
    print(f"Evidence: {out}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT), env.get("PYTHONPATH", "")])
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    sys.path.insert(0, str(ROOT / "src"))
    summary = {"current_capture_only": True, "historical_environment_authenticated": False,
               "reevaluation_requested": args.reevaluate, "errors": [], "jobs": []}
    def run(label, cmd):
        print(f"RUN {label}: {shlex.join(cmd)}", flush=True)
        with (out / f"{label}.log").open("w") as log:
            try:
                rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
            except OSError as exc:
                log.write(str(exc))
                rc = 127
        summary["jobs"].append({"label": label, "argv": cmd, "returncode": rc})
        dump(out / "summary.json", summary)
        return rc
    protected = {}
    try:
        protected = file_hashes([ROOT / "main.py", * (ROOT / "results").rglob("*")])
        dump(out / "protected-before.json", protected)
        for name in protected:
            if name.startswith("results/"):
                dest = out / "archive" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, dest)
        source_files = [ROOT / "main.py", ROOT / "requirements.txt", ROOT / "pyproject.toml"]
        for folder in ("src", "scripts", "configs", "tests"):
            source_files += [p for p in (ROOT / folder).rglob("*") if p.suffix in (".py", ".sh", ".yaml", ".yml")]
        for name in file_hashes(source_files):
            dest = out / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        dump(out / "source-sha256.json", file_hashes(source_files))
        for label, cmd in (("git-head", ["git", "rev-parse", "HEAD"]),
                           ("git-status", ["git", "status", "--porcelain=v1", "--untracked-files=all"]),
                           ("git-diff", ["git", "diff", "--binary", "HEAD"]),
                           ("nvidia-before", ["nvidia-smi", "-q"])):
            run(label, cmd)
        dump(out / "packages.json", sorted(
            [{"name": p.metadata["Name"], "version": p.version} for p in importlib.metadata.distributions()],
            key=lambda p: p["name"].lower()))
        run("pip-freeze", [sys.executable, "-m", "pip", "freeze", "--all"])
        plans, required, splits = [], set(), set()
        for name in SOURCES:
            try:
                source = ROOT / "results" / name
                d = json.loads(source.read_text())
                defaults: dict[str, object] = ({"seed": 0, "tile_chunk": 8} if "curves" in d else
                            {"batch_size": 8, "num_workers": 8})
                post = None
                if name == "p3_sweep_joint_v2/curve_val.json" and "postprocess" not in d:
                    # Historical v2 sweep predates postprocess metadata; explicit reconstruction.
                    post = {"topk_candidates": 1000, "detections_per_img": 500, "box_decode": "bf16-legacy"}
                    defaults["postprocess"] = post
                cmd = command_for(d, out / "reevaluated" / name, post)
                for key in ("ckpt", "coarse_ckpt", "scorer_ckpt"):
                    if d.get(key):
                        required.add(local_path(d[key]))
                splits.add(local_path(d["split_dir"]))
                plans.append({"source": name, "argv": cmd, "archived_sha256": sha256(source),
                              "reconstructed_defaults": defaults, "archived": d})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                summary["errors"].append(f"Source {name}: {exc}")
        dump(out / "reevaluation-plan.json", plans)
        ckpts = required | {p for p in (ROOT / "runs").rglob("*") if p.suffix in (".pt", ".pth")}
        inventory = {}
        for p in sorted(ckpts):
            name = str(p.relative_to(ROOT))
            inventory[name] = {"required": p in required, "exists": p.is_file()}
            if p.is_file():
                inventory[name].update(sha256=sha256(p), bytes=p.stat().st_size)
            elif p in required:
                summary["errors"].append(f"Missing checkpoint: {name}")
        dump(out / "checkpoints.json", inventory)
        datasets = {}
        for split in sorted(splits):
            try:
                coco = json.loads((split / "coco.json").read_text())
                files = [split / "coco.json", *(split / "images" / i["file_name"] for i in coco["images"])]
                missing = [str(p) for p in files if not p.is_file()]
                datasets[str(split.relative_to(ROOT))] = {"n_images": len(coco["images"]),
                    "missing": missing, "sha256": file_hashes(files)}
                if missing:
                    summary["errors"].append(f"Dataset {split}: {len(missing)} missing files")
            except (OSError, ValueError, KeyError) as exc:
                summary["errors"].append(f"Dataset {split}: {exc}")
        dump(out / "datasets.json", datasets)
        if not args.collect_only:
            import pycocotools.cocoeval as ce
            import torch
            import torchvision
            dump(out / "environment.json", {"python": sys.version, "executable": sys.executable,
                "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                "environment_overrides": {k: env.get(k) for k in ("PYTHONPATH", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
                "cocoeval_sha256": sha256(Path(inspect.getfile(ce)))})
            shutil.copy2(inspect.getfile(ce), out / "cocoeval.py")
            gpu_ok = (torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                      and "A100" in torch.cuda.get_device_name(0))
            if not gpu_ok:
                summary["errors"].append("Requires a visible A100 with bf16 support; no CPU substitution")
            for p in sorted(required):
                if not p.is_file():
                    continue
                try:
                    ck = torch.load(p, map_location="cpu", weights_only=True)
                    from bcrs.models.detector import build_detector_from_ckpt
                    from bcrs.train_bcrs import build_scorer
                    state = ck.get("model", ck.get("scorer", {}))
                    if not state or any(not torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point()):
                        raise ValueError("empty or non-finite weights")
                    if "model" in ck:
                        model = build_detector_from_ckpt(ck)
                    else:
                        model = build_scorer(ck["cfg"])
                        model.load_state_dict(state, strict=True)
                    inventory[str(p.relative_to(ROOT))].update(cfg=ck["cfg"], epoch=ck.get("epoch"),
                        strict_load=True, tensor_count=len(state), parameters=sum(t.numel() for t in state.values()))
                    del model, ck, state
                except Exception as exc:
                    summary["errors"].append(f"Checkpoint {p}: {exc}")
            dump(out / "checkpoints.json", inventory)
            rc = run("pytest", [sys.executable, "-m", "pytest", "-q", "-rs", "tests", "--junitxml", str(out / "pytest.xml")])
            if (out / "pytest.xml").is_file():
                summary["tests"] = check_junit(out / "pytest.xml")
            if rc or not summary.get("tests", {}).get("passed_without_skips"):
                summary["errors"].append("Test suite failed, skipped tests, or did not execute both required GPU tests")
            if args.reevaluate and gpu_ok:
                for i, plan in enumerate(plans):
                    d = plan["archived"]
                    inputs = [local_path(d[k]) for k in ("ckpt", "coarse_ckpt", "scorer_ckpt") if d.get(k)]
                    split = datasets.get(str(local_path(d["split_dir"]).relative_to(ROOT)), {})
                    if any(not inventory[str(p.relative_to(ROOT))].get("strict_load") for p in inputs) or not split or split["missing"]:
                        summary["errors"].append(f"Blocked evaluation: {plan['source']} (see input inventory)")
                        continue
                    if "n_images" in d and d["n_images"] != split["n_images"]:
                        summary["errors"].append(f"Blocked evaluation: {plan['source']} image count differs")
                        continue
                    rc = run(f"eval-{i:02d}", plan["argv"])
                    fresh = out / "reevaluated" / plan["source"]
                    if rc or not fresh.is_file():
                        summary["errors"].append(f"Evaluation failed: {plan['source']}")
                        continue
                    comparison = metric_comparison(d, json.loads(fresh.read_text()), args.metric_tolerance)
                    dump(out / "comparisons" / plan["source"], comparison)
                    if not comparison["all_within_tolerance"]:
                        summary["errors"].append(f"Numerical mismatch: {plan['source']} (fresh results retained, archive untouched)")
    except Exception:
        summary["errors"].append(traceback.format_exc())
    finally:
        changed = [name for name, digest in protected.items() if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
        summary["protected_files_changed"] = changed
        if changed:
            summary["errors"].append("Protected input changed during verification; do not attribute this run to the initial snapshot")
        if (out / "checkpoints.json").is_file():
            inventory = json.loads((out / "checkpoints.json").read_text())
            for name, item in inventory.items():
                if item.get("exists") and (not (ROOT / name).is_file() or sha256(ROOT / name) != item["sha256"]):
                    summary["errors"].append(f"Checkpoint changed during verification: {name}")
        if (out / "source-sha256.json").is_file():
            for name, digest in json.loads((out / "source-sha256.json").read_text()).items():
                if not (ROOT / name).is_file() or sha256(ROOT / name) != digest:
                    summary["errors"].append(f"Source changed during verification: {name}")
        if (out / "datasets.json").is_file():
            for dataset in json.loads((out / "datasets.json").read_text()).values():
                for name, digest in dataset["sha256"].items():
                    if not (ROOT / name).is_file() or sha256(ROOT / name) != digest:
                        summary["errors"].append(f"Dataset changed during verification: {name}")
        run("nvidia-after", ["nvidia-smi", "-q"])
        summary["status"] = "evidence_blocked" if summary["errors"] else ("collected_only" if args.collect_only else "current_checks_passed")
        dump(out / "summary.json", summary)
        bundle = shutil.make_archive(str(out), "gztar", root_dir=out)
        print(f"Status: {summary['status']}\nSummary: {out / 'summary.json'}\nReturn bundle: {bundle}\nWeights are NOT included; retain originals unchanged.", flush=True)
    return int(bool(summary["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
