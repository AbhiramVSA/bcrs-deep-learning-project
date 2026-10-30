"""Offline checks for the server verifier; no CUDA, training or real weights."""
import hashlib
import json
import sys

import pytest

import scripts.verify_a100 as audit


def test_all_archived_commands_preserve_protocol_and_generation(tmp_path):
    for name in audit.SOURCES:
        d = json.loads((audit.ROOT / "results" / name).read_text())
        post = ({"topk_candidates": 1000, "detections_per_img": 500, "box_decode": "bf16-legacy"}
                if name == "p3_sweep_joint_v2/curve_val.json" else None)
        cmd = audit.command_for(d, tmp_path / name, post)
        assert cmd[cmd.index("--out") + 1] == str(tmp_path / name)
        assert cmd[cmd.index("--ckpt") + 1] == str(audit.local_path(d["ckpt"]))
        expected_decode = (d.get("postprocess", post or d))["box_decode"]
        assert ("--legacy-bf16-decode" in cmd) == (expected_decode == "bf16-legacy")
        if "curves" in d:
            assert cmd[cmd.index("--overlap") + 1] == str(d["geometry"]["overlap"])
            assert cmd[cmd.index("--seed") + 1] == "0"
            assert ("--whole-image" in cmd) == ("whole" in d["curves"])
            assert cmd[cmd.index("--k-list") + 1] == ",".join(map(str, sorted(
                {int(k) for r in d["curves"] if r != "whole" for k in d["curves"][r]})))
        if d.get("budget_fixed") is not None:
            assert cmd[cmd.index("--budget-fixed") + 1] == str(d["budget_fixed"])


def test_numerical_comparison_is_not_a_significance_test():
    old = {"curves": {"learned": {"3": {"AP": 0.2, "AP_s": 0.13}}}}
    near = {"curves": {"learned": {"3": {"AP": 0.20001, "AP_s": 0.13}}}}
    assert audit.metric_comparison(old, near, 1e-4)["all_within_tolerance"]
    assert not audit.metric_comparison(old, {}, 1e-4)["all_within_tolerance"]
    near["curves"]["learned"]["3"]["AP"] = float("nan")
    assert not audit.metric_comparison(old, near, 1e-4)["all_within_tolerance"]
    near["curves"]["learned"]["3"]["AP"] = 0.21
    assert not audit.metric_comparison(old, near, 1e-4)["all_within_tolerance"]


def test_junit_requires_both_gpu_checks_without_skips(tmp_path):
    path = tmp_path / "pytest.xml"
    cases = "".join(f'<testcase name="{name}"/>' for name in audit.GPU_TESTS)
    path.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    assert audit.check_junit(path)["passed_without_skips"]
    path.write_text(path.read_text().replace("/>", "><skipped/></testcase>", 1))
    assert not audit.check_junit(path)["passed_without_skips"]
    path.write_text("<testsuites/>")
    assert not audit.check_junit(path)["passed_without_skips"]
    path.write_text('<!DOCTYPE x [<!ENTITY a "x">]><testsuites/>')
    with pytest.raises(ValueError, match="DTD/entity"):
        audit.check_junit(path)


def test_hashing_and_path_boundary(tmp_path):
    p = tmp_path / "blob"
    p.write_bytes(b"a" * (2 * 1024 * 1024))
    assert audit.sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="outside repository"):
        audit.local_path("../outside.pth")


def test_collect_only_preserves_inputs_and_bundles_evidence(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "results").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "main.py").write_text("# existing unrelated work\n")
    (root / "runs" / "best.pth").write_bytes(b"synthetic inventory-only file")
    split = root / "data" / "val"
    (split / "images").mkdir(parents=True)
    (split / "images" / "a.jpg").write_bytes(b"synthetic inventory-only image")
    (split / "coco.json").write_text(json.dumps({"images": [{"file_name": "a.jpg"}]}))
    d = {"ckpt": "runs/best.pth", "split_dir": "data/val", "downsample": 1.0,
         "topk_candidates": 1000, "detections_per_img": 500, "box_decode": "fp32", "metrics": {"AP": 0.2}}
    source = root / "results" / "baseline.json"
    source.write_text(json.dumps(d))
    digest = audit.sha256(source)
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "SOURCES", ["baseline.json"])
    monkeypatch.setattr(sys, "argv", ["verify", "--collect-only", "--out", str(tmp_path / "evidence")])
    monkeypatch.chdir(root)
    assert audit.main() == 0
    assert audit.sha256(source) == digest
    out = tmp_path / "evidence"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "collected_only"
    assert not summary["historical_environment_authenticated"]
    assert not summary["protected_files_changed"]
    assert (tmp_path / "evidence.tar.gz").is_file()
    assert json.loads((out / "checkpoints.json").read_text())["runs/best.pth"]["exists"]
    with pytest.raises(SystemExit):
        audit.main()  # existing evidence must never be reused/overwritten


def test_output_and_pickle_safety_guards(monkeypatch):
    for argv in (["--collect-only", "--out", str(audit.ROOT / "results" / "forbidden")],
                 ["--reevaluate"], ["--metric-tolerance", "nan"]):
        monkeypatch.setattr(sys, "argv", ["verify", *argv])
        with pytest.raises(SystemExit):
            audit.main()
