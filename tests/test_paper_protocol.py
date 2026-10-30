"""Checks for the archived metric definition and paper artifact generation.

Run from the repository root: PYTHONPATH=src:. pytest tests/test_paper_protocol.py
These synthetic checks do not attest the historical server environment.
"""
import json

import numpy as np
import pytest
from pycocotools.cocoeval import COCOeval

from bcrs.evaluate import MAX_DETS, coco_metrics
from scripts import make_tables
from scripts.make_curve_figure import cost_axis


def test_standard_coco_summary_uses_mixed_caps():
    ev = COCOeval(iouType="bbox")
    ev.params.maxDets = list(MAX_DETS)
    precision = np.empty((10, 101, 2, 4, 3))
    precision[..., 0] = 0.1
    precision[..., 1] = 0.2
    precision[..., 2] = 0.8
    ev.eval = {"params": ev.params, "counts": [10, 101, 2, 4, 3], "date": "synthetic",
               "precision": precision, "recall": np.full((10, 2, 4, 3), 0.9),
               "scores": np.zeros_like(precision)}
    ev.summarize()
    assert ev.stats[0] == pytest.approx(0.2)  # Overall AP: 100 slot.
    assert ev.stats[1:6] == pytest.approx([0.8] * 5)  # AP50/AP75/size AP: 500 slot.


def test_evaluator_caps_are_per_category_per_image(tmp_path):
    # Two categories with 120 perfect detections each, below the global 500 cap.
    anns, dets = [], []
    for category in (1, 2):
        for i in range(120):
            box = [20 * (i % 20), 20 * (i // 20), 8, 8]
            anns.append({"id": len(anns) + 1, "image_id": 1, "category_id": category,
                         "bbox": box, "area": 64, "iscrowd": 0})
            dets.append({"image_id": 1, "category_id": category, "bbox": box,
                         "score": 1 - i / 200})
    path = tmp_path / "coco.json"
    path.write_text(json.dumps({"info": {}, "images": [{"id": 1, "width": 400, "height": 120}],
                                "categories": [{"id": c, "name": str(c)} for c in (1, 2)],
                                "annotations": anns}))
    m = coco_metrics(path, dets)
    assert m["n_dets"] == 240
    assert m["AP"] == pytest.approx(84 / 101)  # Recall grid points 0.00 through 0.83.
    assert m["AR@100"] == pytest.approx(100 / 120)
    for key in ("AP50", "AP75", "AP_s", "AR@500"):
        assert m[key] == pytest.approx(1)


def test_cost_uses_annotation_area_and_fails_when_missing(tmp_path):
    d = {"split_dir": str(tmp_path), "geometry": {"s": 4, "tile": 512}}
    with pytest.raises(FileNotFoundError):
        make_tables.cost_terms(d)
    (tmp_path / "coco.json").write_text(json.dumps({"images": [
        {"width": 960, "height": 540}, {"width": 1360, "height": 765}]}))
    expected = (1 / 16, 512 ** 2 / ((960 * 540 + 1360 * 765) / 2))
    assert make_tables.cost_terms(d) == pytest.approx(expected)
    assert cost_axis(d) == pytest.approx(expected)
    curve = {str(k): {"avg_tiles": k, "AP": ap} for k, ap in enumerate((0.1, 0.3, 0.5))}
    assert make_tables.auc(curve, 0, 1) == pytest.approx(0.6)


def test_incomplete_seed_is_not_silently_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(make_tables, "R", tmp_path)
    monkeypatch.setattr(make_tables, "OUT", tmp_path)
    (tmp_path / "seed").mkdir()
    (tmp_path / "seed" / "curve_val.json").write_text('{"curves": {}}')
    assert not make_tables.seeds_table({"seeds": ["seed"]})
    assert "missing or incomplete: seed" in (tmp_path / "seeds.tex").read_text()
