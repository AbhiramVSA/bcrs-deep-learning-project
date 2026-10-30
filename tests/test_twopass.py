"""Unit tests for the two-pass fusion machinery (pure CPU, no model)."""
import torch

from bcrs.tiles import tile_grid
from bcrs.twopass import downsample_image, fuse, oracle_tile_scores, unmatched_gt_mask


def _det(boxes, scores, labels):
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "scores": torch.tensor(scores, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def test_downsample_shape():
    img = torch.rand(3, 1500, 2000)
    small = downsample_image(img, 4.0)
    assert small.shape == (3, 375, 500)
    assert downsample_image(img, 1.0) is img


def test_unmatched_gt_mask():
    gt_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0], [500.0, 500.0, 600.0, 600.0]])
    gt_labels = torch.tensor([4, 1])
    det = _det([[5, 5, 95, 95], [500, 500, 600, 600]], [0.9, 0.9], [4, 2])
    miss = unmatched_gt_mask(gt_boxes, gt_labels, det)
    # first GT matched (same class, high IoU); second unmatched (wrong class)
    assert miss.tolist() == [False, True]
    # low-score detections don't count as matches
    det_low = _det([[5, 5, 95, 95]], [0.1], [4])
    assert unmatched_gt_mask(gt_boxes, gt_labels, det_low).tolist() == [True, True]


def test_oracle_scores_count_missed_objects_once():
    tiles = tile_grid(2000, 1500)
    # two GT objects in the same tile region, one detected by coarse
    gt_boxes = torch.tensor([[100.0, 100.0, 140.0, 140.0], [200.0, 200.0, 240.0, 240.0]])
    gt_labels = torch.tensor([1, 1])
    coarse = _det([[100, 100, 140, 140]], [0.9], [1])
    scores = oracle_tile_scores(gt_boxes, gt_labels, coarse, tiles)
    assert scores.sum().item() == 1.0  # exactly the one missed object
    assert scores.max().item() == 1.0


def test_fuse_drops_truncated_and_downweights_coarse():
    tiles = tile_grid(2000, 1500)
    coarse = _det([[100, 100, 150, 150], [1800, 1300, 1900, 1400]], [0.8, 0.7], [4, 4])
    tile_outs = []
    for i in range(len(tiles)):
        if i == 0:
            out = _det([[100, 100, 150, 150], [300, 300, 350, 350]], [0.9, 0.85], [4, 4])
            out["truncated"] = torch.tensor([False, True])  # second one truncated
        else:
            out = _det([], [], [])
            out["truncated"] = torch.zeros(0, dtype=torch.bool)
        tile_outs.append(out)

    fused = fuse(coarse, tile_outs, selected=torch.tensor([0]), tiles=tiles)
    # truncated fine box is gone
    assert not any(abs(b[0] - 300.0) < 1 for b in fused["boxes"].tolist())
    # fine box wins NMS over the down-weighted coarse duplicate at (100,100)
    kept_scores = dict(zip(map(tuple, fused["boxes"].round().tolist()), fused["scores"].tolist()))
    assert abs(kept_scores[(100.0, 100.0, 150.0, 150.0)] - 0.9) < 1e-6
    # coarse box far from any selected tile keeps its score
    assert any(abs(s - 0.7) < 1e-6 for s in fused["scores"].tolist())


def test_fuse_size_aware_keeps_large_coarse_boxes_at_full_weight():
    tiles = tile_grid(2000, 1500)
    # small coarse box (40 px) and large coarse box (300 px), both inside tile 0
    coarse = _det([[100, 100, 140, 140], [50, 50, 350, 350]], [0.8, 0.7], [4, 9])
    empty = _det([], [], [])
    empty["truncated"] = torch.zeros(0, dtype=torch.bool)
    tile_outs = [empty for _ in range(len(tiles))]
    fused = fuse(coarse, tile_outs, selected=torch.tensor([0]), tiles=tiles,
                 coarse_in_tile_weight=0.5, coarse_full_min_side=128.0)
    by_label = {int(l): float(s) for s, l in zip(fused["scores"], fused["labels"])}
    assert abs(by_label[4] - 0.4) < 1e-6   # small: down-weighted
    assert abs(by_label[9] - 0.7) < 1e-6   # large: full weight
    # default (no size rule) down-weights both
    fused0 = fuse(coarse, tile_outs, selected=torch.tensor([0]), tiles=tiles, coarse_in_tile_weight=0.5)
    assert all(abs(float(s) - w) < 1e-6 for s, w in zip(sorted(fused0["scores"].tolist()), [0.35, 0.4]))


def test_fuse_k0_is_pure_coarse():
    tiles = tile_grid(2000, 1500)
    coarse = _det([[10, 10, 60, 60]], [0.5], [2])
    fused = fuse(coarse, [], selected=torch.zeros(0, dtype=torch.long), tiles=tiles)
    assert fused["boxes"].shape[0] == 1
    assert abs(fused["scores"][0].item() - 0.5) < 1e-6
