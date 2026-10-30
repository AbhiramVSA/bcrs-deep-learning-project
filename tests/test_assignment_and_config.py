"""Tests for review findings: unique tile ownership and config-key validation."""
import pytest
import torch

from bcrs.tiles import assign_boxes_to_tiles, boxes_in_tile, tile_grid
from bcrs.train import validate_config_keys


def test_overlap_band_center_matches_two_tiles_but_owner_is_unique():
    tiles = tile_grid(2000, 1500, g=512, overlap=32)
    # a box centered in the overlap band between the first two column tiles
    box = torch.tensor([[470.0, 100.0, 490.0, 120.0]])  # center x=480 in [448,512)
    n_matching = sum(bool(boxes_in_tile(box, t).item()) for t in tiles)
    assert n_matching >= 2  # documents the multi-match behavior
    owner = assign_boxes_to_tiles(box, tiles)
    assert owner.shape == (1,)
    t = tiles[owner[0]]
    assert t[0] <= 480 < t[2] and t[1] <= 110 < t[3]


def test_every_box_gets_exactly_one_owner():
    torch.manual_seed(0)
    tiles = tile_grid(2000, 1500)
    centers_x = torch.rand(200) * 1998 + 1
    centers_y = torch.rand(200) * 1498 + 1
    boxes = torch.stack(
        [centers_x - 1, centers_y - 1, centers_x + 1, centers_y + 1], dim=1
    )
    owners = assign_boxes_to_tiles(boxes, tiles)
    assert owners.shape == (200,)
    assert (owners >= 0).all() and (owners < len(tiles)).all()
    for i in range(200):
        t = tiles[owners[i]]
        assert t[0] <= centers_x[i] < t[2] and t[1] <= centers_y[i] < t[3]


def test_unknown_config_key_rejected():
    validate_config_keys({"run_name": "x", "epochs": 5})  # valid keys pass
    with pytest.raises(SystemExit, match="warmup_iter"):
        validate_config_keys({"warmup_iter": 500})  # typo (missing s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_scorer_accepts_bf16_features_outside_autocast():
    """Sweep-time ranking: bf16 features from an autocast coarse pass, scorer
    called with no autocast context -> must not raise a dtype mismatch."""
    from bcrs.models.scorer import TileScorer

    scorer = TileScorer(in_channels=64, hidden=32, num_blocks=2).cuda().eval()
    feat = torch.randn(1, 64, 4, 4, device="cuda", dtype=torch.bfloat16)
    tiles = [tile_grid(1024, 1024)]
    with torch.no_grad():
        scores = scorer(feat, torch.tensor([0.5], device="cuda"), tiles, 256.0)
    assert scores[0].shape == (len(tiles[0]),)
    assert torch.isfinite(scores[0]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_scorer_runs_on_cuda_with_cpu_tiles():
    """Review finding: rois must be moved to the feature device/dtype."""
    from bcrs.models.scorer import TileScorer

    scorer = TileScorer(in_channels=64, hidden=32, num_blocks=2).cuda()
    feat = torch.randn(1, 64, 4, 4, device="cuda")
    tiles = [tile_grid(1024, 1024)]  # CPU tensors, as produced in the pipeline
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scores = scorer(feat, torch.tensor([0.5], device="cuda"), tiles, 256.0)
    assert scores[0].shape == (len(tiles[0]),)
    assert torch.isfinite(scores[0]).all()
