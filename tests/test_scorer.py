import torch

from bcrs.models.scorer import TileScorer
from bcrs.tiles import tile_grid


def test_scorer_shapes_and_budget_sensitivity():
    torch.manual_seed(0)
    scorer = TileScorer(in_channels=256)
    # coarse image = 2000x1500 / s=4 -> 500x375; P7 stride 128 -> ~4x3 feature map
    feat = torch.randn(2, 256, 3, 4)
    tiles = [tile_grid(2000, 1500), tile_grid(1360, 765)]
    stride_native = 4 * 128.0
    budget = torch.tensor([0.2, 0.8])

    scores = scorer(feat, budget, tiles, stride_native)
    assert len(scores) == 2
    assert scores[0].shape == (len(tiles[0]),)
    assert scores[1].shape == (len(tiles[1]),)
    assert all(torch.isfinite(s).all() for s in scores)

    # FiLM conditioning: same features, different budget -> different scores
    scores_b2 = scorer(feat, torch.tensor([0.8, 0.2]), tiles, stride_native)
    assert not torch.allclose(scores[0], scores_b2[0])


def test_scorer_gradients_flow_to_budget_mlp():
    scorer = TileScorer(in_channels=64, hidden=32, num_blocks=2)
    feat = torch.randn(1, 64, 4, 4)
    tiles = [tile_grid(1024, 1024)]
    scores = scorer(feat, torch.tensor([0.5]), tiles, feat_stride_native=256.0)
    scores[0].sum().backward()
    g = scorer.budget_mlp[0].weight.grad
    assert g is not None and g.abs().sum() > 0
