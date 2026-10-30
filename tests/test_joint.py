import torch

from bcrs.joint import fcos_train_outputs, per_tile_fine_losses, recovered_counts, tile_targets
from bcrs.models import build_detector
from bcrs.tiles import tile_grid


def test_tile_targets_ownership_clip_and_shift():
    tile = torch.tensor([512.0, 0.0, 1024.0, 512.0])
    boxes = torch.tensor([
        [600.0, 100.0, 640.0, 140.0],   # inside -> shifted
        [1000.0, 200.0, 1100.0, 260.0], # center x=1050 outside -> dropped
        [1010.0, 300.0, 1030.0, 330.0], # center 1020 inside, spills past 1024 -> clipped to 512
        [100.0, 100.0, 140.0, 140.0],   # outside
    ])
    labels = torch.tensor([1, 2, 3, 4])
    t = tile_targets(boxes, labels, tile)
    assert t["labels"].tolist() == [1, 3]
    assert t["boxes"][0].tolist() == [88.0, 100.0, 128.0, 140.0]
    assert t["boxes"][1].tolist() == [498.0, 300.0, 512.0, 330.0]
    empty = tile_targets(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), tile)
    assert empty["boxes"].shape == (0, 4)


def test_train_outputs_match_plain_forward_and_give_detections():
    model = build_detector("fcos_r18", pretrained_backbone=False, score_thresh=0.0)
    model.train()
    imgs = [torch.rand(3, 128, 160), torch.rand(3, 96, 128)]
    tgts = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 60.0]]), "labels": torch.tensor([4])},
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    torch.manual_seed(0)
    ref = model(imgs, tgts)
    torch.manual_seed(0)
    losses, feats, dets = fcos_train_outputs(model, imgs, tgts)
    for k in ref:
        assert torch.allclose(ref[k], losses[k]), k
    assert len(feats) == 5 and len(dets) == 2
    assert dets[0]["boxes"].shape[1] == 4 and not dets[0]["boxes"].requires_grad
    assert sum(losses.values()).requires_grad


def _tiles_and_targets():
    torch.manual_seed(0)
    tiles_img = [torch.rand(3, 128, 128), torch.rand(3, 128, 128), torch.rand(3, 96, 128)]
    tgts = [{"boxes": torch.tensor([[8.0, 8.0, 40.0, 48.0]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[30.0, 30.0, 90.0, 100.0], [5.0, 60.0, 25.0, 90.0]]), "labels": torch.tensor([4, 2])},
            {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    return tiles_img, tgts


def test_per_tile_losses_exact_with_frozen_bn():
    """With a frozen backbone, batched per-tile losses equal solo runs exactly
    for full-size tiles; a padded (smaller) tile differs only in the background
    classification term over its padded area, which is inherent to batching."""
    model = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=True)
    model.train()
    tiles_img, tgts = _tiles_and_targets()
    losses, dets, feats = per_tile_fine_losses(model, tiles_img, tgts)
    assert len(losses) == 3 and len(dets) == 3
    for t in range(2):  # full-size tiles: exact
        ref = model([tiles_img[t]], [tgts[t]])
        for k in ref:
            assert torch.allclose(ref[k], losses[t][k], atol=1e-5), (t, k)
    ref = model([tiles_img[2]], [tgts[2]])  # padded, no GT: only cls differs, both tiny
    assert ref["bbox_regression"].item() == losses[2]["bbox_regression"].item() == 0.0
    assert abs(ref["classification"].item() - losses[2]["classification"].item()) < 0.01


def test_trainable_bn_couples_tiles_through_batch_statistics():
    """Documents why joint training uses frozen BN: with trainable BatchNorm in
    train mode, a tile's loss depends on the other tiles in the batch."""
    model = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=False)
    model.train()
    tiles_img, tgts = _tiles_and_targets()
    losses, _, _ = per_tile_fine_losses(model, tiles_img[:2], tgts[:2])
    ref = model([tiles_img[0]], [tgts[0]])
    assert not torch.allclose(ref["classification"], losses[0]["classification"], atol=1e-4)


def test_recovered_counts():
    tile = torch.tensor([512.0, 0.0, 1024.0, 512.0])
    missed = torch.tensor([[600.0, 100.0, 640.0, 140.0], [700.0, 200.0, 740.0, 240.0], [100.0, 100.0, 140.0, 140.0]])
    labels = torch.tensor([1, 1, 1])
    # fine pass finds the first (tile coords 88..128) with the right class, misses the second
    fine = {"boxes": torch.tensor([[90.0, 102.0, 127.0, 139.0], [300.0, 300.0, 340.0, 340.0]]),
            "scores": torch.tensor([0.9, 0.9]), "labels": torch.tensor([1, 1])}
    assert recovered_counts(missed, labels, tile, fine) == 1
    # low-confidence detection does not count
    fine_low = {"boxes": fine["boxes"], "scores": torch.tensor([0.1, 0.9]), "labels": fine["labels"]}
    assert recovered_counts(missed, labels, tile, fine_low) == 0
    assert recovered_counts(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), tile, fine) == 0
