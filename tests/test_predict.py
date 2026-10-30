import pytest
import torch
from PIL import Image

from scripts import predict
from scripts.predict import annotate_detections, select_tiles


def test_select_tiles_uses_only_top_k_and_handles_zero():
    tiles = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    scores = torch.tensor([0.1, 0.9, 0.5])

    indices, selected = select_tiles(scores, tiles, 2)
    assert indices.tolist() == [1, 2]
    assert torch.equal(selected, tiles[[1, 2]])

    empty_indices, empty = select_tiles(scores, tiles, 0)
    assert empty_indices.numel() == 0
    assert empty.shape == (0, 4)


def test_select_tiles_clamps_k_and_rejects_invalid_input():
    tiles = torch.zeros((2, 4))
    scores = torch.ones(2)
    assert len(select_tiles(scores, tiles, 10)[0]) == 2
    with pytest.raises(ValueError, match="zero or greater"):
        select_tiles(scores, tiles, -1)
    with pytest.raises(ValueError, match="same number"):
        select_tiles(scores[:1], tiles, 1)


def test_annotation_and_export_rows_skip_background_label():
    image = Image.new("RGB", (30, 30), color="white")
    detections = {
        "boxes": torch.tensor([[1, 1, 8, 8], [15, 15, 25, 25]], dtype=torch.float32),
        "scores": torch.tensor([0.99, 0.8]),
        "labels": torch.tensor([0, 1]),
    }

    rows = annotate_detections(image, detections)

    assert [row["label"] for row in rows] == [1]
    assert rows[0]["category"] == "pedestrian"
    assert image.getpixel((1, 1)) == (255, 255, 255)
    assert image.getpixel((15, 15)) == (255, 0, 0)


def test_predict_runs_fine_pass_only_on_selected_tiles(monkeypatch):
    tiles = torch.tensor([[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]], dtype=torch.float32)
    fine_tiles = []
    fuse_calls = []
    monkeypatch.setattr(predict, "tile_grid", lambda *args: tiles)
    monkeypatch.setattr(predict, "coarse_pass", lambda *args, **kwargs: ("coarse", [torch.zeros(1)]))

    class Scorer:
        def __call__(self, *args):
            return [torch.tensor([0.1, 0.9, 0.5])]

    def tile_pass(_detector, _image, selected, _device):
        fine_tiles.extend(selected.tolist())
        return []

    def fuse(_coarse, _fine, selected, selected_tiles, **kwargs):
        fuse_calls.append((selected, selected_tiles, kwargs))
        return "fused"

    monkeypatch.setattr(predict, "tile_pass", tile_pass)
    monkeypatch.setattr(predict, "fuse", fuse)
    cfg = {"s": 4, "tile": 10, "overlap": 1, "budget_norm_k": 20}

    result = predict.predict_image(torch.zeros(3, 10, 30), None, Scorer(), cfg, torch.device("cpu"), 1)
    assert result[0] == "fused"
    assert result[1].tolist() == [1]
    assert fine_tiles == [tiles[1].tolist()]
    assert fuse_calls[0][0].tolist() == [0]
    assert torch.equal(fuse_calls[0][1], tiles[[1]])
    assert fuse_calls[0][2]["coarse_full_min_side"] == 2

    fine_tiles.clear()
    predict.predict_image(torch.zeros(3, 10, 30), None, Scorer(), cfg, torch.device("cpu"), 0)
    assert fine_tiles == []
    assert fuse_calls[1][0].numel() == 0
    assert fuse_calls[1][1].shape == (0, 4)
