import torch

from bcrs.tiles import (
    boxes_in_tile,
    crop_tile,
    tile_boxes_to_frame,
    tile_grid,
    truncated_at_tile_border,
)


def test_grid_covers_visdrone_frame():
    g = tile_grid(2000, 1500, g=512, overlap=32)
    assert g.shape[1] == 4
    # full coverage: every pixel corner sample inside some tile
    for x in range(0, 2000, 97):
        for y in range(0, 1500, 97):
            inside = ((g[:, 0] <= x) & (x < g[:, 2]) & (g[:, 1] <= y) & (y < g[:, 3])).any()
            assert inside, (x, y)
    # tiles are g x g except clamped edges
    w = g[:, 2] - g[:, 0]
    h = g[:, 3] - g[:, 1]
    assert (w <= 512).all() and (h <= 512).all()
    # proposal expects on the order of a dozen tiles for a 2000x1500 frame
    assert 12 <= g.shape[0] <= 20, g.shape[0]


def test_small_image_single_tile():
    g = tile_grid(400, 300, g=512, overlap=32)
    assert g.shape[0] == 1
    assert g[0].tolist() == [0.0, 0.0, 400.0, 300.0]


def test_box_assignment_and_mapping():
    tile = torch.tensor([512.0, 0.0, 1024.0, 512.0])
    boxes = torch.tensor([
        [600.0, 100.0, 640.0, 140.0],   # center inside
        [100.0, 100.0, 140.0, 140.0],   # outside
        [1020.0, 500.0, 1060.0, 540.0], # center outside (right of tile)
    ])
    mask = boxes_in_tile(boxes, tile)
    assert mask.tolist() == [True, False, False]

    local = boxes - torch.tensor([512.0, 0.0, 512.0, 0.0])
    back = tile_boxes_to_frame(local, tile)
    assert torch.allclose(back, boxes)


def test_crop_and_truncation():
    img = torch.arange(3 * 100 * 100, dtype=torch.float32).reshape(3, 100, 100)
    tile = torch.tensor([10.0, 20.0, 60.0, 80.0])
    crop = crop_tile(img, tile)
    assert crop.shape == (3, 60, 50)
    assert torch.equal(crop, img[:, 20:80, 10:60])

    # box flush against the tile's right edge (an interior border) is truncated;
    # a box at the tile's left edge is NOT (tile starts at image border x0=0? no, x0=10>0)
    boxes_local = torch.tensor([
        [45.0, 10.0, 50.0, 20.0],  # touches right border (interior) -> truncated
        [10.0, 10.0, 20.0, 20.0],  # interior box -> fine
    ])
    trunc = truncated_at_tile_border(boxes_local, tile, img_w=100, img_h=100)
    assert trunc.tolist() == [True, False]
