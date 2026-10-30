"""Tile geometry for the BCRS fine pass.

A tile grid covers an image with tiles of size g x g and overlap margin w:
interior stride is (g - 2w); tiles are clamped to image bounds, so edge tiles
may shift inward (never shrink below the image itself for tiny images).
All coordinates are in the NATIVE-resolution frame.
"""
from __future__ import annotations

import torch
from torch import Tensor


def tile_grid(img_w: int, img_h: int, g: int = 512, overlap: int = 32) -> Tensor:
    """Return [N, 4] tile boxes (x0, y0, x1, y1) covering the image.

    Stride g - 2*overlap guarantees every point is covered and adjacent tiles
    share >= 2*overlap px of context; last row/col is clamped to the border.
    """
    stride = g - 2 * overlap
    assert stride > 0, "overlap too large for tile size"

    def starts(size: int) -> list[int]:
        if size <= g:
            return [0]
        xs = list(range(0, size - g, stride))
        xs.append(size - g)  # clamp final tile to the border
        return xs

    boxes = []
    for y0 in starts(img_h):
        for x0 in starts(img_w):
            boxes.append([x0, y0, min(x0 + g, img_w), min(y0 + g, img_h)])
    return torch.tensor(boxes, dtype=torch.float32)


def boxes_in_tile(boxes_xyxy: Tensor, tile: Tensor, min_center_inside: bool = True) -> Tensor:
    """Boolean mask of boxes whose center lies inside a tile. NOTE: with the
    overlapping grid from tile_grid, a center in an overlap band satisfies this
    for MORE THAN ONE tile — use assign_boxes_to_tiles for a unique owner."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes_xyxy.device)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    if min_center_inside:
        return (cx >= tile[0]) & (cx < tile[2]) & (cy >= tile[1]) & (cy < tile[3])
    x0 = boxes_xyxy[:, 0].clamp(min=float(tile[0]))
    y0 = boxes_xyxy[:, 1].clamp(min=float(tile[1]))
    x1 = boxes_xyxy[:, 2].clamp(max=float(tile[2]))
    y1 = boxes_xyxy[:, 3].clamp(max=float(tile[3]))
    return (x1 > x0) & (y1 > y0)


def assign_boxes_to_tiles(boxes_xyxy: Tensor, tiles: Tensor) -> Tensor:
    """Unique owner tile per box: among tiles containing the box center, the
    one whose own center is nearest (ties broken by lowest tile index). Used
    wherever per-object counts must not double-count across overlap bands
    (self-derived targets, oracle ranking, per-tile bookkeeping).

    Returns LongTensor [M] of tile indices; grid coverage guarantees every
    in-image center has at least one containing tile."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes_xyxy.device)
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
    contains = (
        (cx[:, None] >= tiles[None, :, 0]) & (cx[:, None] < tiles[None, :, 2])
        & (cy[:, None] >= tiles[None, :, 1]) & (cy[:, None] < tiles[None, :, 3])
    )
    tcx = (tiles[:, 0] + tiles[:, 2]) / 2
    tcy = (tiles[:, 1] + tiles[:, 3]) / 2
    d2 = (cx[:, None] - tcx[None, :]) ** 2 + (cy[:, None] - tcy[None, :]) ** 2
    d2 = d2.masked_fill(~contains, float("inf"))
    return d2.argmin(dim=1)


def crop_tile(image: Tensor, tile: Tensor) -> Tensor:
    """Crop [3,H,W] image to a tile box (no interpolation — a fixed gather)."""
    x0, y0, x1, y1 = (int(v) for v in tile.tolist())
    return image[:, y0:y1, x0:x1]


def tile_boxes_to_frame(boxes_xyxy: Tensor, tile: Tensor) -> Tensor:
    """Map tile-local detection boxes back to frame coordinates."""
    out = boxes_xyxy.clone()
    out[:, [0, 2]] += tile[0]
    out[:, [1, 3]] += tile[1]
    return out


def truncated_at_tile_border(
    boxes_xyxy: Tensor, tile: Tensor, img_w: int, img_h: int, tol: float = 2.0
) -> Tensor:
    """Mask of tile-local boxes touching a NON-image tile border (candidates
    for suppression in fusion: the object likely continues outside the tile)."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=boxes_xyxy.device)
    tw, th = tile[2] - tile[0], tile[3] - tile[1]
    left = (boxes_xyxy[:, 0] <= tol) & (tile[0] > 0)
    top = (boxes_xyxy[:, 1] <= tol) & (tile[1] > 0)
    right = (boxes_xyxy[:, 2] >= tw - tol) & (tile[2] < img_w)
    bottom = (boxes_xyxy[:, 3] >= th - tol) & (tile[3] < img_h)
    return left | top | right | bottom
