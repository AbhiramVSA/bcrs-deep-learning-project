"""Budget-conditioned tile scoring head g_phi (proposal section 4.2).

Input: the coarsest FPN level of the coarse pass (P7 by default) plus a scalar
budget in [0,1]. Four 3x3 conv-GN-ReLU blocks, FiLM modulation from the budget
embedding after each block, then tile-aligned average pooling to one logit per
tile.

Tile pooling: tiles are defined in native-frame coords; on a feature map of
stride `stride` relative to the COARSE image (which is the native image
downsampled by s), a native-frame tile box maps to feature coords by dividing
by (s * stride).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torchvision


class FiLM(nn.Module):
    def __init__(self, budget_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(budget_dim, 2 * channels)

    def forward(self, x: Tensor, budget_emb: Tensor) -> Tensor:
        gamma, beta = self.proj(budget_emb).chunk(2, dim=-1)  # [B, C] each
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class TileScorer(nn.Module):
    def __init__(self, in_channels: int = 256, hidden: int = 128, budget_dim: int = 64,
                 num_blocks: int = 4):
        super().__init__()
        self.budget_mlp = nn.Sequential(
            nn.Linear(1, budget_dim), nn.ReLU(inplace=True),
            nn.Linear(budget_dim, budget_dim),
        )
        blocks, films = [], []
        c = in_channels
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(c, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(32, hidden),
                nn.ReLU(inplace=True),
            ))
            films.append(FiLM(budget_dim, hidden))
            c = hidden
        self.blocks = nn.ModuleList(blocks)
        self.films = nn.ModuleList(films)
        self.out_conv = nn.Conv2d(hidden, 1, 1)

    def forward(self, feat: Tensor, budget: Tensor, tiles: list[Tensor],
                feat_stride_native: float) -> list[Tensor]:
        """feat: [B, C, H, W] coarsest FPN level (from the coarse pass).
        budget: [B] scalar in [0,1]. tiles: per-image [N_i, 4] native-coord
        tile boxes. feat_stride_native: native pixels per feature cell
        (= coarse downsample s * FPN stride).

        Returns per-image score vectors [N_i]."""
        emb = self.budget_mlp(budget.unsqueeze(-1).float())      # [B, D]
        # features may arrive in bf16 from an autocast coarse pass while this
        # module is called outside autocast (e.g. sweep-time ranking): match
        # the parameter dtype; under autocast the convs re-cast as usual.
        x = feat.to(self.out_conv.weight.dtype)
        for block, film in zip(self.blocks, self.films):
            x = film(block(x), emb)
        score_map = self.out_conv(x)                              # [B, 1, H, W]

        # tile-aligned average pooling via roi_align on the score map
        # (rois must match the score map's device AND dtype, incl. under autocast)
        rois = [
            t.to(device=score_map.device, dtype=score_map.dtype) / feat_stride_native
            for t in tiles
        ]
        pooled = torchvision.ops.roi_align(
            score_map, rois, output_size=1, spatial_scale=1.0, aligned=True,
        )                                                         # [sum N_i, 1, 1, 1]
        scores = pooled.flatten()
        out, off = [], 0
        for t in tiles:
            out.append(scores[off:off + len(t)])
            off += len(t)
        return out
