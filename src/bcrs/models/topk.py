"""Differentiable perturbed top-k (Berthet et al. 2020, NeurIPS).

Soft indicator: z_tilde(s) = E_eps[ topk_indicator(s + sigma*eps) ],
estimated with num_samples Monte Carlo draws of Gaussian noise.
Backward uses the score-function Jacobian estimator
    J = E[ z eps^T ] / sigma        (Gaussian noise)
as in Cordonnier et al. 2021 (differentiable patch selection).

`hard_topk_straight_through` gives the training-time configuration from the
proposal: hard selection on the forward pass (fine pass sees exactly the
inference-time input distribution), relaxed gradients on the backward pass.
"""
from __future__ import annotations

import torch
from torch import Tensor


def hard_topk_indicator(scores: Tensor, k: int) -> Tensor:
    """[..., N] scores -> {0,1} indicator of the top-k entries (no gradient)."""
    idx = scores.topk(k, dim=-1).indices
    return torch.zeros_like(scores).scatter_(-1, idx, 1.0)


class _PerturbedTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores: Tensor, k: int, num_samples: int, sigma: float):
        noise = torch.randn(
            (num_samples,) + scores.shape, device=scores.device, dtype=scores.dtype
        )
        perturbed = scores.unsqueeze(0) + sigma * noise          # [S, ..., N]
        indicators = hard_topk_indicator(perturbed, k)           # [S, ..., N]
        ctx.save_for_backward(indicators, noise)
        ctx.sigma = sigma
        ctx.num_samples = num_samples
        return indicators.mean(dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        indicators, noise = ctx.saved_tensors
        # vjp: J^T g with J = E[z eps^T] / sigma
        # per sample: eps * <z, g> summed over ... no — elementwise structure:
        # J_ab = E[z_a eps_b] / sigma ; (J^T g)_b = E[ eps_b * <z, g> ] / sigma
        inner = (indicators * grad_output.unsqueeze(0)).sum(dim=-1, keepdim=True)
        grad_scores = (noise * inner).mean(dim=0) / ctx.sigma
        return grad_scores, None, None, None


def perturbed_topk(scores: Tensor, k: int, num_samples: int = 100, sigma: float = 0.5) -> Tensor:
    """Soft top-k indicator in [0,1], sums to k in expectation. Differentiable."""
    return _PerturbedTopK.apply(scores, k, num_samples, sigma)


def hard_topk_straight_through(
    scores: Tensor, k: int, num_samples: int = 100, sigma: float = 0.5
) -> Tensor:
    """Hard {0,1} top-k on forward, perturbed-top-k gradient on backward."""
    soft = perturbed_topk(scores, k, num_samples, sigma)
    hard = hard_topk_indicator(scores, k)
    return hard + (soft - soft.detach())


def gumbel_topk_straight_through(scores: Tensor, k: int, tau: float = 0.5) -> Tensor:
    """Ablation alternative (Jang et al. 2017 / Xie et al. 2020 style): hard
    top-k forward, backward through k successive Gumbel-softmax draws without
    replacement (relaxed subset sampling). Scores are treated as logits."""
    hard = hard_topk_indicator(scores, k)
    logits = scores.clone()
    soft = torch.zeros_like(scores)
    for _ in range(k):
        g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-10)))
        p = torch.softmax((logits + g) / tau, dim=-1)
        soft = soft + p
        logits = logits + torch.log1p(-p.detach().clamp(max=1 - 1e-6))  # remove mass of the drawn tile
    return hard + (soft - soft.detach())


def hard_topk_no_grad(scores: Tensor, k: int) -> Tensor:
    """Ablation: no gradient through the selection at all (scorer learns from
    BCE/ranking targets only)."""
    return hard_topk_indicator(scores, k)
