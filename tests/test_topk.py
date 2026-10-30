"""Unit tests for the perturbed top-k operator (PLAN.md: gradient correctness
is gated before the operator enters the training graph)."""
import torch

from bcrs.models.topk import hard_topk_indicator, hard_topk_straight_through, perturbed_topk


def test_hard_indicator():
    s = torch.tensor([[0.1, 3.0, -1.0, 2.0]])
    z = hard_topk_indicator(s, 2)
    assert z.tolist() == [[0.0, 1.0, 0.0, 1.0]]


def test_soft_properties():
    torch.manual_seed(0)
    s = torch.randn(8, 12)
    z = perturbed_topk(s, k=3, num_samples=500, sigma=0.5)
    assert z.shape == s.shape
    assert (z >= 0).all() and (z <= 1).all()
    assert torch.allclose(z.sum(-1), torch.full((8,), 3.0), atol=1e-4)
    # low noise -> concentrates on the hard selection
    z_cold = perturbed_topk(s, k=3, num_samples=500, sigma=1e-3)
    assert torch.allclose(z_cold, hard_topk_indicator(s, 3), atol=0.05)


def test_gradient_against_finite_differences():
    """MC-gradient of f(s) = <w, z_tilde(s)> vs central finite differences of
    the true expectation (both estimated with large sample counts)."""
    torch.manual_seed(0)
    n, k, sigma = 6, 2, 0.5
    s = torch.randn(n, requires_grad=True)
    w = torch.randn(n)

    S = 200_000
    z = perturbed_topk(s.unsqueeze(0), k, num_samples=S, sigma=sigma)[0]
    (z * w).sum().backward()
    g_est = s.grad.clone()

    def true_f(x: torch.Tensor) -> float:
        noise = torch.randn(S, n)
        ind = hard_topk_indicator(x.unsqueeze(0) + sigma * noise, k)
        return float((ind.mean(0) * w).sum())

    eps = 0.05
    g_fd = torch.zeros(n)
    torch.manual_seed(1)
    for i in range(n):
        e = torch.zeros(n)
        e[i] = eps
        g_fd[i] = (true_f(s.detach() + e) - true_f(s.detach() - e)) / (2 * eps)

    err = (g_est - g_fd).abs().max().item()
    scale = g_fd.abs().max().item()
    assert err < 0.15 * max(scale, 1.0), f"grad mismatch: {g_est} vs {g_fd} (err {err:.3f})"


def test_straight_through_is_hard_forward():
    torch.manual_seed(0)
    s = torch.randn(4, 10, requires_grad=True)
    z = hard_topk_straight_through(s, k=3, num_samples=50, sigma=0.5)
    assert set(z.detach().unique().tolist()) <= {0.0, 1.0}
    z.sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
