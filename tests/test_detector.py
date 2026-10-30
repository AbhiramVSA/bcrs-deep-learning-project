"""Regression tests for the detector wrapper (CPU, tiny inputs)."""
import torch

from bcrs.models import build_detector


def _tiny_model():
    return build_detector("fcos_r18", pretrained_backbone=False)


def test_empty_target_image_trains():
    """An image whose boxes were all filtered (e.g. sub-pixel at s=4) must not
    break the loss or produce non-finite gradients."""
    model = _tiny_model()
    model.train()
    imgs = [torch.rand(3, 128, 160), torch.rand(3, 96, 128)]
    tgts = [
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)},
        {"boxes": torch.tensor([[8.0, 8.0, 40.0, 48.0]]), "labels": torch.tensor([3])},
    ]
    losses = model(imgs, tgts)
    total = sum(losses.values())
    assert torch.isfinite(total)
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_identity_transform_preserves_resolution():
    """The model must never resize inputs — resolution is owned by the data
    pipeline (PLAN.md section 2)."""
    model = _tiny_model()
    model.eval()
    img = torch.rand(3, 100, 140)
    transformed, _ = model.transform([img])
    # only padding to size_divisible=32 is allowed
    assert transformed.image_sizes[0] == (100, 140)
    assert transformed.tensors.shape[-2:] == (128, 160)


def test_inference_labels_are_visdrone_ids():
    model = _tiny_model()
    model.eval()
    with torch.no_grad():
        out = model([torch.rand(3, 128, 128)])
    assert out[0]["labels"].max().item() <= 10 if out[0]["labels"].numel() else True
    assert out[0]["boxes"].shape[0] <= 500
