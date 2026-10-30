"""fcos_forward_with_features must be exactly equivalent to torchvision's
FCOS.forward in both modes — it only adds the FPN features to the return."""
import torch

from bcrs.models import build_detector
from bcrs.twopass import fcos_forward_with_features


def _inputs():
    torch.manual_seed(0)
    imgs = [torch.rand(3, 160, 224), torch.rand(3, 128, 192)]
    tgts = [
        {"boxes": torch.tensor([[10.0, 10.0, 60.0, 70.0], [100.0, 40.0, 140.0, 90.0]]),
         "labels": torch.tensor([4, 1])},
        {"boxes": torch.tensor([[20.0, 30.0, 90.0, 100.0]]), "labels": torch.tensor([9])},
    ]
    return imgs, tgts


def test_eval_mode_identical_detections_and_feature_shapes():
    model = build_detector("fcos_r18", pretrained_backbone=False, score_thresh=0.0)
    model.eval()
    imgs, _ = _inputs()
    with torch.no_grad():
        ref = model(imgs)
        out, feats = fcos_forward_with_features(model, imgs)
    assert len(feats) == 5  # P3..P7
    # P3 stride 8 on the padded batch canvas (160x224 -> 20x28)
    assert feats[0].shape[-2:] == (20, 28)
    assert feats[-1].shape[-2:] == (2, 2)  # P7 stride 128 -> ceil(160/128)=2, ceil(224/128)=2
    for r, o in zip(ref, out):
        assert torch.equal(r["boxes"], o["boxes"])
        assert torch.equal(r["scores"], o["scores"])
        assert torch.equal(r["labels"], o["labels"])


def test_train_mode_identical_losses():
    model = build_detector("fcos_r18", pretrained_backbone=False)
    model.train()
    imgs, tgts = _inputs()
    torch.manual_seed(1)
    ref = model(imgs, tgts)
    torch.manual_seed(1)
    out, feats = fcos_forward_with_features(model, imgs, tgts)
    assert set(ref) == set(out)
    for k in ref:
        assert torch.allclose(ref[k], out[k]), k
    assert all(f.requires_grad for f in feats)
