import torch
from torchvision.ops.misc import FrozenBatchNorm2d

from bcrs.models import build_detector
from bcrs.models.detector import build_detector_from_ckpt


def test_frozen_bn_replaces_all_backbone_batchnorm():
    m = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=True)
    kinds = {type(mod) for mod in m.backbone.body.modules()}
    assert torch.nn.BatchNorm2d not in kinds
    assert FrozenBatchNorm2d in kinds
    m.train()
    # frozen stats: no BN parameter receives gradients, no running stats update
    assert not any(p.requires_grad for mod in m.backbone.body.modules()
                   if isinstance(mod, FrozenBatchNorm2d) for p in mod.parameters())


def test_checkpoint_roundtrip_respects_frozen_bn_flag():
    for flag in (False, True):
        src = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=flag)
        ckpt = {"model": src.state_dict(), "cfg": {"arch": "fcos_r18", "frozen_bn": flag}}
        dst = build_detector_from_ckpt(ckpt)  # strict load must succeed
        assert (torch.nn.BatchNorm2d in {type(x) for x in dst.backbone.body.modules()}) == (not flag)


def test_old_checkpoints_without_flag_default_to_trainable_bn():
    src = build_detector("fcos_r18", pretrained_backbone=False, frozen_bn=False)
    ckpt = {"model": src.state_dict(), "cfg": {"arch": "fcos_r18"}}  # P1 checkpoints lack the key
    dst = build_detector_from_ckpt(ckpt)
    assert torch.nn.BatchNorm2d in {type(x) for x in dst.backbone.body.modules()}
