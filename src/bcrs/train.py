"""Training entry point.

    python -m bcrs.train --config configs/<name>.yaml

Writes results/<run_name>/metrics.json + train_log.txt (committed to git) and
runs/<run_name>/*.pth checkpoints (gitignored, stay on the machine).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from bcrs.data import VisDroneDataset, collate_fn
from bcrs.evaluate import evaluate_model
from bcrs.models import build_detector
from bcrs.models.detector import count_parameters

DEFAULTS = {
    "arch": "fcos_r18",
    "frozen_bn": False,       # True = FrozenBatchNorm2d backbone (recommended; see detector.py)
    "scale_jitter": None,     # e.g. [0.7, 1.4]: training-time random resize factor (multi-scale aug)
    "ema": False,             # evaluate/checkpoint an exponential moving average of the weights
    "ema_decay": 0.9998,
    "ema_tau": 2000,          # decay ramps up over the first ~tau updates (YOLO-style)
    # None = auto: on for fixed input sizes, OFF when scale_jitter is set (the
    # autotuner re-benchmarks every new input shape; with jitter that is nearly
    # every batch and training slows ~10x — measured 1.9 vs 20.7 img/s).
    "cudnn_benchmark": None,
    "downsample": 4.0,
    "epochs": 15,
    "batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "warmup_iters": 500,
    "grad_clip": 1.0,
    "amp": "bf16",            # bf16 | fp16 | fp32
    "num_workers": 4,
    "seed": 0,
    "eval_every_epochs": 5,
    "log_every_iters": 50,
    "channels_last": False,
    "compile": False,
    "max_train_images": None,  # int for smoke/overfit runs
    "max_val_images": None,
    "init_ckpt": None,         # path to a bcrs checkpoint whose "model" weights initialize the detector
                               # (continued training; the optimizer and schedule start fresh)
    "legacy_bf16_decode": False,  # True = pre-2026-10-05 bf16 box decode in the GIoU loss and eval;
                                  # only to continue a v2 model under the conditions it was trained in
    "overfit": False,          # train set doubles as val set, no flip
    "train_dir": "data/visdrone/VisDrone2019-DET-train",
    "val_dir": "data/visdrone/VisDrone2019-DET-val",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


class ModelEMA:
    """Exponential moving average of model weights (and buffers), evaluated in
    place of the raw weights. Decay ramps from 0 toward `decay` over the first
    ~tau updates so early averages are not dominated by the random init."""

    def __init__(self, model, decay: float = 0.9998, tau: int = 2000):
        self.ema = copy.deepcopy(getattr(model, "_orig_mod", model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay, self.tau, self.updates = decay, tau, 0

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = getattr(model, "_orig_mod", model).state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
            else:
                v.copy_(msd[k])


def build_lr_lambda(total_iters: int, warmup_iters: int):
    def fn(it: int) -> float:
        if it < warmup_iters:
            return (it + 1) / warmup_iters
        p = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return fn


def validate_config_keys(user_cfg: dict) -> None:
    """A typoed key silently training with a default wastes a multi-hour A100
    slot — fail fast instead."""
    unknown = set(user_cfg) - set(DEFAULTS) - {"run_name"}
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)} "
                         f"(valid: {sorted(DEFAULTS) + ['run_name']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    user_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    validate_config_keys(user_cfg)
    cfg = dict(DEFAULTS)
    cfg.update(user_cfg)
    run_name = cfg.get("run_name") or Path(args.config).stem

    results_dir = Path("results") / run_name
    ckpt_dir = Path("runs") / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(results_dir / "train_log.txt")

    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bench = cfg["cudnn_benchmark"]
    if bench is None:
        bench = cfg["scale_jitter"] is None
    torch.backends.cudnn.benchmark = bool(bench)
    log(f"cudnn.benchmark={bool(bench)}" + (" (auto: off because scale_jitter varies input shapes)"
                                             if cfg["cudnn_benchmark"] is None and not bench else ""))

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg["amp"]]
    if amp_dtype is torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
        log("bf16 unsupported on this GPU -> fp16")

    train_ds = VisDroneDataset(
        cfg["train_dir"], downsample=cfg["downsample"], train=not cfg["overfit"],
        max_images=cfg["max_train_images"], seed=cfg["seed"],
        scale_jitter=cfg["scale_jitter"],
    )
    if cfg["overfit"]:
        val_ds = VisDroneDataset(
            cfg["train_dir"], downsample=cfg["downsample"], train=False,
            max_images=cfg["max_train_images"], seed=cfg["seed"],
        )
        val_gt = Path(cfg["train_dir"]) / "coco.json"
    else:
        val_ds = VisDroneDataset(
            cfg["val_dir"], downsample=cfg["downsample"], train=False,
            max_images=cfg["max_val_images"], seed=cfg["seed"],
        )
        val_gt = Path(cfg["val_dir"]) / "coco.json"

    train_dl = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), collate_fn=collate_fn,
        pin_memory=True, persistent_workers=int(cfg["num_workers"]) > 0,
        drop_last=len(train_ds) > int(cfg["batch_size"]),
    )
    val_dl = DataLoader(
        val_ds, batch_size=max(1, int(cfg["batch_size"]) // 2), shuffle=False,
        num_workers=int(cfg["num_workers"]), collate_fn=collate_fn, pin_memory=True,
    )

    model = build_detector(arch=cfg["arch"], frozen_bn=bool(cfg["frozen_bn"]),
                           legacy_bf16_decode=bool(cfg["legacy_bf16_decode"])).to(device)
    if cfg["init_ckpt"]:
        init = torch.load(cfg["init_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(init["model"], strict=True)
        log(f"initialized detector weights from {cfg['init_ckpt']} (epoch {init.get('epoch')})")
    if cfg["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
    if cfg["compile"]:
        model = torch.compile(model)

    ema = ModelEMA(model, float(cfg["ema_decay"]), int(cfg["ema_tau"])) if cfg["ema"] else None

    n_params = count_parameters(model)
    log(f"run={run_name} arch={cfg['arch']} frozen_bn={cfg['frozen_bn']} jitter={cfg['scale_jitter']} "
        f"ema={cfg['ema']} params={n_params/1e6:.2f}M "
        f"device={device} amp={cfg['amp']} s={cfg['downsample']} "
        f"box_decode={'bf16-legacy' if cfg['legacy_bf16_decode'] else 'fp32'} "
        f"train={len(train_ds)} val={len(val_ds)} commit={git_commit()}")

    optim = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]),
    )
    total_iters = len(train_dl) * int(cfg["epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, build_lr_lambda(total_iters, min(int(cfg["warmup_iters"]), max(1, total_iters // 10))),
    )
    scaler = torch.amp.GradScaler(enabled=amp_dtype is torch.float16)

    best_ap = -1.0
    history = []
    it = 0
    for epoch in range(int(cfg["epochs"])):
        model.train()
        t0, seen = time.time(), 0
        for images, targets in train_dl:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [
                {k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets
            ]
            optim.zero_grad(set_to_none=True)
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None else torch.autocast(device_type=device.type, enabled=False)
            )
            with ctx:
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            scale_before = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            # fp16: a skipped (inf/NaN) optimizer step must not consume an LR
            # schedule step; a skip is signalled by the scale decreasing.
            if scaler.get_scale() >= scale_before:
                sched.step()
            if ema is not None:
                ema.update(model)
            it += 1
            seen += len(images)
            if it % int(cfg["log_every_iters"]) == 0:
                parts = " ".join(f"{k}={v.item():.3f}" for k, v in loss_dict.items())
                ips = seen / max(time.time() - t0, 1e-6)
                log(f"e{epoch} it{it}/{total_iters} loss={loss.item():.3f} {parts} "
                    f"lr={sched.get_last_lr()[0]:.2e} {ips:.1f} img/s")

        do_eval = (epoch + 1) % int(cfg["eval_every_epochs"]) == 0 or epoch + 1 == int(cfg["epochs"])
        if do_eval:
            # with EMA on, the averaged weights are what gets evaluated and saved
            eval_model = ema.ema if ema is not None else model
            metrics = evaluate_model(eval_model, val_dl, device, val_gt, amp_dtype=amp_dtype)
            metrics["epoch"] = epoch
            history.append(metrics)
            log(f"eval e{epoch}{' (ema)' if ema else ''}: AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f} "
                f"AP_s={metrics['AP_s']:.4f} AR@500={metrics['AR@500']:.4f}")
            # unwrap torch.compile so checkpoints load into a fresh detector
            raw_model = ema.ema if ema is not None else getattr(model, "_orig_mod", model)
            torch.save({"model": raw_model.state_dict(), "cfg": cfg, "epoch": epoch},
                       ckpt_dir / "last.pth")
            if metrics["AP"] > best_ap:
                best_ap = metrics["AP"]
                torch.save({"model": raw_model.state_dict(), "cfg": cfg, "epoch": epoch},
                           ckpt_dir / "best.pth")

    out = {
        "run_name": run_name,
        "config": cfg,
        "git_commit": git_commit(),
        "n_params": n_params,
        "best_AP": best_ap,
        "final": history[-1] if history else {},
        "history": history,
    }
    (results_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    log(f"done. best AP={best_ap:.4f} -> {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
