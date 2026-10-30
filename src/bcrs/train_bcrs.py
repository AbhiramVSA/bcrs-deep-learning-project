"""BCRS selection-head training (P2b: learned scorer on a FROZEN detector).

    python -m bcrs.train_bcrs --config configs/<name>.yaml

Per iteration: frozen coarse pass at downsample s on native images ->
coarse detections + P7 features; self-derived tile targets (a tile is
positive iff it contains a GT box the coarse pass missed); FiLM-conditioned
scorer on P7 -> per-tile logits; loss = BCE + margin ranking.

Evaluation (no fine pass needed): object-level selection recall@k — the
fraction of coarse-missed objects that fall inside the top-k tiles — for the
learned ranking vs the oracle and random rankings. This is THE metric for the
scoring head (PLAN.md section 1) and is cheap enough to run every epoch.

Checkpoint format: {"scorer": state_dict, "cfg": cfg, "epoch": e}.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bcrs.data import VisDroneDataset, collate_fn
from bcrs.models.detector import build_detector_from_ckpt
from bcrs.models.scorer import TileScorer
from bcrs.tiles import tile_grid
from bcrs.train import Logger, build_lr_lambda, git_commit, set_seed
from bcrs.twopass import downsample_image, fcos_forward_with_features, oracle_tile_scores

DEFAULTS = {
    "detector_ckpt": "runs/p1_uniform_s4/best.pth",
    "s": 4.0,
    "tile": 512,
    "overlap": 32,
    "iou_thr": 0.5,
    "score_thr": 0.3,
    "lambda_bce": 1.0,
    "lambda_rank": 1.0,
    "rank_margin": 1.0,
    "scorer_hidden": 128,
    "scorer_budget_dim": 64,
    "scorer_blocks": 4,
    "epochs": 10,
    "batch_size": 8,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "warmup_iters": 200,
    "amp": "bf16",
    "num_workers": 8,
    "seed": 0,
    "eval_every_epochs": 1,
    "log_every_iters": 50,
    "k_eval": [1, 2, 3, 5, 8],
    "select_metric_k": 3,
    "max_train_images": None,
    "max_val_images": None,
    "train_dir": "data/visdrone/VisDrone2019-DET-train",
    "val_dir": "data/visdrone/VisDrone2019-DET-val",
}


def validate_config_keys(user_cfg: dict) -> None:
    unknown = set(user_cfg) - set(DEFAULTS) - {"run_name"}
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")


def build_scorer(cfg: dict) -> TileScorer:
    return TileScorer(
        in_channels=256, hidden=int(cfg["scorer_hidden"]),
        budget_dim=int(cfg["scorer_budget_dim"]), num_blocks=int(cfg["scorer_blocks"]),
    )


@torch.no_grad()
def coarse_batch(detector, images: list[torch.Tensor], s: float, device, amp_dtype):
    """Frozen coarse pass over a batch of NATIVE images. Returns per-image
    coarse detections in native coords (CPU) and the batched FPN features."""
    smalls, scales = [], []
    for img in images:
        small = downsample_image(img, s)
        smalls.append(small.to(device, non_blocking=True))
        scales.append((img.shape[-1] / small.shape[-1], img.shape[-2] / small.shape[-2]))
    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
           if amp_dtype is not None else torch.autocast(device_type=device.type, enabled=False))
    with ctx:
        dets, feats = fcos_forward_with_features(detector, smalls)
    out = []
    for d, (sx, sy) in zip(dets, scales):
        d = {k: v.float().cpu() for k, v in d.items()}
        d["boxes"][:, [0, 2]] *= sx
        d["boxes"][:, [1, 3]] *= sy
        out.append(d)
    return out, feats


def ranking_loss(scores: torch.Tensor, counts: torch.Tensor, margin: float) -> torch.Tensor | None:
    """Margin ranking over all tile pairs (i, j) with counts_i > counts_j."""
    diff_c = counts[:, None] - counts[None, :]
    pairs = diff_c > 0
    if not pairs.any():
        return None
    diff_s = scores[:, None] - scores[None, :]
    return F.relu(margin - diff_s[pairs]).mean()


def selection_recall(order: torch.Tensor, counts: torch.Tensor, ks: list[int]) -> dict[int, float] | None:
    """Fraction of coarse-missed objects covered by the top-k tiles."""
    total = counts.sum().item()
    if total <= 0:
        return None
    return {k: counts[order[:min(k, len(order))]].sum().item() / total for k in ks}


@torch.no_grad()
def evaluate_selection(detector, scorer, loader, cfg, device, amp_dtype, rng: random.Random) -> dict:
    """Object-level selection recall@k for learned/oracle/random rankings.
    Works with a detector in either mode (restores it afterwards)."""
    det_was_training = detector.training
    detector.eval()
    scorer.eval()
    ks = [int(k) for k in cfg["k_eval"]]
    sums = {name: {k: 0.0 for k in ks} for name in ("learned", "oracle", "random")}
    n_scored = 0
    for images, targets in loader:
        coarse, feats = coarse_batch(detector, images, float(cfg["s"]), device, amp_dtype)
        tiles = [tile_grid(img.shape[-1], img.shape[-2], int(cfg["tile"]), int(cfg["overlap"])) for img in images]
        counts = [
            oracle_tile_scores(t["boxes"], t["labels"], c, tl, float(cfg["iou_thr"]), float(cfg["score_thr"]))
            for t, c, tl in zip(targets, coarse, tiles)
        ]
        budget = torch.full((len(images),), 0.5, device=device)
        ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_dtype is not None else torch.autocast(device_type=device.type, enabled=False))
        with ctx:
            scores = scorer(feats[-1], budget, tiles, float(cfg["s"]) * 128.0)
        for sc, cnt in zip(scores, counts):
            sc = sc.float().cpu()
            orders = {
                "learned": torch.argsort(sc, descending=True),
                "oracle": torch.argsort(cnt, descending=True),
                "random": torch.tensor(rng.sample(range(len(cnt)), len(cnt)), dtype=torch.long),
            }
            rec = {name: selection_recall(o, cnt, ks) for name, o in orders.items()}
            if rec["oracle"] is None:
                continue
            n_scored += 1
            for name in sums:
                for k in ks:
                    sums[name][k] += rec[name][k]
    out = {f"{name}_recall@{k}": (sums[name][k] / max(1, n_scored)) for name in sums for k in ks}
    out["n_images_with_misses"] = n_scored
    scorer.train()
    if det_was_training:
        detector.train()
    return out


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
    rng = random.Random(int(cfg["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg["amp"]]
    if amp_dtype is torch.float16:
        raise SystemExit("use bf16 or fp32 for scorer training (no GradScaler path here)")

    det_ckpt = torch.load(cfg["detector_ckpt"], map_location="cpu", weights_only=False)
    detector = build_detector_from_ckpt(det_ckpt).to(device)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)

    scorer = build_scorer(cfg).to(device)
    train_ds = VisDroneDataset(cfg["train_dir"], downsample=1.0, train=True,
                               max_images=cfg["max_train_images"], seed=int(cfg["seed"]))
    val_ds = VisDroneDataset(cfg["val_dir"], downsample=1.0, train=False,
                             max_images=cfg["max_val_images"], seed=int(cfg["seed"]))
    train_dl = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                          num_workers=int(cfg["num_workers"]), collate_fn=collate_fn,
                          pin_memory=True, drop_last=len(train_ds) > int(cfg["batch_size"]),
                          persistent_workers=int(cfg["num_workers"]) > 0)
    val_dl = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
                        num_workers=int(cfg["num_workers"]), collate_fn=collate_fn, pin_memory=True)

    optim = torch.optim.AdamW(scorer.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    total_iters = len(train_dl) * int(cfg["epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, build_lr_lambda(total_iters, min(int(cfg["warmup_iters"]), max(1, total_iters // 10))))
    log(f"run={run_name} detector={cfg['detector_ckpt']} ({det_ckpt['cfg']['arch']}, frozen) "
        f"s={cfg['s']} tile={cfg['tile']}/{cfg['overlap']} scorer_params="
        f"{sum(p.numel() for p in scorer.parameters())/1e3:.1f}K train={len(train_ds)} val={len(val_ds)} "
        f"amp={cfg['amp']} commit={git_commit()}")

    sel_k = int(cfg["select_metric_k"])
    best, history, it = -1.0, [], 0
    for epoch in range(int(cfg["epochs"])):
        scorer.train()
        t0, seen = time.time(), 0
        for images, targets in train_dl:
            coarse, feats = coarse_batch(detector, images, float(cfg["s"]), device, amp_dtype)
            tiles = [tile_grid(img.shape[-1], img.shape[-2], int(cfg["tile"]), int(cfg["overlap"])) for img in images]
            counts = [
                oracle_tile_scores(t["boxes"], t["labels"], c, tl, float(cfg["iou_thr"]), float(cfg["score_thr"]))
                for t, c, tl in zip(targets, coarse, tiles)
            ]
            budget = torch.rand(len(images), device=device)  # exercises FiLM; targets are budget-free in P2b
            ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
                   if amp_dtype is not None else torch.autocast(device_type=device.type, enabled=False))
            with ctx:
                scores = scorer(feats[-1], budget, tiles, float(cfg["s"]) * 128.0)
            bce_terms, rank_terms = [], []
            for sc, cnt in zip(scores, counts):
                cnt_d = cnt.to(sc.device)
                bce_terms.append(F.binary_cross_entropy_with_logits(sc.float(), (cnt_d > 0).float()))
                r = ranking_loss(sc.float(), cnt_d, float(cfg["rank_margin"]))
                if r is not None:
                    rank_terms.append(r)
            loss_bce = torch.stack(bce_terms).mean()
            loss_rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=device)
            loss = float(cfg["lambda_bce"]) * loss_bce + float(cfg["lambda_rank"]) * loss_rank
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            optim.step()
            sched.step()
            it += 1
            seen += len(images)
            if it % int(cfg["log_every_iters"]) == 0:
                pos = sum(int((c > 0).sum()) for c in counts)
                tot = sum(len(c) for c in counts)
                log(f"e{epoch} it{it}/{total_iters} loss={loss.item():.4f} bce={loss_bce.item():.4f} "
                    f"rank={loss_rank.item():.4f} pos_tiles={pos}/{tot} lr={sched.get_last_lr()[0]:.2e} "
                    f"{seen / max(time.time() - t0, 1e-6):.1f} img/s")

        if (epoch + 1) % int(cfg["eval_every_epochs"]) == 0 or epoch + 1 == int(cfg["epochs"]):
            m = evaluate_selection(detector, scorer, val_dl, cfg, device, amp_dtype, random.Random(0))
            m["epoch"] = epoch
            history.append(m)
            log(f"eval e{epoch}: " + " ".join(
                f"{n}@{sel_k}={m[f'{n}_recall@{sel_k}']:.3f}" for n in ("learned", "oracle", "random"))
                + f" (n={m['n_images_with_misses']})")
            state = {"scorer": scorer.state_dict(), "cfg": cfg, "epoch": epoch}
            torch.save(state, ckpt_dir / "scorer_last.pth")
            if m[f"learned_recall@{sel_k}"] > best:
                best = m[f"learned_recall@{sel_k}"]
                torch.save(state, ckpt_dir / "scorer_best.pth")

    out = {"run_name": run_name, "config": cfg, "git_commit": git_commit(),
           "best_learned_recall": best, "final": history[-1] if history else {}, "history": history}
    (results_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    log(f"done. best learned_recall@{sel_k}={best:.4f} -> {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
