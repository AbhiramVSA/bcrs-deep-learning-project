"""Joint BCRS training (P2c): detector + budget-conditioned scorer, end to end.

    python -m bcrs.train_joint --config configs/<name>.yaml

Per iteration (proposal Algorithm 2):
  1. sample k ~ U{k_min..k_max}, budget B = k / budget_norm_k
  2. coarse pass (train mode) on s-downsampled images -> L_coarse, P7 feats,
     coarse detections (no extra forward)
  3. self-derived tile targets: count of GT missed by the coarse pass per tile
  4. scorer(P7, B) -> tile logits; hard top-k forward / perturbed top-k backward
  5. fine pass on the selected native-res tiles (same detector weights) ->
     per-tile losses; L_fine = mean over tiles (detector gradient); the
     relaxed indicator couples each tile's fine loss to the scores
     (lambda_sel, gradient reaches only the scorer)
  6. optional target refinement: for selected tiles the target becomes the
     number of missed objects the fine pass actually recovered
  7. L = L_coarse + L_fine + lambda_sel*L_sel + lambda_bce*BCE + lambda_rank*rank

Evaluation each eval interval: coarse-only AP, selection recall@k
(learned/oracle/random), and BCRS AP at k=eval_k with learned selection.
Checkpoints: runs/<run>/detector_{best,last}.pth (bcrs.train format) and
scorer_{best,last}.pth (bcrs.train_bcrs format) -> usable by sweep_curve.py.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

from bcrs.data import VisDroneDataset, collate_fn
from bcrs.evaluate import coco_metrics
from bcrs.joint import fcos_train_outputs, per_tile_fine_losses, recovered_counts, tile_targets
from bcrs.models.detector import build_detector_from_ckpt
from bcrs.models.topk import gumbel_topk_straight_through, hard_topk_no_grad, hard_topk_straight_through
from bcrs.tiles import crop_tile, tile_grid
from bcrs.train import Logger, build_lr_lambda, git_commit, set_seed
from bcrs.train_bcrs import build_scorer, evaluate_selection, ranking_loss
from bcrs.twopass import coarse_pass, downsample_image, fuse, oracle_tile_scores, tile_pass, unmatched_gt_mask

DEFAULTS = {
    "detector_ckpt": "runs/p1_uniform_s2/best.pth",
    "scorer_ckpt": None,
    "s": 4.0,
    "tile": 512,
    "overlap": 32,
    "iou_thr": 0.5,
    "score_thr": 0.3,
    "k_min": 1,
    "k_max": 8,
    "budget_norm_k": 20,
    "lambda_coarse": 1.0,
    "lambda_fine": 1.0,
    "lambda_sel": 0.5,
    "lambda_bce": 1.0,
    "lambda_rank": 1.0,
    "rank_margin": 1.0,
    "topk_samples": 32,
    "topk_sigma": 0.5,
    "refine_targets": True,
    # ablations: selection_grad in {perturbed, gumbel, none}; target_mode in
    # {self_derived, objectness} (objectness = count of ALL GT in the tile,
    # i.e. density-style supervision that ignores what the coarse pass found)
    "selection_grad": "perturbed",
    "target_mode": "self_derived",
    "scorer_hidden": 128,
    "scorer_budget_dim": 64,
    "scorer_blocks": 4,
    "epochs": 12,
    "batch_size": 4,
    "lr": 5.0e-5,
    "scorer_lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "warmup_iters": 500,
    "grad_clip": 1.0,
    "amp": "bf16",
    # OFF by default: the tile batch size changes with every sampled k, so
    # cuDNN autotuning would re-benchmark backward kernels almost every
    # iteration (measured: 139 s for one first-shape backward on an RTX 3060).
    "cudnn_benchmark": False,
    "num_workers": 8,
    "seed": 0,
    "eval_every_epochs": 2,
    "eval_k": 3,
    "k_eval": [1, 2, 3, 5, 8],
    "log_every_iters": 50,
    "max_train_images": None,
    "max_val_images": None,
    "train_dir": "data/visdrone/VisDrone2019-DET-train",
    "val_dir": "data/visdrone/VisDrone2019-DET-val",
}


def validate_config_keys(user_cfg: dict) -> None:
    unknown = set(user_cfg) - set(DEFAULTS) - {"run_name"}
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")


def _autocast(device, amp_dtype):
    return (torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None
            else contextlib.nullcontext())


def scale_targets(t: dict, sx: float, sy: float, device) -> dict:
    """Scale native GT to coarse coords, dropping sub-pixel remnants."""
    b = t["boxes"].clone()
    b[:, [0, 2]] *= sx
    b[:, [1, 3]] *= sy
    keep = ((b[:, 2] - b[:, 0]) >= 1.0) & ((b[:, 3] - b[:, 1]) >= 1.0)
    return {"boxes": b[keep].to(device), "labels": t["labels"][keep].to(device)}


@torch.no_grad()
def evaluate_bcrs(detector, scorer, loader, cfg, device, amp_dtype, gt_json) -> dict:
    """Coarse-only AP and BCRS AP at k=eval_k with the learned ranking."""
    detector.eval()
    scorer.eval()
    s, k = float(cfg["s"]), int(cfg["eval_k"])
    dets_coarse, dets_bcrs, img_ids = [], [], []
    for images, targets in loader:
        for image, target in zip(images, targets):
            img_id = int(target["image_id"])
            img_ids.append(img_id)
            _, h, w = image.shape
            tiles = tile_grid(w, h, int(cfg["tile"]), int(cfg["overlap"]))
            n = len(tiles)
            coarse, feats = coarse_pass(detector, image, s, device, amp_dtype, return_features=True)
            kk = min(k, n)
            b = torch.tensor([kk / float(cfg["budget_norm_k"])], device=device)
            with _autocast(device, amp_dtype):
                sc = scorer(feats[-1], b, [tiles], s * 128.0)[0].float().cpu()
            sel = torch.argsort(sc, descending=True)[:kk]
            sel_tiles = tiles[sel]
            tile_outs = tile_pass(detector, image, sel_tiles, device, amp_dtype)
            fused = fuse(coarse, tile_outs, torch.arange(len(sel_tiles)), sel_tiles)
            for det, acc in ((coarse, dets_coarse), (fused, dets_bcrs)):
                for (x0, y0, x1, y1), sco, lab in zip(det["boxes"].tolist(), det["scores"].tolist(),
                                                      det["labels"].tolist()):
                    if lab >= 1:
                        acc.append({"image_id": img_id, "category_id": int(lab),
                                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(sco)})
    mc = coco_metrics(gt_json, dets_coarse, img_ids=img_ids)
    mb = coco_metrics(gt_json, dets_bcrs, img_ids=img_ids)
    out = {f"coarse_{key}": mc[key] for key in ("AP", "AP50", "AP_s", "AR@500")}
    out.update({f"bcrs@{k}_{key}": mb[key] for key in ("AP", "AP50", "AP_s", "AR@500")})
    detector.train()
    scorer.train()
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
    torch.backends.cudnn.benchmark = bool(cfg["cudnn_benchmark"])
    amp_dtype = {"bf16": torch.bfloat16, "fp32": None}[cfg["amp"]]

    det_ckpt = torch.load(cfg["detector_ckpt"], map_location="cpu", weights_only=False)
    detector = build_detector_from_ckpt(det_ckpt).to(device)
    det_arch_cfg = {"arch": det_ckpt["cfg"]["arch"], "frozen_bn": bool(det_ckpt["cfg"].get("frozen_bn", False))}
    scorer = build_scorer(cfg).to(device)
    if cfg["scorer_ckpt"]:
        scorer.load_state_dict(torch.load(cfg["scorer_ckpt"], map_location="cpu", weights_only=False)["scorer"])

    train_ds = VisDroneDataset(cfg["train_dir"], downsample=1.0, train=True,
                               max_images=cfg["max_train_images"], seed=int(cfg["seed"]))
    val_ds = VisDroneDataset(cfg["val_dir"], downsample=1.0, train=False,
                             max_images=cfg["max_val_images"], seed=int(cfg["seed"]))
    train_dl = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                          num_workers=int(cfg["num_workers"]), collate_fn=collate_fn,
                          drop_last=len(train_ds) > int(cfg["batch_size"]),
                          persistent_workers=int(cfg["num_workers"]) > 0)
    val_dl = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
                        num_workers=int(cfg["num_workers"]), collate_fn=collate_fn)
    val_gt = Path(cfg["val_dir"]) / "coco.json"

    optim = torch.optim.AdamW([
        {"params": [p for p in detector.parameters() if p.requires_grad], "lr": float(cfg["lr"])},
        {"params": scorer.parameters(), "lr": float(cfg["scorer_lr"])},
    ], weight_decay=float(cfg["weight_decay"]))
    total_iters = len(train_dl) * int(cfg["epochs"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, build_lr_lambda(total_iters, min(int(cfg["warmup_iters"]), max(1, total_iters // 10))))
    log(f"run={run_name} detector={cfg['detector_ckpt']} ({det_arch_cfg}) scorer_init={cfg['scorer_ckpt']} "
        f"s={cfg['s']} tile={cfg['tile']}/{cfg['overlap']} k~U[{cfg['k_min']},{cfg['k_max']}] "
        f"lambda_sel={cfg['lambda_sel']} refine={cfg['refine_targets']} sel_grad={cfg['selection_grad']} targets={cfg['target_mode']} train={len(train_ds)} "
        f"val={len(val_ds)} amp={cfg['amp']} commit={git_commit()}")

    s = float(cfg["s"])
    best, history, it = -1.0, [], 0
    for epoch in range(int(cfg["epochs"])):
        detector.train()
        scorer.train()
        t0, seen = time.time(), 0
        for images, targets in train_dl:
            k = rng.randint(int(cfg["k_min"]), int(cfg["k_max"]))
            b_val = k / float(cfg["budget_norm_k"])

            # --- coarse pass (train mode) ---
            smalls, coarse_tgts, scales = [], [], []
            for img, t in zip(images, targets):
                small = downsample_image(img, s)
                sx, sy = small.shape[-1] / img.shape[-1], small.shape[-2] / img.shape[-2]
                smalls.append(small.to(device, non_blocking=True))
                coarse_tgts.append(scale_targets(t, sx, sy, device))
                scales.append((1.0 / sx, 1.0 / sy))
            with _autocast(device, amp_dtype):
                coarse_losses, feats, coarse_dets = fcos_train_outputs(detector, smalls, coarse_tgts)
            loss_coarse = sum(coarse_losses.values())

            # --- self-derived targets (native coords, CPU) ---
            tiles, counts, missed = [], [], []
            for img, t, d, (ux, uy) in zip(images, targets, coarse_dets, scales):
                d = {kk_: v.float().cpu() for kk_, v in d.items()}
                d["boxes"][:, [0, 2]] *= ux
                d["boxes"][:, [1, 3]] *= uy
                tl = tile_grid(img.shape[-1], img.shape[-2], int(cfg["tile"]), int(cfg["overlap"]))
                tiles.append(tl)
                if cfg["target_mode"] == "objectness":
                    empty = {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
                    counts.append(oracle_tile_scores(t["boxes"], t["labels"], empty, tl,
                                                     float(cfg["iou_thr"]), float(cfg["score_thr"])))
                else:
                    counts.append(oracle_tile_scores(t["boxes"], t["labels"], d, tl,
                                                     float(cfg["iou_thr"]), float(cfg["score_thr"])))
                m = unmatched_gt_mask(t["boxes"], t["labels"], d, float(cfg["iou_thr"]), float(cfg["score_thr"]))
                missed.append((t["boxes"][m], t["labels"][m]))

            # --- scorer + hard-forward / relaxed-backward selection ---
            budget = torch.full((len(images),), b_val, device=device)
            with _autocast(device, amp_dtype):
                scores = scorer(feats[-1], budget, tiles, s * 128.0)
            zs, crops, crop_tgts, owners = [], [], [], []
            for i, (img, t, tl, sc) in enumerate(zip(images, targets, tiles, scores)):
                kk = min(k, len(tl))
                mode = cfg["selection_grad"]
                if mode == "perturbed":
                    z = hard_topk_straight_through(sc.float(), kk, int(cfg["topk_samples"]), float(cfg["topk_sigma"]))
                elif mode == "gumbel":
                    z = gumbel_topk_straight_through(sc.float(), kk, float(cfg["topk_sigma"]))
                elif mode == "none":
                    z = hard_topk_no_grad(sc.float(), kk)
                else:
                    raise SystemExit(f"unknown selection_grad: {mode}")
                zs.append(z)
                for j in z.detach().nonzero().flatten().tolist():
                    crops.append(crop_tile(img, tl[j]).to(device, non_blocking=True))
                    tt = tile_targets(t["boxes"], t["labels"], tl[j])
                    crop_tgts.append({kk_: v.to(device) for kk_, v in tt.items()})
                    owners.append((i, j))

            # --- fine pass on selected tiles ---
            with _autocast(device, amp_dtype):
                tile_losses, tile_dets, _ = per_tile_fine_losses(detector, crops, crop_tgts)
            per_tile = torch.stack([sum(l.values()) for l in tile_losses])
            loss_fine = per_tile.mean()
            coupling = torch.stack([zs[i][j] - zs[i][j].detach() for (i, j) in owners])
            loss_sel = (coupling * per_tile.detach()).mean()

            # --- target refinement with what the fine pass actually recovered ---
            if cfg["refine_targets"] and cfg["target_mode"] == "self_derived":
                for (i, j), d in zip(owners, tile_dets):
                    d = {kk_: v.float().cpu() for kk_, v in d.items()}
                    counts[i][j] = float(recovered_counts(missed[i][0], missed[i][1], tiles[i][j], d,
                                                          float(cfg["iou_thr"]), float(cfg["score_thr"])))

            # --- scorer losses ---
            bce_terms, rank_terms = [], []
            for sc, cnt in zip(scores, counts):
                cnt_d = cnt.to(sc.device)
                bce_terms.append(F.binary_cross_entropy_with_logits(sc.float(), (cnt_d > 0).float()))
                r = ranking_loss(sc.float(), cnt_d, float(cfg["rank_margin"]))
                if r is not None:
                    rank_terms.append(r)
            loss_bce = torch.stack(bce_terms).mean()
            loss_rank = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=device)

            loss = (float(cfg["lambda_coarse"]) * loss_coarse + float(cfg["lambda_fine"]) * loss_fine
                    + float(cfg["lambda_sel"]) * loss_sel + float(cfg["lambda_bce"]) * loss_bce
                    + float(cfg["lambda_rank"]) * loss_rank)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(list(detector.parameters()) + list(scorer.parameters()),
                                               float(cfg["grad_clip"]))
            optim.step()
            sched.step()
            it += 1
            seen += len(images)
            if it % int(cfg["log_every_iters"]) == 0:
                log(f"e{epoch} it{it}/{total_iters} loss={loss.item():.3f} coarse={loss_coarse.item():.3f} "
                    f"fine={loss_fine.item():.3f} bce={loss_bce.item():.3f} rank={loss_rank.item():.3f} "
                    f"k={k} tiles={len(crops)} lr={sched.get_last_lr()[0]:.2e} "
                    f"{seen / max(time.time() - t0, 1e-6):.1f} img/s")

        if (epoch + 1) % int(cfg["eval_every_epochs"]) == 0 or epoch + 1 == int(cfg["epochs"]):
            m = evaluate_bcrs(detector, scorer, val_dl, cfg, device, amp_dtype, val_gt)
            m.update(evaluate_selection(detector, scorer, val_dl, cfg, device, amp_dtype, random.Random(0)))
            m["epoch"] = epoch
            history.append(m)
            ek = int(cfg["eval_k"])
            log(f"eval e{epoch}: coarse AP={m['coarse_AP']:.4f} | BCRS@{ek} AP={m[f'bcrs@{ek}_AP']:.4f} "
                f"AP_s={m[f'bcrs@{ek}_AP_s']:.4f} AR@500={m[f'bcrs@{ek}_AR@500']:.3f} | "
                f"sel_recall@{ek}: learned={m[f'learned_recall@{ek}']:.3f} oracle={m[f'oracle_recall@{ek}']:.3f} "
                f"random={m[f'random_recall@{ek}']:.3f}")
            det_state = {"model": detector.state_dict(), "cfg": {**det_arch_cfg, "joint_cfg": cfg}, "epoch": epoch}
            sc_state = {"scorer": scorer.state_dict(), "cfg": cfg, "epoch": epoch}
            torch.save(det_state, ckpt_dir / "detector_last.pth")
            torch.save(sc_state, ckpt_dir / "scorer_last.pth")
            if m[f"bcrs@{ek}_AP"] > best:
                best = m[f"bcrs@{ek}_AP"]
                torch.save(det_state, ckpt_dir / "detector_best.pth")
                torch.save(sc_state, ckpt_dir / "scorer_best.pth")

    out = {"run_name": run_name, "config": cfg, "git_commit": git_commit(),
           "best_bcrs_AP": best, "final": history[-1] if history else {}, "history": history}
    (results_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    log(f"done. best BCRS@{cfg['eval_k']} AP={best:.4f} -> {results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
