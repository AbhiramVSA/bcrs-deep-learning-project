"""NaN diagnostic: run a few real training steps and report the first
iteration where the forward loss or gradient norm goes non-finite.

    python scripts/diag_nan.py --amp bf16 --batch-size 4 --iters 60
    python scripts/diag_nan.py --amp fp32 --batch-size 4 --iters 60
    python scripts/diag_nan.py --amp bf16 --batch-size 1 --iters 60
    python scripts/diag_nan.py --amp bf16 --batch-size 4 --min-width 1900

Same optimizer / warmup ramp / grad clip / channels_last as bcrs.train.
On a non-finite step it dumps the offending batch's image sizes, box counts,
box size extremes and which loss component broke, then stops.
"""
from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from bcrs.data import VisDroneDataset, collate_fn
from bcrs.models import build_detector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--split-dir", default="data/visdrone/VisDrone2019-DET-train")
    ap.add_argument("--min-width", type=int, default=0, help="only images at least this wide")
    ap.add_argument("--downsample", type=float, default=1.0)
    ap.add_argument("--no-channels-last", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ds = VisDroneDataset(args.split_dir, downsample=args.downsample, train=True)
    if args.min_width:
        ds.images = [im for im in ds.images if im["width"] >= args.min_width]
        print(f"filtered to {len(ds)} images with width >= {args.min_width}")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate_fn, drop_last=True)

    model = build_detector("fcos_r18").to(dev)
    if not args.no_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.amp]
    scaler = torch.amp.GradScaler(enabled=amp is torch.float16)
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} amp={args.amp} "
          f"bs={args.batch_size} channels_last={not args.no_channels_last} clip={not args.no_clip}")

    for it, (imgs, tgts) in enumerate(dl):
        if it >= args.iters:
            break
        imgs = [x.to(dev) for x in imgs]
        tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (it + 1) / 500)
        opt.zero_grad(set_to_none=True)
        ctx = torch.autocast("cuda", dtype=amp) if amp else torch.autocast("cuda", enabled=False)
        with ctx:
            ld = model(imgs, tgts)
            loss = sum(ld.values())
        vals = {k: v.item() for k, v in ld.items()}
        fwd_ok = all(math.isfinite(v) for v in vals.values())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if not args.no_clip else float("inf")).item()
        scaler.step(opt)
        scaler.update()
        shapes = [tuple(x.shape[1:]) for x in imgs]
        if it % 10 == 0 or not fwd_ok or not math.isfinite(gn):
            print(f"it{it} loss={loss.item():.4f} " + " ".join(f"{k}={v:.3f}" for k, v in vals.items())
                  + f" gradnorm={gn:.3f} shapes={shapes}", flush=True)
        if not fwd_ok or not math.isfinite(gn):
            print(f"FIRST NON-FINITE at it{it}: forward_finite={fwd_ok} gradnorm={gn}")
            for t in tgts:
                b = t["boxes"]
                if b.numel():
                    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                    print(f"  image_id={int(t['image_id'])} nboxes={b.shape[0]} "
                          f"w[min,max]=({w.min():.1f},{w.max():.1f}) h[min,max]=({h.min():.1f},{h.max():.1f}) "
                          f"maxcoord={b.max():.0f} labels={sorted(set(t['labels'].tolist()))}")
                else:
                    print(f"  image_id={int(t['image_id'])} nboxes=0")
            return
    print(f"DONE: no non-finite values in {args.iters} iterations")


if __name__ == "__main__":
    main()
