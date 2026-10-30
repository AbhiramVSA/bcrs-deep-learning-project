"""Run one image through a trained BCRS detector and save its detections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from bcrs.models.detector import build_detector_from_ckpt
from bcrs.tiles import tile_grid
from bcrs.train_bcrs import build_scorer
from bcrs.twopass import coarse_pass, fuse, tile_pass


CATEGORIES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}


def select_tiles(scores: torch.Tensor, tiles: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k < 0:
        raise ValueError("k must be zero or greater")
    if scores.ndim != 1 or len(scores) != len(tiles):
        raise ValueError("scores and tiles must have the same number of entries")
    indices = torch.argsort(scores.float(), descending=True)[:min(k, len(tiles))]
    return indices, tiles[indices]


@torch.inference_mode()
def predict_image(image: torch.Tensor, detector, scorer, cfg: dict, device: torch.device, k: int):
    s, tile, overlap = float(cfg["s"]), int(cfg["tile"]), int(cfg["overlap"])
    budget_norm_k = float(cfg["budget_norm_k"])
    if k < 0:
        raise ValueError("k must be zero or greater")
    if s <= 0 or tile <= 0 or overlap < 0 or 2 * overlap >= tile:
        raise ValueError("invalid scale or tile geometry in checkpoint config")
    if budget_norm_k <= 0:
        raise ValueError("budget_norm_k must be greater than zero")
    tiles = tile_grid(image.shape[-1], image.shape[-2], tile, overlap)
    coarse, features = coarse_pass(detector, image, s, device, return_features=True)
    budget = torch.tensor([min(k, len(tiles)) / budget_norm_k], device=device)
    scores = scorer(features[-1], budget, [tiles], s * 128.0)[0].float().cpu()
    indices, selected_tiles = select_tiles(scores, tiles, k)
    fine = tile_pass(detector, image, selected_tiles, device) if len(selected_tiles) else []
    fused = fuse(coarse, fine, torch.arange(len(selected_tiles)), selected_tiles,
                 coarse_full_min_side=2 * overlap)
    return fused, indices, scores, selected_tiles


def annotate_detections(image: Image.Image, detections: dict) -> list[dict]:
    draw = ImageDraw.Draw(image)
    rows = []
    for box, score, label in zip(detections["boxes"].tolist(), detections["scores"].tolist(),
                                 detections["labels"].tolist()):
        if int(label) < 1:
            continue
        category = CATEGORIES.get(int(label), f"class-{int(label)}")
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1] - 12)), f"{category} {score:.2f}", fill="red")
        rows.append({"box_xyxy": box, "score": float(score), "label": int(label), "category": category})
    return rows


def _load_checkpoint(path: Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--detector-ckpt", required=True, type=Path)
    parser.add_argument("--scorer-ckpt", required=True, type=Path)
    parser.add_argument("--k", required=True, type=int, help="maximum number of native-resolution tiles")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    for path in (args.image, args.detector_ckpt, args.scorer_ckpt):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if args.k < 0:
        parser.error("--k must be zero or greater")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")

    device = torch.device(args.device)
    detector_ckpt = _load_checkpoint(args.detector_ckpt)
    scorer_ckpt = _load_checkpoint(args.scorer_ckpt)
    detector_cfg = detector_ckpt.get("cfg")
    cfg = detector_cfg.get("joint_cfg") if isinstance(detector_cfg, dict) else None
    if not isinstance(cfg, dict) or "budget_norm_k" not in cfg:
        parser.error("detector checkpoint must include the BCRS joint config")
    if "cfg" not in scorer_ckpt or "scorer" not in scorer_ckpt:
        parser.error("scorer checkpoint must include cfg and scorer weights")

    detector = build_detector_from_ckpt(detector_ckpt, topk_candidates=1000,
                                        detections_per_img=500).to(device).eval()
    scorer = build_scorer(scorer_ckpt["cfg"]).to(device)
    scorer.load_state_dict(scorer_ckpt["scorer"], strict=True)
    scorer.eval()

    with Image.open(args.image) as source:
        original = source.convert("RGB")
    image = to_tensor(original)
    detections, indices, scores, selected_tiles = predict_image(image, detector, scorer, cfg, device, args.k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated = original.copy()
    rows = annotate_detections(annotated, detections)
    annotated.save(args.output_dir / "annotated.png")

    selected = [{"index": int(index), "box_xyxy": [float(v) for v in box],
                 "score": float(scores[index])}
                for index, box in zip(indices.tolist(), selected_tiles.tolist())]
    (args.output_dir / "predictions.json").write_text(json.dumps({
        "image": str(args.image), "k_requested": args.k, "tiles_selected": selected,
        "detections": rows,
    }, indent=2) + "\n")
    print(f"Saved {len(rows)} detections and {len(selected)} selected tiles to {args.output_dir}")


if __name__ == "__main__":
    main()
