#!/usr/bin/env python3
"""
Stage-wise visualization of SegFormer-B2 processing one bruise image:
input -> overlapping patch embedding -> encoder stages 1-4 -> All-MLP
decoder -> bruise probability map -> thresholded binary mask.

Hooks the real forward pass of a trained checkpoint -- these are the
model's actual intermediate activations on the given image, not a
schematic/illustration.

Usage (on ORC, with a trained checkpoint):
    python scripts/visualize_segformer_stages.py \
        --checkpoint /home/tbommawa/bruise_segformer_yolo_v1_runs/segformer_b2_teacher/best_model.pt \
        --pretrained /home/tbommawa/paul_all_wavelength_segformer_project/pretrained_weights/segformer_mit_b2 \
        --image /path/to/one_bruise_image.jpg \
        --threshold 0.55 \
        --out stage_visualization.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.models import SegformerWrapper, build_segformer


def to_heatmap(feature_map: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """(C, H, W) -> mean over channels -> normalized 0..1 -> resized to out_h x out_w."""
    fmap = feature_map.float().mean(dim=0).cpu().numpy()
    fmap -= fmap.min()
    if fmap.max() > 0:
        fmap /= fmap.max()
    return cv2.resize(fmap, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def overlay(rgb_uint8: np.ndarray, heat01: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    heat_color = (plt.cm.jet(heat01)[..., :3] * 255).astype("uint8")
    return (rgb_uint8 * (1 - alpha) + heat_color * alpha).astype("uint8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    ap.add_argument("--pretrained", required=True, help="Path to segformer_mit_b2 pretrained dir")
    ap.add_argument("--image", required=True, help="Path to one bruise image")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="stage_visualization.png")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SegformerWrapper(build_segformer(args.pretrained, 1)).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    # ---- capture the raw patch-embedding output (post-LayerNorm, pre-transformer-blocks) ----
    patch_embed_capture = {}

    def patch_embed_hook(module, inputs, output):
        embeddings, height, width = output  # (B, N, C), int, int
        b, n, c = embeddings.shape
        patch_embed_capture["fmap"] = embeddings.transpose(1, 2).reshape(b, c, height, width)[0]

    stage0_patch_embed = model.backbone.encoder.patch_embeddings[0]
    handle = stage0_patch_embed.register_forward_hook(patch_embed_hook)

    # ---- load and preprocess the image ----
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    x = ((img_resized.astype("float32") / 255.0) - mean) / std
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        encoder_out = model.backbone(pixel_values=x, output_hidden_states=True)
        stage_fmaps = [hs[0] for hs in encoder_out.hidden_states]  # 4 stages, each (C_i, H_i, W_i)
        logits = model(x)  # full forward (encoder + decode_head), upsampled to img_size x img_size
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    handle.remove()
    binary_mask = (prob >= args.threshold).astype("uint8")

    H = W = args.img_size
    panels = [("Input image", img_resized, None)]

    patch_heat = to_heatmap(patch_embed_capture["fmap"], H, W)
    panels.append(("Patch embedding\n(overlapping, pre-transformer)", overlay(img_resized, patch_heat), None))

    stage_titles = [
        "Stage 1\n(local color/texture)",
        "Stage 2\n(grouping bruise evidence)",
        "Stage 3\n(irregular bruise shape)",
        "Stage 4\n(whole-image context)",
    ]
    for title, fmap in zip(stage_titles, stage_fmaps):
        heat = to_heatmap(fmap, H, W)
        panels.append((title, overlay(img_resized, heat), None))

    panels.append(("Decoder output\n(bruise probability map)", overlay(img_resized, prob), "prob"))
    panels.append((f"Thresholded mask\n(threshold={args.threshold})", binary_mask * 255, "mask"))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, (title, img, kind) in zip(axes.flat, panels):
        if kind == "mask":
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in axes.flat[len(panels):]:
        ax.axis("off")

    fig.suptitle("SegFormer-B2 stage-wise processing of one bruise image", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
