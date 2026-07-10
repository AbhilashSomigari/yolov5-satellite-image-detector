#!/usr/bin/env python3
"""
yolov5_ig_strict.py — YOLOv5 + Captum Integrated Gradients (STRICT, VERBOSE)

Key features:
  • Absolute paths + up-front write permission test (text + PNG).
  • Detailed logging of CWD, image count, model shapes, and save locations.
  • Fail-fast with full tracebacks (no swallowed warnings).
  • Pre-NMS differentiable scalar (mean(topK(sigmoid(obj)*sigmoid(cls)))).
  • Optional debug: save resized input fed to model.
  • PNG save with OpenCV -> PIL fallback (verified).

Run (example):
  python -u yolov5_ig_strict.py \
    --weights yolov5s.pt \
    --images data/images \
    --out_overlays data/ig_overlays \
    --out_numpy data/ig_numpy \
    --imgsz 640 \
    --steps 64 \
    --topk 50 \
    --target_class -1 \
    --baseline black \
    --debug_save_inputs 0
"""

from __future__ import annotations
from pathlib import Path
import argparse
import traceback
import sys
import os
import math
import cv2
import numpy as np
import torch

# PIL fallback for saving PNGs
try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

# YOLOv5 internals (assumes this file is run inside a YOLOv5 clone or PYTHONPATH set)
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from captum.attr import IntegratedGradients

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# -------------------- utilities --------------------
def abspath(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def print_kv(k: str, v) -> None:
    print(f"{k:<12} : {v}")

def _ensure_dir_writable(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    test_txt = d / "__write_test.txt"
    test_png = d / "__write_test.png"
    test_txt.write_text("ok\n", encoding="utf-8")
    # tiny white png test
    arr = np.full((8, 8, 3), 255, np.uint8)
    ok = cv2.imwrite(str(test_png), arr)
    if not ok:
        if _HAS_PIL:
            _PILImage.fromarray(arr[..., ::-1]).save(str(test_png))  # RGB order
        else:
            raise IOError(f"OpenCV cannot write PNG in {d} and PIL not available")
    if not (test_txt.exists() and test_png.exists()):
        raise IOError(f"Write test failed in {d}")


def load_image_for_yolov5(path: Path, imgsz: int, device: torch.device):
    im0 = cv2.imread(str(path))
    if im0 is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
    lb, ratio, pad = letterbox(im0, new_shape=imgsz, auto=False)
    lb = lb.astype(np.float32) / 255.0
    lb = lb.transpose(2, 0, 1)  # HWC->CHW
    xin = torch.from_numpy(lb).unsqueeze(0).to(device)  # [1,3,H,W]
    xin.requires_grad_(True)
    return im0, xin  # original RGB, input tensor [1,3,H,W]


def to_overlay(rgb_uint8: np.ndarray, attributions: torch.Tensor | np.ndarray, alpha=0.45) -> np.ndarray:
    if isinstance(attributions, torch.Tensor):
        A = attributions.detach().cpu().numpy()
    else:
        A = attributions
    if A.ndim == 4:
        A = A[0]          # [3,H,W]
    if A.ndim != 3 or A.shape[0] != 3:
        raise ValueError(f"Unexpected IG shape for overlay: {A.shape}")
    A = np.abs(A).mean(axis=0)  # [H,W]
    if not np.isfinite(A).all():
        raise ValueError("IG has NaN/Inf values; check model/target/baseline")

    A = A - A.min()
    denom = A.max() + 1e-12
    A = (A / denom * 255.0).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(A, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    H0, W0 = rgb_uint8.shape[:2]
    heat_rgb = cv2.resize(heat_rgb, (W0, H0), interpolation=cv2.INTER_LINEAR)

    out = (alpha * heat_rgb + (1 - alpha) * rgb_uint8).clip(0, 255).astype(np.uint8)
    return out


def _save_png(png_path: Path, rgb_uint8: np.ndarray) -> None:
    # Try OpenCV first (expects BGR)
    ok = cv2.imwrite(str(png_path), cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        if _HAS_PIL:
            _PILImage.fromarray(rgb_uint8).save(str(png_path))
        else:
            raise IOError(f"OpenCV failed to write {png_path} and PIL not available")
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise IOError(f"PNG save verification failed: {png_path}")


class YoloIGScalar(torch.nn.Module):
    """
    Convert YOLOv5 pre-NMS outputs into a single differentiable scalar:
      scalar = mean(topK(sigmoid(obj) * sigmoid(cls_prob)))
    If target_class >= 0, use that class; otherwise pick max class per anchor.
    """
    def __init__(self, model: DetectMultiBackend, target_class: int, topk: int):
        super().__init__()
        self.model = model
        self.target_class = target_class
        self.topk = topk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x, augment=False, visualize=False)
        pred = out[0] if isinstance(out, (list, tuple)) else out

        if pred.ndim == 2:   # [N, 5+nc]
            pred = pred.unsqueeze(0)
        if pred.ndim != 3 or pred.shape[-1] < 5:
            raise RuntimeError(f"Model forward returned unexpected shape: {tuple(pred.shape)}")

        obj = torch.sigmoid(pred[..., 4])   # [B,N]
        Cdim = pred.shape[-1] - 5
        if Cdim > 0:
            cls_logits = pred[..., 5:]      # [B,N,NC]
            cls_prob = torch.sigmoid(cls_logits)
            if self.target_class is not None and self.target_class >= 0:
                if self.target_class >= Cdim:
                    raise ValueError(f"--target_class={self.target_class} out of range [0,{Cdim-1}]")
                cls_sel = cls_prob[..., self.target_class]  # [B,N]
            else:
                cls_sel, _ = cls_prob.max(dim=-1)          # [B,N]
        else:
            cls_sel = torch.ones_like(obj)

        scores = obj * cls_sel                # [B,N]
        N = scores.shape[1]
        if N == 0:
            # return a 1-element zero tensor (NOT a 0-dim scalar)
            return scores.sum().new_tensor([0.0])

        k = max(1, min(self.topk, N))
        topk_vals, _ = torch.topk(scores, k=k, dim=1)
        scalar = topk_vals.mean(dim=1).sum()  # 0-dim scalar
        return scalar.unsqueeze(0)            # <-- make it shape [1]
        out = self.model(x, augment=False, visualize=False)
        pred = out[0] if isinstance(out, (list, tuple)) else out

        if pred.ndim == 2:   # [N, 5+nc]
            pred = pred.unsqueeze(0)
        if pred.ndim != 3 or pred.shape[-1] < 5:
            raise RuntimeError(f"Model forward returned unexpected shape: {tuple(pred.shape)}")

        obj = torch.sigmoid(pred[..., 4])   # [B,N]
        Cdim = pred.shape[-1] - 5
        if Cdim > 0:
            cls_logits = pred[..., 5:]      # [B,N,NC]
            cls_prob = torch.sigmoid(cls_logits)
            if self.target_class is not None and self.target_class >= 0:
                if self.target_class >= Cdim:
                    raise ValueError(f"--target_class={self.target_class} out of range [0,{Cdim-1}]")
                cls_sel = cls_prob[..., self.target_class]  # [B,N]
            else:
                cls_sel, _ = cls_prob.max(dim=-1)          # [B,N]
        else:
            # Model without class dimension (unlikely); treat as 1
            cls_sel = torch.ones_like(obj)

        scores = obj * cls_sel  # [B,N]
        N = scores.shape[1]
        if N == 0:
            # no anchors?
            return (scores.sum() * 0.0)  # scalar 0 with grad

        k = max(1, min(self.topk, N))
        topk_vals, _ = torch.topk(scores, k=k, dim=1)
        scalar = topk_vals.mean(dim=1)      # [B]
        return scalar.sum()                  # scalar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to YOLOv5 .pt")
    ap.add_argument("--images", type=str, required=True, help="Folder containing input images")
    ap.add_argument("--out_overlays", type=str, required=True, help="Folder to write overlay PNGs")
    ap.add_argument("--out_numpy", type=str, required=True, help="Folder to write IG NPYs")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size")
    ap.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or CUDA id like '0'")
    ap.add_argument("--steps", type=int, default=64, help="IG steps")
    ap.add_argument("--topk", type=int, default=50, help="Top-K anchors for scalar objective")
    ap.add_argument("--target_class", type=int, default=-1, help="-1 auto; otherwise fixed class id")
    ap.add_argument("--baseline", type=str, default="black", choices=["black", "blur"], help="IG baseline")
    ap.add_argument("--debug_save_inputs", type=int, default=0, help="1 to save resized input per image")
    args = ap.parse_args()

    # Resolve absolute paths + print
    cwd = abspath(".")
    weights_p = abspath(args.weights)
    images_d = abspath(args.images)
    out_over = abspath(args.out_overlays)
    out_npy = abspath(args.out_numpy)

    print_kv("[CWD]", cwd)
    print_kv("weights", weights_p)
    print_kv("images", images_d)
    print_kv("overlays", out_over)
    print_kv("numpy", out_npy)

    # Check inputs
    if not weights_p.exists():
        raise FileNotFoundError(f"Weights not found: {weights_p}")
    if not images_d.exists():
        raise FileNotFoundError(f"Images folder not found: {images_d}")

    # Ensure writable outputs
    _ensure_dir_writable(out_over)
    _ensure_dir_writable(out_npy)

    # Device + model
    device = select_device('0' if (args.device.lower() in {"auto", ""} and torch.cuda.is_available()) else args.device)
    print_kv("device", device)

    model = DetectMultiBackend(str(weights_p), device=device, dnn=False, data=None, fp16=False)
    model.to(device)
    model.eval()

    # Wrap scalar head for IG
    head = YoloIGScalar(model, target_class=args.target_class, topk=args.topk)
    ig = IntegratedGradients(head)

    # Collect images
    img_paths = [p for p in images_d.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print_kv("images_found", len(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {images_d}")

    ok_count = 0
    fail_count = 0

    for i, p in enumerate(sorted(img_paths), start=1):
        print(f"\n[{i}/{len(img_paths)}] {p.name}")
        try:
            rgb, xin = load_image_for_yolov5(p, imgsz=args.imgsz, device=device)
            print_kv("input.shape", tuple(xin.shape))

            # Optional debug: save resized input
            if args.debug_save_inputs:
                dbg = (xin.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                dbg_png = out_over / f"{p.stem}.__debug_input.png"
                _save_png(dbg_png, dbg)

            # Baseline
            if args.baseline == "black":
                baseline = torch.zeros_like(xin)
            else:
                # blur baseline
                x_np = (xin.detach().cpu().numpy()[0].transpose(1, 2, 0) * 255.0).astype(np.uint8)
                x_np = cv2.GaussianBlur(x_np, (31, 31), 0)
                baseline = torch.from_numpy(x_np.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

            # IG (no torch.no_grad())
            torch.set_grad_enabled(True)
            attributions = ig.attribute(
                xin,
                baselines=baseline,
                n_steps=args.steps,
                method="riemann_trapezoid",
                internal_batch_size=None
            )  # [1,3,H,W]

            if not torch.isfinite(attributions).all():
                raise ValueError("IG produced NaN/Inf values")

            # Save NPY
            npy_path = out_npy / f"{p.stem}.ig.npy"
            np.save(str(npy_path), attributions.detach().cpu().numpy())
            if not npy_path.exists() or npy_path.stat().st_size == 0:
                raise IOError(f"Failed to write NPY: {npy_path}")

            # Save overlay
            overlay = to_overlay(rgb, attributions, alpha=0.45)
            png_path = out_over / f"{p.stem}.ig.png"
            _save_png(png_path, overlay)

            print(f"[SAVE] -> {png_path.name} | {npy_path.name}")
            ok_count += 1

        except Exception as e:
            fail_count += 1
            print("!!! ERROR on", p.name)
            traceback.print_exc()

    print("\n====== SUMMARY ======")
    print_kv("processed", len(img_paths))
    print_kv("saved_ok", ok_count)
    print_kv("failed", fail_count)
    print_kv("overlays_dir", out_over)
    print_kv("numpy_dir", out_npy)


if __name__ == "__main__":
    main()
