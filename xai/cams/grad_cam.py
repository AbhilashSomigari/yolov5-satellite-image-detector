#!/usr/bin/env python3
"""
xai/cams/grad_cam.py — YOLOv5 + class-targeted Grad-CAM overlays (no EigenCAM)

Adds:
  • Saves PNG overlays under OUTPUTS_DIR/overlays with the SAME basename as the input image
  • Saves raw combined CAM .npy under OUTPUTS_DIR/numpy with the SAME basename as the input image
  • Preserves the relative subfolder structure of IMAGES_DIR

Run (from the repo root):
  source venv/bin/activate
  python -u xai/cams/grad_cam.py
"""

from pathlib import Path
import os, sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ==== Make the repo root importable (models/, utils/) ====
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # repo root (xai/cams/ -> xai/ -> root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ==== YOLOv5 imports ====
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

# ==== Grad-CAM (classic) ====
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

print("[BOOT] xai/cams/grad_cam.py (Grad-CAM only)")
print(f"[VERSIONS] python={sys.version.split()[0]} torch={torch.__version__} cv2={cv2.__version__}")

# ==== CONFIG (paths are relative to the repo root; override with your own if needed) ====
WEIGHTS     = ROOT / "weights.pt"
IMAGES_DIR  = ROOT / "data" / "images"
OUTPUTS_DIR = ROOT / "data" / "gradcam"             # root for outputs
IMG_SIZE    = (1024, 1024)
CONF_T      = 0.25
IOU_T       = 0.45
IMG_EXTS    = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

HEATMAP_ALPHA       = 0.35
SAVE_PER_CLASS_HEAT = False
PREFER_MPS          = True

# --------- Utilities ----------
def _normalize_to_rgb01(bgr_uint8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)

def _auto_find_last_conv2d(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last

def preprocess_bgr_for_yolo(im0_bgr: np.ndarray, new_shape, stride, device):
    im = letterbox(im0_bgr, new_shape=new_shape, stride=stride, auto=False)[0]  # HWC uint8
    im = im.astype(np.float32) / 255.0
    im = im.transpose(2, 0, 1)  # HWC->CHW
    im = np.ascontiguousarray(im)
    t = torch.from_numpy(im).to(device)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t

class ClassScoreTarget:
    """
    Target for YOLO-style raw outputs: (bs, N, 5+nc).
    Backprop mean sigmoid score of class_idx over N predictions.
    """
    def __init__(self, class_idx: int):
        self.class_idx = class_idx
    def __call__(self, model_outputs: torch.Tensor):
        logits = model_outputs[0] if isinstance(model_outputs, (list, tuple)) else model_outputs
        cls_logits = logits[..., 5:]                        # (bs, N, nc)
        cls_scores = cls_logits.sigmoid()[..., self.class_idx]  # (bs, N)
        return cls_scores.mean(dim=1).sum()                 # scalar

def _select_mac_device():
    if PREFER_MPS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# --------- Load model ----------
# Minimize thread contention on macOS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try: cv2.setNumThreads(0)
except Exception: pass

device = select_device(_select_mac_device())
print("[INFO] device =", device)

if not WEIGHTS.exists():
    raise SystemExit(f"[FATAL] weights not found: {WEIGHTS}")

print("[INFO] loading weights:", WEIGHTS)
model = DetectMultiBackend(str(WEIGHTS), device=device, dnn=False, fp16=False)
model.eval()
stride, names = model.stride, model.names
imgsz = check_img_size([IMG_SIZE[0], IMG_SIZE[1]], s=stride)
print("[INFO] stride =", stride, "imgsz =", imgsz)

# Use the raw PyTorch model for Grad-CAM
inner_model = model.model if hasattr(model, "model") else model
# Ensure parameters require grad (some conversions/wrappers may disable)
for p in inner_model.parameters():
    p.requires_grad_(True)

target_layer = _auto_find_last_conv2d(inner_model)
print(f"[INFO] Grad-CAM target layer: {target_layer}")

cam = GradCAM(model=inner_model, target_layers=[target_layer])

# --------- Collect images ----------
images_root = Path(IMAGES_DIR)
img_paths = [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
print(f"[INFO] images_root={images_root}")
print(f"[INFO] found {len(img_paths)} images")
for p in img_paths[:10]:
    print("   -", p)
if not img_paths:
    raise SystemExit(f"[FATAL] No images found under: {images_root}")

# --------- Outputs ----------
out_root      = Path(OUTPUTS_DIR)
out_png_root  = out_root / "overlays"
out_npy_root  = out_root / "numpy"
out_root.mkdir(parents=True, exist_ok=True)
out_png_root.mkdir(parents=True, exist_ok=True)
out_npy_root.mkdir(parents=True, exist_ok=True)

print("[INFO] starting detection + Grad-CAM...")
for img_path in tqdm(img_paths, desc="Detecting"):
    im0_bgr = cv2.imread(str(img_path))
    if im0_bgr is None:
        print("[WARN] could not read", img_path); continue

    # Preprocess for detection AND CAM; make sure same device
    im = preprocess_bgr_for_yolo(im0_bgr, tuple(imgsz), stride, model.device)

    # ---- Detection (no grad needed here) ----
    with torch.no_grad():
        raw_pred = model(im, augment=False, visualize=False)  # (bs, N, 5+nc)
    pred = non_max_suppression(raw_pred, CONF_T, IOU_T, classes=None, agnostic=False)

    # Unique detected classes for this image
    detected_classes = set()
    for det in pred:
        if det is not None and len(det):
            for cls in det[:, 5].tolist():
                detected_classes.add(int(cls))

    rgb_base = _normalize_to_rgb01(im0_bgr)

    # ---- Grad-CAM per class (force grad ON) ----
    heatmaps = []
    if detected_classes:
        # Important: re-use the SAME image tensor but with requires_grad for CAM path
        im_cam = im.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            for cls_id in sorted(detected_classes):
                targets = [ClassScoreTarget(cls_id)]
                grayscale_cam = cam(input_tensor=im_cam, targets=targets)  # (1, Hc, Wc) with grad path
                heat = grayscale_cam[0]
                heat_resized = cv2.resize(heat, (im0_bgr.shape[1], im0_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                heatmaps.append(heat_resized)

                if SAVE_PER_CLASS_HEAT:
                    overlay_rgb_pc = show_cam_on_image(rgb_base, heat_resized, use_rgb=True, image_weight=1.0 - HEATMAP_ALPHA)
                    rel = img_path.relative_to(images_root)
                    # keep same basename, but add ".class{cls}.png" in the *overlays* folder
                    out_c_png = (out_png_root / rel).with_suffix(f".class{cls_id}.png")
                    out_c_png.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_c_png), cv2.cvtColor(overlay_rgb_pc, cv2.COLOR_RGB2BGR))

    # Combine and overlay
    if heatmaps:
        combined_heat = np.maximum.reduce(heatmaps)
        combined_heat = np.clip(combined_heat, 0.0, 1.0).astype(np.float32)
        overlay_rgb = show_cam_on_image(rgb_base, combined_heat, use_rgb=True, image_weight=1.0 - HEATMAP_ALPHA)
    else:
        # No detections -> save zeros heatmap and just the original image as overlay
        combined_heat = np.zeros((im0_bgr.shape[0], im0_bgr.shape[1]), dtype=np.float32)
        overlay_rgb   = (rgb_base * 255).astype(np.uint8)

    # Draw detections on top
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    annot = Annotator(overlay_bgr, line_width=3, example=str(names))
    for det in pred:
        if det is None or not len(det):
            continue
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0_bgr.shape).round()
        for *xyxy, conf, cls in det:
            c = int(cls)
            label = f"{names[c] if isinstance(names,(list,tuple,dict)) else c} {float(conf):.2f}"
            annot.box_label(xyxy, label, color=colors(c, True))

    # ===== Save with SAME BASENAME as input image =====
    rel = img_path.relative_to(images_root)

    # 1) PNG overlay
    out_png = (out_png_root / rel).with_suffix(".png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    ok_png = cv2.imwrite(str(out_png), annot.result())

    # 2) NPY raw combined heatmap (float32 in [0,1], same HxW as original image)
    out_npy = (out_npy_root / rel).with_suffix(".npy")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), combined_heat)

    print(f"[SAVE] PNG: {out_png} ok={ok_png}")
    print(f"[SAVE] NPY: {out_npy} shape={combined_heat.shape} dtype={combined_heat.dtype}")

print("[DONE] Grad-CAM overlays (.png) and heatmaps (.npy) saved in separate folders.")
