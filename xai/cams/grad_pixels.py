#!/usr/bin/env python3
"""
batch_infer_gradcam_mac_fix.py — YOLOv5 + class-targeted Grad-CAM overlays (no EigenCAM)
+ Save numeric heatmaps & TOP-%-BY-MASS masks in a separate data folder.

Key points:
  • Top-p is by MASS (cumulative intensity), not by pixel COUNT.
  • Precise tie-handling at the cutoff (include just enough equal-threshold pixels).
  • Optional weighted masks (keep original intensities inside selected set).
  • Uses INTER_NEAREST for resizing heatmaps to avoid smoothing/dulling.

Run:
  cd ~/Downloads/YOLOv5_Detector/yolov5
  source ../venv/bin/activate
  python -u batch_infer_gradcam_mac_fix.py
"""

from pathlib import Path
import os, sys, json
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ==== YOLOv5 imports ====
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

# ==== Grad-CAM (classic) ====
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

print("[BOOT] batch_infer_gradcam_mac_fix.py (Grad-CAM only)")
print(f"[VERSIONS] python={sys.version.split()[0]} torch={torch.__version__} cv2={cv2.__version__}")

# =========================
# =======  CONFIG  ========
# =========================
WEIGHTS       = Path("weights.pt").expanduser()
IMAGES_DIR    = Path("data/images").expanduser()            # input images root
OUTPUTS_DIR   = Path("data/gradcam").expanduser()           # colored overlays (visuals)
VALUES_DIR    = Path("data/grad_pixelvalues").expanduser()  # numeric arrays + masks

IMG_SIZE      = (1024, 1024)
CONF_T        = 0.25
IOU_T         = 0.45
IMG_EXTS      = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

# Make overlays less "dull" visually
HEATMAP_ALPHA       = 0.50
SAVE_PER_CLASS_HEAT = False
PREFER_MPS          = True

# ---- Heatmap save/export knobs ----
SAVE_HEATMAP_NUMPY   = True     # .npy arrays (recommended)
SAVE_HEATMAP_CSV     = False    # .csv (large; only if you need it)
SAVE_FLOAT_PNG       = False    # grayscale 0..255 debug image of the heatmap

# ---- TOP-%-BY-MASS config ----
TOP_MASS_PERCENTS    = [1, 5, 10, 50, 100]   # percentage of total intensity mass
SAVE_TOP_MASKS       = True
SAVE_WEIGHTED_MASKS  = True   # also save float arrays with original intensities inside the selected set

# =========================
# ======  Utilities  ======
# =========================
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
        cls_logits = logits[..., 5:]                            # (bs, N, nc)
        cls_scores = cls_logits.sigmoid()[..., self.class_idx]  # (bs, N)
        return cls_scores.mean(dim=1).sum()                     # scalar

def _select_mac_device():
    if PREFER_MPS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_heatmap_arrays(base_out_values: Path, heat: np.ndarray, tag: str):
    """
    Save heatmap numeric arrays for later analysis.
    heat: float32 [H,W] in [0,1]
    tag: e.g., 'class7' or 'combined'
    Writes into VALUES_DIR subtree.
    """
    if SAVE_HEATMAP_NUMPY:
        p_npy = base_out_values.with_suffix(f".{tag}.heat.npy")
        _ensure_parent(p_npy)
        np.save(str(p_npy), heat.astype(np.float32))
    if SAVE_HEATMAP_CSV:
        p_csv = base_out_values.with_suffix(f".{tag}.heat.csv")
        _ensure_parent(p_csv)
        np.savetxt(str(p_csv), heat.astype(np.float32), delimiter=",")
    if SAVE_FLOAT_PNG:
        p_png = base_out_values.with_suffix(f".{tag}.heat.gray.png")
        _ensure_parent(p_png)
        cv2.imwrite(str(p_png), (heat.clip(0,1) * 255).astype(np.uint8))

# ======================================
# === TOP-%-BY-MASS (with tie control) ==
# ======================================
def top_mass_masks_strict(heat: np.ndarray, percents=(1,5,10,50,100), return_weighted=False):
    """
    Mass-based top-p% masks:
      - Sort pixels by intensity (desc)
      - Select the smallest set whose cumulative SUM reaches p% of total mass
      - Handle ties at the threshold by including only as many '==thr' pixels as needed

    Returns dict: p -> (binary_mask[H,W] uint8{0,1}, threshold_float, weighted_mask[H,W] float32 in [0,1] or None)
    """
    H = heat.shape[0]
    W = heat.shape[1]
    h = heat.astype(np.float32).clip(0, 1)
    flat = h.reshape(-1)
    total = float(flat.sum())

    if total <= 0.0:
        out = {}
        for p in percents:
            mask = np.zeros((H, W), dtype=np.uint8)
            wmask = np.zeros((H, W), dtype=np.float32) if return_weighted else None
            out[p] = (mask, 0.0, wmask)
        return out

    # Descending order indices and values
    order = np.argsort(-flat)   # high -> low
    vals  = flat[order]
    csum  = np.cumsum(vals)

    out = {}
    for p in percents:
        target_mass = (p / 100.0) * total
        idx = int(np.searchsorted(csum, target_mass, side="left"))
        idx = min(max(idx, 0), len(vals) - 1)

        thr = float(vals[idx])
        # Strict '>' to avoid tie bloat:
        sel_strict = (flat > thr)

        # How much mass we already have with '>'
        mass_strict = float(flat[sel_strict].sum())
        need = target_mass - mass_strict

        sel = sel_strict.copy()
        if need > 0:
            equals = (flat == thr)
            # pick only as many equal-threshold pixels as needed by mass
            eq_indices = np.nonzero(equals)[0]
            # They all have value==thr; number needed:
            # count_needed * thr >= need  =>  count_needed = ceil(need / thr)
            if thr > 0:
                count_needed = int(np.ceil(need / thr))
            else:
                # degenerate: thr==0 (lots of zeros). If we reached here, include none (keep compact)
                count_needed = 0

            if count_needed > 0 and len(eq_indices) > 0:
                # Choose a deterministic subset: the earliest in order among equals
                # eq_indices_in_order: equals intersected with 'order' sequence
                eq_in_order = [i for i in order if i in set(eq_indices)]
                take = min(count_needed, len(eq_in_order))
                if take > 0:
                    sel[np.array(eq_in_order[:take], dtype=np.int64)] = True

        # Build masks
        mask = sel.reshape(H, W).astype(np.uint8)

        if return_weighted:
            wmask = np.zeros((H, W), dtype=np.float32)
            wmask[mask == 1] = h[mask == 1]
        else:
            wmask = None

        out[p] = (mask, thr, wmask)

    return out

def save_top_mass_artifacts(base_out_values: Path, tag: str, heat: np.ndarray, percents=(1,5,10,50,100), save_weighted=False):
    """
    Save top-mass masks (binary PNGs), thresholds JSON, and optional weighted masks (.npy).
    """
    res = top_mass_masks_strict(heat, percents, return_weighted=save_weighted)
    thresholds = {}
    for p, (mask, thr, wmask) in res.items():
        thresholds[str(p)] = thr
        p_mask = base_out_values.with_suffix(f".{tag}.top{p}p.mask.png")
        _ensure_parent(p_mask)
        cv2.imwrite(str(p_mask), (mask.astype(np.uint8) * 255))

        if save_weighted and wmask is not None:
            p_w = base_out_values.with_suffix(f".{tag}.top{p}p.weighted.npy")
            _ensure_parent(p_w)
            np.save(str(p_w), wmask.astype(np.float32))

    p_json = base_out_values.with_suffix(f".{tag}.top_thresholds.json")
    _ensure_parent(p_json)
    with open(p_json, "w") as f:
        json.dump({"percents": list(percents), "thresholds": thresholds}, f, indent=2)

# =========================
# ========  Init  =========
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)
except Exception:
    pass

device = select_device("mps" if (PREFER_MPS and torch.backends.mps.is_available()) else "cpu")
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
for p in inner_model.parameters():
    p.requires_grad_(True)

target_layer = _auto_find_last_conv2d(inner_model)
print(f"[INFO] Grad-CAM target layer: {target_layer}")

cam = GradCAM(model=inner_model, target_layers=[target_layer])

# Collect images
images_root = Path(IMAGES_DIR)
img_paths = [p for p in images_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
print(f"[INFO] images_root={images_root}")
print(f"[INFO] found {len(img_paths)} images")
for p in img_paths[:10]:
    print("   -", p)
if not img_paths:
    raise SystemExit(f"[FATAL] No images found under: {images_root}")

# Ensure base output dirs exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)  # color overlays
VALUES_DIR.mkdir(parents=True, exist_ok=True)   # numeric values + masks

# =========================
# ========  Main  =========
# =========================
print("[INFO] starting detection + Grad-CAM...")
for img_path in tqdm(img_paths, desc="Detecting"):
    im0_bgr = cv2.imread(str(img_path))
    if im0_bgr is None:
        print("[WARN] could not read", img_path); continue

    # Preprocess for detection AND CAM; same device
    im = preprocess_bgr_for_yolo(im0_bgr, tuple(imgsz), stride, model.device)

    # ---- Detection (no grad) ----
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
        im_cam = im.clone().detach().requires_grad_(True)

        with torch.enable_grad():
            for cls_id in sorted(detected_classes):
                targets = [ClassScoreTarget(cls_id)]
                grayscale_cam = cam(input_tensor=im_cam, targets=targets)  # (1, Hc, Wc)
                heat = grayscale_cam[0]
                # Use NEAREST to avoid smoothing/dulling and large tie-plateaus
                heat_resized = cv2.resize(
                    heat, (im0_bgr.shape[1], im0_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                heat_resized = np.clip(heat_resized, 0.0, 1.0)
                heatmaps.append(heat_resized)

                # ---- Optional: per-class color overlay ----
                if SAVE_PER_CLASS_HEAT:
                    overlay_rgb_pc = show_cam_on_image(
                        rgb_base, heat_resized, use_rgb=True,
                        image_weight=1.0 - HEATMAP_ALPHA
                    )
                    rel = img_path.relative_to(images_root)
                    out_c = (OUTPUTS_DIR / rel).with_suffix(f".class{cls_id}.gradcam.png")
                    _ensure_parent(out_c)
                    cv2.imwrite(str(out_c), cv2.cvtColor(overlay_rgb_pc, cv2.COLOR_RGB2BGR))

                # ---- Save per-class arrays + TOP-%-BY-MASS artifacts ----
                rel = img_path.relative_to(images_root)
                base_out_values = (VALUES_DIR / rel).with_suffix("")
                tag = f"class{cls_id}"
                save_heatmap_arrays(base_out_values, heat_resized, tag)
                if SAVE_TOP_MASKS and TOP_MASS_PERCENTS:
                    save_top_mass_artifacts(
                        base_out_values, tag, heat_resized,
                        percents=TOP_MASS_PERCENTS, save_weighted=SAVE_WEIGHTED_MASKS
                    )

    # ---- Combine and overlay ----
    if heatmaps:
        combined_heat = np.maximum.reduce(heatmaps)
        combined_heat = np.clip(combined_heat, 0.0, 1.0)
        overlay_rgb = show_cam_on_image(
            rgb_base, combined_heat, use_rgb=True, image_weight=1.0 - HEATMAP_ALPHA
        )

        # Save combined arrays + TOP-%-BY-MASS artifacts
        rel = img_path.relative_to(images_root)
        base_out_values = (VALUES_DIR / rel).with_suffix("")
        save_heatmap_arrays(base_out_values, combined_heat, "combined")
        if SAVE_TOP_MASKS and TOP_MASS_PERCENTS:
            save_top_mass_artifacts(
                base_out_values, "combined", combined_heat,
                percents=TOP_MASS_PERCENTS, save_weighted=SAVE_WEIGHTED_MASKS
            )
    else:
        overlay_rgb = (rgb_base * 255).astype(np.uint8)

    # ---- Draw detections on top (for the overlay image only) ----
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

    # ---- Save colored overlay to OUTPUTS_DIR ----
    rel = img_path.relative_to(images_root)
    out_img = (OUTPUTS_DIR / rel).with_suffix(".gradcam.png")
    _ensure_parent(out_img)
    ok = cv2.imwrite(str(out_img), annot.result())
    print("[SAVE] overlay:", out_img, "ok:", ok)

print("[DONE] Grad-CAM overlays + TOP-%-BY-MASS values saved.")
