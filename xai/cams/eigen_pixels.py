#!/usr/bin/env python3
"""
YOLOv5 + EigenCAM (class-agnostic) with:
- Prediction boxes (red) + "Class 92.4%" on the box
- Ground-truth boxes (green) if LABELS_DIR provided (no per-box text)
- EigenCAM overlay (optionally stronger alpha)
- Legend at the top

Also saves quantitative artifacts for research:
- data/eigencam_values/...:
    * *.eigen.heat.npy            (float32 [H,W] in [0,1])
    * *.eigen.top{1,5,10,50,100}p.mask.png   (binary masks)
    * *.eigen.top{...}p.weighted.npy         (OPTIONAL: intensities inside mask)
    * *.eigen.top_thresholds.json            (per-p thresholds)

Run from the yolov5 folder:
  cd /home/WVU-AD/as00391/Downloads/YOLOv5_Detector/yolov5
  source ../venv/bin/activate
  python -u batch_infer_eigencam_mac_topmass.py
"""

from pathlib import Path
import os, sys, json
import cv2
import numpy as np
import torch
from tqdm import tqdm

# YOLOv5 local imports (assume we are inside the yolov5 folder)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# ---------- Grad-CAM (EigenCAM) ----------
_HAS_CAM = True
try:
    from pytorch_grad_cam import EigenCAM
except Exception as e:
    print("[WARN] pytorch-grad-cam not available -> EigenCAM disabled.", file=sys.stderr)
    print("       pip install pytorch-grad-cam", file=sys.stderr)
    _HAS_CAM = False

# ========= CONFIG =========
WEIGHTS       = Path("weights.pt").expanduser()
IMAGES_DIR    = Path("data/images").expanduser()
OUTPUTS_DIR   = Path("data/eigencam").expanduser()            # overlays
VALUES_DIR    = Path("data/eigencam_values").expanduser()     # numeric + masks
LABELS_DIR    = Path("data/labels").expanduser()              # optional (YOLO txt)

IMG_SIZE      = (1024, 1024)
CONF_T        = 0.25
IOU_T         = 0.45
IMG_EXTS      = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

# Device/options
PREFER_MPS    = True
CAM_ALPHA     = 0.50        # make overlay a bit stronger (less “dull”)
LINE_THICK    = 2

# Label text styling
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.6
FONT_THICK    = 2
TEXT_PAD_X    = 5
TEXT_PAD_Y    = 4

# Saving knobs
SAVE_HEATMAP_NUMPY  = True
SAVE_HEATMAP_CSV    = False
SAVE_HEATMAP_GRAY   = False   # grayscale debug PNG of heatmap (0..255)

# Mass-based top-p (percent of total intensity)
TOP_MASS_PERCENTS   = [1, 5, 10, 50, 100]
SAVE_TOP_MASKS      = True
SAVE_WEIGHTED_MASKS = True    # also save *.weighted.npy (original intensities inside the selected set)
# =========================


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _letterbox_resize(im, new_size, stride):
    return letterbox(im, new_shape=new_size, stride=stride, auto=False)[0]


def _yolo_txt_to_xyxy(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path or not txt_path.exists():
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            parts = s.split()
            if len(parts) < 5: continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w/2.0) * img_w
            y1 = (cy - h/2.0) * img_h
            x2 = (cx + w/2.0) * img_w
            y2 = (cy + h/2.0) * img_h
            boxes.append([cls,
                          max(0, x1), max(0, y1),
                          min(img_w-1, x2), min(img_h-1, y2)])
    return boxes


def _prepare_tensor_bgr(im0, new_shape, stride, device):
    """BGR uint8 -> torch float32 [1,3,H,W] normalized."""
    im = _letterbox_resize(im0, new_shape, stride)  # BGR uint8
    im = im.astype(np.float32) / 255.0
    im = im.transpose(2, 0, 1)                      # CHW
    im = np.ascontiguousarray(im)
    t  = torch.from_numpy(im).to(device)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t


def _overlay_cam_on_bgr(im_bgr_uint8, cam_gray_0to1, alpha=CAM_ALPHA):
    cam_uint8 = np.uint8(255 * np.clip(cam_gray_0to1, 0.0, 1.0))
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # BGR
    overlay = cv2.addWeighted(im_bgr_uint8, 1.0, heatmap, alpha, 0)
    return overlay


def _pick_last_conv_layer(yolo_pt_model: torch.nn.Module):
    last_conv = None
    for m in yolo_pt_model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    return last_conv


class TensorOutputWrapper(torch.nn.Module):
    """
    Wrap a model so it ALWAYS returns a single Tensor for Grad-CAM.
    (YOLOv5 can return tuples/lists; CAM expects a Tensor.)
    """
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        if torch.is_tensor(out):
            return out
        if isinstance(out, (list, tuple)):
            for o in out:
                if torch.is_tensor(o):
                    return o
            try:
                return torch.as_tensor(out, device=x.device)
            except Exception:
                return torch.zeros((x.shape[0], 1), device=x.device)
        try:
            return torch.as_tensor(out, device=x.device)
        except Exception:
            return torch.zeros((x.shape[0], 1), device=x.device)


def _class_name(names, idx):
    if isinstance(names, (list, tuple)):
        if 0 <= idx < len(names): return str(names[idx])
        return str(idx)
    if isinstance(names, dict):
        return str(names.get(idx, idx))
    return str(idx)


def _draw_label(img, box_xyxy, label_text, color=(0, 0, 255)):
    x1, y1, x2, y2 = map(int, box_xyxy)
    (tw, th), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICK)
    bg_x1 = x1
    bg_y1 = max(0, y1 - th - 2*TEXT_PAD_Y)
    bg_x2 = x1 + tw + 2*TEXT_PAD_X
    bg_y2 = y1
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
    tx = x1 + TEXT_PAD_X
    ty = y1 - TEXT_PAD_Y
    cv2.putText(img, label_text, (tx, ty), FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)


def _select_mac_device():
    if PREFER_MPS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------- Saving helpers ----------
def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def save_eigen_arrays(base_out_values: Path, heat: np.ndarray, tag: str):
    """
    Save EigenCAM heatmap arrays.
    heat: float32 [H,W] in [0,1]; tag usually 'eigen'.
    """
    if SAVE_HEATMAP_NUMPY:
        p_npy = base_out_values.with_suffix(f".{tag}.heat.npy")
        _ensure_parent(p_npy)
        np.save(str(p_npy), heat.astype(np.float32))
    if SAVE_HEATMAP_CSV:
        p_csv = base_out_values.with_suffix(f".{tag}.heat.csv")
        _ensure_parent(p_csv)
        np.savetxt(str(p_csv), heat.astype(np.float32), delimiter=",")
    if SAVE_HEATMAP_GRAY:
        p_png = base_out_values.with_suffix(f".{tag}.heat.gray.png")
        _ensure_parent(p_png)
        cv2.imwrite(str(p_png), (heat.clip(0,1) * 255).astype(np.uint8))

# ---- TOP-%-BY-MASS with strict tie handling ----
def top_mass_masks_strict(heat: np.ndarray, percents=(1,5,10,50,100), return_weighted=False):
    """
    Mass-based top-p% masks:
      - Sort pixels by intensity (desc)
      - Select the smallest set whose cumulative SUM reaches p% of total mass
      - Handle ties at the threshold by including only as many '==thr' pixels as needed

    Returns dict: p -> (binary_mask[H,W] uint8{0,1}, threshold_float, weighted_mask[H,W] float32 in [0,1] or None)
    """
    H, W = heat.shape[:2]
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

    order = np.argsort(-flat)     # high -> low
    vals  = flat[order]
    csum  = np.cumsum(vals)

    out = {}
    for p in percents:
        target_mass = (p / 100.0) * total
        idx = int(np.searchsorted(csum, target_mass, side="left"))
        idx = min(max(idx, 0), len(vals) - 1)

        thr = float(vals[idx])
        sel_strict = (flat > thr)           # strict to avoid tie bloat
        mass_strict = float(flat[sel_strict].sum())
        need = target_mass - mass_strict

        sel = sel_strict.copy()
        if need > 0:
            equals = (flat == thr)
            eq_indices = np.nonzero(equals)[0]
            if thr > 0 and len(eq_indices) > 0:
                # Deterministic subset: follow descending order
                eq_in_order = [i for i in order if i in set(eq_indices)]
                count_needed = int(np.ceil(need / thr))
                take = min(count_needed, len(eq_in_order))
                if take > 0:
                    sel[np.array(eq_in_order[:take], dtype=np.int64)] = True

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


def main():
    # ---- device + model ----
    device_str = _select_mac_device()
    device = select_device(device_str)
    print(f"[INFO] device = {device}")

    if not WEIGHTS.exists():
        raise SystemExit(f"[FATAL] weights not found: {WEIGHTS}")

    dmb = DetectMultiBackend(str(WEIGHTS), device=device, dnn=False, fp16=False)
    stride, names = dmb.stride, dmb.names
    imgsz = check_img_size([IMG_SIZE[0], IMG_SIZE[1]], s=stride)
    print(f"[INFO] stride = {stride}, imgsz = {imgsz}")

    images_root = Path(IMAGES_DIR)
    out_root    = Path(OUTPUTS_DIR);     out_root.mkdir(parents=True, exist_ok=True)
    values_root = Path(VALUES_DIR);      values_root.mkdir(parents=True, exist_ok=True)
    labels_root = Path(LABELS_DIR) if LABELS_DIR else None

    img_paths = [p for p in images_root.rglob("*") if p.is_file() and _is_image(p)]
    print(f"[INFO] found {len(img_paths)} images under {images_root}")
    for p in img_paths[:10]:
        print("   -", p)
    if not img_paths:
        raise SystemExit(f"[FATAL] No images found under: {images_root}")

    # ---- EigenCAM init ----
    cam_ctx = None
    if _HAS_CAM and hasattr(dmb, "model") and isinstance(dmb.model, torch.nn.Module):
        target_layer = _pick_last_conv_layer(dmb.model)
        if target_layer is None:
            print("[WARN] No Conv2d layer found for CAM; EigenCAM disabled.")
        else:
            wrapped = TensorOutputWrapper(dmb.model).eval()
            cam_ctx = EigenCAM(model=wrapped, target_layers=[target_layer])
            print(f"[INFO] EigenCAM enabled on: {target_layer.__class__.__name__}")
    else:
        print("[INFO] EigenCAM disabled.")

    # ---- loop ----
    if cam_ctx is not None:
        with cam_ctx as cam:
            _run_loop(dmb, cam, images_root, out_root, values_root, labels_root, imgsz, stride, names)
    else:
        _run_loop(dmb, None, images_root, out_root, values_root, labels_root, imgsz, stride, names)


def _run_loop(dmb, cam, images_root, out_root, values_root, labels_root, imgsz, stride, names):
    img_list = [p for p in images_root.rglob("*") if p.is_file() and _is_image(p)]
    for img_path in tqdm(img_list, desc="Detecting"):
        im0 = cv2.imread(str(img_path))  # BGR
        if im0 is None:
            print(f"[WARN] could not read {img_path}")
            continue

        canvas = im0.copy()

        # --- EigenCAM: compute + overlay AND save numeric artifacts
        eigen_resized = None
        if cam is not None:
            inp_cam = _prepare_tensor_bgr(im0, tuple(imgsz), stride, dmb.device)
            with torch.no_grad():
                # EigenCAM returns (B, Hc, Wc); we take [0]
                grayscale_cam = cam(input_tensor=inp_cam, eigen_smooth=True)[0]
            # Use NEAREST to preserve peak values (avoid dull smoothing)
            eigen_resized = cv2.resize(grayscale_cam, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
            eigen_resized = np.clip(eigen_resized, 0.0, 1.0)

            # Overlay for visualization
            canvas = _overlay_cam_on_bgr(canvas, eigen_resized, alpha=CAM_ALPHA)

            # Save numeric artifacts for research
            rel = img_path.relative_to(images_root)
            base_out_values = (values_root / rel).with_suffix("")
            save_eigen_arrays(base_out_values, eigen_resized, tag="eigen")
            if SAVE_TOP_MASKS and TOP_MASS_PERCENTS:
                save_top_mass_artifacts(
                    base_out_values, "eigen", eigen_resized,
                    percents=TOP_MASS_PERCENTS, save_weighted=SAVE_WEIGHTED_MASKS
                )

        # --- Detection
        inp = _prepare_tensor_bgr(im0, tuple(imgsz), stride, dmb.device)
        with torch.no_grad():
            pred = dmb(inp, augment=False, visualize=False)
        pred = non_max_suppression(pred, CONF_T, IOU_T, classes=None, agnostic=False)

        # Predictions: red rectangles + "Class 92.4%" label
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(inp.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = [int(v.item() if hasattr(v, "item") else v) for v in xyxy]
                    cls_idx = int(cls.item() if hasattr(cls, "item") else cls)
                    cls_name = _class_name(names, cls_idx)
                    score = float(conf)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), LINE_THICK)
                    _draw_label(canvas, (x1, y1, x2, y2), f"{cls_name} {score*100:.1f}%", color=(0, 0, 255))

        # Ground truth: green rectangles
        if labels_root is not None:
            rel = img_path.relative_to(images_root)
            lab_path = (labels_root / rel).with_suffix(".txt")
            gts = _yolo_txt_to_xyxy(lab_path, img_w=im0.shape[1], img_h=im0.shape[0])
            for cls_id, gx1, gy1, gx2, gy2 in gts:
                cv2.rectangle(canvas, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (0, 255, 0), LINE_THICK)

        # Legend at top
        cv2.putText(canvas, "Green: Ground Truth Box", (10, 30), FONT, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Red: Prediction Box",   (10, 60), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # Save alongside original tree
        rel = img_path.relative_to(images_root)
        out_img = (out_root / rel).with_suffix(".png")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_img), canvas)
        print("[SAVE]", out_img, "ok:", ok)


if __name__ == "__main__":
    # macOS niceties: avoid OpenMP thread storms
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    main()
