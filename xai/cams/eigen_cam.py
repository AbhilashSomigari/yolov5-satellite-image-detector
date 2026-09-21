#!/usr/bin/env python3
"""
YOLOv5 batch inference with:
- Prediction boxes (red) + "Class 92.4%" on the box
- Ground-truth boxes (green) if LABELS_DIR provided (no per-box text)
- EigenCAM overlay (optional)
- Legend at the top

macOS + VS Code notes:
- Works on CPU or Apple Silicon (MPS) automatically.
- No CUDA or env vars required on Mac.
- Run from the repo root:

  source venv/bin/activate
  python -u xai/cams/eigen_cam.py
"""

from pathlib import Path
import os, sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Make the repo root importable (models/, utils/)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # repo root (xai/cams/ -> xai/ -> root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# ---------- Grad-CAM (EigenCAM) ----------
_HAS_CAM = True
try:
    from pytorch_grad_cam import EigenCAM
except Exception as e:
    print("[WARN] pytorch-grad-cam not available -> Grad-CAM disabled.", file=sys.stderr)
    print("       pip install pytorch-grad-cam", file=sys.stderr)
    _HAS_CAM = False

# ========= CONFIG (edit for your Mac paths if needed) =========
# Use expanduser so "~" works on macOS home folders
WEIGHTS     = ROOT / "weights.pt"
IMAGES_DIR  = ROOT / "data" / "images"
OUTPUTS_DIR = ROOT / "data" / "eigencam"
LABELS_DIR  = ROOT / "data" / "labels"

IMG_SIZE   = (1024, 1024)
CONF_T     = 0.25
IOU_T      = 0.45
IMG_EXTS   = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

# Try to use Apple Silicon GPU (MPS) if available; otherwise CPU
PREFER_MPS = True
CAM_ALPHA  = 0.35
LINE_THICK = 2

# label text styling
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 2
TEXT_PAD_X = 5
TEXT_PAD_Y = 4
# =============================================================


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
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                continue
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
    im = _letterbox_resize(im0, new_shape, stride)          # BGR uint8
    im = im.astype(np.float32) / 255.0
    im = im.transpose(2, 0, 1)                              # CHW
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
    """
    Prefer Apple MPS if available; else CPU.
    select_device('mps') is supported in YOLOv5.
    """
    if PREFER_MPS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    # ---- images ----
    images_root = Path(IMAGES_DIR)
    out_root    = Path(OUTPUTS_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    labels_root = Path(LABELS_DIR) if LABELS_DIR else None

    img_paths = [p for p in images_root.rglob("*") if p.is_file() and _is_image(p)]
    print(f"[INFO] found {len(img_paths)} images under {images_root}")
    for p in img_paths[:10]:
        print("   -", p)
    if not img_paths:
        raise SystemExit(f"[FATAL] No images found under: {images_root}")

    # ---- Grad-CAM ----
    cam_ctx = None
    if _HAS_CAM and hasattr(dmb, "model") and isinstance(dmb.model, torch.nn.Module):
        target_layer = _pick_last_conv_layer(dmb.model)
        if target_layer is None:
            print("[WARN] No Conv2d layer found for CAM; CAM disabled.")
        else:
            wrapped = TensorOutputWrapper(dmb.model).eval()
            # Important for MPS: ensure model and input tensors are on the same device
            cam_ctx = EigenCAM(model=wrapped, target_layers=[target_layer])
            print(f"[INFO] EigenCAM enabled on: {target_layer.__class__.__name__}")
    else:
        print("[INFO] Grad-CAM disabled.")

    # ---- loop ----
    if cam_ctx is not None:
        # Context manager ensures hooks are cleaned (important on macOS as well)
        with cam_ctx as cam:
            _run_loop(dmb, cam, images_root, out_root, labels_root, imgsz, stride, names)
    else:
        _run_loop(dmb, None, images_root, out_root, labels_root, imgsz, stride, names)


def _run_loop(dmb, cam, images_root, out_root, labels_root, imgsz, stride, names):
    img_list = [p for p in images_root.rglob("*") if p.is_file() and _is_image(p)]
    for img_path in tqdm(img_list, desc="Detecting"):
        im0 = cv2.imread(str(img_path))  # BGR
        if im0 is None:
            print(f"[WARN] could not read {img_path}")
            continue

        canvas = im0.copy()

        # --- EigenCAM overlay first (optional)
        if cam is not None:
            inp_cam = _prepare_tensor_bgr(im0, tuple(imgsz), stride, dmb.device)
            with torch.no_grad():
                grayscale_cam = cam(input_tensor=inp_cam, targets=None, eigen_smooth=True)[0]
            cam_resized = cv2.resize(grayscale_cam, (im0.shape[1], im0.shape[0]))
            canvas = _overlay_cam_on_bgr(canvas, cam_resized)

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
        cv2.putText(canvas, "Red: Prediction Box", (10, 60), FONT, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

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
