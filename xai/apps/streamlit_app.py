#!/usr/bin/env python3
"""
xai/apps/streamlit_app.py — YOLOv5 + Grad-CAM + EigenCAM + F_Norm (Mac/MPS-friendly)

- Uses the SAME Grad-CAM logic as xai/cams/grad_cam.py:
    * DetectMultiBackend for YOLO
    * inner_model = model.model for CAM
    * ClassScoreTarget on raw YOLO outputs (bs, N, 5+nc)
    * Per-class heatmaps combined with max() over classes
    * show_cam_on_image() for overlay

- Additional:
    * EigenCAM with same targets
    * F_Norm metrics (raw ratio, area-corrected ratio, F_norm in [0,1])
      for both Grad-CAM and EigenCAM.

Run (from the repo root):
    streamlit run xai/apps/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn

# ==== Make the repo root importable (models/, utils/) ====
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # repo root (xai/apps/ -> xai/ -> root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.augmentations import letterbox
from utils.plots import colors

# ==== Grad-CAM ====
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ----------------------------------
# 0. Small helpers (copied/adapted from your script)
# ----------------------------------
HEATMAP_ALPHA = 0.35  # same as offline script


def _normalize_to_rgb01(bgr_uint8: np.ndarray) -> np.ndarray:
    """BGR uint8 -> RGB float in [0,1]."""
    rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _auto_find_last_conv2d(module: nn.Module) -> nn.Module:
    """Return the last Conv2d inside a module."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last


def preprocess_bgr_for_yolo(im0_bgr: np.ndarray, new_shape, stride, device):
    """Letterbox + normalize, identical to your batch script."""
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
    Same as in batch_infer_gradcam_mac_fix.py.
    """

    def __init__(self, class_idx: int):
        self.class_idx = class_idx

    def __call__(self, model_outputs: torch.Tensor):
        logits = model_outputs[0] if isinstance(model_outputs, (list, tuple)) else model_outputs
        cls_logits = logits[..., 5:]                         # (bs, N, nc)
        cls_scores = cls_logits.sigmoid()[..., self.class_idx]  # (bs, N)
        return cls_scores.mean(dim=1).sum()                  # scalar


def _select_mac_device_str():
    """Return 'mps' if available, else 'cpu' (string for DetectMultiBackend)."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def draw_yolo_boxes_overlay(img_rgb: np.ndarray, detections: torch.Tensor,
                            names, thickness: int = 2) -> np.ndarray:
    """Draw YOLO boxes+labels on RGB image."""
    overlay = img_rgb.copy()
    if detections is None or len(detections) == 0:
        return overlay

    h, w, _ = overlay.shape
    for *xyxy, conf, cls in detections.cpu().numpy():
        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        c = int(cls)
        label = f"{names[c] if isinstance(names, (list, tuple, dict)) else c} {float(conf):.2f}"
        color = colors(c, True)  # (r,g,b)
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            overlay, label, (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return overlay


# ----------------------------------
# 1. F_Norm metrics (your definition)
# ----------------------------------
def compute_f_metrics(heatmap: np.ndarray, box_xyxy, img_shape):
    """
    Implements your F_Norm definition:

      - Normalize heatmap to [0,1].
      - Resize to original image size.
      - Split to inside vs outside predicted box.
      - Compute:
          * sum_in, sum_out
          * dens_in = sum_in / n_in
          * dens_out = sum_out / n_out
          * raw_ratio   = sum_in / sum_out
          * area_ratio  = dens_in / dens_out
          * F_norm      = dens_in / (dens_in + dens_out)

    Returns dict with:
        raw_ratio, area_ratio, f_norm, sum_in, sum_out, dens_in, dens_out.
    """
    H_img, W_img, _ = img_shape
    eps = 1e-8

    hm = heatmap.astype(np.float32)
    hm -= hm.min()
    if hm.max() > 0:
        hm /= hm.max()

    hm_resized = cv2.resize(hm, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(x1, W_img - 1))
    x2 = max(0, min(x2, W_img - 1))
    y1 = max(0, min(y1, H_img - 1))
    y2 = max(0, min(y2, H_img - 1))

    mask = np.zeros((H_img, W_img), dtype=np.float32)
    mask[y1:y2 + 1, x1:x2 + 1] = 1.0
    inv_mask = 1.0 - mask

    sum_in = float((hm_resized * mask).sum())
    sum_out = float((hm_resized * inv_mask).sum())

    n_in = float(mask.sum())
    n_out = float(inv_mask.sum())

    dens_in = sum_in / (n_in + eps)
    dens_out = sum_out / (n_out + eps)

    raw_ratio = sum_in / (sum_out + eps)
    area_ratio = dens_in / (dens_out + eps)
    f_norm = dens_in / (dens_in + dens_out + eps)

    return {
        "raw_ratio": raw_ratio,
        "area_ratio": area_ratio,
        "f_norm": f_norm,
        "sum_in": sum_in,
        "sum_out": sum_out,
        "dens_in": dens_in,
        "dens_out": dens_out,
    }


# ----------------------------------
# 2. Model + CAM loaders (cached)
# ----------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_meta(weights_path: str, base_img_size: int = 640):
    device_str = _select_mac_device_str()
    model = DetectMultiBackend(
        weights_path,
        device=device_str,
        dnn=False,
        fp16=False,
    )
    model.eval()

    # check image size against model stride (like your script)
    stride, names = model.stride, model.names
    imgsz = check_img_size([base_img_size, base_img_size], s=stride)

    # inner model for CAM
    inner_model = model.model if hasattr(model, "model") else model
    for p in inner_model.parameters():
        p.requires_grad_(True)

    target_layer = _auto_find_last_conv2d(inner_model)

    # Grad-CAM + EigenCAM
    gradcam = GradCAM(model=inner_model, target_layers=[target_layer])
    eigencam = EigenCAM(model=inner_model, target_layers=[target_layer])

    return model, inner_model, gradcam, eigencam, imgsz, stride, names


# ----------------------------------
# 3. Streamlit app
# ----------------------------------
def main():
    st.set_page_config(page_title="YOLOv5 Detector", layout="wide")
    st.title("YOLOv5 Detector")

    st.markdown(
        "This app performs YOLOv5 object detection with Grad-CAM and EigenCAM visualizations, "
  
    )

    # Sidebar
    st.sidebar.header("Settings")
    default_weights = str(ROOT /  "weights.pt")
    weights_path = st.sidebar.text_input("YOLOv5 weights (.pt)", value=default_weights)
    conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    iou_thres = st.sidebar.slider("IoU threshold", 0.0, 1.0, 0.45, 0.01)
    img_size = st.sidebar.selectbox("Base inference size", [640, 1024, 512], index=0)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    # Load image (PIL -> RGB -> BGR)
    pil_img = Image.open(uploaded_file).convert("RGB")
    im0_rgb = np.array(pil_img)
    im0_bgr = cv2.cvtColor(im0_rgb, cv2.COLOR_RGB2BGR)
    H0, W0, _ = im0_bgr.shape

    st.subheader("Original Image")
    st.image(im0_rgb, use_column_width=True)

    # Load model + CAM
    try:
        model, inner_model, gradcam, eigencam, imgsz, stride, names = load_model_and_meta(
            weights_path, base_img_size=img_size
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Preprocess for YOLO/CAM
    im = preprocess_bgr_for_yolo(im0_bgr, tuple(imgsz), stride, model.device)

    # YOLO detection
    st.subheader("Detection")
    with st.spinner("Running YOLOv5 inference..."):
        with torch.no_grad():
            raw_pred = model(im, augment=False, visualize=False)
        pred_list = non_max_suppression(raw_pred, conf_thres, iou_thres, classes=None, agnostic=False)

    det = pred_list[0]
    if det is None or len(det) == 0:
        st.warning("No detections found.")
        return

    # Scale boxes to original resolution
    det_scaled = det.clone()
    scale_boxes(im.shape[2:], det_scaled[:, :4], im0_bgr.shape)

    # Detection overlay
    det_overlay_rgb = draw_yolo_boxes_overlay(im0_rgb, det_scaled, names)
    st.image(det_overlay_rgb, caption="YOLOv5 Detection Overlay", use_column_width=True)

    # Pick top detection (by confidence) for F_Norm reference
    top_idx = det_scaled[:, 4].argmax()
    top_box = det_scaled[top_idx, :4].cpu().numpy()
    top_conf = float(det_scaled[top_idx, 4].cpu().item())
    top_cls = int(det_scaled[top_idx, 5].cpu().item())
    st.markdown(
        f"**Top detection** – class: `{top_cls}`, conf: `{top_conf:.3f}`, "
        f"box: `{[int(v) for v in top_box]}`"
    )

    # CAMs: per-class heatmaps like your script, then max over classes
    with st.spinner("Computing Grad-CAM and EigenCAM (class-targeted)..."):
        detected_classes = sorted({int(c) for c in det[:, 5].tolist()})

        rgb_base01 = _normalize_to_rgb01(im0_bgr)
        heatmaps_grad = []
        heatmaps_eig = []

        im_cam = im.clone().detach().requires_grad_(True)

        for cls_id in detected_classes:
            targets = [ClassScoreTarget(cls_id)]

            # Grad-CAM
            grayscale_cam_g = gradcam(input_tensor=im_cam, targets=targets)  # (1,Hc,Wc)
            heat_g = grayscale_cam_g[0]
            heat_g_resized = cv2.resize(
                heat_g, (W0, H0), interpolation=cv2.INTER_LINEAR
            )
            heatmaps_grad.append(heat_g_resized)

            # EigenCAM
            grayscale_cam_e = eigencam(input_tensor=im_cam, targets=targets)
            heat_e = grayscale_cam_e[0]
            heat_e_resized = cv2.resize(
                heat_e, (W0, H0), interpolation=cv2.INTER_LINEAR
            )
            heatmaps_eig.append(heat_e_resized)

        if heatmaps_grad:
            combined_grad = np.maximum.reduce(heatmaps_grad)
            combined_grad = np.clip(combined_grad, 0.0, 1.0).astype(np.float32)
        else:
            combined_grad = np.zeros((H0, W0), dtype=np.float32)

        if heatmaps_eig:
            combined_eig = np.maximum.reduce(heatmaps_eig)
            combined_eig = np.clip(combined_eig, 0.0, 1.0).astype(np.float32)
        else:
            combined_eig = np.zeros((H0, W0), dtype=np.float32)

    # Overlays (same show_cam_on_image style as your script)
    grad_overlay_rgb = show_cam_on_image(
        rgb_base01, combined_grad, use_rgb=True, image_weight=1.0 - HEATMAP_ALPHA
    )
    eig_overlay_rgb = show_cam_on_image(
        rgb_base01, combined_eig, use_rgb=True, image_weight=1.0 - HEATMAP_ALPHA
    )

    # Optionally draw boxes on top of CAM overlays
    grad_overlay_rgb = draw_yolo_boxes_overlay(grad_overlay_rgb, det_scaled, names)
    eig_overlay_rgb = draw_yolo_boxes_overlay(eig_overlay_rgb, det_scaled, names)

    col1, col2 = st.columns(2)
    with col1:
        st.image(grad_overlay_rgb, caption="Grad-CAM Overlay", use_column_width=True)
    with col2:
        st.image(eig_overlay_rgb, caption="EigenCAM Overlay", use_column_width=True)

    # F_Norm metrics (combined maps vs top box)
    metrics_grad = compute_f_metrics(combined_grad, top_box, im0_rgb.shape)
    metrics_eig = compute_f_metrics(combined_eig, top_box, im0_rgb.shape)

    st.write("---")
    st.subheader("Q-Metric")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Grad-CAM")
        st.markdown(f"- **Raw ratio (sum_in / sum_out):** `{metrics_grad['raw_ratio']:.4f}`")
        st.markdown(f"- **Area-corrected ratio (dens_in / dens_out):** `{metrics_grad['area_ratio']:.4f}`")
        st.markdown(f"- **F_norm (dens_in / (dens_in + dens_out))**: `{metrics_grad['f_norm']:.4f}`")

    with c2:
        st.markdown("#### EigenCAM ")
        st.markdown(f"- **Raw ratio (sum_in / sum_out):** `{metrics_eig['raw_ratio']:.4f}`")
        st.markdown(f"- **Area-corrected ratio (dens_in / dens_out):** `{metrics_eig['area_ratio']:.4f}`")
        st.markdown(f"- **F_norm (dens_in / (dens_in + dens_out))**: `{metrics_eig['f_norm']:.4f}`")


if __name__ == "__main__":
    # Reduce OpenBLAS/OpenMP thread spam a bit (optional, like your script)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass

    main()
