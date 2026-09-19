#!/usr/bin/env python3
# --- YOLOv5 batch infer + save + metrics (YOLO labels) ---
# Run from the repo root: python -u xai/batch_infer.py
from pathlib import Path
import sys
import os
import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---- (macOS stability knobs) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    cv2.setNumThreads(0)
except Exception:
    pass

# Make the repo root importable (models/, utils/)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # repo root (xai/ -> root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

# =====================
# CONFIGURE THESE (all relative to the repo root; override with env vars if needed)
# =====================
REPO_ROOT = ROOT

WEIGHTS         = str(REPO_ROOT / "weights.pt")
IMAGES_DIR      = str(REPO_ROOT / "data" / "images")
OUTPUTS_DIR     = str(REPO_ROOT / "data" / "outputs")       # annotated images root
PRED_BOXES_DIR  = str(REPO_ROOT / "data" / "pred_boxes")    # prediction .txt root (YOLO format)
LABELS_DIR      = str(REPO_ROOT / "data" / "labels")        # GT labels (YOLO txt) for metrics; set None to skip

DEVICE      = ""            # ''(auto) | 'mps' | 'cpu'
IMG_SIZE    = (1024, 1024)  # (w, h)
CONF_TH     = 0.25
IOU_TH      = 0.45
METRIC_IOU  = 0.50
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SAVE_TXT_PREDS = False      # optional extra per-image debug file near annotated image
# =====================

# ---------- Helpers ----------
def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def _letterbox_resize(im, new_size, stride):
    return letterbox(im, new_size, stride=stride, auto=False)[0]

def _yolo_txt_to_xyxy(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path or not txt_path.exists():
        return boxes
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip().lstrip("\ufeff")
                if not s or s.startswith("#"):
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                x1 = (cx - w / 2.0) * img_w
                y1 = (cy - h / 2.0) * img_h
                x2 = (cx + w / 2.0) * img_w
                y2 = (cy + h / 2.0) * img_h
                x1 = max(0.0, min(x1, img_w - 1)); y1 = max(0.0, min(y1, img_h - 1))
                x2 = max(0.0, min(x2, img_w - 1)); y2 = max(0.0, min(y2, img_h - 1))
                if x2 <= x1 or y2 <= y1: continue
                boxes.append([cls, x1, y1, x2, y2])
    except Exception:
        pass
    return boxes

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-9
    return inter / union

def _ap_from_pr(precisions, recalls):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p = mpre[mrec >= t].max() if np.any(mrec >= t) else 0.0
        ap += p / 101.0
    return ap

def _evaluate_map(pred_by_image, gt_by_image, num_classes, iou_thresh=0.5, conf_th=0.25):
    npos_cls = np.zeros(num_classes, dtype=int)
    for gts in gt_by_image.values():
        for g in gts:
            if 0 <= g["cls"] < num_classes:
                npos_cls[g["cls"]] += 1

    micro_tp = micro_fp = micro_fn = 0
    for img_id, preds in pred_by_image.items():
        gts = gt_by_image.get(img_id, [])
        matched = np.zeros(len(gts), dtype=bool)
        for p in sorted(preds, key=lambda x: -x["conf"]):
            if p["conf"] < conf_th: continue
            c = p["cls"]
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if matched[j] or g["cls"] != c: continue
                iou = _iou_xyxy(p["box"], g["box"])
                if iou > best_iou: best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                matched[best_j] = True; micro_tp += 1
            else:
                micro_fp += 1
        micro_fn += int((~matched).sum())

    precision_micro = micro_tp / max(micro_tp + micro_fp, 1e-9)
    recall_micro    = micro_tp / max(micro_tp + micro_fn, 1e-9)

    predictions_per_class = {c: [] for c in range(num_classes)}
    for img_id, preds in pred_by_image.items():
        for p in preds:
            c = p["cls"]
            if 0 <= c < num_classes:
                predictions_per_class[c].append((img_id, p["conf"], p["box"]))

    per_class_AP = {}
    for c in range(num_classes):
        preds_c = predictions_per_class[c]
        preds_c.sort(key=lambda x: -x[1])
        tp_flags = np.zeros(len(preds_c), dtype=int)
        fp_flags = np.zeros(len(preds_c), dtype=int)
        used_per_image = {img_id: np.zeros(len(gt_by_image.get(img_id, [])), dtype=bool) for img_id in gt_by_image}
        for i, (img_id, conf, p_box) in enumerate(preds_c):
            gts = gt_by_image.get(img_id, [])
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if used_per_image[img_id][j] or g["cls"] != c: continue
                iou = _iou_xyxy(p_box, g["box"])
                if iou > best_iou: best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                tp_flags[i] = 1; used_per_image[img_id][best_j] = True
            else:
                fp_flags[i] = 1
        tp_cum = np.cumsum(tp_flags); fp_cum = np.cumsum(fp_flags)
        denom = (tp_cum + fp_cum)
        precisions = (tp_cum / np.maximum(denom, 1e-9))
        recalls    = (tp_cum / float(npos_cls[c])) if npos_cls[c] > 0 else np.zeros_like(precisions)
        per_class_AP[c] = float(_ap_from_pr(precisions, recalls)) if (npos_cls[c] > 0 and len(precisions) > 0) else 0.0

    mAP = float(np.mean(list(per_class_AP.values()))) if per_class_AP else 0.0
    return {"mAP@0.5": mAP, "per_class_AP": {int(k): float(v) for k, v in per_class_AP.items()},
            "precision": float(precision_micro), "recall": float(recall_micro),
            "tp": int(micro_tp), "fp": int(micro_fp), "fn": int(micro_fn)}

def _pick_device(user_choice: str):
    if user_choice: return user_choice
    return "mps" if torch.backends.mps.is_available() else "cpu"

def run_folder(
    images_root: str,
    output_root: str,
    labels_root: str = None,
    conf_thres: float = CONF_TH,
    iou_thres: float = IOU_TH,
    metric_iou: float = METRIC_IOU,
    save_txt_preds: bool = SAVE_TXT_PREDS,
):
    images_root  = Path(images_root)
    output_root  = Path(output_root)            # annotated images root
    labels_root  = Path(labels_root) if labels_root else None
    pred_txt_root = Path(PRED_BOXES_DIR)        # predicted YOLO txt root

    (output_root).mkdir(parents=True, exist_ok=True)
    pred_txt_root.mkdir(parents=True, exist_ok=True)

    pred_by_image, gt_by_image = {}, {}

    img_paths = [p for p in images_root.rglob("*") if p.is_file() and _is_image(p)]
    if not img_paths:
        print(f"[FATAL] No images found under: {images_root}")
        return None, None

    device = select_device(_pick_device(DEVICE))
    print(f"[INFO] device: {device} (requested='{DEVICE or 'auto'}')")
    print("[INFO] loading weights:", WEIGHTS)
    model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, fp16=False)
    stride, names, _ = model.stride, model.names, model.pt
    imgsz = check_img_size([IMG_SIZE[0], IMG_SIZE[1]], s=stride)

    for img_path in tqdm(img_paths, desc="Detecting"):
        # --- paths & I/O ---
        rel_img = img_path.relative_to(images_root)
        out_annot_path = (output_root / rel_img).with_suffix(".png")
        out_annot_path.parent.mkdir(parents=True, exist_ok=True)

        im0 = cv2.imread(str(img_path))
        if im0 is None:
            print(f"[WARN] could not read {img_path}")
            continue

        # preprocess
        img = _letterbox_resize(im0, imgsz, stride).transpose(2, 0, 1)  # HWC->CHW
        img = torch.from_numpy(img).to(model.device).float() / 255.0
        if img.ndim == 3: img = img[None]

        # inference
        with torch.no_grad():
            pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        annotator = Annotator(im0.copy(), line_width=3, example=str(names))
        preds_for_metrics = []

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    label = f"{names[c] if isinstance(names, (list, tuple, dict)) else c} {float(conf):.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    preds_for_metrics.append({"cls": c, "conf": float(conf), "box": [x1, y1, x2, y2]})

        # save annotated image
        cv2.imwrite(str(out_annot_path), annotator.result())

        # ===== SAVE PREDICTION .TXT IN A SEPARATE FOLDER, WITH THE SAME FILENAME AS LABELS =====
        # Build the relative txt path to MATCH labels naming (same subfolders & basename).
        if labels_root is not None:
            # expected labels relative path (e.g., images/foo/bar/img.jpg -> labels/foo/bar/img.txt)
            rel_label_txt = rel_img.with_suffix(".txt")  # basename same as label
            pred_txt_path = (pred_txt_root / rel_label_txt)
        else:
            # fallback: mirror images structure if labels_root not provided
            pred_txt_path = (pred_txt_root / rel_img).with_suffix(".txt")

        pred_txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Write YOLO-format predictions: cls cx cy w h conf  (cx,cy,w,h normalized to [0,1])
        H, W = im0.shape[:2]
        with open(pred_txt_path, "w", encoding="utf-8") as f:
            for p in preds_for_metrics:
                x1, y1, x2, y2 = p["box"]
                w = (x2 - x1); h = (y2 - y1)
                cx = x1 + w / 2.0; cy = y1 + h / 2.0
                cxn, cyn = cx / W, cy / H
                wn, hn   = w / W,  h / H
                f.write(f"{p['cls']} {cxn:.6f} {cyn:.6f} {wn:.6f} {hn:.6f} {p['conf']:.6f}\n")

        # Optional extra debug file next to the annotated image (pixel coords)
        if save_txt_preds:
            dbg_txt = out_annot_path.with_suffix(".pred.txt")
            with open(dbg_txt, "w", encoding="utf-8") as f:
                for p in preds_for_metrics:
                    x1, y1, x2, y2 = p["box"]
                    f.write(f"{p['cls']} {p['conf']:.6f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

        # store for metrics
        img_id = str(rel_img).replace("\\", "/")
        pred_by_image[img_id] = preds_for_metrics

        # GT for metrics
        if labels_root is not None:
            label_path = (labels_root / rel_img).with_suffix(".txt")
            gts = []
            if label_path.exists():
                gt_boxes = _yolo_txt_to_xyxy(label_path, W, H)
                for cls_id, gx1, gy1, gx2, gy2 in gt_boxes:
                    gts.append({"cls": int(cls_id), "box": [float(gx1), float(gy1), float(gx2), float(gy2)]})
            gt_by_image[img_id] = gts

    # Metrics
    metrics = None
    if labels_root is not None:
        if isinstance(names, (list, tuple)):
            num_classes = len(names)
        else:
            found = [p["cls"] for arr in pred_by_image.values() for p in arr]
            num_classes = int(max(found, default=0) + 1)
        metrics = _evaluate_map(
            pred_by_image, gt_by_image, num_classes=num_classes, iou_thresh=metric_iou, conf_th=conf_thres
        )

        print("\n==== Detection Metrics ====")
        print(f"mAP@0.5 : {metrics['mAP@0.5']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        if isinstance(names, (list, tuple)):
            for c, ap in metrics["per_class_AP"].items():
                label = names[c] if 0 <= c < len(names) else str(c)
                print(f"AP@0.5[{label}]: {ap:.4f}")
        else:
            for c, ap in metrics["per_class_AP"].items():
                print(f"AP@0.5[class {c}]: {ap:.4f}")

    return metrics, pred_by_image

def main():
    metrics, _ = run_folder(
        images_root=IMAGES_DIR,
        output_root=OUTPUTS_DIR,     # annotated images saved here
        labels_root=LABELS_DIR,      # set to None to skip metrics
        conf_thres=CONF_TH,
        iou_thres=IOU_TH,
        metric_iou=METRIC_IOU,
        save_txt_preds=SAVE_TXT_PREDS,
    )
    print("\n[DONE] all images processed.")
    if metrics:
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    print("[BOOT] batch_infer_mac.py")
    import numpy as _np, torch as _torch, cv2 as _cv2
    print(f"[VERSIONS] python={sys.version.split()[0]} torch={_torch.__version__} numpy={_np.__version__} cv2={_cv2.__version__}")
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
