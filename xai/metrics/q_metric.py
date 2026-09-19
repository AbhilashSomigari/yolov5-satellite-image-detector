#!/usr/bin/env python3
"""
Quantifying metric with Localization Curve (single-object)

Supports:
- Heatmaps: .npy (H,W) or image (.png/.jpg/.bmp/.tif). Auto-normalized to [0,1].
- Top-K masks: optional dir with <image_id>_top{K}.(npy|png). If missing, derived from heatmap by top-K *mass*.
- Boxes:
  * --gt_boxes and --pred_boxes may be a single file (legacy: "image_id x1 y1 x2 y2 ...")
    OR a directory of per-image .txt files.
  * Per-image .txt formats accepted:
      A) Plain pixels:            x1 y1 x2 y2 [conf]
      B) Pixels with class:       cls x1 y1 x2 y2 [conf]
      C) YOLO normalized (rel):   [cls] cx cy w h [conf]  where 0..1, converted to pixels by (W,H)
  * If multiple preds in a file, pick the highest confidence; if none, use first line.
  * If multiple GTs, use first line (single-object setting).

Metric (per K in {1,5,10,50,100}):
  Define regions using GT (G) and Prediction (P):
    I = G ∩ P
    P_only = P \ G
    GT_only = G \ P
    O = outside (¬G ∪ ¬P)
  For a Top-K mask Mk, score is:
    Score_K = ( wI*mass(I) + wP*mass(P_only) + wGT*mass(GT_only) + wO*mass(O) ) / total_mass
  where mass(R) = sum( heatmap * Mk * R ), total_mass = sum(heatmap * Mk).
  Localization curve = K vs Score_K; AUC via trapezoidal rule (K treated as x).

Outputs:
  - CSV with per-image scores at each K and AUC
  - Printed dataset means

Run (from the repo root):
  python3 xai/metrics/q_metric.py \
    --heatmaps_dir "data/gradcam" \
    --masks_dir    "data/grad_pixelvalues" \
    --gt_boxes     "data/labels" \
    --pred_boxes   "data/pred_boxes" \
    --out_csv      "results/quant_metric.csv"
"""

from pathlib import Path
import argparse, csv, sys, warnings
import numpy as np
import cv2

# ---------- Weights (you can edit) ----------
W_I       = 1.00   # reward on intersection (correct & faithful)
W_P_ONLY  = 0.75   # mild reward: context inside prediction but outside GT
W_GT_ONLY = 0.50   # small reward: focuses on GT even if prediction missed
W_O       = 0.00   # no reward; set negative to penalize spillover

K_LIST = [1, 5, 10, 50, 100]  # percent

# -------------- Utilities --------------

def _is_number(s):
    try:
        float(s); return True
    except Exception:
        return False

def _as_int_box(x1, y1, x2, y2):
    x1 = int(round(float(x1))); y1 = int(round(float(y1)))
    x2 = int(round(float(x2))); y2 = int(round(float(y2)))
    return (x1, y1, x2, y2)

def _xywh_rel_to_xyxy_px(cx, cy, w, h, W, H):
    # YOLO rel cx,cy,w,h -> pixel x1,y1,x2,y2 (inclusive-ish)
    cx = float(cx) * W
    cy = float(cy) * H
    w  = float(w)  * W
    h  = float(h)  * H
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return _as_int_box(x1, y1, x2, y2)

def clamp_box(box, H, W):
    x1,y1,x2,y2 = box
    x1 = max(0, min(W-1, x1))
    y1 = max(0, min(H-1, y1))
    x2 = max(0, min(W-1, x2))
    y2 = max(0, min(H-1, y2))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return (x1,y1,x2,y2)

def box_mask(box, H, W):
    x1,y1,x2,y2 = clamp_box(box, H, W)
    m = np.zeros((H, W), dtype=bool)
    m[y1:y2+1, x1:x2+1] = True
    return m

def safe_sum(arr):
    s = float(arr.sum())
    return s if np.isfinite(s) else 0.0

# -------------- Heatmap / Mask I/O --------------

def load_heatmap(path):
    """
    Return float32 HxW in [0,1].
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        h = np.load(p)
        if h.ndim == 3:
            h = h.squeeze()
        h = h.astype(np.float32)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read heatmap image: {p}")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img.astype(np.float32)
    h_min, h_max = float(h.min()), float(h.max())
    if h_max > h_min:
        h = (h - h_min) / (h_max - h_min)
    else:
        h = np.zeros_like(h, dtype=np.float32)
    return h

def load_mask(path, H, W):
    """
    Return bool mask or None if not found. Resizes to (H,W) if needed.
    """
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() == ".npy":
        m = np.load(p)
        if m.ndim == 3:
            m = m.squeeze()
        m = (m > 0).astype(np.uint8)
    else:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = (img > 0).astype(np.uint8)
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return m.astype(bool)

def derive_topk_mask_from_heatmap(heatmap, k_percent):
    """
    Build top-K% *mass* mask: smallest threshold t s.t. sum(h[h>=t]) >= K% total mass.
    """
    h = heatmap.astype(np.float32).ravel()
    total = float(h.sum())
    if total <= 0:
        return np.zeros_like(heatmap, dtype=bool)
    idx = np.argsort(-h)
    h_sorted = h[idx]
    cumsum = np.cumsum(h_sorted)
    target = (k_percent / 100.0) * total
    j = np.searchsorted(cumsum, target, side="left")
    thr = h_sorted[-1] if j >= h_sorted.size else h_sorted[j]
    return (heatmap >= thr)

# -------------- Parsing box lines --------------

def _parse_box_line(line, W, H):
    """
    Parse a single line into (x1,y1,x2,y2, conf) in pixels.

    Accepts:
      - plain pixels:            x1 y1 x2 y2 [conf]
      - pixels with class:       cls x1 y1 x2 y2 [conf]
      - YOLO normalized (rel):   [cls] cx cy w h [conf]
      - YOLOv5 pred txt:         cls conf cx cy w h  (conf is 2nd token)

    Returns (box, conf_or_None) or (None, None).
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return (None, None)
    toks = s.replace(",", " ").split()
    if len(toks) < 4:
        return (None, None)

    def is_rel(vals):
        try:
            vs = [float(v) for v in vals]
            return all(0.0 <= v <= 1.0 for v in vs)
        except Exception:
            return False

    # Case 1: x1 y1 x2 y2 [conf] (pixels)
    if len(toks) >= 4 and all(_is_number(t) for t in toks[:4]) and not is_rel(toks[:4]):
        x1,y1,x2,y2 = toks[:4]
        conf = float(toks[4]) if len(toks) >= 5 and _is_number(toks[4]) else None
        return (_as_int_box(x1,y1,x2,y2), conf)

    # Case 2: cls x1 y1 x2 y2 [conf] (pixels with class)
    if len(toks) >= 5 and all(_is_number(t) for t in toks[1:5]) and not is_rel(toks[1:5]):
        x1,y1,x2,y2 = toks[1:5]
        conf = float(toks[5]) if len(toks) >= 6 and _is_number(toks[5]) else None
        return (_as_int_box(x1,y1,x2,y2), conf)

    # Case 3: YOLO normalized (various orders)

    # 3a) YOLOv5 predictions: cls conf cx cy w h  (conf in 2nd column)
    if len(toks) >= 6 and _is_number(toks[0]) and _is_number(toks[1]) and is_rel(toks[2:6]):
        cx,cy,w,h = toks[2:6]
        conf = float(toks[1])
        return (_xywh_rel_to_xyxy_px(cx,cy,w,h,W,H), conf)

    # 3b) cls cx cy w h [conf]  (conf at the end or absent)
    if len(toks) >= 5 and is_rel(toks[1:5]):
        cx,cy,w,h = toks[1:5]
        conf = float(toks[5]) if len(toks) >= 6 and _is_number(toks[5]) else None
        return (_xywh_rel_to_xyxy_px(cx,cy,w,h,W,H), conf)

    # 3c) cx cy w h [conf]  (no class)
    if len(toks) >= 4 and is_rel(toks[:4]):
        cx,cy,w,h = toks[:4]
        conf = float(toks[4]) if len(toks) >= 5 and _is_number(toks[4]) else None
        return (_xywh_rel_to_xyxy_px(cx,cy,w,h,W,H), conf)

    return (None, None)

    """
    Parse a single line into (x1,y1,x2,y2, conf) in pixels.
    Accepts:
      - plain pixels:            x1 y1 x2 y2 [conf]
      - pixels with class:       cls x1 y1 x2 y2 [conf]
      - YOLO normalized (rel):   [cls] cx cy w h [conf]  (0..1)
    Return (box, conf_or_None) or (None, None).
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return (None, None)
    toks = s.replace(",", " ").split()
    if len(toks) < 4:
        return (None, None)

    def is_rel(vals):
        try:
            vs = [float(v) for v in vals]
            return all(0.0 <= v <= 1.0 for v in vs)
        except Exception:
            return False

    # Case 1: x1 y1 x2 y2 [conf] (not all 0..1)
    if len(toks) >= 4 and all(_is_number(t) for t in toks[:4]) and not is_rel(toks[:4]):
        x1,y1,x2,y2 = toks[:4]
        conf = float(toks[4]) if len(toks) >= 5 and _is_number(toks[4]) else None
        return (_as_int_box(x1,y1,x2,y2), conf)

    # Case 2: cls x1 y1 x2 y2 [conf] (not all 0..1)
    if len(toks) >= 5 and all(_is_number(t) for t in toks[1:5]) and not is_rel(toks[1:5]):
        x1,y1,x2,y2 = toks[1:5]
        conf = float(toks[5]) if len(toks) >= 6 and _is_number(toks[5]) else None
        return (_as_int_box(x1,y1,x2,y2), conf)

    # Case 3: YOLO normalized
    # 3a) cx cy w h [conf]
    if len(toks) >= 4 and is_rel(toks[:4]):
        cx,cy,w,h = toks[:4]
        conf = float(toks[4]) if len(toks) >= 5 and _is_number(toks[4]) else None
        return (_xywh_rel_to_xyxy_px(cx,cy,w,h,W,H), conf)

    # 3b) cls cx cy w h [conf]
    if len(toks) >= 5 and is_rel(toks[1:5]):
        cx,cy,w,h = toks[1:5]
        conf = float(toks[5]) if len(toks) >= 6 and _is_number(toks[5]) else None
        return (_xywh_rel_to_xyxy_px(cx,cy,w,h,W,H), conf)

    return (None, None)

# -------------- Box readers (file or directory) --------------

def is_dir_or_file(p):
    p = Path(p)
    if p.is_dir():  return "dir"
    if p.is_file(): return "file"
    return "missing"

def read_box_file_to_dict(path):
    """
    Legacy single-file map: 'image_id x1 y1 x2 y2 [..]' -> {image_id: (x1,y1,x2,y2)}
    """
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = [t for chunk in line.replace(",", " ").split() for t in chunk.split()]
            if len(toks) < 5:
                warnings.warn(f"[{path}] Line {ln}: expected 'image_id x1 y1 x2 y2 ...', got: {line}")
                continue
            image_id = toks[0]
            try:
                x1 = float(toks[1]); y1 = float(toks[2]); x2 = float(toks[3]); y2 = float(toks[4])
            except Exception:
                warnings.warn(f"[{path}] Line {ln}: could not parse box coords: {line}")
                continue
            box = _as_int_box(x1,y1,x2,y2)
            if image_id in d:
                warnings.warn(f"[{path}] Multiple rows for {image_id}; keeping the first.")
                continue
            d[image_id] = box
    return d

def read_one_box_from_per_image_txt(txt_path, W, H, is_pred=False):
    """
    Read one .txt (possibly multi-line). For preds, pick highest conf if present.
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return None

    best = None
    best_conf = -1.0
    first_box = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            box, conf = _parse_box_line(line, W, H)
            if box is None:
                continue
            if first_box is None:
                first_box = box
            if is_pred:
                c = float(conf) if conf is not None else 0.0
                if c > best_conf:
                    best_conf = c
                    best = box

    return (best if is_pred and best is not None else first_box)

def _candidate_ids(image_id: str):
    """
    Generate likely label IDs from an image_id derived from heatmap filename.
    Handles suffixes like '.gradcam', '_gradcam', '-gradcam', and 'something.ext'.
    """
    cands = [image_id]

    # Strip after first dot (e.g., "foo.gradcam" -> "foo")
    if "." in image_id:
        cands.append(image_id.split(".")[0])

    # Remove common gradcam suffixes
    for suf in [".gradcam", "_gradcam", "-gradcam"]:
        if image_id.endswith(suf):
            cands.append(image_id[: -len(suf)])

    # Deduplicate
    out, seen = [], set()
    for c in cands:
        if c and c not in seen:
            out.append(c); seen.add(c)
    return out

def get_box_for_image(image_id, boxes_arg, W, H, is_pred=False, legacy_map=None):
    """
    Resolve a box for a given image_id from either:
      - a single file (legacy map), or
      - a directory of <id>.txt files (try multiple candidate ids).
    """
    mode = is_dir_or_file(boxes_arg)
    if mode == "file":
        if legacy_map is None:
            legacy_map = read_box_file_to_dict(boxes_arg)
        # try candidate ids
        for cid in _candidate_ids(image_id):
            if cid in legacy_map:
                return legacy_map[cid], legacy_map
        return None, legacy_map

    elif mode == "dir":
        boxdir = Path(boxes_arg)
        # Try exact candidates
        for cid in _candidate_ids(image_id):
            txt = boxdir / f"{cid}.txt"
            if txt.exists():
                return read_one_box_from_per_image_txt(txt, W, H, is_pred=is_pred), None
        # Fuzzy: prefix/suffix containment
        for txt in boxdir.glob("*.txt"):
            stem = txt.stem
            if stem in image_id or image_id in stem:
                box = read_one_box_from_per_image_txt(txt, W, H, is_pred=is_pred)
                if box is not None:
                    return box, None
        return None, None

    else:
        warnings.warn(f"Boxes path not found: {boxes_arg}")
        return None, legacy_map

# -------------- Scoring --------------

def score_for_k(heatmap, topk_mask, gt_box, pred_box):
    H, W = heatmap.shape
    M_GT  = box_mask(gt_box, H, W)
    M_P   = box_mask(pred_box, H, W)
    M_I        = M_GT & M_P
    M_P_only   = M_P & (~M_GT)
    M_GT_only  = M_GT & (~M_P)
    M_O        = ~(M_GT | M_P)

    Mk = topk_mask
    Hk = heatmap * Mk.astype(np.float32)

    mass_total   = safe_sum(Hk)
    if mass_total <= 0:
        return 0.0

    mass_I       = safe_sum(Hk * M_I)
    mass_P_only  = safe_sum(Hk * M_P_only)
    mass_GT_only = safe_sum(Hk * M_GT_only)
    mass_O       = safe_sum(Hk * M_O)

    num = (W_I * mass_I +
           W_P_ONLY * mass_P_only +
           W_GT_ONLY * mass_GT_only +
           W_O * mass_O)
    return num / mass_total

def trapezoid_auc(xs, ys):
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    return float(np.trapz(ys, xs)) / (xs[-1] - xs[0] + 1e-12)

# -------------- Main --------------

def main():
    ap = argparse.ArgumentParser(description="Quantifying metric with Localization Curve (single-object)")
    ap.add_argument("--heatmaps_dir", required=True, help="Dir with heatmaps (.npy or image files)")
    ap.add_argument("--masks_dir", required=False, default=None,
                    help="Dir with <image_id>_top{K}.(png|npy) masks; if missing, derived from heatmap")
    ap.add_argument("--gt_boxes", required=True, help="GT boxes: single file OR directory of per-image .txts")
    ap.add_argument("--pred_boxes", required=True, help="Pred boxes: single file OR directory of per-image .txts")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--images_list", required=False, default=None,
                    help="Optional list of image_ids; else inferred from heatmaps_dir filenames")
    args = ap.parse_args()

    heatmaps_dir = Path(args.heatmaps_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else None

    # Collect image_ids
    if args.images_list:
        with open(args.images_list, "r", encoding="utf-8") as f:
            image_ids = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    else:
        image_ids = sorted({p.stem for p in heatmaps_dir.glob("*")
                            if p.suffix.lower() in [".npy",".png",".jpg",".jpeg",".bmp",".tif",".tiff"]})
    if not image_ids:
        print("No images found. Provide --images_list or populate --heatmaps_dir.", file=sys.stderr)
        sys.exit(1)

    # Preload legacy maps if single files provided
    gt_legacy = read_box_file_to_dict(args.gt_boxes) if is_dir_or_file(args.gt_boxes) == "file" else None
    pr_legacy = read_box_file_to_dict(args.pred_boxes) if is_dir_or_file(args.pred_boxes) == "file" else None

    rows = []
    kxs = np.array(K_LIST, dtype=np.float64)
    k_names = [f"top{k}" for k in K_LIST]

    for image_id in image_ids:
        # find heatmap file
        hp = None
        npy = heatmaps_dir / f"{image_id}.npy"
        if npy.exists():
            hp = npy
        else:
            for ext in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
                cand = heatmaps_dir / f"{image_id}{ext}"
                if cand.exists():
                    hp = cand
                    break
        if hp is None:
            warnings.warn(f"[{image_id}] Missing heatmap; skipping.")
            continue

        heatmap = load_heatmap(hp)
        H, W = heatmap.shape

        # per-image boxes (file or dir) with robust ID matching
        gt_box, gt_legacy = get_box_for_image(image_id, args.gt_boxes, W, H, is_pred=False, legacy_map=gt_legacy)
        pr_box, pr_legacy = get_box_for_image(image_id, args.pred_boxes, W, H, is_pred=True,  legacy_map=pr_legacy)

        if gt_box is None:
            warnings.warn(f"[{image_id}] Missing GT box; skipping.")
            continue
        if pr_box is None:
            warnings.warn(f"[{image_id}] Missing Pred box; skipping.")
            continue

        scores = []
        for k in K_LIST:
            Mk = None
            if masks_dir is not None:
                base = f"{image_id}_top{k}"
                for ext in [".npy", ".png"]:
                    cand = masks_dir / f"{base}{ext}"
                    Mk = load_mask(cand, H, W)
                    if Mk is not None:
                        break
            if Mk is None:
                Mk = derive_topk_mask_from_heatmap(heatmap, k)
            s = score_for_k(heatmap, Mk, gt_box, pr_box)
            scores.append(s)

        auc = trapezoid_auc(kxs, np.array(scores, dtype=np.float64))
        rows.append({"image_id": image_id, **{k_names[i]: scores[i] for i in range(len(K_LIST))}, "AUC": auc})

    # Write CSV
    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id"] + k_names + ["AUC"]
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Print means
    if rows:
        means = {k: float(np.mean([r[k] for r in rows])) for k in k_names + ["AUC"]}
        print("Dataset means:")
        for k in k_names + ["AUC"]:
            print(f"  {k}: {means[k]:.4f}")
        print(f"Wrote: {outp}")
    else:
        print("No rows written (check inputs).")

if __name__ == "__main__":
    main()
