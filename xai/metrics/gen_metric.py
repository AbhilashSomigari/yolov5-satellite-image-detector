#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-metric for arbitrary heatmaps (Grad-CAM, EigenCAM, IG, etc.)
Now includes tqdm progress bar.

Usage (from the repo root):
python xai/metrics/gen_metric.py \
  --images_dir data/images \
  --pred_dir data/pred_boxes \
  --heatmaps_dir data/heatmaps \
  --out_dir data/qmetric_out \
  --gt_dir data/labels
"""

from __future__ import annotations
from pathlib import Path
import argparse, csv, math
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm  # 👈 added for progress bar

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_TOPK_LIST = [1, 5, 10, 50, 100]

# ----------------------------- Utility -----------------------------
def log(m): print(f"[INFO] {m}", flush=True)
def warn(m): print(f"[WARN] {m}", flush=True)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_image_size(p: Path) -> Tuple[int, int]:
    with Image.open(p) as im:
        return im.width, im.height

def yolo_to_xyxy(xywh, W, H):
    x,y,w,h = xywh
    cx,cy = x*W, y*H
    bw,bh = w*W, h*H
    x1 = int(round(cx - bw/2)); y1 = int(round(cy - bh/2))
    x2 = int(round(cx + bw/2)); y2 = int(round(cy + bh/2))
    x1 = max(0, min(x1, W-1)); x2 = max(0, min(x2, W-1))
    y1 = max(0, min(y1, H-1)); y2 = max(0, min(y2, H-1))
    return (x1,y1,x2,y2)

def box_area(b): x1,y1,x2,y2 = b; return max(0, x2-x1) * max(0, y2-y1)
def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1,iy1,ix2,iy2 = max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    iw,ih = max(0,ix2-ix1),max(0,iy2-iy1)
    inter = iw*ih
    den = box_area(a)+box_area(b)-inter
    return inter/den if den>0 else 0.0

def valid_xyxy(b):
    x1,y1,x2,y2=b
    return (x2 > x1) and (y2 > y1)

# ----------------------------- Load YOLO -----------------------------
def load_yolo(txt: Path, is_pred: bool):
    if not txt.exists(): return []
    rows=[]
    for line in txt.read_text().splitlines():
        line=line.strip()
        if not line: 
            continue
        parts = line.replace(",", " ").split()
        try:
            if is_pred:
                if len(parts) >= 6:
                    # Try cls conf x y w h
                    try:
                        cls=int(float(parts[0])); conf=float(parts[1])
                        x,y,w,h=map(float,parts[2:6])
                        rows.append({"cls":cls,"conf":conf,"xywh":(x,y,w,h)})
                    except Exception: pass
                    # Try cls x y w h conf
                    try:
                        cls=int(float(parts[0]))
                        x,y,w,h=map(float,parts[1:5]); conf=float(parts[5])
                        rows.append({"cls":cls,"conf":conf,"xywh":(x,y,w,h)})
                    except Exception: pass
                elif len(parts)==5:
                    cls=int(float(parts[0])); x,y,w,h=map(float,parts[1:5])
                    rows.append({"cls":cls,"conf":1.0,"xywh":(x,y,w,h)})
            else:
                if len(parts)>=5:
                    cls=int(float(parts[0])); x,y,w,h=map(float,parts[1:5])
                    rows.append({"cls":cls,"xywh":(x,y,w,h)})
        except Exception:
            continue
    return rows

def pick_boxes(preds,gts,W,H):
    preds_xy=[(p,yolo_to_xyxy(p["xywh"],W,H)) for p in preds]
    gts_xy=[(g,yolo_to_xyxy(g["xywh"],W,H)) for g in gts]
    preds_xy=[(p,px) for (p,px) in preds_xy if valid_xyxy(px)]
    gts_xy=[(g,gx) for (g,gx) in gts_xy if valid_xyxy(gx)]
    if preds_xy and gts_xy:
        best=(None,None,-1.0)
        for p,px in preds_xy:
            for g,gx in gts_xy:
                v=iou_xyxy(px,gx)
                if v>best[2]: best=(p,g,v)
        if best[0] is not None: return best
    if preds_xy:
        p=max((p for p,_ in preds_xy), key=lambda d:d.get("conf",0.0))
        return p,None,float("nan")
    if gts_xy:
        g=max((g for g,_ in gts_xy), key=lambda g: box_area(yolo_to_xyxy(g["xywh"],W,H)))
        return None,g,float("nan")
    return None,None,float("nan")

# ----------------------------- Heatmaps -----------------------------
def load_heatmap(heat_dir: Path, basename: str, size: Tuple[int,int]) -> Optional[np.ndarray]:
    W,H=size; arr=None
    npy=(heat_dir/basename).with_suffix(".npy")
    if npy.exists():
        try:
            arr=np.load(npy)
            if arr.ndim==3: arr=arr.max(axis=0)
            if arr.shape!=(H,W): arr=np.array(Image.fromarray(arr).resize((W,H),Image.NEAREST))
        except Exception: arr=None
    if arr is None:
        for ext in [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]:
            p=(heat_dir/basename).with_suffix(ext)
            if p.exists():
                im=Image.open(p).convert("L")
                if im.size!=(W,H): im=im.resize((W,H),Image.NEAREST)
                arr=np.asarray(im,dtype=np.float32)
                break
    if arr is None: return None
    arr=np.clip(arr,0,None)
    m=float(arr.max())
    return (arr/m if m>0 else np.zeros_like(arr,dtype=np.float32))

# ----------------------------- Top-K-by-mass -----------------------------
def monotone_topk_masks_by_mass(hm: np.ndarray, topk_list: List[int]) -> Dict[int, np.ndarray]:
    H,W=hm.shape; flat=hm.reshape(-1); tot=float(flat.sum())
    masks={}; 
    if tot<=0:
        for k in topk_list: masks[k]=np.ones((H,W),np.uint8) if k==100 else np.zeros((H,W),np.uint8)
        return masks
    order=np.argsort(-flat); csum=np.cumsum(flat[order])
    chosen=np.zeros_like(flat,np.uint8); prev_mass=0.0
    for k in sorted(topk_list):
        if k==100: masks[k]=np.ones((H,W),np.uint8); continue
        target=(k/100.0)*tot
        if prev_mass>=target:
            masks[k]=chosen.reshape(H,W).copy(); continue
        idx=np.searchsorted(csum,target,"left")
        current=int(chosen.sum())
        new_sel=order[current:idx+1]; chosen[new_sel]=1
        prev_mass=float(flat[chosen==1].sum())
        masks[k]=chosen.reshape(H,W).copy()
    return masks

def region_masks(W,H,pred_xyxy,gt_xyxy):
    P=np.zeros((H,W),np.uint8); G=np.zeros((H,W),np.uint8)
    if pred_xyxy:
        x1,y1,x2,y2=pred_xyxy
        if x2>x1 and y2>y1: P[y1:y2,x1:x2]=1
    if gt_xyxy:
        x1,y1,x2,y2=gt_xyxy
        if x2>x1 and y2>y1: G[y1:y2,x1:x2]=1
    I=(P&G).astype(np.uint8)
    return {"inter":I,"pred_only":(P&(1-G)).astype(np.uint8),
            "gt_only":(G&(1-P)).astype(np.uint8),
            "union":(P|G).astype(np.uint8),
            "outside":(1-(P|G)).astype(np.uint8)}

def frac(hm,region,support):
    s=support.astype(bool); r=(region.astype(bool))&s
    den=float(hm[s].sum()); 
    return float(hm[r].sum())/den if den>0 else float("nan")

def trapezoid_auc(xs,ys):
    if len(xs)<2:return float("nan")
    area=0.0
    for i in range(len(xs)-1):
        dx=xs[i+1]-xs[i]; area+=0.5*dx*(ys[i]+ys[i+1])
    w=xs[-1]-xs[0]; return area/w if w>0 else float("nan")

def save_mask(path: Path,mask):
    ensure_dir(path.parent); Image.fromarray(mask*255).save(path)

def overlay_preview(image_path: Path, mask, out_path: Path, alpha=0.45):
    ensure_dir(out_path.parent)
    with Image.open(image_path).convert("RGB") as im:
        im=im.copy().convert("RGBA")
        a=(mask.astype(np.uint8)*int(255*alpha)).astype(np.uint8)
        red=Image.new("RGBA",im.size,(255,0,0,0)); red.putalpha(Image.fromarray(a,"L"))
        Image.alpha_composite(im,red).convert("RGB").save(out_path)

# ----------------------------- MAIN -----------------------------
def run(images_dir,pred_dir,heatmaps_dir,out_dir,gt_dir,topk):
    ensure_dir(out_dir)
    masks_root=out_dir/"masks"; previews_root=out_dir/"previews"
    ensure_dir(masks_root); ensure_dir(previews_root)
    imgs=sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    log(f"Images: {len(imgs)}")

    per_rows=[]; k_cols=[f"score_top{k}p" for k in topk]

    # ✅ tqdm progress bar
    for img in tqdm(imgs, desc="Processing images", unit="img", ncols=100):
        base=img.stem; W,H=read_image_size(img)
        preds=load_yolo((pred_dir/base).with_suffix(".txt"),True)
        gts=load_yolo((gt_dir/base).with_suffix(".txt"),False) if gt_dir else []
        if not preds and not gts: continue
        pred,gt,best_iou=pick_boxes(preds,gts,W,H)
        if pred is None and gt is None: continue
        pred_xy=yolo_to_xyxy(pred["xywh"],W,H) if pred else None
        gt_xy=yolo_to_xyxy(gt["xywh"],W,H) if gt else None
        hm=load_heatmap(heatmaps_dir,base,(W,H))
        if hm is None: continue

        kmasks=monotone_topk_masks_by_mass(hm,topk)
        regs=region_masks(W,H,pred_xy,gt_xy)
        topk_scores={}
        for k in topk:
            kmask=kmasks[k]; save_mask(masks_root/f"{k}p"/f"{base}.png",kmask)
            overlay_preview(img,kmask,previews_root/f"{k}p"/f"{base}.png")
            if (pred_xy and gt_xy):
                q=frac(hm, regs["union"] if best_iou==0.0 else regs["inter"], kmask)
            elif (pred_xy and not gt_xy):
                q=frac(hm, regs["pred_only"]|regs["inter"], kmask)
            elif (gt_xy and not pred_xy):
                q=frac(hm, regs["gt_only"]|regs["inter"], kmask)
            else:
                q=float("nan")
            topk_scores[f"score_top{k}p"]=q
        xs=[k/100.0 for k in topk]
        ys=[(0.0 if (v is None or (isinstance(v,float) and math.isnan(v))) else float(v))
             for v in (topk_scores[c] for c in k_cols)]
        auc=trapezoid_auc(xs,ys)
        per_rows.append({"rel_path":img.relative_to(images_dir).as_posix(),
                         "best_iou":best_iou,
                         "picked_pred_cls":pred["cls"] if pred else -1,
                         "picked_gt_cls":gt["cls"] if gt else -1,
                         **topk_scores,"AUC_topK":auc})

    if not per_rows: warn("No rows produced."); return
    per_csv=out_dir/"results_per_image.csv"
    fieldnames=["rel_path","best_iou","picked_pred_cls","picked_gt_cls",*k_cols,"AUC_topK"]
    with per_csv.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader()
        for r in per_rows: w.writerow(r)
    log(f"Wrote {per_csv}")

    vals={c:[] for c in k_cols+["AUC_topK"]}
    for r in per_rows:
        for c in vals:
            v=r[c]
            if v is not None and not (isinstance(v,float) and math.isnan(v)): vals[c].append(float(v))
    means={c:(float(np.mean(v)) if v else float("nan")) for c,v in vals.items()}
    summary=out_dir/"results_summary.txt"
    with summary.open("w") as f:
        f.write(f"N_images\t{len(per_rows)}\n")
        for c in k_cols: f.write(f"mean_{c}\t{means[c]:.6f}\n")
        f.write(f"mean_AUC_topK\t{means['AUC_topK']:.6f}\n")
    log(f"Wrote {summary}")

# ----------------------------- ENTRY -----------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--images_dir",required=True,type=Path)
    ap.add_argument("--pred_dir",required=True,type=Path)
    ap.add_argument("--heatmaps_dir",required=True,type=Path)
    ap.add_argument("--out_dir",required=True,type=Path)
    ap.add_argument("--gt_dir",type=Path,default=None)
    args=ap.parse_args()
    run(args.images_dir,args.pred_dir,args.heatmaps_dir,args.out_dir,args.gt_dir,DEFAULT_TOPK_LIST)
