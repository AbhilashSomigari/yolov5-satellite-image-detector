# YOLOv5 Satellite Image Detector

A [YOLOv5](https://github.com/ultralytics/yolov5) fork used to train/evaluate an object detector on
aerial/satellite-style imagery, and to study **why** the detector fires where it does — via Grad-CAM,
EigenCAM, Integrated Gradients, and a custom localization-faithfulness metric.

## Repository layout

```
.
├── train.py, val.py, detect.py, export.py, benchmarks.py, hubconf.py   # stock YOLOv5 CLI (unmodified)
├── models/, utils/, classify/, segment/                                # stock YOLOv5 library code
├── data/
│   ├── images/, images2/          # raw input frames (tracked)
│   └── labels/                    # YOLO-format ground-truth boxes (tracked)
│       # generated artifacts (gradcam/, eigencam*/, ig_numpy/, outputs/, pred_boxes/, qmetric_out/)
│       # are gitignored — see "Reproducing generated data" below
├── xai/                            # explainability pipeline (this project's main contribution)
│   ├── batch_infer.py              # batch YOLO inference -> annotated images + pred boxes + metrics
│   ├── cams/
│   │   ├── grad_cam.py             # class-targeted Grad-CAM overlays
│   │   ├── grad_pixels.py          # Grad-CAM + numeric top-K%-by-mass heatmap masks
│   │   ├── eigen_cam.py            # class-agnostic EigenCAM overlays
│   │   ├── eigen_pixels.py         # EigenCAM + numeric top-K%-by-mass heatmap masks
│   │   └── integrated_gradients.py # Captum Integrated Gradients attributions
│   ├── metrics/
│   │   ├── q_metric.py             # localization-curve AUC, single ground-truth object per image
│   │   └── gen_metric.py           # same metric, generalized to arbitrary heatmap sources
│   ├── apps/
│   │   ├── streamlit_app.py        # interactive Streamlit UI (detection + Grad-CAM/EigenCAM + F_norm)
│   │   └── gradio_app.py           # equivalent Gradio UI
│   ├── tools/
│   │   ├── sum_csv_column.py       # sum/average a metrics CSV column
│   │   └── strip_gradcam_suffix.py # filename cleanup utility
│   └── requirements-xai.txt        # extra deps needed only for the XAI pipeline
├── experiments/                    # archived, frozen configs from earlier detection runs (see below)
├── out_metric/, results/           # generated metric outputs (gitignored)
└── requirements.txt                # core YOLOv5 dependencies
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt              # core YOLOv5
pip install -r xai/requirements-xai.txt       # only if you're running the explainability pipeline
```

All custom scripts under `xai/` and `experiments/` are meant to be run **from the repo root**, e.g.:

```bash
python -u xai/batch_infer.py
python -u xai/cams/grad_cam.py
streamlit run xai/apps/streamlit_app.py
```

They locate `models/` and `utils/` by walking up from their own file location, so this works regardless
of your current shell directory as long as you invoke them with a path into this repo.

## Core YOLOv5 usage

Standard Ultralytics workflows are unmodified — see the [YOLOv5 docs](https://docs.ultralytics.com/yolov5)
for full detail. Quick examples:

```bash
python detect.py --weights yolov5s.pt --source data/images   # inference
python train.py  --data your_data.yaml --weights yolov5s.pt  # training
python val.py     --weights yolov5s.pt --data your_data.yaml  # validation
```

## Explainability (XAI) pipeline

The core research question: **does the detector's saliency actually land on the object it claims to
detect?** The pipeline:

1. **Generate attributions** — run one of `xai/cams/grad_cam.py`, `xai/cams/eigen_cam.py`,
   `xai/cams/grad_pixels.py` / `eigen_pixels.py` (also save numeric top-K%-by-mass masks), or
   `xai/cams/integrated_gradients.py` over `data/images/` to produce heatmaps in `data/gradcam/`,
   `data/eigencam*/`, or `data/ig_numpy/`.
2. **Get predictions** — `xai/batch_infer.py` runs the detector and writes YOLO-format boxes to
   `data/pred_boxes/`.
3. **Score faithfulness** — `xai/metrics/q_metric.py` (or `gen_metric.py` for arbitrary heatmap sources)
   compares each heatmap's top-K%-by-mass region against the ground-truth vs. predicted box, and reports
   a per-image localization-curve AUC (K ∈ {1, 5, 10, 50, 100}) to `results/` / `out_metric/`.
4. **Explore interactively** — `xai/apps/streamlit_app.py` or `xai/apps/gradio_app.py` run detection +
   Grad-CAM + EigenCAM + an F_norm faithfulness score on a single uploaded image.

### Reproducing generated data

`data/images/`, `data/images2/`, and `data/labels/` are tracked (raw inputs). Everything the pipeline
*produces* — `data/gradcam/`, `data/gradcam2/`, `data/grad_pixelvalues/`, `data/eigencam/`,
`data/eigencam_values/`, `data/ig_numpy/`, `data/outputs/`, `data/pred_boxes/`, `data/qmetric_out/`,
`out_metric/`, `results/` — is gitignored, since it's multi-gigabyte and fully reproducible by re-running
the scripts above against a weights file (`weights.pt`, not tracked either — supply your own).

## Experiments

`experiments/` holds three frozen `detect.py` variants from earlier detection runs on a different
machine, kept as a record of what was tried rather than as reusable scripts (their default
`--weights`/`--data` paths point at that machine's filesystem and won't resolve here):

| Script | Run | What it tested |
| --- | --- | --- |
| `detect_exp35_fullres2048.py` | exp35 | Full-resolution tiles, inference at 2048px |
| `detect_exp40_wavelet128.py` | exp40 | Wavelet-preprocessed 128px tiles |
| `detect_exp44_sr1024_test128.py` | exp44 | Super-resolution pipeline, trained at 1024px / tested at 128px |

Pass `--weights`, `--source`, and `--data` explicitly to reuse any of them against your own model/data.

## Credits

Built on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5), licensed AGPL-3.0 (see
[LICENSE](LICENSE)). Grad-CAM/EigenCAM via
[`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam); Integrated Gradients via
[Captum](https://captum.ai/).
