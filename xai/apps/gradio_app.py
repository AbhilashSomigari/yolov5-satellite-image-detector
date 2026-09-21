# --- Imports and Setup ---
# xai/apps/gradio_app.py — run from the repo root: python xai/apps/gradio_app.py
import sys
from pathlib import Path

import torch, cv2, numpy as np
import gradio as gr
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Make the repo root importable (models/, utils/)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # repo root (xai/apps/ -> xai/ -> root)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Import YOLOv5 DetectMultiBackend and utilities
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# Load the YOLOv5 model with DetectMultiBackend
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # Use MPS on Apple Silicon if available, else CPU
WEIGHTS = str(ROOT / "weights.pt")
model = DetectMultiBackend(WEIGHTS, device=device, dnn=False, fp16=False)
model.eval()  # set model to evaluation mode
stride, class_names = model.stride, model.names

# Use the raw PyTorch model for Grad-CAM/EigenCAM (same as xai/cams/grad_cam.py):
# hooking/backprop needs the actual nn.Module, not the DetectMultiBackend wrapper.
inner_model = model.model if hasattr(model, "model") else model
for p in inner_model.parameters():
    p.requires_grad_(True)


def _auto_find_last_conv2d(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last


target_layer = _auto_find_last_conv2d(inner_model)


class TensorOutputWrapper(torch.nn.Module):
    """
    Wrap a model so it ALWAYS returns a single Tensor for Grad-CAM.
    (YOLOv5 returns a (predictions, features) tuple in eval mode; CAM's own
    target-auto-selection when targets=None expects a plain Tensor.)
    """
    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (list, tuple)) else out


cam_model = TensorOutputWrapper(inner_model).eval()

# Define a custom Target class for GradCAM to target a specific detection (class and bbox index)
class YoloDetectionTarget:
    def __init__(self, category_id, bbox_index):
        self.category_id = category_id
        self.bbox_index = bbox_index
    def __call__(self, model_output):
        # Use objectness score * class probability of the target as the target value
        return (model_output[0, self.bbox_index, 4] * model_output[0, self.bbox_index, 5 + self.category_id])

# --- Gradio Inference Function ---
def process_image(image):
    """
    Run object detection on the input image and produce:
    - Detection result image with bounding boxes
    - Grad-CAM heatmap overlay image
    - Eigen-CAM heatmap overlay image
    """
    # Convert input image to BGR (numpy array) for processing
    orig_img = np.array(image)  # input is PIL Image or numpy in RGB
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = orig_img.shape[:2]

    # Letterbox resize to model input size (e.g., 640x640) with padding
    img_resized, ratio, (dw, dh) = letterbox(orig_img, new_shape=640, stride=int(stride))
    # letterbox() returns half-padding (dw, dh); recover the actual border pixel counts
    # the same way it computed them internally.
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # Convert resized image to RGB and normalize to [0,1]
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_resized.astype(np.float32) / 255.0  # normalized float image for model and heatmap overlay
    # Prepare tensor
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)

    # Run inference (no grad needed for detection)
    with torch.no_grad():
        pred = model(tensor)  # forward pass
        if isinstance(pred, (list, tuple)):
            pred = pred[0]  # get tensor if model returned a tuple (DetectMultiBackend returns first element)
        # Run Non-Maximum Suppression to filter detections
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000)
    det = pred[0]  # detections for the single image

    # Initialize output images as copies of the original
    det_img = orig_img.copy()
    grad_cam_img = None
    eigen_cam_img = None

    # If there are detections, draw them and prepare Grad-CAM target
    if det is not None and len(det):
        # Rescale coordinates from padded image back to original image size
        det[:, :4] = scale_boxes(img_resized.shape[:2], det[:, :4], (orig_h, orig_w)).round()
        # Draw bounding boxes and labels on detection image
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            cls = int(cls)
            label = f"{class_names[cls] if cls < len(class_names) else cls}:{conf:.2f}"
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(det_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
        # Determine the highest confidence detection for Grad-CAM targeting
        det_cpu = det.cpu().numpy()
        best_idx = np.argmax(det_cpu[:, 4])  # index of detection with highest confidence
        best_cls_id = int(det_cpu[best_idx, 5])
        # Setup Grad-CAM with target
        targets = [YoloDetectionTarget(category_id=best_cls_id, bbox_index=best_idx)]
        cam = GradCAM(model=cam_model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]  # shape: (H_pad, W_pad)
        # Remove padding from CAM and upscale to original image size
        cam_nopad = grayscale_cam[top: grayscale_cam.shape[0]-bottom, left: grayscale_cam.shape[1]-right] if (grayscale_cam is not None) else grayscale_cam
        cam_nopad = cam_nopad if cam_nopad.size else grayscale_cam  # fallback if no padding
        cam_orig_resized = cv2.resize(cam_nopad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        # Overlay Grad-CAM heatmap on the original image (in RGB)
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        orig_img_rgb_norm = orig_img_rgb.astype(np.float32) / 255.0
        grad_cam_img = show_cam_on_image(orig_img_rgb_norm, cam_orig_resized, use_rgb=True)
    else:
        # If no detections, just put a message on the original image
        cv2.putText(det_img, "No objects detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    # Compute Eigen-CAM (gradient-free) on the same target layer (no specific target needed)
    cam_eigen = EigenCAM(model=cam_model, target_layers=[target_layer])
    grayscale_cam_eigen = cam_eigen(input_tensor=tensor, targets=None)[0]  # shape: (H_pad, W_pad)
    # Remove padding and resize EigenCAM heatmap to original size
    cam_eigen_nopad = grayscale_cam_eigen[top: grayscale_cam_eigen.shape[0]-bottom, left: grayscale_cam_eigen.shape[1]-right] if grayscale_cam_eigen is not None else grayscale_cam_eigen
    cam_eigen_nopad = cam_eigen_nopad if cam_eigen_nopad.size else grayscale_cam_eigen
    cam_eigen_resized = cv2.resize(cam_eigen_nopad, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    # Overlay Eigen-CAM heatmap on original image
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img_rgb_norm = orig_img_rgb.astype(np.float32) / 255.0
    eigen_cam_img = show_cam_on_image(orig_img_rgb_norm, cam_eigen_resized, use_rgb=True)

    # Convert detection image to RGB for output
    det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
    return det_img_rgb, (grad_cam_img if grad_cam_img is not None else orig_img_rgb), eigen_cam_img

# --- Gradio Interface ---
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil"),  # user uploads an image (we convert to PIL for consistent handling)
    outputs=[
        gr.Image(label="Detections"),
        gr.Image(label="Grad-CAM"),
        gr.Image(label="Eigen-CAM")
    ],
    title="YOLOv5 Detection with Grad-CAM and Eigen-CAM",
    description="Upload an image to see YOLOv5 object detection results with Grad-CAM and Eigen-CAM visualization overlays."
)

# Launch the Gradio app (if running as a script, otherwise use demo.launch() in interactive environments)
if __name__ == "__main__":
    demo.launch()
