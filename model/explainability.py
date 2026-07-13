"""
CV Module — Explainable AI overlay for the Chest X-ray Deep Learning model.

Real Grad-CAM needs gradients flowing back through actual convolutional
layers of a trained CNN (TensorFlow/PyTorch). Since Phase 2's DL model is
a conv-style feature extractor + MLPClassifier (no framework available,
no internet to install one), a real Grad-CAM isn't mathematically
applicable here. Instead this module uses **Occlusion Sensitivity
Mapping** — a classic, model-agnostic CV explainability technique that
works with ANY classifier (not just CNNs):

  1. Slide a small opaque patch across the image.
  2. At each position, re-run the FULL pipeline (conv feature extraction
     + MLP) on the occluded image and measure how much the predicted
     "Pneumonia" probability drops compared to the unoccluded image.
  3. A big probability drop means that patch was important for the
     model's decision -> high heatmap value there.
  4. Colorize the resulting sensitivity map and overlay it on the
     original X-ray, exactly like a Grad-CAM heatmap is presented.

This produces a genuinely faithful "why did the model decide this"
visualization of our own Phase-2 model, rather than a generic edge map.
"""
import numpy as np
from PIL import Image
import matplotlib.cm as cm

def occlusion_heatmap(img_uint8, model, scaler, feature_fn, target_class=1,
                       patch=8, stride=8, baseline_value=None):
    h, w = img_uint8.shape
    if baseline_value is None:
        baseline_value = float(np.median(img_uint8))

    base_feat = feature_fn(img_uint8).reshape(1, -1)
    base_prob = model.predict_proba(scaler.transform(base_feat))[0, target_class]

    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            occluded = img_uint8.copy()
            y2, x2 = min(y + patch, h), min(x + patch, w)
            occluded[y:y2, x:x2] = baseline_value
            feat = feature_fn(occluded).reshape(1, -1)
            prob = model.predict_proba(scaler.transform(feat))[0, target_class]
            drop = max(0.0, base_prob - prob)   # importance = how much occluding it hurt confidence
            heat[y:y2, x:x2] += drop
            counts[y:y2, x:x2] += 1

    counts[counts == 0] = 1
    heat = heat / counts
    if heat.max() > 1e-8:
        heat = heat / heat.max()
    return heat, base_prob

def overlay_heatmap(img_uint8, heat, alpha=0.45):
    """Colorize heatmap (jet colormap) and alpha-blend over the grayscale image."""
    base_rgb = np.stack([img_uint8]*3, axis=-1).astype(np.float32)
    colored = (cm.jet(heat)[:, :, :3] * 255).astype(np.float32)  # HxWx3
    blended = (1 - alpha) * base_rgb + alpha * colored
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def hottest_region_bbox(heat, threshold=0.65):
    """Bounding box around the most influential (hottest) region, for a highlight box."""
    ys, xs = np.where(heat >= threshold)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
