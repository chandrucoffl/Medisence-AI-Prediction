"""
Synthetic Chest X-ray generator (Normal vs Pneumonia).

NOTE: Real chest X-ray datasets (e.g. Kaggle "Chest X-Ray Pneumonia") could
not be downloaded in this environment (no internet access). To keep the DL
pipeline fully functional end-to-end, this script procedurally generates
X-ray-like grayscale images:
  - Normal: smooth lung fields with faint rib shadows, no opacities.
  - Pneumonia: same base anatomy + randomly placed cloudy "consolidation"
    blobs (bright, diffuse patches) in one or both lung fields, which is
    the classic radiographic sign of pneumonia.
This lets the CNN-style feature extractor + deep neural network train and
be evaluated on a realistic *pattern-classification* task. Swapping in a
real chest X-ray dataset later requires no code changes beyond pointing
the loader at real image files instead of this generator.
"""
import numpy as np

IMG_SIZE = 64

def _rib_shadows(img, rng):
    h, w = img.shape
    for i in range(5):
        x = int(w * (0.15 + i * 0.14))
        curve = 8 * np.sin(np.linspace(0, np.pi, h))
        for y in range(h):
            xx = int(x + curve[y])
            if 0 <= xx < w:
                img[y, max(0, xx-1):xx+1] += 12
    return img

def _lung_field(rng, noisy=False):
    img = np.full((IMG_SIZE, IMG_SIZE), 95, dtype=np.float32)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    # two lung ovals (darker / air-filled)
    for cx in (IMG_SIZE*0.32, IMG_SIZE*0.68):
        cy = IMG_SIZE * 0.55
        mask = ((xx - cx) / (IMG_SIZE*0.22))**2 + ((yy - cy) / (IMG_SIZE*0.38))**2 < 1
        img[mask] -= 35
    # baseline sensor noise: variable strength so "normal" images span a
    # wide noise range too, not just perfectly clean renders. This stops
    # the classifier from treating "any texture/noise" as itself
    # diagnostic -- it has to key on the localized opacity pattern instead.
    base_noise_std = rng.uniform(4, 6) if not noisy else rng.uniform(14, 26)
    img += rng.normal(0, base_noise_std, img.shape)
    if noisy:
        # occasional speckle/JPEG-grain-like texture and mild brightness jitter,
        # simulating a real photographed/rescanned image
        if rng.random() < 0.6:
            speckle = rng.normal(0, rng.uniform(8, 18), img.shape)
            mask = rng.random(img.shape) < 0.35
            img[mask] += speckle[mask]
        img *= rng.uniform(0.85, 1.15)
        img += rng.uniform(-15, 15)
    img = _rib_shadows(img, rng)
    return img

def _add_opacity(img, rng, n_blobs=None):
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    n_blobs = n_blobs or rng.integers(1, 4)
    for _ in range(n_blobs):
        cx = rng.uniform(w*0.2, w*0.8)
        cy = rng.uniform(h*0.35, h*0.8)
        r  = rng.uniform(6, 13)
        intensity = rng.uniform(40, 70)
        d2 = (xx - cx)**2 + (yy - cy)**2
        blob = intensity * np.exp(-d2 / (2 * r * r))
        img += blob
    return img

def generate_dataset(n_per_class=250, seed=42):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_per_class):
        noisy = rng.random() < 0.45   # ~45% of normals are noisy/textured too
        img = _lung_field(rng, noisy=noisy)
        X.append(np.clip(img, 0, 255).astype(np.uint8)); y.append(0)  # Normal
    for _ in range(n_per_class):
        noisy = rng.random() < 0.45
        img = _lung_field(rng, noisy=noisy)
        img = _add_opacity(img, rng)
        X.append(np.clip(img, 0, 255).astype(np.uint8)); y.append(1)  # Pneumonia
    X = np.array(X); y = np.array(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]

if __name__ == "__main__":
    X, y = generate_dataset()
    np.save("/home/claude/MediSense/data/xray/images.npy", X)
    np.save("/home/claude/MediSense/data/xray/labels.npy", y)
    print("Generated:", X.shape, "Normal:", (y==0).sum(), "Pneumonia:", (y==1).sum())
