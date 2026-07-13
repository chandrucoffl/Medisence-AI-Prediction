"""Shared conv-style feature extraction, used by both training and the Flask app."""
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import convolve, gaussian_filter
from skimage.measure import block_reduce

SOBEL_X = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T
LAPLACIAN = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)

def preprocess_upload(pil_img, size=64):
    """Turn ANY uploaded photo (any aspect ratio, orientation, lighting,
    color mode) into a clean, consistent square input for the model.

    - convert to grayscale (handles RGB/RGBA/CMYK/etc. uploads)
    - center-crop to a square using the shorter side, instead of naively
      squishing width/height to 64x64, which badly distorts non-square
      photos (most real phone photos of an X-ray film are not square)
    - histogram-equalize so differing lighting/exposure/contrast across
      photos gets normalized onto a consistent scale
    """
    img = pil_img.convert("L")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    img = ImageOps.equalize(img)
    return img

def conv_feature_extract(img):
    img = img.astype(np.float32)
    edge_x = np.abs(convolve(img, SOBEL_X))
    edge_y = np.abs(convolve(img, SOBEL_Y))
    laplace = np.abs(convolve(img, LAPLACIAN))
    blob = gaussian_filter(img, sigma=3)
    blob_response = np.abs(img - gaussian_filter(img, sigma=1))

    maps = [edge_x, edge_y, laplace, blob, blob_response]
    pooled = []
    for m in maps:
        m = np.maximum(m, 0)
        p = block_reduce(m, block_size=(8, 8), func=np.max)
        pooled.append(p.flatten())
    feat = np.concatenate(pooled)

    # Per-image L2 normalization: makes the classifier reason about the
    # *shape*/distribution of energy across filters and spatial positions
    # rather than its absolute magnitude. Without this, any image with
    # naturally higher baseline texture/noise than our clean synthetic
    # training renders (e.g. a real photo with JPEG grain, skin, fabric,
    # camera noise) gets an inflated feature norm and is misread as
    # "pneumonia opacity" almost every time, regardless of content.
    norm = np.linalg.norm(feat)
    if norm > 1e-6:
        feat = feat / norm
    return feat
