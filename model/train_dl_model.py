"""
DL Module — Chest X-ray Pneumonia Detector.

Architecture (built without TensorFlow/PyTorch, which could not be
installed in this offline sandbox — no internet access to fetch the
packages or pretrained ImageNet weights):

  Input image (any size/aspect ratio, uploaded by a doctor)
      -> preprocess_upload(): grayscale + center-crop-to-square +
         histogram equalization -> clean 64x64 square
      -> Convolution bank (Sobel-X, Sobel-Y, Laplacian edge filters +
         Gaussian blob/consolidation filter)         [real 2D convolution]
      -> ReLU
      -> Max-pooling (8x8 blocks)                     [spatial downsampling]
      -> Flatten -> feature vector
      -> Deep Neural Network (MLPClassifier, hidden layers 256-128-64,
         trained via backpropagation + Adam)          [the "deep" classifier]
      -> Softmax -> Normal / Pneumonia

This mirrors the conv -> pool -> dense structure of a real CNN, using
NumPy/SciPy for the convolutional feature extraction stage and a genuine
multi-layer backprop network for the classification head.

── TRAINING ON REAL X-RAYS ─────────────────────────────────────────
By default this trains on procedurally-generated synthetic X-ray-like
patterns (no real dataset could be downloaded in this offline sandbox).
If you have internet access, you can get MUCH better real-world accuracy
by training on the free Kaggle "Chest X-Ray Images (Pneumonia)" dataset:

  1. Download & unzip: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  2. Run:  python train_dl_model.py --real-data "/path/to/chest_xray/train"
     (folder must contain NORMAL/ and PNEUMONIA/ subfolders of images —
     this is exactly how the Kaggle download is already structured)
  3. Restart the Flask app — it will pick up the newly retrained model.
"""
import os, sys, pickle, argparse, glob
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "xray"))
from generate_xray_images import generate_dataset
from feature_utils import conv_feature_extract, preprocess_upload

IMG_SIZE = 64

def build_feature_matrix(images_uint8):
    return np.array([conv_feature_extract(im) for im in images_uint8])

def load_real_dataset(root_dir):
    """Loads images from <root_dir>/NORMAL/*.jpeg and <root_dir>/PNEUMONIA/*.jpeg
    (the standard Kaggle chest_xray folder layout), running each through the
    exact same preprocess_upload() pipeline used at inference time."""
    X, y = [], []
    for label, cls in [(0, "NORMAL"), (1, "PNEUMONIA")]:
        files = glob.glob(os.path.join(root_dir, cls, "*"))
        print(f"  Loading {len(files)} images from {cls}/ ...")
        for fp in files:
            try:
                img = Image.open(fp)
                img = preprocess_upload(img, size=IMG_SIZE)
                X.append(np.array(img, dtype=np.uint8))
                y.append(label)
            except Exception as e:
                print(f"    skipped {fp}: {e}")
    return np.array(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=str, default=None,
        help="Path to a folder with NORMAL/ and PNEUMONIA/ subfolders of real X-ray images "
             "(e.g. the Kaggle chest_xray/train folder). If omitted, trains on synthetic data.")
    parser.add_argument("--n-per-class", type=int, default=250,
        help="Number of synthetic images per class (ignored if --real-data is set).")
    args = parser.parse_args()

    if args.real_data:
        print(f"Loading REAL chest X-ray images from {args.real_data} ...")
        X_img, y = load_real_dataset(args.real_data)
        data_source = f"real Kaggle-format dataset ({args.real_data})"
    else:
        print("Generating synthetic chest X-ray dataset (no --real-data supplied)...")
        X_img_raw, y = generate_dataset(n_per_class=args.n_per_class)
        # run through the SAME preprocess_upload pipeline used at inference time,
        # so training and real-world inference are perfectly consistent
        X_img = np.array([np.array(preprocess_upload(Image.fromarray(im), size=IMG_SIZE)) for im in X_img_raw])
        data_source = "synthetic procedurally-generated patterns"

    print(f"Dataset: {X_img.shape[0]} images, {X_img.shape[0]-int(y.sum())} normal / {int(y.sum())} pneumonia")

    print("Running convolutional feature extraction (Sobel/Laplacian/blob filters + max-pool)...")
    X_feat = build_feature_matrix(X_img)
    print("Feature matrix:", X_feat.shape)

    # ── Template-similarity OOD guard ──
    # Feature vectors are now L2-normalized (see feature_utils.py), so their
    # norm is always ~1 and can no longer signal "this looks unusual". Instead,
    # keep pixel-domain "what do normal/pneumonia chest X-rays look like"
    # templates and measure how well new uploads correlate with EITHER one.
    # Real photos or unrelated images correlate far worse than anything in
    # training (of either class), which lets the app show an honest
    # "uncertain" result instead of a confident-but-wrong verdict.
    def make_template(imgs):
        t = imgs.astype(np.float64).mean(axis=0)
        flat = t.flatten() - t.mean()
        return t, flat, np.linalg.norm(flat) + 1e-8

    def corr(img2d, flat_ref, norm_ref):
        v = img2d.astype(np.float64).flatten() - img2d.astype(np.float64).mean()
        vn = np.linalg.norm(v) + 1e-8
        return float(np.dot(v, flat_ref) / (vn * norm_ref))

    normal_template, normal_flat, normal_norm = make_template(X_img[y == 0])
    pneu_template, pneu_flat, pneu_norm = make_template(X_img[y == 1])

    # best-of-either-class correlation for every training image = the
    # in-distribution reference range
    all_best_corrs = np.array([
        max(corr(im, normal_flat, normal_norm), corr(im, pneu_flat, pneu_norm))
        for im in X_img.astype(np.float64)
    ])
    ood_corr_threshold = float(np.percentile(all_best_corrs, 3))
    print(f"  Template best-match correlation range: [{all_best_corrs.min():.3f}, {all_best_corrs.max():.3f}]  "
          f"-> OOD threshold: {ood_corr_threshold:.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("Training deep neural network classifier (256-128-64 hidden layers)...")
    clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu",
                         solver="adam", alpha=1e-4, max_iter=800,
                         early_stopping=True, random_state=42)
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"\n  🫁 Chest X-ray Pneumonia Detector   Accuracy: {acc*100:.2f}%   AUC: {auc*100:.2f}%")
    print(f"  Trained on: {data_source}")

    bundle = {
        "model": clf, "scaler": scaler,
        "name": "Chest X-ray Pneumonia Detector", "icon": "🫁", "color": "#6f42c1",
        "description": "CNN-style convolutional feature extraction + deep neural network "
                        "classifies chest X-rays as Normal or Pneumonia.",
        "accuracy": round(acc*100, 2), "auc": round(auc*100, 2),
        "img_size": IMG_SIZE,
        "data_source": data_source,
        "is_synthetic": args.real_data is None,
        "normal_template": normal_template, "pneumonia_template": pneu_template,
        "ood_corr_threshold": ood_corr_threshold,
    }
    out_path = os.path.join(os.path.dirname(__file__), "xray_dl.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    print("✅ Saved model to", out_path)

if __name__ == "__main__":
    main()
