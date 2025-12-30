import os
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms.functional as F

import joblib
from skimage.feature import hog as sk_hog

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent.parent


# -----------------------
# Digit CNN (same as training)
# -----------------------
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# -----------------------
# Letters CNN (same as training)
# -----------------------
class LettersCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 256),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def fix_emnist_pil(img):
    # rotate -90 then flip horizontally (your training fix)
    img = F.rotate(img, -90)
    img = F.hflip(img)
    return img


def load_hog_meta(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    if isinstance(meta, dict):
        for key in ("hog", "hog_params", "hog_kwargs"):
            if key in meta and isinstance(meta[key], dict):
                return meta[key]
        keys = ["orientations", "pixels_per_cell", "cells_per_block", "block_norm", "transform_sqrt"]
        flat = {k: meta[k] for k in keys if k in meta}
        return flat if flat else None
    return None


def hog_features_u8(img_u8: np.ndarray, meta: dict | None = None) -> np.ndarray:
    kwargs = dict(
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    if isinstance(meta, dict):
        for k in ("orientations", "block_norm", "transform_sqrt"):
            if k in meta:
                kwargs[k] = meta[k]
        for k in ("pixels_per_cell", "cells_per_block"):
            if k in meta:
                v = meta[k]
                if isinstance(v, list):
                    v = tuple(v)
                kwargs[k] = v

    return sk_hog(img_u8, **kwargs).astype(np.float32)


def tensor_to_u8(img_tensor: torch.Tensor) -> np.ndarray:
    return (img_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)


def fix_emnist_u8(img_u8: np.ndarray) -> np.ndarray:
    img = np.rot90(img_u8, k=-1)
    img = np.fliplr(img)
    return img.astype(np.uint8)


def get_feat_dim(clf) -> int:
    try:
        if hasattr(clf, "named_steps") and "logreg" in clf.named_steps:
            return int(clf.named_steps["logreg"].coef_.shape[1])
    except Exception:
        pass
    try:
        return int(clf.coef_.shape[1])
    except Exception:
        pass
    return int(getattr(clf, "n_features_in_", 0) or 0)


def should_use_hog(clf) -> bool:
    feat_dim = get_feat_dim(clf)
    if feat_dim == 0:
        return True
    return feat_dim != 784


def build_baseline_features(dataset, use_hog: bool, hog_meta: dict | None = None, label_shift: int = 0, fix_emnist: bool = False):
    n = len(dataset)
    sample_img, _ = dataset[0]
    sample_u8 = tensor_to_u8(sample_img)
    if fix_emnist:
        sample_u8 = fix_emnist_u8(sample_u8)
    if use_hog:
        feat_dim = hog_features_u8(sample_u8, hog_meta).shape[0]
    else:
        feat_dim = sample_u8.size

    X = np.zeros((n, feat_dim), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)

    for i in range(n):
        img, label = dataset[i]
        img_u8 = tensor_to_u8(img)
        if fix_emnist:
            img_u8 = fix_emnist_u8(img_u8)

        if use_hog:
            X[i] = hog_features_u8(img_u8, hog_meta)
        else:
            X[i] = (img_u8.astype(np.float32) / 255.0).reshape(-1)

        y[i] = int(label) + label_shift

        if (i + 1) % 2000 == 0 or i + 1 == n:
            print(f"Built baseline features: {i + 1}/{n}")

    return X, y


def save_cm_png(y_true, y_pred, title, out_path, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if labels is not None and len(labels) <= 26:
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def preprocess_steps_png(img_path, out_path):
    # Optional: only if you provide a sample image
    import cv2

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    gray = img.copy()

    # Otsu threshold (invert so ink is white)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # crop around ink
    ys, xs = np.where(bw > 0)
    if len(xs) == 0 or len(ys) == 0:
        crop = bw
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = bw[y0:y1 + 1, x0:x1 + 1]

    # pad to square
    h, w = crop.shape
    m = max(h, w)
    pad_y = (m - h) // 2
    pad_x = (m - w) // 2
    sq = np.pad(
        crop,
        ((pad_y, m - h - pad_y), (pad_x, m - w - pad_x)),
        mode="constant"
    )

    # resize to 28x28
    final28 = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_AREA)

    steps = [gray, bw, crop, sq, final28]
    titles = ["Original", "Threshold", "Cropped", "Padded", "Final 28×28"]

    plt.figure(figsize=(12, 3))
    for i, (im, t) in enumerate(zip(steps, titles), start=1):
        plt.subplot(1, 5, i)
        plt.imshow(im, cmap="gray")
        plt.title(t, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits_cnn", default=str(ROOT / "models" / "digit_model.pth"))
    ap.add_argument("--letters_cnn", default=str(ROOT / "models" / "letters_model.pth"))
    ap.add_argument("--out_dir", default=str(ROOT / "figures"))
    ap.add_argument("--data_dir", default=str(ROOT / "data"))

    # Optional: only for preprocessing pipeline figure
    ap.add_argument("--sample_image", default=None, help="Optional: image path to generate preprocess pipeline figure")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    digits_cnn = Path(args.digits_cnn)
    if not digits_cnn.is_absolute():
        digits_cnn = ROOT / digits_cnn

    letters_cnn = Path(args.letters_cnn)
    if not letters_cnn.is_absolute():
        letters_cnn = ROOT / letters_cnn

    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = ROOT / "models"
    letters_hog_meta = load_hog_meta(models_dir / "letters_logreg_meta.json")

    # Optional preprocessing pipeline figure
    if args.sample_image:
        sample_image = Path(args.sample_image)
        if not sample_image.is_absolute():
            sample_image = ROOT / sample_image
        out_pp = out_dir / "preprocess_pipeline.png"
        preprocess_steps_png(str(sample_image), str(out_pp))
        print("Saved:", out_pp)
    else:
        print("Skipping preprocessing pipeline figure (no --sample_image provided).")

    # ---------------- DIGITS CNN ----------------
    if not digits_cnn.exists():
        print(f"Digits CNN not found: {digits_cnn} (skipping digits CNN eval)")
    else:
        digits_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        mnist_test = datasets.MNIST(str(data_dir), train=False, download=True, transform=digits_tf)

        Xd = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))])
        yd = np.array([mnist_test[i][1] for i in range(len(mnist_test))], dtype=np.int64)

        model = DigitCNN().to(device)
        model.load_state_dict(torch.load(digits_cnn, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(Xd), 512):
                out = model(Xd[i:i + 512].to(device))
                preds.append(out.argmax(dim=1).cpu().numpy())

        pd = np.concatenate(preds)
        acc = accuracy_score(yd, pd)

        report_path = out_dir / "digits_cnn_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Digits CNN accuracy: {acc:.4f}\n\n")
            f.write(classification_report(yd, pd))

        cm_path = out_dir / "cm_digits_cnn.png"
        save_cm_png(yd, pd, "Digits CNN Confusion Matrix", cm_path)

        print("Saved:", report_path)
        print("Saved:", cm_path)

    # ---------------- LETTERS CNN ----------------
    if not letters_cnn.exists():
        print(f"Letters CNN not found: {letters_cnn} (skipping letters CNN eval)")
    else:
        letters_tf = transforms.Compose([
            transforms.Lambda(fix_emnist_pil),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        emnist_test = datasets.EMNIST(str(data_dir), split="letters", train=False, download=True, transform=letters_tf)

        Xl = torch.stack([emnist_test[i][0] for i in range(len(emnist_test))])
        yl = np.array([emnist_test[i][1] - 1 for i in range(len(emnist_test))], dtype=np.int64)
        labels = [chr(ord("A") + i) for i in range(26)]

        ckpt = torch.load(letters_cnn, map_location=device)
        model = LettersCNN(num_classes=26).to(device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, len(Xl), 512):
                out = model(Xl[i:i + 512].to(device))
                preds.append(out.argmax(dim=1).cpu().numpy())

        pl = np.concatenate(preds)
        acc = accuracy_score(yl, pl)

        report_path = out_dir / "letters_cnn_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Letters CNN accuracy: {acc:.4f}\n\n")
            f.write(classification_report(yl, pl, target_names=labels))

        cm_path = out_dir / "cm_letters_cnn.png"
        save_cm_png(yl, pl, "Letters CNN Confusion Matrix", cm_path, labels=labels)

        print("Saved:", report_path)
        print("Saved:", cm_path)

    # ---------------- DIGITS BASELINE ----------------
    digits_baseline_path = models_dir / "digits_logreg.pkl"
    if not digits_baseline_path.exists():
        print(f"Digits baseline not found: {digits_baseline_path} (skipping digits baseline eval)")
    else:
        clf = joblib.load(digits_baseline_path)
        use_hog = should_use_hog(clf)

        digits_tf = transforms.ToTensor()
        mnist_test = datasets.MNIST(str(data_dir), train=False, download=True, transform=digits_tf)

        Xd, yd = build_baseline_features(mnist_test, use_hog=use_hog)
        pd = clf.predict(Xd)
        acc = accuracy_score(yd, pd)

        report_path = out_dir / "digits_baseline_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Digits baseline accuracy: {acc:.4f}\n")
            f.write(f"Features: {'HOG' if use_hog else 'raw'}\n\n")
            f.write(classification_report(yd, pd))

        cm_path = out_dir / "cm_digits_baseline.png"
        save_cm_png(yd, pd, "Digits Baseline Confusion Matrix", cm_path)

        print("Saved:", report_path)
        print("Saved:", cm_path)

    # ---------------- LETTERS BASELINE ----------------
    letters_baseline_path = models_dir / "letters_logreg.pkl"
    if not letters_baseline_path.exists():
        print(f"Letters baseline not found: {letters_baseline_path} (skipping letters baseline eval)")
    else:
        clf = joblib.load(letters_baseline_path)
        use_hog = should_use_hog(clf)

        letters_tf = transforms.ToTensor()
        emnist_test = datasets.EMNIST(str(data_dir), split="letters", train=False, download=True, transform=letters_tf)
        labels = [chr(ord("A") + i) for i in range(26)]

        Xl, yl = build_baseline_features(
            emnist_test,
            use_hog=use_hog,
            hog_meta=letters_hog_meta,
            label_shift=-1,
            fix_emnist=True,
        )
        pl = clf.predict(Xl)
        acc = accuracy_score(yl, pl)

        report_path = out_dir / "letters_baseline_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Letters baseline accuracy: {acc:.4f}\n")
            f.write(f"Features: {'HOG' if use_hog else 'raw'}\n\n")
            f.write(classification_report(yl, pl, target_names=labels))

        cm_path = out_dir / "cm_letters_baseline.png"
        save_cm_png(yl, pl, "Letters Baseline Confusion Matrix", cm_path, labels=labels)

        print("Saved:", report_path)
        print("Saved:", cm_path)


if __name__ == "__main__":
    main()
