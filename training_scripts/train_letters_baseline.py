import os
import argparse
from pathlib import Path

import numpy as np
import joblib
from tqdm import tqdm

import torch
from torchvision import datasets, transforms

from skimage.feature import hog
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = MODELS_DIR / "letters_logreg.pkl"
META_OUT = MODELS_DIR / "letters_logreg_meta.json"


# hog settings
def hog_features(img28_uint8: np.ndarray) -> np.ndarray:
    feat = hog(
        img28_uint8,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


# emnist letters
to_tensor = transforms.ToTensor()

def load_emnist_letters(train=True):
    return datasets.EMNIST(
        root=str(DATA_DIR),
        split="letters",
        train=train,
        download=True,
        transform=to_tensor
    )


def tensor_to_uint8_emnist(img_tensor: torch.Tensor) -> np.ndarray:
    """
    EMNIST fix:
      rotate -90 then flip left-right
    """
    img = (img_tensor.squeeze(0).numpy() * 255.0).astype(np.uint8)
    img = np.rot90(img, k=-1)
    img = np.fliplr(img)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_n", type=int, default=60000, help="How many training samples (default: 60000). Full train is ~88800.")
    parser.add_argument("--test_n", type=int, default=14800, help="How many test samples (default: 14800 = full test).")
    parser.add_argument("--C", type=float, default=3.0, help="Inverse regularization strength (default: 3.0)")
    parser.add_argument("--max_iter", type=int, default=2500, help="Max iterations for LogisticRegression (default: 2500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Retrain even if model exists")
    args = parser.parse_args()

    # skip if exists
    if MODEL_OUT.exists() and not args.force:
        print(f"letters baseline already exists -> {MODEL_OUT}")
        print("Use --force if you want to retrain.")
        return

    print("Loading EMNIST Letters (A–Z) for baseline...")
    train_ds = load_emnist_letters(train=True)
    test_ds  = load_emnist_letters(train=False)

    rng = np.random.default_rng(args.seed)

    # subset
    train_n = min(args.train_n, len(train_ds))
    test_n  = min(args.test_n, len(test_ds))

    train_idx = rng.permutation(len(train_ds))[:train_n]
    test_idx  = rng.permutation(len(test_ds))[:test_n]

    print("Building HOG features:")
    print(f"  Train samples: {train_n}/{len(train_ds)}")
    print(f"  Test samples : {test_n}/{len(test_ds)}")

    X_train = np.zeros((train_n, hog_features(np.zeros((28, 28), dtype=np.uint8)).shape[0]), dtype=np.float32)
    y_train = np.zeros((train_n,), dtype=np.int32)

    for j, i in enumerate(tqdm(train_idx, desc="HOG train")):
        img, label = train_ds[int(i)]
        img_u8 = tensor_to_uint8_emnist(img)
        X_train[j] = hog_features(img_u8)
        y_train[j] = int(label) - 1  # 1..26 -> 0..25

    X_test = np.zeros((test_n, X_train.shape[1]), dtype=np.float32)
    y_test = np.zeros((test_n,), dtype=np.int32)

    for j, i in enumerate(tqdm(test_idx, desc="HOG test")):
        img, label = test_ds[int(i)]
        img_u8 = tensor_to_uint8_emnist(img)
        X_test[j] = hog_features(img_u8)
        y_test[j] = int(label) - 1

    print("\nTraining baseline: StandardScaler + LogisticRegression (multinomial)...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            solver="saga",
            multi_class="multinomial",
            max_iter=args.max_iter,
            n_jobs=-1,
            verbose=1,
            random_state=args.seed,
            C=args.C
        ))
    ])

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nLetters baseline accuracy (HOG+LR) on {test_n} test samples: {acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, preds))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(clf, MODEL_OUT)
    print("\nSaved baseline model to:", MODEL_OUT)

    # meta
    try:
        import json
        meta = {
            "train_n": int(train_n),
            "test_n": int(test_n),
            "seed": int(args.seed),
            "C": float(args.C),
            "max_iter": int(args.max_iter),
            "hog": {
                "orientations": 9,
                "pixels_per_cell": [4, 4],
                "cells_per_block": [2, 2],
                "block_norm": "L2-Hys"
            },
            "emnist_fix": "rotate(-90) + hflip",
            "labels": "EMNIST letters are 1..26; shifted to 0..25"
        }
        with open(META_OUT, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("Saved meta to:", META_OUT)
    except Exception:
        pass


if __name__ == "__main__":
    main()
