import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, Subset

# HOG
from skimage.feature import hog as sk_hog


# model
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


# paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
CNN_OUT = MODELS_DIR / "digit_model.pth"
BASELINE_OUT = MODELS_DIR / "digits_logreg.pkl"
BASELINE_OUT_COMPAT = MODELS_DIR / "logreg_model.pkl"  # compat


# cnn train
def train_digits_cnn(epochs: int = 15):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=15,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomApply([
            transforms.ElasticTransform(alpha=34.0, sigma=4.0)
        ], p=0.3),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading MNIST dataset with heavy augmentation...")
    train_dataset = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=train_transform)
    val_dataset = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CNN on: {device}")

    model = DigitCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    crit = nn.CrossEntropyLoss()

    print("\nStarting training with heavy augmentation...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}\n")

    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (img, lbl) in enumerate(train_loader):
            img, lbl = img.to(device), lbl.to(device)
            opt.zero_grad()
            out = model(img)
            loss = crit(out, lbl)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_total += lbl.size(0)
            train_correct += (pred == lbl).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                pred = out.argmax(dim=1)
                val_total += lbl.size(0)
                val_correct += (pred == lbl).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        avg_loss = train_loss / max(1, len(train_loader))

        print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
        print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Validation Acc: {val_acc:.2f}%\n")

        if val_acc > best_acc:
            best_acc = val_acc
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CNN_OUT)
            print(f"Best CNN model saved! (Val Acc: {val_acc:.2f}%) -> {CNN_OUT}\n")

        scheduler.step()

    print("\nCNN training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"CNN model saved to: {CNN_OUT}")


# baseline (HOG + LR)
def hog_features(img28_u8: np.ndarray) -> np.ndarray:
    feat = sk_hog(
        img28_u8,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return feat.astype(np.float32)


def loader_to_hog_numpy(loader: DataLoader):
    X_list, y_list = [], []
    total = len(loader)

    for i, (img, lbl) in enumerate(loader, start=1):
        # to uint8
        imgs_u8 = (img.squeeze(1).numpy() * 255.0).astype(np.uint8)

        feats = [hog_features(im) for im in imgs_u8]
        X_list.append(np.stack(feats, axis=0))
        y_list.append(lbl.numpy().astype(np.int64))

        if i % 20 == 0 or i == total:
            print(f"Built HOG features for {i}/{total} batches...")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def train_digits_baseline(train_n: int = 40000, test_n: int = 10000, seed: int = 42):
    # use raw pixels for HOG
    baseline_transform = transforms.ToTensor()

    print("\nLoading MNIST for baseline (HOG + LR)...")
    full_train = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=baseline_transform)
    full_test = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=baseline_transform)

    rng = np.random.default_rng(seed)

    if train_n and train_n < len(full_train):
        idx = rng.permutation(len(full_train))[:train_n]
        train_ds = Subset(full_train, idx)
    else:
        train_ds = full_train

    if test_n and test_n < len(full_test):
        idx = rng.permutation(len(full_test))[:test_n]
        test_ds = Subset(full_test, idx)
    else:
        test_ds = full_test

    print(f"Baseline train samples: {len(train_ds)}")
    print(f"Baseline test samples : {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    print("\nExtracting HOG features (this takes a bit)...")
    X_train, y_train = loader_to_hog_numpy(train_loader)
    X_test, y_test = loader_to_hog_numpy(test_loader)

    print("\nTraining Logistic Regression baseline (HOG)...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=2000,
            solver="saga",
            multi_class="multinomial",
            n_jobs=-1,
            verbose=1,
            random_state=seed,
            C=2.0
        )),
    ])

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nDigits baseline test accuracy (HOG+LR): {acc:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, BASELINE_OUT)
    print(f"Saved baseline to: {BASELINE_OUT}")

    # compat save
    joblib.dump(clf, BASELINE_OUT_COMPAT)
    print(f"Also saved baseline copy to: {BASELINE_OUT_COMPAT}")


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="CNN epochs (default: 15)")
    parser.add_argument("--force", action="store_true", help="Retrain even if model files exist")
    parser.add_argument("--only", choices=["cnn", "baseline", "both"], default="both")

    # baseline opts
    parser.add_argument("--baseline_train_n", type=int, default=40000, help="Digits baseline train samples (default 40000)")
    parser.add_argument("--baseline_test_n", type=int, default=10000, help="Digits baseline test samples (default 10000)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    run_cnn = args.only in ("cnn", "both")
    run_baseline = args.only in ("baseline", "both")

    if run_cnn and CNN_OUT.exists() and not args.force:
        print(f"CNN already exists -> skipping: {CNN_OUT} (use --force to retrain)")
        run_cnn = False

    if run_baseline and BASELINE_OUT.exists() and not args.force:
        print(f"Baseline already exists -> skipping: {BASELINE_OUT} (use --force to retrain)")
        run_baseline = False

    if not run_cnn and not run_baseline:
        print("Nothing to do (everything already trained).")
        return

    if run_cnn:
        train_digits_cnn(epochs=args.epochs)

    if run_baseline:
        train_digits_baseline(
            train_n=args.baseline_train_n,
            test_n=args.baseline_test_n,
            seed=args.seed
        )

    print("\nDone. Models saved in the models/ folder.")


if __name__ == "__main__":
    main()
