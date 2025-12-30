# advanced model training

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


# EMNIST rotate/flip
def emnist_fix_pil(img: Image.Image) -> Image.Image:
    a = np.array(img)
    a = np.fliplr(np.rot90(a, 1))
    return Image.fromarray(a)


# EMNIST labels: 1..26 -> 0..25
def emnist_target_fix(y):
    return int(y) - 1


# custom CNN
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(task: str, data_dir: str, img_size: int, batch: int, num_workers: int):
    aug = transforms.RandomAffine(
        degrees=10,
        translate=(0.08, 0.08),
        scale=(0.9, 1.1),
        shear=5,
        fill=0,
    )

    if task == "digits":
        tf_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            aug,
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        tf_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        train_ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tf_train)
        test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=tf_test)
        num_classes = 10

    else:
        tf_train = transforms.Compose([
            transforms.Lambda(emnist_fix_pil),
            transforms.Resize((img_size, img_size)),
            aug,
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        tf_test = transforms.Compose([
            transforms.Lambda(emnist_fix_pil),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        train_ds = datasets.EMNIST(
            root=data_dir,
            split="letters",
            train=True,
            download=True,
            transform=tf_train,
            target_transform=emnist_target_fix,
        )
        test_ds = datasets.EMNIST(
            root=data_dir,
            split="letters",
            train=False,
            download=True,
            transform=tf_test,
            target_transform=emnist_target_fix,  
        )
        num_classes = 26

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, num_classes


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["digits", "letters"])
    ap.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.img_size is None:
        args.img_size = 28

    if args.out is None:
        args.out = str(
            ROOT / "models" / ("digits_advanced.pth" if args.task == "digits" else "letters_advanced.pth")
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader, num_classes = build_loaders(
        task=args.task,
        data_dir=str(data_dir),
        img_size=args.img_size,
        batch=args.batch,
        num_workers=args.num_workers,
    )

    model = AdvancedCNN(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            running += float(loss.item()) * x.size(0)
            n += x.size(0)

        test_acc = eval_acc(model, test_loader, device)
        train_loss = running / max(1, n)
        print(f"[{args.task}] Epoch {ep}/{args.epochs}  loss={train_loss:.4f}  test_acc={test_acc*100:.2f}%")

        if test_acc > best:
            best = test_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "task": args.task,
                    "img_size": args.img_size,
                    "norm_mean": 0.5,
                    "norm_std": 0.5,
                },
                str(out_path),
            )
            print(f"✅ Saved best -> {out_path} (acc={best*100:.2f}%)")

    print(f"Done. Best test_acc={best*100:.2f}%  file={out_path}")


if __name__ == "__main__":
    main()
