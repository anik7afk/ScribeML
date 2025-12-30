import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.transforms.functional as F


# letters cnn
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


# emnist fix
def fix_emnist_pil(img):
    img = F.rotate(img, -90)
    img = F.hflip(img)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    data_dir = root / "data"

    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "letters_model.pth"

    if out_path.exists() and not args.force:
        print(f"✅ letters model already exists -> {out_path}")
        print("Use --force if you want to retrain.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # aug
    train_transform = transforms.Compose([
        transforms.Lambda(fix_emnist_pil),

        # small aug
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.25),
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.25),
        transforms.RandomAutocontrast(p=0.20),
        transforms.RandomInvert(p=0.10),

        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=12,
            translate=(0.08, 0.08),
            scale=(0.9, 1.1),
            shear=8
        ),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(fix_emnist_pil),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading EMNIST Letters (A–Z only)...")
    train_dataset = datasets.EMNIST(
        str(data_dir),
        split="letters",
        train=True,
        download=True,
        transform=train_transform
    )

    val_dataset = datasets.EMNIST(
        str(data_dir),
        split="letters",
        train=False,
        download=True,
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = LettersCNN(num_classes=26).to(device)

    # AdamW
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    # label smoothing
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    classes = [chr(ord("A") + i) for i in range(26)]

    print("\nStarting training...")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Classes: 26 (A–Z)")
    print("Note: EMNIST letters labels are 1..26, so we shift to 0..25 during training.\n")

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (img, lbl) in enumerate(train_loader):
            img = img.to(device)
            lbl = (lbl - 1).to(device)  # 1..26 -> 0..25

            opt.zero_grad()
            out = model(img)
            loss = crit(out, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # clip grads
            opt.step()

            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_total += lbl.size(0)
            train_correct += (pred == lbl).sum().item()

            if (batch_idx + 1) % 200 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for img, lbl in val_loader:
                img = img.to(device)
                lbl = (lbl - 1).to(device)
                out = model(img)
                pred = out.argmax(dim=1)
                val_total += lbl.size(0)
                val_correct += (pred == lbl).sum().item()

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        avg_loss = train_loss / max(1, len(train_loader))

        print(f"\nEpoch [{epoch+1}/{args.epochs}] Summary:")
        print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Acc: {val_acc:.2f}%\n")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": 26,
                "classes": classes,
                "split": "letters",
                "emnist_fix": "rotate(-90) + hflip",
                "normalize": {"mean": 0.1307, "std": 0.3081}
            }, out_path)
            print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%) -> {out_path}\n")

        scheduler.step()

    print("Training complete!")
    print(f"Best val accuracy: {best_acc:.2f}%")
    print(f"Saved to: {out_path}")
    print("Class mapping: 0->A, 1->B, ... 25->Z")


if __name__ == "__main__":
    main()
