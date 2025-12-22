import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

# CNN model (same as original)
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64*5*5, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, x): return self.fc(self.conv(x))

# Data augmentation
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

# Validation transform
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
print("Loading MNIST dataset with heavy augmentation...")
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
val_dataset = datasets.MNIST('./data', train=False, download=True, transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

model = DigitCNN().to(device)
opt = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
crit = nn.CrossEntropyLoss()

print("\nStarting training with heavy augmentation...")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}\n")

best_acc = 0.0

# Train for 15 epochs
for epoch in range(15):
    # Train
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
        _, pred = torch.max(out, 1)
        train_total += lbl.size(0)
        train_correct += (pred == lbl).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/15], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    
    # Validate
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for img, lbl in val_loader:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            _, pred = torch.max(out, 1)
            val_total += lbl.size(0)
            val_correct += (pred == lbl).sum().item()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    avg_loss = train_loss / len(train_loader)
    
    print(f"\nEpoch [{epoch+1}/15] Summary:")
    print(f"Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Validation Acc: {val_acc:.2f}%\n")
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/digit_model.pth")
        print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)\n")
    
    scheduler.step()

print(f"\nTraining complete!")
print(f"Best validation accuracy: {best_acc:.2f}%")
print(f"Model saved to: models/digit_model.pth")