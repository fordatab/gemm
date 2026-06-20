"""
PyTorch port of the Modern VGG-style CNN in this repo (modernnet.cc).

Mirrors the C++/CUDA network so the two can be compared:
  - same layer shapes, kernel sizes, pooling, and classifier
  - same loss (cross-entropy) and optimizer (SGD, momentum + Nesterov + L2 decay)
  - same input pipeline: Fashion-MNIST resized to 86x86, scaled to [0, 1]

Architecture (input 1x86x86):
  Block 1: Conv 3x3  1->16   -> 84x84, ReLU, MaxPool 2x2/2 -> 42x42
  Block 2: Conv 3x3  16->32  -> 40x40, ReLU, MaxPool 2x2/2 -> 20x20
  Block 3: Conv 3x3  32->64  -> 18x18, ReLU, MaxPool 2x2/2 ->  9x9
  Block 4: Conv 3x3  64->128 ->  7x7,  ReLU, MaxPool 2x2/2 ->  3x3
  Flatten: 3*3*128 = 1152
  Classifier: FC 1152->128, ReLU, FC 128->10  (softmax folded into the loss)

Usage:
  python pytorch_net.py --epochs 10 --batch 128 --lr 0.01
  python pytorch_net.py --summary           # just print the model + param counts
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ModernNet(nn.Module):
    """VGG-style CNN matching createModernNet_* in modernnet.cc."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Convs use no padding (the C++ Conv defaults pad_h=pad_w=0, stride=1),
        # so each 3x3 conv shrinks H and W by 2. MaxPool is 2x2 stride 2.
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),    # 86 -> 84
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 84 -> 42

            nn.Conv2d(16, 32, kernel_size=3),   # 42 -> 40
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 40 -> 20

            nn.Conv2d(32, 64, kernel_size=3),   # 20 -> 18
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 18 -> 9

            nn.Conv2d(64, 128, kernel_size=3),  # 9 -> 7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 7 -> 3 (floor)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x  # logits; softmax is folded into CrossEntropyLoss


def make_loaders(batch_size: int, data_dir: str = "./data"):
    """Fashion-MNIST resized to 86x86, scaled to [0, 1] (ToTensor does /255)."""
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.Resize(86),
        transforms.ToTensor(),  # -> float32 in [0, 1], shape 1x86x86
    ])
    train_set = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
    test_set = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ModernNet().to(device)
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader, test_loader = make_loaders(args.batch)

    criterion = nn.CrossEntropyLoss()
    # Matches SGD opt(lr, 5e-4, 0.9, true) in modern_main.cc:
    #   lr=lr, weight_decay=5e-4 (L2), momentum=0.9, nesterov=True
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if i == 0 or (i + 1) % 10 == 0:
                print(f"\rEpoch {epoch+1}/{args.epochs} | Batch {i+1}/{len(train_loader)}"
                      f" | Loss: {running/(i+1):.4f}", end="", flush=True)

        acc = evaluate(model, test_loader, device)
        print(f"\nEpoch {epoch+1} complete. Test Accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), args.out)
    print(f"Saved weights to {args.out}")


def main():
    p = argparse.ArgumentParser(description="PyTorch port of the repo's Modern CNN")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--out", type=str, default="./build/modern-weights-torch.pt")
    p.add_argument("--summary", action="store_true",
                   help="print model and parameter count, then exit")
    args = p.parse_args()

    if args.summary:
        model = ModernNet()
        print(model)
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
        # Shape sanity check against the C++ comments.
        x = torch.zeros(1, 1, 86, 86)
        feat = model.features(x)
        print(f"Feature map before flatten: {tuple(feat.shape)}  (expect 1, 128, 3, 3)")
        return

    train(args)


if __name__ == "__main__":
    main()
