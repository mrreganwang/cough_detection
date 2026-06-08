#!/usr/bin/env python3
"""
train.py
--------
Trains a CNN cough classifier on log-mel spectrograms.

Model: EfficientNet-style lightweight CNN with residual blocks.
  - Input: (batch, 1, 128, ~87) log-mel spectrogram
  - Output: probability of cough

Training strategy:
  - Focal loss (down-weights easy negatives, focuses on hard cases)
  - MixUp augmentation (interpolates between samples in feature space)
  - Cosine annealing LR schedule with warm restarts
  - Early stopping on validation AUC
  - Label smoothing to prevent overconfidence
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─── Dataset ─────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        # Add channel dim: (N, 1, n_mels, T)
        self.X = torch.from_numpy(X).unsqueeze(1)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """SpecAugment: frequency and time masking on the spectrogram."""
        # Frequency masking
        if torch.rand(1) < 0.5:
            f = int(torch.randint(0, 20, (1,)))
            f0 = int(torch.randint(0, max(1, x.shape[-2] - f), (1,)))
            x[:, f0:f0+f, :] = x.mean()

        # Time masking
        if torch.rand(1) < 0.5:
            t = int(torch.randint(0, 15, (1,)))
            t0 = int(torch.randint(0, max(1, x.shape[-1] - t), (1,)))
            x[:, :, t0:t0+t] = x.mean()

        return x


# ─── Model Architecture ───────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """Residual block with squeeze-and-excitation attention."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # Squeeze-and-excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        se_weight = self.se(out).view(-1, out.shape[1], 1, 1)
        out = out * se_weight
        return self.act(out + residual)


class CoughCNN(nn.Module):
    """
    Lightweight CNN for cough vs. non-cough classification.
    Designed to run in real-time on a CPU laptop.

    Input:  (batch, 1, 128, T) log-mel spectrogram
    Output: (batch, 1) logit
    """
    def __init__(self, n_mels: int = 128, dropout: float = 0.4):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(1, 32, kernel=3, stride=1, padding=1),
            ConvBNReLU(32, 32, kernel=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),   # 128→64 freq, T→T/2
        )

        self.stage1 = nn.Sequential(
            ConvBNReLU(32, 64, stride=2),  # 64→32 freq, T/2→T/4
            ResBlock(64),
        )

        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128, stride=2),  # 32→16 freq, T/4→T/8
            ResBlock(128),
            ResBlock(128),
        )

        self.stage3 = nn.Sequential(
            ConvBNReLU(128, 256, stride=2),  # 16→8 freq
            ResBlock(256),
        )

        # Global average + max pooling (captures both mean and peak activations)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        avg = self.global_pool(x).flatten(1)
        max_ = self.global_max(x).flatten(1)
        x = torch.cat([avg, max_], dim=1)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self(x))


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for binary classification.
    Down-weights easy examples, focuses training on hard cases.
    alpha: weight for positive class (use > 0.5 if positives are rare)
    gamma: focusing parameter (2.0 is standard)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        targets = targets * (1 - self.ls) + 0.5 * self.ls

        bce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets, reduction="none")
        p = torch.sigmoid(logits.squeeze())
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ─── MixUp ────────────────────────────────────────────────────────────────────

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """MixUp data augmentation — interpolates pairs of samples."""
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return x_mix, y_mix


# ─── Training loop ────────────────────────────────────────────────────────────

def evaluate(model, loader, device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze()
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    probs = 1 / (1 + np.exp(-logits))

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    # Accuracy at threshold 0.5
    preds = (probs >= 0.5).astype(int)
    acc = (preds == labels).mean()
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)

    return {"auc": auc, "ap": ap, "acc": acc, "precision": prec, "recall": rec, "f1": f1}


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    data_path = PROC_DIR / "dataset.npz"
    if not data_path.exists():
        print(f"❌  Dataset not found at {data_path}")
        print("   Run: python src/prepare_dataset.py")
        sys.exit(1)

    print(f"Loading dataset from {data_path}...")
    d = np.load(data_path)
    X_train, y_train = d["X_train"], d["y_train"]
    X_val, y_val = d["X_val"], d["y_val"]
    X_test, y_test = d["X_test"], d["y_test"]

    # Load feature params for later use in inference
    feature_params = {
        "sr": int(d["sr"]),
        "n_mels": int(d["n_mels"]),
        "n_samples": int(d["n_samples"]),
        "fmin": int(d["fmin"]),
        "fmax": int(d["fmax"]),
        "n_fft": int(d["n_fft"]),
        "hop_length": int(d["hop_length"]),
    }

    print(f"  Train: {len(X_train)} ({y_train.sum():.0f} coughs)")
    print(f"  Val:   {len(X_val)} ({y_val.sum():.0f} coughs)")
    print(f"  Test:  {len(X_test)} ({y_test.sum():.0f} coughs)")

    train_ds = SpectrogramDataset(X_train, y_train, augment=True)
    val_ds = SpectrogramDataset(X_val, y_val, augment=False)
    test_ds = SpectrogramDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type != "cpu")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # Model
    n_mels = X_train.shape[1]
    model = CoughCNN(n_mels=n_mels, dropout=args.dropout).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: CoughCNN ({n_params:,} parameters)")

    # Class weights — coughs are fewer, upweight them
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max(1, (y_train == 1).sum()) * 0.5],
        dtype=torch.float32,
    ).to(device)
    criterion = FocalLoss(alpha=0.75, gamma=2.0, label_smoothing=0.05)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # ── Training loop ──
    best_auc = 0.0
    best_epoch = 0
    patience_count = 0

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print("-" * 70)
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val AUC':>9} {'Val F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # MixUp augmentation
            if args.mixup > 0 and torch.rand(1) < 0.5:
                x, y = mixup_batch(x, y, alpha=args.mixup)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)

        if epoch % 2 == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device)
            print(
                f"{epoch:>6} {avg_loss:>11.4f} {metrics['auc']:>9.4f} "
                f"{metrics['f1']:>8.4f} {metrics['precision']:>10.4f} {metrics['recall']:>8.4f}"
            )

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_epoch = epoch
                patience_count = 0
                # Save best model
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_config": {"n_mels": n_mels, "dropout": args.dropout},
                    "feature_params": feature_params,
                    "best_auc": best_auc,
                    "epoch": epoch,
                }, MODEL_DIR / "best_model.pt")
            else:
                patience_count += 2  # we check every 2 epochs

            if patience_count >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (best was {best_epoch})")
                break

    print("-" * 70)
    print(f"\nBest validation AUC: {best_auc:.4f} at epoch {best_epoch}")

    # ── Test set evaluation ──
    print("\nLoading best model for test set evaluation...")
    ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, device)
    print("\nTest Set Results:")
    print(f"  AUC:       {test_metrics['auc']:.4f}")
    print(f"  AP:        {test_metrics['ap']:.4f}")
    print(f"  Accuracy:  {test_metrics['acc']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")

    # Save metrics
    with open(MODEL_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n✅  Model saved to {MODEL_DIR / 'best_model.pt'}")
    print("Next step: python src/evaluate.py  (detailed analysis)")
    print("       or: python src/realtime_detect.py  (run live)")


def main():
    parser = argparse.ArgumentParser(description="Train the cough detection CNN")
    parser.add_argument("--epochs", type=int, default=10, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--mixup", type=float, default=0.2, help="MixUp alpha (0=off)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
