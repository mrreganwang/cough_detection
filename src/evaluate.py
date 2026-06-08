#!/usr/bin/env python3
"""
evaluate.py
-----------
Detailed evaluation of the trained cough detector:
  - ROC curve + optimal threshold selection
  - Precision-recall curve
  - Confusion matrix
  - False positive analysis (what non-cough sounds trip the detector?)
  - Threshold sweep table (helps you pick the right threshold for your use case)

Run after training: python src/evaluate.py
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless plotting
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, classification_report,
    average_precision_score,
)

ROOT = Path(__file__).parent.parent
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

sys.path.insert(0, str(Path(__file__).parent))
from train import CoughCNN, SpectrogramDataset


def load_model(device):
    ckpt_path = MODEL_DIR / "best_model.pt"
    if not ckpt_path.exists():
        print(f"❌  No model found at {ckpt_path}")
        print("   Run: python src/train.py")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["model_config"]
    model = CoughCNN(n_mels=cfg["n_mels"], dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("feature_params", {})


def get_predictions(model, loader, device):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs if probs.ndim > 0 else probs.reshape(1))
            all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def plot_roc(fpr, tpr, roc_auc, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, "b-", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Cough Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_pr(precision, recall, ap, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, "g-", lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Cough Detector")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion(cm, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No Cough", "Pred: Cough"])
    ax.set_yticklabels(["True: No Cough", "True: Cough"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def find_optimal_threshold(fpr, tpr, thresholds):
    """Find threshold that maximizes Youden's J = sensitivity + specificity - 1."""
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]


def threshold_sweep_table(probs, labels):
    """Print a table of metrics at various thresholds to help users choose."""
    print("\nThreshold Sweep (choose based on your use case):")
    print("-" * 70)
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FPR':>8} {'TPR':>8}")
    print("-" * 70)

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        preds = (probs >= thresh).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        fpr = fp / max(1, fp + tn)
        print(f"{thresh:>10.2f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f} {fpr:>8.4f} {rec:>8.4f}")

    print("-" * 70)
    print("\nGuidance:")
    print("  - For low false positives (near a mic all day): use threshold 0.75–0.85")
    print("  - For balanced detection:                       use threshold 0.60–0.70")
    print("  - For maximum sensitivity (miss nothing):       use threshold 0.40–0.50")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    print("Cough Detector — Evaluation")
    print("=" * 60)

    model, feature_params = load_model(device)

    # Load test set
    data_path = PROC_DIR / "dataset.npz"
    d = np.load(data_path)
    X_test, y_test = d["X_test"], d["y_test"]

    test_ds = SpectrogramDataset(X_test, y_test, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"Test set: {len(X_test)} samples ({y_test.sum():.0f} coughs, {(y_test==0).sum():.0f} non-coughs)")
    print("\nRunning inference...")

    probs, labels = get_predictions(model, test_loader, device)

    # ── Metrics ──
    roc_auc_val = 0.0
    try:
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        roc_auc_val = auc(fpr, tpr)
        opt_thresh, opt_j = find_optimal_threshold(fpr, tpr, roc_thresholds)
    except Exception:
        fpr = tpr = roc_thresholds = np.array([0, 1])
        opt_thresh = 0.5
        opt_j = 0

    ap = average_precision_score(labels, probs)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, probs)

    print(f"\nROC AUC:  {roc_auc_val:.4f}")
    print(f"Avg Prec: {ap:.4f}")
    print(f"Optimal threshold (Youden's J): {opt_thresh:.3f}")

    # At optimal threshold
    preds = (probs >= opt_thresh).astype(int)
    cm = confusion_matrix(labels, preds)
    print("\nClassification Report (at optimal threshold):")
    print(classification_report(labels, preds, target_names=["No Cough", "Cough"]))

    # ── Plots ──
    print("Saving plots to models/...")
    plot_roc(fpr, tpr, roc_auc_val, MODEL_DIR / "roc_curve.png")
    plot_pr(precision_curve, recall_curve, ap, MODEL_DIR / "pr_curve.png")
    plot_confusion(cm, MODEL_DIR / "confusion_matrix.png")

    # ── Threshold sweep ──
    threshold_sweep_table(probs, labels)

    # ── Save recommended threshold ──
    # Use a slightly more conservative threshold than optimal for real-world use
    recommended = min(opt_thresh + 0.05, 0.85)
    print(f"\nRecommended threshold for real-time use: {recommended:.2f}")
    print("(You can override this with --threshold in realtime_detect.py)")

    config = {
        "recommended_threshold": float(recommended),
        "optimal_threshold_youdens_j": float(opt_thresh),
        "roc_auc": float(roc_auc_val),
        "average_precision": float(ap),
        "feature_params": feature_params,
    }
    with open(MODEL_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {MODEL_DIR / 'config.json'}")


if __name__ == "__main__":
    main()
