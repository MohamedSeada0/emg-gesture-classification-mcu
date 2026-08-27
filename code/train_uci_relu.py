#!/usr/bin/env python3
"""Train the ReLU-based Network_Small_Quantizable baseline on the processed UCI EMG dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from semg_network_small import Network_Small_Quantizable
from train_uci import build_dataloaders, evaluate_model, format_per_class_accuracy, load_processed_data, per_class_accuracy
from uci_data_loader import UCI_SEMG_Dataset

try:
    from sklearn.metrics import confusion_matrix
except Exception:  # pragma: no cover
    confusion_matrix = None


SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(train_windows, num_classes: int = 6):
    counts = np.zeros(num_classes, dtype=np.float64)
    for _, label in train_windows:
        label_idx = int(label)
        if 0 <= label_idx < num_classes:
            counts[label_idx] += 1.0

    counts = np.maximum(counts, 1.0)
    inv_freq = np.sum(counts) / (num_classes * counts)
    return torch.tensor(inv_freq, dtype=torch.float32)


def main():
    set_seed(SEED)

    parser = argparse.ArgumentParser(description="Train the ReLU-based small CNN on the UCI EMG dataset")
    parser.add_argument("--data_path", type=str, default="processed_uci_data.pkl", help="Path to processed pickle from uci_data_loader.py")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", "--learning_rate", type=float, default=5e-4, dest="learning_rate", help="Base learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience in epochs without lower validation loss.")
    parser.add_argument("--use_class_weights", action="store_true", help="Use inverse-frequency class weights in CrossEntropyLoss.")
    parser.add_argument("--use_augmentation", action="store_true", help="Enable lightweight training-only data augmentation on EMG windows.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {data_path}. Run 'python code/uci_data_loader.py' first."
        )

    payload = load_processed_data(data_path)
    train_windows = payload["train_windows"]
    val_windows = payload["val_windows"]
    test_windows = payload["test_windows"]

    if len(train_windows) == 0 or len(val_windows) == 0 or len(test_windows) == 0:
        raise ValueError("Train/val/test windows must all be non-empty for training.")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_windows, val_windows, test_windows, args.batch_size, use_augmentation=args.use_augmentation
    )

    num_classes = 6
    class_weights = compute_class_weights(train_windows, num_classes=num_classes)
    if args.use_class_weights:
        print(f"Using class weights for {num_classes} classes: {class_weights.tolist()}")
    else:
        print("Using unweighted CrossEntropyLoss.")

    model = Network_Small_Quantizable(num_classes=num_classes)
    device = torch.device(args.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if args.use_class_weights else None)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    training_history = []
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "training_history_relu.json"

    print(f"Training ReLU baseline with seed={SEED}, lr={args.learning_rate}, patience={args.patience}, batch_size={args.batch_size}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            if logits.shape != (inputs.shape[0], num_classes):
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}; expected (batch, {num_classes})")
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
            running_total += inputs.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss = 0.0
        val_total = 0
        val_correct = 0
        val_targets = []
        val_preds = []
        with torch.no_grad():
            model.eval()
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)
                val_correct += (torch.argmax(logits, dim=1) == targets).sum().item()
                val_targets.extend(targets.detach().cpu().tolist())
                val_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        val_class_acc, _, _ = per_class_accuracy(val_targets, val_preds, num_classes=num_classes)
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "per_class_acc": val_class_acc.tolist(),
            }
        )
        history_path.write_text(json.dumps(training_history, indent=2), encoding="utf-8")
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"val_per_class_acc=[{format_per_class_accuracy(val_class_acc)}]"
        )

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(val_loss)

        if epochs_without_improvement >= args.patience:
            stopped_early = True
            print(
                f"Early stopping triggered at epoch {epoch:02d}: no validation loss improvement for {args.patience} consecutive epochs. "
                f"Best val_loss was {best_val_loss:.4f} at epoch {best_epoch:02d}."
            )
            break

    model_dir = Path(__file__).resolve().parents[1] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / "uci_fp32_relu_baseline.pt"
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, save_path)
    print(f"Saved ReLU model checkpoint to {save_path} at epoch {best_epoch:02d} with lowest validation loss {best_val_loss:.4f}")
    if stopped_early:
        print(f"Training stopped early at epoch {epoch:02d} due to patience={args.patience}.")

    model.load_state_dict(best_state)
    model.eval()
    test_acc, test_targets, test_preds = evaluate_model(model, test_loader, device)
    class_acc, _, _ = per_class_accuracy(test_targets, test_preds, num_classes=num_classes)

    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Per-class accuracy: {format_per_class_accuracy(class_acc)}")
    if confusion_matrix is not None:
        cm = confusion_matrix(test_targets, test_preds, labels=list(range(num_classes)))
        print("Confusion matrix:")
        print(cm)
    else:
        print("sklearn is not available; confusion matrix skipped.")


if __name__ == "__main__":
    main()
