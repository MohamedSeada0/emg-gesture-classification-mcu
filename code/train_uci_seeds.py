from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from semg_network_small import Network_Small
from train_uci import build_dataloaders, compute_class_weights, evaluate_model, load_processed_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "seed_results.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FP32 PReLU UCI baseline with a fixed seed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="processed_uci_data.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", "--learning_rate", type=float, default=5e-4, dest="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--use_augmentation", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data_path)
    if not data_path.exists():
        data_path = ROOT / args.data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")

    payload = load_processed_data(data_path)
    train_windows = payload["train_windows"]
    val_windows = payload["val_windows"]
    test_windows = payload["test_windows"]
    train_loader, val_loader, test_loader = build_dataloaders(
        train_windows, val_windows, test_windows, args.batch_size, use_augmentation=args.use_augmentation
    )

    num_classes = 6
    class_weights = compute_class_weights(train_windows, num_classes=num_classes)
    device = torch.device(args.device)
    model = Network_Small(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if args.use_class_weights else None)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            model.eval()
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)

        val_loss = val_loss / max(val_total, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        best_state = model.state_dict()
    model.load_state_dict(best_state)
    model.eval()
    test_acc, _, _ = evaluate_model(model, test_loader, device)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(f"seed={args.seed}, accuracy={test_acc:.4f}\n")

    print(f"seed={args.seed}, accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()
