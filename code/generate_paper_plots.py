#!/usr/bin/env python3
"""Generate paper-ready accuracy and confusion-matrix figures from live UCI results."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
except ImportError as exc:
    print(
        "scikit-learn is required for confusion matrices and metrics. "
        "Install it with: python -m pip install scikit-learn",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

from quantize_int8 import (
    BASELINE_PATH,
    DATA_PATH,
    DYNAMIC_PATH,
    STATIC_PATH,
    NUM_CLASSES,
    quantize_dynamic,
    quantize_static,
)
from semg_network_small import Network_Small
from uci_data_loader import UCI_SEMG_Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT_DIR / "results" / "training_history.json"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
REPORT_PATH = ROOT_DIR / "results" / "confusion_matrices_report.txt"
CLASS_LABELS = ["Rest", "Fist", "Wrist Flex", "Wrist Ext", "Radial Dev", "Ulnar Dev"]


def load_baseline():
    model = Network_Small(num_classes=NUM_CLASSES)
    checkpoint = torch.load(BASELINE_PATH, map_location="cpu")
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    return model.cpu().eval()


def load_test_loader():
    with open(DATA_PATH, "rb") as handle:
        payload = pickle.load(handle)
    dataset = UCI_SEMG_Dataset(payload["test_windows"], augment=False)
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)


def predict(model, loader):
    model.eval()
    targets = []
    predictions = []
    with torch.inference_mode():
        for inputs, batch_targets in loader:
            batch_predictions = model(inputs).argmax(dim=1)
            targets.extend(batch_targets.cpu().numpy().tolist())
            predictions.extend(batch_predictions.cpu().numpy().tolist())
    return np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)


def reconstruct_models():
    baseline = load_baseline()
    with open(DATA_PATH, "rb") as handle:
        payload = pickle.load(handle)
    calibration_loader = DataLoader(
        UCI_SEMG_Dataset(payload["train_windows"], augment=False),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    dynamic = quantize_dynamic(load_baseline()).eval()
    static = quantize_static(load_baseline(), calibration_loader).eval()
    return [
        ("FP32 Baseline", baseline),
        ("int8_dynamic", dynamic),
        ("int8_static", static),
    ]


def save_figure(figure, filename: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Generated {path} from live/file data.")


def generate_accuracy_curve():
    with open(HISTORY_PATH, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    epochs = [entry["epoch"] for entry in history]
    train_accuracy = [100.0 * entry["train_acc"] for entry in history]
    validation_accuracy = [100.0 * entry["val_acc"] for entry in history]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_accuracy, color="black", linewidth=2, label="Training")
    axis.plot(epochs, validation_accuracy, color="red", linewidth=2, label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Training and Validation Accuracy — FP32 Baseline (Network_Small)")
    axis.legend()
    axis.grid(True, color="#d9d9d9", linewidth=0.8)
    save_figure(figure, "accuracy_curve.png")


def generate_confusion_matrices(models, test_loader):
    report_lines = []
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    labels = np.arange(NUM_CLASSES)

    for axis, (name, model) in zip(axes, models):
        targets, predictions = predict(model, test_loader)
        matrix = confusion_matrix(targets, predictions, labels=labels)
        accuracy = accuracy_score(targets, predictions)
        macro_f1 = f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)

        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        axis.set_title(f"{name}\nAcc={accuracy:.4f}, F1_macro={macro_f1:.4f}")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(labels, CLASS_LABELS, rotation=35, ha="right")
        axis.set_yticks(labels, CLASS_LABELS)
        threshold = matrix.max() / 2.0 if matrix.size else 0.0
        for row in range(NUM_CLASSES):
            for column in range(NUM_CLASSES):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        report_lines.extend(
            [
                name,
                f"accuracy={accuracy:.8f}",
                f"f1_macro={macro_f1:.8f}",
                "labels=" + ", ".join(f"{index}:{label}" for index, label in enumerate(CLASS_LABELS)),
                "confusion_matrix:",
                np.array2string(matrix),
                "",
            ]
        )
        print(f"{name}: accuracy={accuracy:.8f}, macro F1={macro_f1:.8f}")
        print(matrix)

    save_figure(figure, "confusion_matrices.png")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote raw confusion matrices and F1 scores to {REPORT_PATH} from live test predictions.")


def main():
    required_files = [BASELINE_PATH, DATA_PATH, HISTORY_PATH]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Required files are missing: " + ", ".join(missing))

    test_loader = load_test_loader()
    models = reconstruct_models()
    generate_accuracy_curve()
    generate_confusion_matrices(models, test_loader)


if __name__ == "__main__":
    main()
