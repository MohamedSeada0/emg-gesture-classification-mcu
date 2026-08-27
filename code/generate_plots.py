#!/usr/bin/env python3
"""Generate live-data figures for the UCI EMG model and quantization comparison."""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from torch.utils.data import DataLoader

from quantize_int8 import (
    BASELINE_PATH,
    DATA_PATH,
    DYNAMIC_PATH,
    STATIC_PATH,
    NUM_CLASSES,
    benchmark_single_sample,
    file_size_kb,
    quantize_dynamic,
    quantize_static,
    save_model,
)
from semg_network import Network_XL
from semg_network_small import Network_Small
from uci_data_loader import UCI_SEMG_Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT_DIR / "results" / "training_history.json"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
INPUT_SHAPE = (1, 1, 52, 8)
CLASS_LABELS = ["rest", "fist", "wrist flexion", "wrist extension", "radial deviation", "ulnar deviation"]


def load_payload():
    with open(DATA_PATH, "rb") as handle:
        return pickle.load(handle)


def load_baseline():
    model = Network_Small(num_classes=NUM_CLASSES)
    checkpoint = torch.load(BASELINE_PATH, map_location="cpu")
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    return model.cpu().eval()


def evaluate_predictions(model, loader):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.inference_mode():
        for inputs, targets in loader:
            predictions = model(inputs).argmax(dim=1)
            all_targets.extend(targets.numpy().tolist())
            all_predictions.extend(predictions.numpy().tolist())

    targets = np.asarray(all_targets, dtype=np.int64)
    predictions = np.asarray(all_predictions, dtype=np.int64)
    accuracy = float(np.mean(targets == predictions))
    class_accuracy = []
    for class_index in range(NUM_CLASSES):
        class_mask = targets == class_index
        class_accuracy.append(float(np.mean(predictions[class_mask] == targets[class_mask])) if np.any(class_mask) else 0.0)
    return accuracy, np.asarray(class_accuracy, dtype=np.float64), targets, predictions


def save_figure(figure, filename: str, source: str):
    path = FIGURES_DIR / filename
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Generated {path.name} from {source}.")


def model_results(models, test_loader):
    results = {}
    for name, model, path in models:
        accuracy, class_accuracy, targets, predictions = evaluate_predictions(model, test_loader)
        mean_ms, std_ms = benchmark_single_sample(model, runs=100)
        size_kb = file_size_kb(path)
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "class_accuracy": class_accuracy,
            "targets": targets,
            "predictions": predictions,
            "size_kb": size_kb,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        }
        print(
            f"Live {name}: accuracy={accuracy:.4f}, per_class_acc={class_accuracy.tolist()}, "
            f"size={size_kb:.2f} KB, inference={mean_ms:.4f} +/- {std_ms:.4f} ms/sample"
        )
    return results


def plot_accuracy_comparison(results):
    names = list(results)
    values = [results[name]["accuracy"] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(names, values, color=["#3568a8", "#4f9d69", "#d97941"])
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Test Accuracy Comparison")
    axis.tick_params(axis="x", rotation=15)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    save_figure(figure, "accuracy_comparison.png", "fresh full-test evaluation of all three loaded/quantized models")


def plot_size_comparison(results):
    names = list(results)
    values = [results[name]["size_kb"] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(names, values, color=["#3568a8", "#4f9d69", "#d97941"])
    axis.set_ylabel("File size (KB)")
    axis.set_title("Saved Model File Size")
    axis.tick_params(axis="x", rotation=15)
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:.1f}", ha="center", va="bottom")
    save_figure(figure, "size_comparison.png", "os.path.getsize values from the saved model files")


def plot_inference_comparison(results):
    names = list(results)
    means = [results[name]["mean_ms"] for name in names]
    deviations = [results[name]["std_ms"] for name in names]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(names, means, yerr=deviations, capsize=5, color=["#3568a8", "#4f9d69", "#d97941"])
    axis.set_ylabel("Inference time per sample (ms)")
    axis.set_title("Single-Sample Inference Time")
    axis.tick_params(axis="x", rotation=15)
    save_figure(figure, "inference_time_comparison.png", "100 fresh batch-size-one timing runs per model")


def plot_accuracy_vs_size(results):
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, result in results.items():
        axis.scatter(result["size_kb"], result["accuracy"], s=100, label=name)
        axis.annotate(name, (result["size_kb"], result["accuracy"]), xytext=(6, 6), textcoords="offset points")
    axis.set_xlabel("File size (KB)")
    axis.set_ylabel("Test accuracy")
    axis.set_title("Accuracy vs. Saved Model Size")
    axis.legend()
    save_figure(figure, "accuracy_vs_size_tradeoff.png", "the same fresh accuracy and saved-file-size measurements")


def plot_training_curves():
    with open(HISTORY_PATH, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    best_entry = min(history, key=lambda entry: entry["val_loss"])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, train_loss, label="train loss")
    axis.plot(epochs, val_loss, label="validation loss")
    axis.scatter([best_entry["epoch"]], [best_entry["val_loss"]], color="red", label=f"best checkpoint epoch {best_entry['epoch']}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training and Validation Loss")
    axis.legend()
    save_figure(figure, "training_curves.png", "results/training_history.json saved by train_uci.py")


def plot_per_class_accuracy(results):
    class_accuracy = results["FP32 baseline"]["class_accuracy"]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(CLASS_LABELS, class_accuracy, color="#3568a8")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("FP32 Baseline Per-Class Test Accuracy")
    axis.tick_params(axis="x", rotation=25)
    save_figure(figure, "per_class_accuracy.png", "fresh FP32 predictions on the full test set")


def plot_dataset_distribution(payload):
    split_names = ["train", "val", "test"]
    split_windows = [payload[f"{name}_windows"] for name in split_names]
    counts = np.asarray(
        [[sum(int(label) == class_index for _, label in windows) for class_index in range(NUM_CLASSES)] for windows in split_windows],
        dtype=np.int64,
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    positions = np.arange(NUM_CLASSES)
    width = 0.25
    for split_index, split_name in enumerate(split_names):
        axis.bar(positions + (split_index - 1) * width, counts[split_index], width, label=split_name)
    axis.set_xticks(positions, CLASS_LABELS, rotation=25, ha="right")
    axis.set_ylabel("Window count")
    axis.set_title("Dataset Class Distribution")
    axis.legend()
    save_figure(figure, "dataset_class_distribution.png", "class counts read directly from processed_uci_data.pkl")


def module_description(name, module):
    description = [name or "Network_XL"]
    if isinstance(module, torch.nn.Conv2d):
        description.append(f"Conv2d {module.in_channels}->{module.out_channels} k={tuple(module.kernel_size)}")
    elif isinstance(module, torch.nn.Linear):
        description.append(f"Linear {module.in_features}->{module.out_features}")
    elif isinstance(module, torch.nn.BatchNorm2d):
        description.append(f"BatchNorm2d {module.num_features}")
    elif isinstance(module, torch.nn.MaxPool2d):
        description.append(f"MaxPool2d k={module.kernel_size}")
    else:
        description.append(type(module).__name__)
    return "\n".join(description)


def plot_model_architecture():
    model = Network_XL(7).eval()
    layer_modules = [
        (name, module) for name, module in model.named_modules()
        if name and len(list(module.children())) == 0
    ]
    figure, axis = plt.subplots(figsize=(max(12, len(layer_modules) * 1.8), 4.5))
    axis.axis("off")
    axis.set_title("Network_XL Architecture from Live Module Inspection")
    for index, (name, module) in enumerate(layer_modules):
        x_position = index * 1.8
        box = FancyBboxPatch(
            (x_position, 1.2), 1.45, 1.5, boxstyle="round,pad=0.03", facecolor="#e5eef8", edgecolor="#3568a8"
        )
        axis.add_patch(box)
        axis.text(x_position + 0.725, 1.95, module_description(name, module), ha="center", va="center", fontsize=7)
        if index < len(layer_modules) - 1:
            axis.annotate("", xy=(x_position + 1.8, 1.95), xytext=(x_position + 1.45, 1.95), arrowprops={"arrowstyle": "->"})
    axis.set_xlim(-0.2, len(layer_modules) * 1.8)
    axis.set_ylim(0.8, 3.2)
    save_figure(figure, "model_architecture_diagram.png", "Network_XL.named_modules() layer types and parameters")


def plot_model_size_reduction():
    xl_model = Network_XL(7)
    small_model = Network_Small(6)
    xl_parameters = sum(parameter.numel() for parameter in xl_model.parameters())
    small_parameters = sum(parameter.numel() for parameter in small_model.parameters())
    reduction_percent = (xl_parameters - small_parameters) / xl_parameters * 100.0

    figure, axis = plt.subplots(figsize=(8, 5))
    names = ["Network_XL", "Network_Small"]
    values = [xl_parameters, small_parameters]
    axis.bar(names, values, color=["#3568a8", "#d97941"])
    axis.set_ylabel("Parameter count")
    axis.set_title(f"Model Size Reduction: {reduction_percent:.2f}%")
    for index, value in enumerate(values):
        axis.text(index, value, f"{value:,}", ha="center", va="bottom")
    save_figure(figure, "model_size_reduction_diagram.png", "live parameter counts from Network_XL(7) and Network_Small(6)")


def main():
    required_files = [BASELINE_PATH, DATA_PATH, HISTORY_PATH]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Required source files are missing: " + ", ".join(missing))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    payload = load_payload()
    train_loader = DataLoader(UCI_SEMG_Dataset(payload["train_windows"], augment=False), batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(UCI_SEMG_Dataset(payload["test_windows"], augment=False), batch_size=256, shuffle=False, num_workers=0)
    baseline_model = load_baseline()
    dynamic_model = quantize_dynamic(load_baseline())
    static_model = quantize_static(load_baseline(), train_loader)
    save_model(dynamic_model, DYNAMIC_PATH)
    save_model(static_model, STATIC_PATH)

    models = [
        ("FP32 baseline", baseline_model, BASELINE_PATH),
        ("int8_dynamic", dynamic_model, DYNAMIC_PATH),
        ("int8_static", static_model, STATIC_PATH),
    ]
    results = model_results(models, test_loader)

    plot_accuracy_comparison(results)
    plot_size_comparison(results)
    plot_inference_comparison(results)
    plot_accuracy_vs_size(results)
    plot_training_curves()
    plot_per_class_accuracy(results)
    plot_dataset_distribution(payload)
    plot_model_architecture()
    plot_model_size_reduction()
    print(f"All figures saved to {FIGURES_DIR}.")


if __name__ == "__main__":
    main()
