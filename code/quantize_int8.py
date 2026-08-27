#!/usr/bin/env python3
"""Compare FP32, dynamic INT8, and static FX INT8 UCI EMG models."""

from __future__ import annotations

import copy
import pickle
import time
import traceback
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch.utils.data import DataLoader

from semg_network_small import Network_Small, Network_Small_Quantizable
from uci_data_loader import UCI_SEMG_Dataset


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT_DIR / "models" / "uci_fp32_baseline.pt"
DATA_PATH = ROOT_DIR / "processed_uci_data.pkl"
RESULTS_PATH = ROOT_DIR / "results" / "quantization_comparison.txt"
DYNAMIC_PATH = ROOT_DIR / "models" / "uci_int8_dynamic.pt"
STATIC_PATH = ROOT_DIR / "models" / "uci_int8_static.pt"
NUM_CLASSES = 6
INPUT_SHAPE = (1, 1, 52, 8)


def load_baseline() -> Network_Small:
    model = Network_Small(num_classes=NUM_CLASSES)
    checkpoint = torch.load(BASELINE_PATH, map_location="cpu")
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    model.eval()
    return model.cpu()


def load_data():
    with open(DATA_PATH, "rb") as handle:
        payload = pickle.load(handle)

    train_dataset = UCI_SEMG_Dataset(payload["train_windows"], augment=False)
    test_dataset = UCI_SEMG_Dataset(payload["test_windows"], augment=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


def file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0


def evaluate_accuracy(model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += int((predictions == targets).sum())
            total += int(targets.numel())
    return correct / max(total, 1)


def benchmark_single_sample(model: torch.nn.Module, runs: int = 100) -> tuple[float, float]:
    model.eval()
    sample = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(10):
            model(sample)

        elapsed_ms = []
        for _ in range(runs):
            start = time.perf_counter()
            model(sample)
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    return mean(elapsed_ms), stdev(elapsed_ms) if len(elapsed_ms) > 1 else 0.0


def save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )


def quantize_static(model: torch.nn.Module, calibration_loader: DataLoader) -> torch.nn.Module:
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from torch.ao.quantization.qconfig_mapping import QConfigMapping

    quantizable_model = Network_Small_Quantizable(num_classes=NUM_CLASSES)
    baseline_state = model.state_dict()
    quantizable_state = {
        name: value for name, value in baseline_state.items()
        if name in quantizable_model.state_dict()
    }
    quantizable_model.load_state_dict(quantizable_state, strict=False)
    quantizable_model.eval()
    print("Static quantization uses ReLU; dynamic quantization uses the original PReLU model.")

    supported_engines = list(torch.backends.quantized.supported_engines)
    backend = next((candidate for candidate in ("x86", "fbgemm", "qnnpack", "onednn") if candidate in supported_engines), None)
    if backend is None:
        raise RuntimeError(f"No supported static quantization backend found. Available engines: {supported_engines}")
    torch.backends.quantized.engine = backend
    qconfig = get_default_qconfig(backend)
    qconfig_mapping = (
        QConfigMapping()
        .set_global(None)
        .set_object_type(torch.nn.Conv2d, qconfig)
        .set_object_type(torch.nn.ReLU, qconfig)
        .set_object_type(torch.nn.Linear, qconfig)
    )
    prepared = prepare_fx(quantizable_model, qconfig_mapping, example_inputs=(torch.zeros(INPUT_SHAPE),))

    calibration_batches = min(1000, len(calibration_loader))
    print(f"Static calibration: running {calibration_batches} training batches (backend={backend}).")
    with torch.inference_mode():
        for batch_index, (inputs, _) in enumerate(calibration_loader):
            prepared(inputs)
            if batch_index + 1 >= calibration_batches:
                break

    return convert_fx(prepared)


def measure_variant(name: str, model: torch.nn.Module, path: Path | None, test_loader: DataLoader):
    if path is not None:
        save_model(model, path)
    accuracy = evaluate_accuracy(model, test_loader)
    mean_ms, std_ms = benchmark_single_sample(model)
    size_path = path if path is not None else BASELINE_PATH
    size_kb = file_size_kb(size_path)
    print(
        f"{name}: accuracy={accuracy:.4f}, size={size_kb:.2f} KB "
        f"({size_kb / 1024.0:.2f} MB), inference={mean_ms:.4f} +/- {std_ms:.4f} ms/sample"
    )
    return {
        "name": name,
        "accuracy": accuracy,
        "size_kb": size_kb,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
    }


def format_summary(results):
    lines = [
        "Model | Accuracy | Size (KB) | Inference time per sample (ms)",
        "--- | ---: | ---: | ---:",
    ]
    for result in results:
        if result is None:
            continue
        if "error" in result:
            lines.append(f"{result['name']} | ERROR | N/A | N/A")
            continue
        lines.append(
            f"{result['name']} | {result['accuracy']:.4f} | {result['size_kb']:.2f} | "
            f"{result['mean_ms']:.4f} +/- {result['std_ms']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"FP32 baseline checkpoint not found: {BASELINE_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Processed UCI dataset not found: {DATA_PATH}")

    train_loader, test_loader = load_data()
    baseline_model = load_baseline()
    baseline_size_kb = file_size_kb(BASELINE_PATH)
    print(f"FP32 baseline file size: {baseline_size_kb:.2f} KB ({baseline_size_kb / 1024.0:.2f} MB)")

    results = []
    results.append(measure_variant("FP32 baseline", baseline_model, None, test_loader))

    dynamic_result = None
    try:
        dynamic_model = quantize_dynamic(copy.deepcopy(baseline_model))
        dynamic_result = measure_variant("int8_dynamic", dynamic_model, DYNAMIC_PATH, test_loader)
    except Exception:
        print("ERROR during dynamic quantization or evaluation:")
        traceback.print_exc()
        dynamic_result = {"name": "int8_dynamic", "error": True}
    results.append(dynamic_result)

    static_result = None
    try:
        static_model = quantize_static(copy.deepcopy(baseline_model), train_loader)
        static_result = measure_variant("int8_static", static_model, STATIC_PATH, test_loader)
    except Exception:
        print("ERROR during static FX quantization, calibration, or evaluation:")
        traceback.print_exc()
        static_result = {"name": "int8_static", "error": True}
    results.append(static_result)

    summary = format_summary(results)
    print("\nFinal quantization comparison:")
    print(summary, end="")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(summary, encoding="utf-8")
    print(f"Saved comparison table to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
