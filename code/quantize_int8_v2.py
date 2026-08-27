#!/usr/bin/env python3
"""Compare FP32-PReLU, FP32-ReLU, dynamic INT8, and static INT8 ReLU variants."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from semg_network_small import Network_Small, Network_Small_Quantizable
from uci_data_loader import UCI_SEMG_Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT_DIR / "models" / "uci_fp32_baseline.pt"
RELU_BASELINE_PATH = ROOT_DIR / "models" / "uci_fp32_relu_baseline.pt"
DYNAMIC_PATH = ROOT_DIR / "models" / "uci_int8_dynamic.pt"
STATIC_PATH = ROOT_DIR / "models" / "uci_int8_static.pt"
RESULTS_PATH = ROOT_DIR / "results" / "quantization_comparison_v2.txt"
NUM_CLASSES = 6
INPUT_SHAPE = (1, 1, 52, 8)


def load_processed_data(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_data():
    payload = load_processed_data(ROOT_DIR / "processed_uci_data.pkl")
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


def save_model(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def quantize_static(model: torch.nn.Module, calibration_loader: DataLoader) -> torch.nn.Module:
    from torch.ao.quantization import get_default_qconfig
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    from torch.ao.quantization.qconfig_mapping import QConfigMapping

    quantizable_model = Network_Small_Quantizable(num_classes=NUM_CLASSES)
    quantizable_model.load_state_dict(model.state_dict(), strict=True)
    quantizable_model.eval()

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
    print(f"Static calibration: running {calibration_batches} training batches with backend={backend}.")
    with torch.inference_mode():
        for batch_index, (inputs, _) in enumerate(calibration_loader):
            prepared(inputs)
            if batch_index + 1 >= calibration_batches:
                break

    return convert_fx(prepared)


def measure_peak_activation_memory(model: torch.nn.Module, input_shape: tuple[int, ...] = INPUT_SHAPE) -> float:
    model.eval()
    sample = torch.zeros(input_shape, dtype=torch.float32)

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            record_shapes=False,
        ) as profiler:
            with torch.inference_mode():
                model(sample)

        prof_rows = list(profiler.key_averages())
        if prof_rows:
            peak_bytes = 0
            for row in prof_rows:
                value = getattr(row, "self_cpu_memory_usage", None)
                if value is None:
                    value = getattr(row, "cpu_memory_usage", None)
                if value is not None:
                    peak_bytes = max(int(value), peak_bytes)
            return peak_bytes / 1024.0
    except Exception:
        pass

    peak_bytes = 0
    running_total = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal peak_bytes, running_total
        tensors = output if isinstance(output, (tuple, list)) else [output]
        live_bytes = 0
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                live_bytes += int(tensor.numel() * tensor.element_size())
        running_total += live_bytes
        peak_bytes = max(peak_bytes, running_total)

    for module in model.modules():
        if list(module.children()):
            continue
        if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.PReLU, torch.nn.MaxPool2d, torch.nn.Linear, torch.nn.Dropout2d)):
            handles.append(module.register_forward_hook(hook))

    with torch.inference_mode():
        model(sample)

    for handle in handles:
        handle.remove()

    return peak_bytes / 1024.0


def load_baseline_model(path: Path, model_cls):
    model = model_cls(num_classes=NUM_CLASSES)
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint
    model.eval()
    return model.cpu()


def build_report_line(name: str, acc: float, size_kb: float, mem_kb: float) -> str:
    return f"{name} | {acc:.4f} | {size_kb:.2f} | {mem_kb:.2f}"


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(f"FP32-PReLU checkpoint not found: {BASELINE_PATH}")
    if not RELU_BASELINE_PATH.exists():
        raise FileNotFoundError(f"FP32-ReLU checkpoint not found: {RELU_BASELINE_PATH}")
    if not ROOT_DIR.joinpath("processed_uci_data.pkl").exists():
        raise FileNotFoundError(f"Processed UCI dataset not found: {ROOT_DIR / 'processed_uci_data.pkl'}")

    train_loader, test_loader = load_data()
    print("Step 1: loading the original FP32-PReLU and ReLU baselines")
    fp32_prelu = load_baseline_model(BASELINE_PATH, Network_Small)
    fp32_relu = load_baseline_model(RELU_BASELINE_PATH, Network_Small_Quantizable)

    print("Step 2: evaluating the FP32-PReLU baseline on the held-out test set")
    fp32_prelu_acc = evaluate_accuracy(fp32_prelu, test_loader)
    fp32_prelu_size = file_size_kb(BASELINE_PATH)
    fp32_prelu_mem = measure_peak_activation_memory(fp32_prelu)
    print(f"FP32-PReLU accuracy={fp32_prelu_acc:.4f}, size={fp32_prelu_size:.2f} KB, peak_activation_memory={fp32_prelu_mem:.2f} KB")

    print("Step 3: evaluating the FP32-ReLU baseline on the held-out test set")
    fp32_relu_acc = evaluate_accuracy(fp32_relu, test_loader)
    fp32_relu_size = file_size_kb(RELU_BASELINE_PATH)
    fp32_relu_mem = measure_peak_activation_memory(fp32_relu)
    print(f"FP32-ReLU accuracy={fp32_relu_acc:.4f}, size={fp32_relu_size:.2f} KB, peak_activation_memory={fp32_relu_mem:.2f} KB")

    print("Step 4: generating dynamic INT8 quantization from the PReLU model")
    dynamic_model = quantize_dynamic(copy.deepcopy(fp32_prelu))
    dynamic_model.eval()
    dynamic_acc = evaluate_accuracy(dynamic_model, test_loader)
    dynamic_size = file_size_kb(DYNAMIC_PATH) if DYNAMIC_PATH.exists() else 0.0
    if not DYNAMIC_PATH.exists():
        save_model(dynamic_model, DYNAMIC_PATH)
        dynamic_size = file_size_kb(DYNAMIC_PATH)
    dynamic_mem = measure_peak_activation_memory(dynamic_model)
    print(f"INT8-dynamic accuracy={dynamic_acc:.4f}, size={dynamic_size:.2f} KB, peak_activation_memory={dynamic_mem:.2f} KB")

    print("Step 5: generating static INT8 quantization from the ReLU model")
    static_model = quantize_static(copy.deepcopy(fp32_relu), train_loader)
    static_model.eval()
    static_acc = evaluate_accuracy(static_model, test_loader)
    static_size = file_size_kb(STATIC_PATH) if STATIC_PATH.exists() else 0.0
    if not STATIC_PATH.exists():
        save_model(static_model, STATIC_PATH)
        static_size = file_size_kb(STATIC_PATH)
    static_mem = measure_peak_activation_memory(static_model)
    print(f"INT8-static-ReLU accuracy={static_acc:.4f}, size={static_size:.2f} KB, peak_activation_memory={static_mem:.2f} KB")

    rows = [
        {"Model": "FP32-PReLU", "Test Accuracy": fp32_prelu_acc, "Model Size (KB)": fp32_prelu_size, "Peak Activation Memory (KB)": fp32_prelu_mem},
        {"Model": "FP32-ReLU", "Test Accuracy": fp32_relu_acc, "Model Size (KB)": fp32_relu_size, "Peak Activation Memory (KB)": fp32_relu_mem},
        {"Model": "INT8-dynamic", "Test Accuracy": dynamic_acc, "Model Size (KB)": dynamic_size, "Peak Activation Memory (KB)": dynamic_mem},
        {"Model": "INT8-static-ReLU", "Test Accuracy": static_acc, "Model Size (KB)": static_size, "Peak Activation Memory (KB)": static_mem},
    ]

    lines = [
        "Model | Test Accuracy | Model Size (KB) | Peak Activation Memory (KB)",
        "--- | ---: | ---: | ---:",
    ]
    for row in rows:
        lines.append(
            f"{row['Model']} | {row['Test Accuracy']:.4f} | {row['Model Size (KB)']:.2f} | {row['Peak Activation Memory (KB)']:.2f}"
        )
    output = "\n".join(lines) + "\n"

    print("\nStep 6: final comparison table")
    print(output, end="")
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(output, encoding="utf-8")
    print(f"Saved final comparison table to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
