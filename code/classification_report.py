from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report

from semg_network_small import Network_Small
from train_uci import build_dataloaders, load_processed_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "uci_fp32_baseline.pt"
DATA_PATH = ROOT / "processed_uci_data.pkl"
RESULTS_PATH = ROOT / "results" / "classification_report.txt"
TARGET_NAMES = ["Rest", "Fist", "Wrist Flex", "Wrist Ext", "Radial Dev", "Ulnar Dev"]


def main() -> None:
    payload = load_processed_data(DATA_PATH)
    train_windows = payload["train_windows"]
    val_windows = payload["val_windows"]
    test_windows = payload["test_windows"]
    train_loader, _, test_loader = build_dataloaders(train_windows, val_windows, test_windows, batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network_Small(num_classes=6).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    y_true = []
    y_pred = []
    with torch.inference_mode():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    report = classification_report(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        target_names=TARGET_NAMES,
        digits=4,
    )
    print(report)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
