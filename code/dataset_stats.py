from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "processed_uci_data.pkl"
RESULTS_PATH = ROOT / "results" / "dataset_stats.txt"


def load_processed_data(data_path: str | Path):
    with open(data_path, "rb") as f:
        return pickle.load(f)


def summarize(windows):
    counts = Counter(int(label) for _, label in windows)
    return {int(k): int(v) for k, v in sorted(counts.items())}


def main() -> None:
    payload = load_processed_data(DATA_PATH)
    lines = []

    for split_name in ("train_windows", "val_windows", "test_windows"):
        windows = payload.get(split_name, [])
        counts = summarize(windows)
        lines.append(f"{split_name}: {len(windows)} windows")
        lines.append(f"class_distribution: {counts}")

    text = "\n".join(lines) + "\n"
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
