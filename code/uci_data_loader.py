#!/usr/bin/env python3
"""UCI EMG gesture data loader for the EMG_data_for_gestures-master dataset."""

from __future__ import annotations

import csv
import pickle
import random
from collections import Counter
from pathlib import Path
from statistics import mode
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils import data


def _detect_delimiter(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if "\t" in first_line:
            return "\t"
        if "," in first_line:
            return ","
    raise ValueError(f"Could not detect delimiter for file: {path}")


def load_all_subjects(root_dir: str | Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Load all raw_data_*.txt files under the UCI dataset root.

    Returns a list of (subject_id, emg_array, label_array) tuples.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    all_subject_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
    subject_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])

    for subject_dir in subject_dirs:
        files = sorted(subject_dir.glob("*_raw_data_*.txt"))
        if not files:
            continue

        for raw_file in files:
            delimiter = _detect_delimiter(raw_file)
            rows = []
            with open(raw_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                next(reader, None)
                for row in reader:
                    if not row or all(cell.strip() == "" for cell in row):
                        continue
                    if len(row) < 10:
                        continue
                    try:
                        values = [float(cell.strip()) for cell in row[:10]]
                    except ValueError:
                        continue
                    rows.append(values)

            if not rows:
                raise ValueError(f"No valid rows parsed from {raw_file}")

            data_raw = np.asarray(rows, dtype=np.float32)
            if data_raw.ndim == 1:
                data_raw = data_raw.reshape(1, -1)
            if data_raw.shape[1] < 10:
                raise ValueError(f"Expected at least 10 columns in {raw_file}, got {data_raw.shape}")

            emg = data_raw[:, 1:9].astype(np.float32)
            labels = data_raw[:, 9].astype(np.int64)
            all_subject_data.append((subject_dir.name, emg, labels))

    if not all_subject_data:
        raise ValueError(f"No raw_data_*.txt files were found under: {root_dir}")

    return all_subject_data


def create_windows(
    emg_array: np.ndarray,
    label_array: np.ndarray,
    window_size: int = 52,
    step: int = 5,
    drop_zero: bool = True,
    rest_keep_fraction: float = 0.1,
    rng_seed: Optional[int] = None,
) -> List[Tuple[np.ndarray, int]]:
    """Slide a window over the EMG stream and assign the majority class per window.

    This mirrors the original repo's logic: majority vote over the window, drop zero/unmarked data,
    and keep only a small random fraction of rest windows (label==1) if requested. Gesture class 7
    (extended palm) is removed entirely because it is not represented reliably enough across subjects.
    """
    if emg_array.shape[0] != label_array.shape[0]:
        raise ValueError(
            f"EMG shape {emg_array.shape} and label shape {label_array.shape} do not match: "
            "same number of time samples required."
        )
    if emg_array.shape[1] != 8:
        raise ValueError(f"Expected 8 EMG channels, got shape {emg_array.shape}")
    if window_size <= 0 or step <= 0:
        raise ValueError(f"window_size and step must be positive; got {window_size}, {step}")
    if not 0.0 <= rest_keep_fraction <= 1.0:
        raise ValueError(f"rest_keep_fraction must be in [0,1], got {rest_keep_fraction}")

    rng = np.random.RandomState(rng_seed)
    windows: List[Tuple[np.ndarray, int]] = []

    for start in range(0, len(emg_array) - window_size + 1, step):
        segment = emg_array[start : start + window_size]
        segment_labels = label_array[start : start + window_size]
        majority_label = int(mode(segment_labels.astype(int).tolist()))

        if majority_label in (0, 7):
            continue
        if drop_zero and majority_label == 0:
            continue
        if majority_label == 1 and rest_keep_fraction < 1.0:
            if rng.rand() > rest_keep_fraction:
                continue

        # The usable gesture classes are 1..6; map them to 0..5 for a six-class network.
        remapped_label = majority_label - 1
        windows.append((segment.copy(), remapped_label))

    return windows


def normalize_windows(
    windows: Sequence[Tuple[np.ndarray, int]],
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[List[Tuple[np.ndarray, int]], float, float]:
    """Apply rectification + z-score normalization.

    If mean/std are not provided, calculate them from the windows and return them for reuse on val/test.
    """
    if not windows:
        raise ValueError("Cannot normalize empty window list")

    raw_windows = np.stack([np.abs(window) for window, _ in windows], axis=0)
    if mean is None or std is None:
        mean = float(raw_windows.mean())
        std = float(raw_windows.std())
        if std == 0.0:
            std = 1.0

    normalized = []
    for window, label in windows:
        rectified = np.abs(window).astype(np.float32)
        normalized_window = (rectified - mean) / std
        normalized.append((normalized_window.astype(np.float32), int(label)))

    return normalized, mean, std


def subject_wise_split(
    subject_data: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Split by subject ID so each subject appears in only one set.

    Class 7 is removed from the dataset entirely; this split is now a simple subject-wise ratio split.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    subject_ids = sorted({subject_id for subject_id, _, _ in subject_data})
    rng = random.Random(seed)
    shuffled = subject_ids[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if test_count < 0:
        val_count = max(0, val_count + test_count)
        test_count = total - train_count - val_count

    if train_count == 0 and total > 0:
        train_count = 1
        val_count = max(0, int(total * val_ratio))
        test_count = total - train_count - val_count

    train_subjects = shuffled[:train_count]
    val_subjects = shuffled[train_count : train_count + val_count]
    test_subjects = shuffled[train_count + val_count :]

    return train_subjects, val_subjects, test_subjects


class UCI_SEMG_Dataset(data.Dataset):
    """Dataset class for UCI EMG windows. Returns samples as [1, window_size, 8] tensors."""

    def __init__(self, windows: Sequence[Tuple[np.ndarray, int]], augment: bool = False):
        self.windows = list(windows)
        self.augment = augment

    def _augment_window(self, window: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            return window

        signal_std = float(np.std(window))
        noise_std = 0.02 * signal_std if signal_std > 0 else 0.02
        noise = np.random.normal(0.0, noise_std, size=window.shape).astype(np.float32)
        augmented = window + noise

        scale = np.random.uniform(0.9, 1.1)
        augmented = augmented * scale
        return augmented.astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        window, label = self.windows[index]
        window = np.asarray(window, dtype=np.float32)
        if window.shape != (52, 8):
            raise ValueError(f"Window shape mismatch at index {index}: expected (52, 8), got {window.shape}")

        if self.augment:
            window = self._augment_window(window)

        sample = torch.from_numpy(window).float().unsqueeze(0)  # [1, 52, 8]
        target = torch.tensor(int(label), dtype=torch.long)
        return sample, target


def _summarize_split(split_name: str, windows: Sequence[Tuple[np.ndarray, int]]) -> None:
    counts = Counter(int(label) for _, label in windows)
    print(f"{split_name}: {len(windows)} windows")
    print(f"  class counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1] / "EMG_data_for_gestures-master"
    processed_path = Path(__file__).resolve().parents[1] / "processed_uci_data.pkl"

    subject_data = load_all_subjects(root_dir)
    train_subjects, val_subjects, test_subjects = subject_wise_split(subject_data, seed=42)

    train_windows: List[Tuple[np.ndarray, int]] = []
    val_windows: List[Tuple[np.ndarray, int]] = []
    test_windows: List[Tuple[np.ndarray, int]] = []

    for subject_id, emg_array, label_array in subject_data:
        subject_windows = create_windows(emg_array, label_array, window_size=52, step=5, drop_zero=True, rest_keep_fraction=0.1, rng_seed=42)
        if subject_id in train_subjects:
            train_windows.extend(subject_windows)
        elif subject_id in val_subjects:
            val_windows.extend(subject_windows)
        elif subject_id in test_subjects:
            test_windows.extend(subject_windows)

    print(f"subject split counts: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
    if len(train_windows) == 0:
        raise RuntimeError("Training set is empty after windowing. Check dataset structure or split logic.")

    train_windows, mean, std = normalize_windows(train_windows)
    val_windows, _, _ = normalize_windows(val_windows, mean, std)
    test_windows, _, _ = normalize_windows(test_windows, mean, std)

    _summarize_split("train", train_windows)
    _summarize_split("val", val_windows)
    _summarize_split("test", test_windows)

    processed = {
        "train_windows": train_windows,
        "val_windows": val_windows,
        "test_windows": test_windows,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "test_subjects": test_subjects,
        "mean": mean,
        "std": std,
    }
    with open(processed_path, "wb") as f:
        pickle.dump(processed, f)
    print(f"Saved processed dataset to {processed_path}")
