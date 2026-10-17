"""Training and evaluation pipeline for MSCANet.

The script supports two subject-independent evaluation protocols:

1. Stratified group K-fold cross-validation, where all EEG segments from the
   same participant remain in the same fold.
2. Leave-one-subject-out (LOSO) cross-validation, where one participant is
   held out for testing in each iteration.

Input files are expected to be preprocessed and segmented EEGLAB ``.set``
files. The preprocessing pipeline used in the manuscript (filtering, notch
filtering, re-referencing, artifact removal, and non-overlapping segmentation)
should therefore be completed before running this script.
"""

from __future__ import annotations

import argparse
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from thop import profile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mscanet_model import EEG_MSCA_Net

warnings.filterwarnings("ignore")


@dataclass
class ExperimentConfig:
    """Experiment settings matching the manuscript configuration."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    seed: int = 42
    n_splits: int = 10
    num_workers: int = 0
    dropout: float = 0.3


class EEGTrialDataset(Dataset):
    """Simple PyTorch dataset for segmented EEG trials."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

        if self.X.ndim != 3:
            raise ValueError(
                f"X must be three-dimensional [N, C, T], received {tuple(self.X.shape)}."
            )
        if self.y.ndim != 1:
            raise ValueError(f"y must be one-dimensional [N], received {tuple(self.y.shape)}.")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"X and y contain different numbers of samples: "
                f"X={self.X.shape[0]}, y={self.y.shape[0]}."
            )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_set_file(set_path: str, target_sfreq: float | None = None):
    """Read one epoched EEGLAB ``.set`` file with MNE."""
    try:
        epochs = mne.read_epochs_eeglab(
            set_path,
            uint16_codec="latin1",
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {set_path}. Ensure that the file contains "
            f"pre-segmented EEGLAB epochs. Original error: {exc!r}"
        ) from exc

    if target_sfreq is not None:
        epochs.resample(target_sfreq)

    data = epochs.get_data().astype(np.float32)
    if data.ndim != 3:
        raise RuntimeError(
            f"Expected [trials, channels, time_points], received {data.shape} "
            f"from {set_path}."
        )

    return data, list(epochs.ch_names), float(epochs.info["sfreq"])


def get_label_from_filename(filename: str) -> int | None:
    """Infer the binary class label from the file prefix."""
    name = os.path.basename(filename).lower()
    if name.startswith("control"):
        return 0
    if name.startswith("pd"):
        return 1
    return None


def load_all_subjects(
    data_dir: str,
    target_sfreq: float | None = None,
    per_trial_normalization: bool = False,
) -> List[Dict]:
    """Load all valid participant files from a directory.

    Files are expected to start with ``Control`` or ``PD``. All participants
    must share the same channel order, sampling frequency, and segment length.

    Args:
        data_dir: Directory containing epoched ``.set`` files.
        target_sfreq: Optional resampling frequency.
        per_trial_normalization: If True, z-score each channel within each EEG
            segment before cross-validation. Fold-wise normalization is applied
            later using training-set statistics in all cases.
    """
    data_path = Path(data_dir)
    set_files = sorted(data_path.glob("*.set"))
    if not set_files:
        raise RuntimeError(f"No .set files were found in {data_dir}.")

    subjects: List[Dict] = []
    for file_path in set_files:
        label = get_label_from_filename(file_path.name)
        if label is None:
            print(f"Skipping unrecognized file name: {file_path.name}")
            continue

        data, ch_names, sfreq = read_set_file(str(file_path), target_sfreq)

        if per_trial_normalization:
            mean = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True) + 1e-6
            data = (data - mean) / std

        subjects.append(
            {
                "subject_id": file_path.stem,
                "label": label,
                "data": data.astype(np.float32),
                "ch_names": ch_names,
                "sfreq": sfreq,
            }
        )

    if len(subjects) < 2:
        raise RuntimeError("At least two valid participants are required.")

    n_control = sum(subject["label"] == 0 for subject in subjects)
    n_pd = sum(subject["label"] == 1 for subject in subjects)
    if n_control == 0 or n_pd == 0:
        raise RuntimeError("Both Control and PD participants are required.")

    channel_counts = {subject["data"].shape[1] for subject in subjects}
    time_lengths = {subject["data"].shape[2] for subject in subjects}
    sampling_rates = {subject["sfreq"] for subject in subjects}
    reference_channels = subjects[0]["ch_names"]

    if len(channel_counts) != 1:
        raise RuntimeError(f"Inconsistent channel counts: {sorted(channel_counts)}")
    if len(time_lengths) != 1:
        raise RuntimeError(f"Inconsistent segment lengths: {sorted(time_lengths)}")
    if len(sampling_rates) != 1:
        raise RuntimeError(f"Inconsistent sampling frequencies: {sorted(sampling_rates)}")
    for subject in subjects[1:]:
        if subject["ch_names"] != reference_channels:
            raise RuntimeError(
                f"Channel names or channel order differ for {subject['subject_id']}."
            )

    print(
        f"Loaded {len(subjects)} participants "
        f"({n_control} Control, {n_pd} PD)."
    )
    print(
        f"EEG shape: channels={next(iter(channel_counts))}, "
        f"time_points={next(iter(time_lengths))}, "
        f"sampling_rate={next(iter(sampling_rates)):.2f} Hz"
    )
    return subjects


def build_segment_arrays(subjects: Sequence[Dict]):
    """Concatenate subject data while retaining subject-level group labels."""
    X = np.concatenate([subject["data"] for subject in subjects], axis=0)
    y = np.concatenate(
        [np.full(subject["data"].shape[0], subject["label"], dtype=int) for subject in subjects]
    )
    groups = np.concatenate(
        [np.full(subject["data"].shape[0], idx, dtype=int) for idx, subject in enumerate(subjects)]
    )
    return X, y, groups


def standardize_from_training(train_X: np.ndarray, test_X: np.ndarray):
    """Z-score train and test data using statistics estimated only from training data."""
    mean = train_X.mean(axis=(0, 2), keepdims=True)
    std = train_X.std(axis=(0, 2), keepdims=True) + 1e-6
    return (train_X - mean) / std, (test_X - mean) / std


def compute_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> Dict[str, float]:
    """Compute binary classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": auc,
    }


def create_model(n_channels: int, dropout: float, device: torch.device) -> nn.Module:
    """Instantiate MSCANet with the manuscript architecture."""
    return EEG_MSCA_Net(
        n_channels=n_channels,
        n_classes=2,
        base_channels=16,
        mid_channels=16,
        out_channels=64,
        temporal_kernel=25,
        msca_reduction=4,
        msca_kernel_sizes=(7, 25, 51),
        dropout=dropout,
    ).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    all_true: List[int] = []
    all_pred: List[int] = []

    progress = tqdm(loader, desc=f"Train {epoch:03d}/{total_epochs:03d}", ncols=110)
    for X, y in progress:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        pred = logits.argmax(dim=1)
        all_true.extend(y.detach().cpu().numpy().tolist())
        all_pred.extend(pred.detach().cpu().numpy().tolist())

        progress.set_postfix(
            loss=f"{total_loss / len(all_true):.4f}",
            acc=f"{accuracy_score(all_true, all_pred):.4f}",
        )

    return total_loss / len(loader.dataset), accuracy_score(all_true, all_pred)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluate a trained model and return segment-level predictions."""
    model.eval()
    total_loss = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * X.size(0)
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        y_prob.extend(probs[:, 1].cpu().numpy().tolist())

    return (
        total_loss / len(loader.dataset),
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(y_prob),
    )


def aggregate_by_subject(
    segment_true: np.ndarray,
    segment_prob: np.ndarray,
    segment_groups: np.ndarray,
):
    """Average segment probabilities to obtain one prediction per participant."""
    subject_true, subject_pred, subject_prob, subject_ids = [], [], [], []

    for subject_idx in np.unique(segment_groups):
        mask = segment_groups == subject_idx
        labels = np.unique(segment_true[mask])
        if len(labels) != 1:
            raise RuntimeError(f"Multiple labels detected for subject group {subject_idx}.")

        probability = float(np.mean(segment_prob[mask]))
        subject_ids.append(int(subject_idx))
        subject_true.append(int(labels[0]))
        subject_prob.append(probability)
        subject_pred.append(int(probability >= 0.5))

    return (
        np.asarray(subject_true),
        np.asarray(subject_pred),
        np.asarray(subject_prob),
        np.asarray(subject_ids),
    )


def save_split_information(
    output_dir: Path,
    protocol: str,
    fold: int,
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> None:
    """Append subject-level train/test assignments to a CSV file."""
    rows = []
    for subject_id in train_subjects:
        rows.append({"protocol": protocol, "fold": fold, "subject_id": subject_id, "set": "train"})
    for subject_id in test_subjects:
        rows.append({"protocol": protocol, "fold": fold, "subject_id": subject_id, "set": "test"})

    path = output_dir / f"{protocol}_subject_splits.csv"
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_subject_kfold_cv(
    subjects: Sequence[Dict],
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
    """Run subject-level K-fold cross-validation using a fixed train/test protocol.

    Protocol used for each fold:
    1. Split participants at the subject level into training and held-out test sets.
    2. Estimate normalization statistics from the training set only.
    3. Train MSCANet for exactly ``config.epochs`` epochs using training subjects only.
    4. Evaluate the trained model once on the held-out test subjects.
    5. Save the fixed segment-level predictions for independent verification.

    Fold-wise metrics are summarized as mean ± sample standard deviation across folds.
    Predictions from all held-out folds are also concatenated to compute pooled
    out-of-fold (OOF) metrics and a pooled OOF confusion matrix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove outputs from a previous run so results from different runs are not mixed.
    for filename in (
        "kfold_subject_splits.csv",
        "kfold_fold_summary.csv",
        "kfold_segment_predictions.csv",
        "kfold_subject_predictions.csv",
        "kfold_pooled_oof_confusion_matrix.csv",
        "kfold_fold_mean_sd.csv",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()

    X, y, groups = build_segment_arrays(subjects)

    # Local segment index within each participant. This makes every saved prediction
    # traceable to a unique subject/segment pair without relying on array row order.
    local_segment_ids = np.concatenate(
        [np.arange(subject["data"].shape[0], dtype=int) for subject in subjects]
    )

    splitter = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.seed,
    )

    all_seg_true: List[int] = []
    all_seg_pred: List[int] = []
    all_seg_prob: List[float] = []
    all_seg_groups: List[int] = []
    prediction_rows: List[Dict] = []
    fold_rows: List[Dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        print(f"\n{'=' * 18} Fold {fold_idx}/{config.n_splits} {'=' * 18}")

        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])

        # Safety check: no participant can appear in both train and test sets.
        overlap = np.intersect1d(train_groups, test_groups)
        if overlap.size:
            raise RuntimeError(
                f"Subject leakage detected in fold {fold_idx}: group IDs {overlap.tolist()}"
            )

        train_subject_ids = [subjects[idx]["subject_id"] for idx in train_groups]
        test_subject_ids = [subjects[idx]["subject_id"] for idx in test_groups]
        save_split_information(
            output_dir,
            "kfold",
            fold_idx,
            train_subject_ids,
            test_subject_ids,
        )

        # Normalization parameters are estimated from the training set only.
        train_X, test_X = standardize_from_training(X[train_idx], X[test_idx])
        train_y, test_y = y[train_idx], y[test_idx]

        train_loader = DataLoader(
            EEGTrialDataset(train_X, train_y),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        test_loader = DataLoader(
            EEGTrialDataset(test_X, test_y),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = create_model(train_X.shape[1], config.dropout, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # Fixed training duration. The test set is not evaluated during training and
        # is not used for checkpoint selection, early stopping, or hyperparameter tuning.
        for epoch in range(1, config.epochs + 1):
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch,
                config.epochs,
            )

        # Exactly one final evaluation on the held-out test set for this fold.
        test_loss, seg_true, seg_pred, seg_prob = predict(
            model, test_loader, criterion, device
        )
        seg_metrics = compute_binary_metrics(seg_true, seg_pred, seg_prob)
        cm = confusion_matrix(seg_true, seg_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        n_correct = int((seg_true == seg_pred).sum())

        # Internal arithmetic verification: the reported fold accuracy must be exactly
        # reproducible from the integer number of correct predictions.
        direct_accuracy = n_correct / len(seg_true)
        if not np.isclose(seg_metrics["accuracy"], direct_accuracy, atol=1e-12):
            raise RuntimeError(
                f"Accuracy mismatch in fold {fold_idx}: "
                f"metric={seg_metrics['accuracy']}, direct={direct_accuracy}."
            )

        print(
            f"Fold {fold_idx} held-out test accuracy: "
            f"{seg_metrics['accuracy'] * 100:.2f}% "
            f"({n_correct}/{len(seg_true)} correct)"
        )

        # Save one row for every fixed held-out segment prediction.
        for row_pos, global_idx in enumerate(test_idx):
            group_idx = int(groups[global_idx])
            prediction_rows.append(
                {
                    "fold": fold_idx,
                    "subject_id": subjects[group_idx]["subject_id"],
                    "segment_id": int(local_segment_ids[global_idx]),
                    "true_label": int(seg_true[row_pos]),
                    "pred_label": int(seg_pred[row_pos]),
                    "probability_pd": float(seg_prob[row_pos]),
                }
            )

        all_seg_true.extend(seg_true.tolist())
        all_seg_pred.extend(seg_pred.tolist())
        all_seg_prob.extend(seg_prob.tolist())
        all_seg_groups.extend(groups[test_idx].tolist())

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train_subjects": len(train_groups),
                "n_test_subjects": len(test_groups),
                "train_subjects": ";".join(train_subject_ids),
                "test_subjects": ";".join(test_subject_ids),
                "n_train_segments": len(train_idx),
                "n_test_segments": len(test_idx),
                "n_correct": n_correct,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "test_loss": float(test_loss),
                "accuracy": seg_metrics["accuracy"],
                "sensitivity": seg_metrics["sensitivity"],
                "specificity": seg_metrics["specificity"],
                "precision": seg_metrics["precision"],
                "f1": seg_metrics["f1"],
                "balanced_accuracy": seg_metrics["balanced_accuracy"],
                "auc": seg_metrics["auc"],
            }
        )

    # ---------- Fold-wise results: mean ± SD across folds ----------
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(output_dir / "kfold_fold_summary.csv", index=False)

    prediction_df = pd.DataFrame(prediction_rows)
    prediction_df.to_csv(output_dir / "kfold_segment_predictions.csv", index=False)

    metric_columns = [
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "auc",
    ]
    fold_mean_sd: Dict[str, Dict[str, float]] = {}
    fold_summary_rows = []
    for metric_name in metric_columns:
        values = fold_df[metric_name].astype(float).to_numpy()
        mean_value = float(np.nanmean(values))
        # Sample SD (ddof=1), conventionally reported for cross-validation folds.
        sd_value = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
        fold_mean_sd[metric_name] = {"mean": mean_value, "sd": sd_value}
        fold_summary_rows.append(
            {"metric": metric_name, "mean": mean_value, "sd": sd_value}
        )

    pd.DataFrame(fold_summary_rows).to_csv(
        output_dir / "kfold_fold_mean_sd.csv", index=False
    )

    # ---------- Pooled out-of-fold results ----------
    all_seg_true_arr = np.asarray(all_seg_true)
    all_seg_pred_arr = np.asarray(all_seg_pred)
    all_seg_prob_arr = np.asarray(all_seg_prob)
    all_seg_groups_arr = np.asarray(all_seg_groups)

    pooled_segment_metrics = compute_binary_metrics(
        all_seg_true_arr, all_seg_pred_arr, all_seg_prob_arr
    )

    pooled_cm = confusion_matrix(all_seg_true_arr, all_seg_pred_arr, labels=[0, 1])
    pooled_cm_df = pd.DataFrame(
        pooled_cm,
        index=["True_HC", "True_PD"],
        columns=["Pred_HC", "Pred_PD"],
    )
    pooled_cm_df.to_csv(output_dir / "kfold_pooled_oof_confusion_matrix.csv")

    # ---------- Subject-level results from pooled OOF probabilities ----------
    sub_true, sub_pred, sub_prob, sub_ids = aggregate_by_subject(
        all_seg_true_arr, all_seg_prob_arr, all_seg_groups_arr
    )
    subject_metrics = compute_binary_metrics(sub_true, sub_pred, sub_prob)

    subject_results = pd.DataFrame(
        {
            "subject_index": sub_ids,
            "subject_id": [subjects[idx]["subject_id"] for idx in sub_ids],
            "true_label": sub_true,
            "pred_label": sub_pred,
            "probability_pd": sub_prob,
        }
    )
    subject_results.to_csv(output_dir / "kfold_subject_predictions.csv", index=False)

    return {
        "fold_mean_sd": fold_mean_sd,
        "pooled_segment_level": pooled_segment_metrics,
        "subject_level": subject_metrics,
    }


def run_loso_cv(
    subjects: Sequence[Dict],
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run leave-one-subject-out cross-validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = output_dir / "loso_subject_splits.csv"
    if split_file.exists():
        split_file.unlink()

    X, y, groups = build_segment_arrays(subjects)
    splitter = LeaveOneGroupOut()

    subject_true, subject_pred, subject_prob, subject_names = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        test_group = int(np.unique(groups[test_idx])[0])
        test_subject_id = subjects[test_group]["subject_id"]
        print(f"\n{'=' * 18} LOSO {fold_idx}/{len(subjects)} {'=' * 18}")
        print(f"Held-out participant: {test_subject_id}")

        train_groups = np.unique(groups[train_idx])
        save_split_information(
            output_dir,
            "loso",
            fold_idx,
            [subjects[idx]["subject_id"] for idx in train_groups],
            [test_subject_id],
        )

        train_X, test_X = standardize_from_training(X[train_idx], X[test_idx])
        train_y, test_y = y[train_idx], y[test_idx]

        train_loader = DataLoader(
            EEGTrialDataset(train_X, train_y),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )
        test_loader = DataLoader(
            EEGTrialDataset(test_X, test_y),
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = create_model(train_X.shape[1], config.dropout, device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        for epoch in range(1, config.epochs + 1):
            train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, config.epochs)

        _, seg_true, _, seg_prob = predict(model, test_loader, criterion, device)
        probability = float(np.mean(seg_prob))
        prediction = int(probability >= 0.5)

        subject_true.append(int(seg_true[0]))
        subject_pred.append(prediction)
        subject_prob.append(probability)
        subject_names.append(test_subject_id)

    results = pd.DataFrame(
        {
            "subject_id": subject_names,
            "true_label": subject_true,
            "pred_label": subject_pred,
            "probability_pd": subject_prob,
        }
    )
    results.to_csv(output_dir / "loso_subject_predictions.csv", index=False)

    return compute_binary_metrics(subject_true, subject_pred, subject_prob)


def evaluate_model_complexity(
    model: nn.Module,
    input_shape: Tuple[int, int],
    device: torch.device,
    iterations: int = 200,
) -> Dict[str, float]:
    """Measure parameter count, FLOPs, and network-only inference latency."""
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(1, *input_shape, device=device)

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    if device.type == "cuda":
        for _ in range(20):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "parameters_m": params / 1e6,
        "flops_m": flops / 1e6,
        "inference_ms": elapsed / iterations * 1000.0,
    }


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    """Print a compact metric summary."""
    print(f"\n{title}")
    for key, value in metrics.items():
        if np.isnan(value):
            print(f"  {key}: N/A")
        else:
            print(f"  {key}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSCANet training and evaluation")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing epoched .set files")
    parser.add_argument("--output-dir", type=str, default="./results", help="Directory for outputs")
    parser.add_argument("--protocol", choices=["kfold", "loso"], default="kfold")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-sfreq", type=float, default=None)
    parser.add_argument(
        "--per-trial-normalization",
        action="store_true",
        help="Apply an additional per-segment channel-wise z-score before CV",
    )
    parser.add_argument(
        "--complexity",
        action="store_true",
        help="Report model parameters, FLOPs, and network-only inference latency",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        n_splits=args.n_splits,
        num_workers=args.num_workers,
        dropout=args.dropout,
    )

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    subjects = load_all_subjects(
        data_dir=args.data_dir,
        target_sfreq=args.target_sfreq,
        per_trial_normalization=args.per_trial_normalization,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.complexity:
        n_channels = subjects[0]["data"].shape[1]
        n_times = subjects[0]["data"].shape[2]
        model = create_model(n_channels, config.dropout, device)
        complexity = evaluate_model_complexity(model, (n_channels, n_times), device)
        print_metrics("Model complexity", complexity)

    if args.protocol == "kfold":
        metrics = run_subject_kfold_cv(subjects, output_dir, config, device)

        print("\nFold-wise metrics (mean ± sample SD across folds)")
        for metric_name, summary in metrics["fold_mean_sd"].items():
            mean_value = summary["mean"]
            sd_value = summary["sd"]
            if np.isnan(mean_value):
                print(f"  {metric_name}: N/A")
            else:
                print(f"  {metric_name}: {mean_value:.4f} ± {sd_value:.4f}")

        print_metrics(
            "Pooled out-of-fold segment-level metrics",
            metrics["pooled_segment_level"],
        )
        print_metrics("Subject-level metrics", metrics["subject_level"])
    else:
        metrics = run_loso_cv(subjects, output_dir, config, device)
        print_metrics("LOSO subject-level metrics", metrics)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()
