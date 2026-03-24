"""
train_mlp.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Description
-----------
This module trains a compact MLP model for the NNIA-based recommendation system.

It loads engineered dataset files, performs data cleaning and splitting,
applies feature standardization, and trains a 16 -> 8 -> 2 classifier using
a controlled model search strategy.

The training process includes validation-based model selection, class balancing,
and evaluation across train, validation, and test splits.

Model parameters, scaler statistics, and a detailed training report are saved
for downstream quantization and hardware-aligned inference.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Python package path fix for direct script execution
# =============================================================================
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent              # Python/ott_recommender
PYTHON_ROOT = THIS_DIR.parent            # Python/

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from ott_recommender.feature_encoder import (
    INPUT_SIZE,
    NUM_CLASSES,
    FEATURE_NAMES,
    CLASS_NAMES,
    CLASS_NAME_TO_LABEL,
)


# =============================================================================
# Configuration
# =============================================================================
RANDOM_SEED = 42
HIDDEN_SIZE = 8

PROJECT_ROOT = THIS_DIR.parent.parent
PYTHON_DIR = PROJECT_ROOT / "Python"
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DATASETS_DIR = ARTIFACTS_DIR / "datasets"

MODEL_OUT = ARTIFACTS_DIR / "trained_mlp_model.npz"
REPORT_OUT = ARTIFACTS_DIR / "train_report.json"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Alignment checks
# =============================================================================
def validate_alignment() -> None:
    assert INPUT_SIZE == 16, f"Expected INPUT_SIZE=16, got {INPUT_SIZE}"
    assert NUM_CLASSES == 2, f"Expected NUM_CLASSES=2, got {NUM_CLASSES}"
    assert HIDDEN_SIZE == 8, f"Expected HIDDEN_SIZE=8, got {HIDDEN_SIZE}"
    assert len(FEATURE_NAMES) == INPUT_SIZE
    assert len(CLASS_NAMES) == NUM_CLASSES

    expected_classes = ["Not Recommended", "Recommended"]
    assert CLASS_NAMES == expected_classes, (
        f"Class alignment mismatch. Expected {expected_classes}, got {CLASS_NAMES}"
    )


# =============================================================================
# Dataset discovery
# =============================================================================
RAW_MOVIELENS_NAMES = {
    "movies.csv",
    "ratings.csv",
    "tags.csv",
    "links.csv",
    "genome-scores.csv",
    "genome-tags.csv",
}

PREFERRED_SINGLE_DATASET_NAMES = [
    "ott_dataset_full.csv",
    "engineered_dataset.csv",
    "ott_engineered_dataset.csv",
    "movie_dataset_engineered.csv",
    "preference_dataset.csv",
    "mlp_dataset.csv",
    "train_dataset.csv",
    "dataset.csv",
]

PREFERRED_TRAIN_NAMES = [
    "ott_dataset_train.csv",
]

PREFERRED_TEST_NAMES = [
    "ott_dataset_test.csv",
]


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "target",
        "label",
        "class",
        "preference_label",
        "preference_bucket",
        "bucket",
        "y",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def looks_like_engineered_dataset(csv_path: Path) -> bool:
    if csv_path.name.lower() in RAW_MOVIELENS_NAMES:
        return False

    try:
        df_head = pd.read_csv(csv_path, nrows=5)
    except Exception:
        return False

    feature_hits = sum(1 for c in FEATURE_NAMES if c in df_head.columns)
    if feature_hits != len(FEATURE_NAMES):
        return False

    target_col = detect_target_column(df_head)
    return target_col is not None


def search_csvs_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def find_named_file(search_roots: Sequence[Path], candidate_names: Sequence[str]) -> Optional[Path]:
    for root in search_roots:
        if not root.exists():
            continue
        for name in candidate_names:
            p = root / name
            if p.exists() and looks_like_engineered_dataset(p):
                return p
    return None


def collect_candidate_csvs() -> List[Path]:
    search_roots = [
        ARTIFACT_DATASETS_DIR,
        ARTIFACTS_DIR,
        DATA_DIR,
        DATASETS_DIR,
    ]

    candidates: List[Path] = []
    for root in search_roots:
        candidates.extend(search_csvs_under(root))

    seen = set()
    unique_candidates = []
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(p)

    return unique_candidates


def find_split_dataset_files() -> Tuple[Optional[Path], Optional[Path]]:
    search_roots = [
        ARTIFACT_DATASETS_DIR,
        ARTIFACTS_DIR,
        DATA_DIR,
        PROJECT_ROOT,
    ]

    train_file = find_named_file(search_roots, PREFERRED_TRAIN_NAMES)
    test_file = find_named_file(search_roots, PREFERRED_TEST_NAMES)
    return train_file, test_file


def find_single_dataset_file() -> Optional[Path]:
    search_roots = [
        ARTIFACT_DATASETS_DIR,
        ARTIFACTS_DIR,
        DATA_DIR,
        DATASETS_DIR,
    ]

    named = find_named_file(search_roots, PREFERRED_SINGLE_DATASET_NAMES)
    if named is not None:
        return named

    for p in collect_candidate_csvs():
        if looks_like_engineered_dataset(p):
            return p

    return None


def run_create_dataset() -> None:
    script_path = PYTHON_DIR / "ott_recommender" / "create_dataset.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find create_dataset.py at: {script_path}")

    print("\n============================================================")
    print("Prepared dataset not found.")
    print("Auto-running create_dataset.py ...")
    print("============================================================")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )

    print("\n----- create_dataset.py stdout -----")
    print(result.stdout if result.stdout else "[no stdout]")

    if result.stderr:
        print("\n----- create_dataset.py stderr -----")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"create_dataset.py failed with return code {result.returncode}"
        )


def get_dataset_files() -> Tuple[Optional[Path], Optional[Path], Optional[Path], str]:
    train_file, test_file = find_split_dataset_files()
    if train_file is not None and test_file is not None:
        return train_file, test_file, None, "prepared_split"

    full_file = find_single_dataset_file()
    if full_file is not None:
        return None, None, full_file, "single_dataset"

    run_create_dataset()

    train_file, test_file = find_split_dataset_files()
    if train_file is not None and test_file is not None:
        return train_file, test_file, None, "prepared_split"

    full_file = find_single_dataset_file()
    if full_file is not None:
        return None, None, full_file, "single_dataset"

    candidates = collect_candidate_csvs()
    raise FileNotFoundError(
        "No usable engineered dataset found after running create_dataset.py.\n\n"
        "CSV files seen:\n" + "\n".join(f"  - {str(p)}" for p in candidates[:60])
    )


# =============================================================================
# Label helpers
# =============================================================================
def validate_label(label: int) -> None:
    label = int(label)
    if label < 0 or label >= NUM_CLASSES:
        raise ValueError(f"Label out of range: {label}")


def normalize_class_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Class name must be string-like, got {type(name)}")

    s = name.strip().lower()

    mapping = {
        "not recommended": "Not Recommended",
        "not_recommended": "Not Recommended",
        "notrecommended": "Not Recommended",
        "not suitable": "Not Recommended",
        "less aligned": "Not Recommended",
        "recommended": "Recommended",
        "strong match": "Recommended",
        "good fit": "Recommended",
        "aligned": "Recommended",
    }

    if s in mapping:
        return mapping[s]

    title = s.title()
    if title in CLASS_NAMES:
        return title

    raise ValueError(f"Unknown class label text: {name}")


def convert_targets_to_ids(series: pd.Series) -> np.ndarray:
    out: List[int] = []

    for value in series.tolist():
        if isinstance(value, str):
            cls_name = normalize_class_name(value)
            label = CLASS_NAME_TO_LABEL[cls_name]
        else:
            try:
                label = int(value)
            except Exception as exc:
                raise TypeError(f"Bad label value: {value}") from exc
            validate_label(label)

        out.append(label)

    y = np.asarray(out, dtype=np.int64)

    for label in np.unique(y):
        validate_label(int(label))

    return y


# =============================================================================
# Cleaning
# =============================================================================
def clean_dataframe(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
    keep_cols = feature_cols + [target_col]

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df[keep_cols].copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=keep_cols).copy()
    dropped_nan = before - len(df)

    for col in feature_cols:
        df[col] = df[col].clip(lower=0.0, upper=1.0)

    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped_dup = before_dup - len(df)

    print("\n================ DATA CLEANING ================")
    print(f"Rows after load        : {before}")
    print(f"Dropped NaN rows       : {dropped_nan}")
    print(f"Dropped duplicate rows : {dropped_dup}")
    print(f"Rows after cleaning    : {len(df)}")

    return df


def load_feature_label_dataframe(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    feature_cols = FEATURE_NAMES[:]
    target_col = detect_target_column(df)

    if target_col is None:
        raise ValueError(f"Target column could not be detected in: {csv_path}")

    print(f"\nUsing dataset file      : {csv_path}")
    print(f"Raw dataset shape       : {df.shape}")
    print(f"Detected target column  : {target_col}")
    print(f"Feature count           : {len(feature_cols)}")

    df = clean_dataframe(df, feature_cols, target_col)

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = convert_targets_to_ids(df[target_col])

    if X.shape[1] != INPUT_SIZE:
        raise ValueError(f"Expected {INPUT_SIZE} features, got {X.shape[1]}")

    return df, X, y


# =============================================================================
# Balancing
# =============================================================================
def oversample_training_set(
    X: np.ndarray,
    y: np.ndarray,
    rng_seed: int = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)

    classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))

    X_parts = []
    y_parts = []

    for cls, count in zip(classes, counts):
        idx = np.where(y == cls)[0]
        X_cls = X[idx]
        y_cls = y[idx]

        if count < max_count:
            extra_idx = rng.choice(idx, size=max_count - count, replace=True)
            X_extra = X[extra_idx]
            y_extra = y[extra_idx]
            X_cls = np.vstack([X_cls, X_extra])
            y_cls = np.concatenate([y_cls, y_extra])

        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X_bal = np.vstack(X_parts)
    y_bal = np.concatenate(y_parts)

    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


# =============================================================================
# Metrics helpers
# =============================================================================
def class_distribution(y: np.ndarray) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for label in y:
        counts[CLASS_NAMES[int(label)]] += 1
    return counts


def safe_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    return cm.astype(int).tolist()


def evaluate_split(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> Dict[str, object]:
    y_pred = model.predict(X)

    acc = float(accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y, y_pred, average="weighted"))

    report = classification_report(
        y,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n================ {split_name.upper()} METRICS ================")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Macro F1     : {macro_f1:.4f}")
    print(f"Weighted F1  : {weighted_f1:.4f}")

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": safe_confusion_matrix(y, y_pred),
        "classification_report": report,
    }


# =============================================================================
# Model search
# =============================================================================
def build_candidate_configs() -> List[Dict[str, object]]:
    return [
        {
            "hidden_layer_sizes": (8,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-5,
            "learning_rate_init": 1.5e-3,
            "max_iter": 1200,
            "early_stopping": True,
            "n_iter_no_change": 50,
        },
        {
            "hidden_layer_sizes": (8,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 5e-5,
            "learning_rate_init": 1.0e-3,
            "max_iter": 1200,
            "early_stopping": True,
            "n_iter_no_change": 50,
        },
        {
            "hidden_layer_sizes": (8,),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate_init": 8.0e-4,
            "max_iter": 1400,
            "early_stopping": True,
            "n_iter_no_change": 60,
        },
        {
            "hidden_layer_sizes": (8,),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 1e-5,
            "learning_rate_init": 1.2e-3,
            "max_iter": 1400,
            "early_stopping": True,
            "n_iter_no_change": 60,
        },
        {
            "hidden_layer_sizes": (8,),
            "activation": "tanh",
            "solver": "adam",
            "alpha": 1e-4,
            "learning_rate_init": 8.0e-4,
            "max_iter": 1500,
            "early_stopping": True,
            "n_iter_no_change": 65,
        },
        {
            "hidden_layer_sizes": (8,),
            "activation": "logistic",
            "solver": "adam",
            "alpha": 1e-5,
            "learning_rate_init": 1.0e-3,
            "max_iter": 1500,
            "early_stopping": True,
            "n_iter_no_change": 65,
        },
        {
            "hidden_layer_sizes": (8,),
            "activation": "relu",
            "solver": "lbfgs",
            "alpha": 1e-4,
            "learning_rate_init": 1.0e-3,
            "max_iter": 800,
            "early_stopping": False,
            "n_iter_no_change": 50,
        },
        {
            "hidden_layer_sizes": (8,),
            "activation": "tanh",
            "solver": "lbfgs",
            "alpha": 1e-4,
            "learning_rate_init": 1.0e-3,
            "max_iter": 800,
            "early_stopping": False,
            "n_iter_no_change": 50,
        },
    ]


def train_best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[MLPClassifier, Dict[str, object], List[Dict[str, object]]]:
    configs = build_candidate_configs()

    best_model: Optional[MLPClassifier] = None
    best_summary: Optional[Dict[str, object]] = None
    history: List[Dict[str, object]] = []
    best_score = -1.0

    for i, cfg in enumerate(configs, start=1):
        print("\n============================================================")
        print(f"Training candidate {i}/{len(configs)}")
        print(json.dumps(cfg, indent=2))

        model_kwargs = dict(
            hidden_layer_sizes=cfg["hidden_layer_sizes"],
            activation=cfg["activation"],
            solver=cfg["solver"],
            alpha=cfg["alpha"],
            learning_rate_init=cfg["learning_rate_init"],
            max_iter=cfg["max_iter"],
            random_state=RANDOM_SEED,
        )

        if cfg["solver"] != "lbfgs":
            model_kwargs["early_stopping"] = cfg["early_stopping"]
            model_kwargs["n_iter_no_change"] = cfg["n_iter_no_change"]
            model_kwargs["validation_fraction"] = 0.15

        model = MLPClassifier(**model_kwargs)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_acc = float(accuracy_score(y_val, val_pred))
        val_macro_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_weighted_f1 = float(f1_score(y_val, val_pred, average="weighted"))

        per_class_f1 = f1_score(
            y_val,
            val_pred,
            average=None,
            labels=np.arange(NUM_CLASSES),
            zero_division=0,
        )
        notrec_f1 = float(per_class_f1[0])
        rec_f1 = float(per_class_f1[1])

        score = (
            0.50 * val_acc
            + 0.25 * val_macro_f1
            + 0.15 * val_weighted_f1
            + 0.05 * notrec_f1
            + 0.05 * rec_f1
        )

        item = {
            "candidate_index": i,
            "config": cfg,
            "val_accuracy": val_acc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "val_not_recommended_f1": notrec_f1,
            "val_recommended_f1": rec_f1,
            "epochs_ran": int(getattr(model, "n_iter_", 0)),
            "loss_curve_last": float(model.loss_curve_[-1]) if hasattr(model, "loss_curve_") and len(model.loss_curve_) else None,
            "selection_score": score,
        }
        history.append(item)

        print(f"Validation Accuracy          : {val_acc:.4f}")
        print(f"Validation Macro F1          : {val_macro_f1:.4f}")
        print(f"Validation Weighted F1       : {val_weighted_f1:.4f}")
        print(f"Validation NotRec F1         : {notrec_f1:.4f}")
        print(f"Validation Recommended F1    : {rec_f1:.4f}")
        print(f"Selection Score              : {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_summary = item

    if best_model is None or best_summary is None:
        raise RuntimeError("Model search failed to produce a valid model.")

    return best_model, best_summary, history


# =============================================================================
# Save
# =============================================================================
def save_model_artifact(
    model: MLPClassifier,
    scaler: StandardScaler,
    best_summary: Dict[str, object],
    dataset_name: str,
    training_mode: str,
) -> None:
    W1 = model.coefs_[0].astype(np.float64)
    W2 = model.coefs_[1].astype(np.float64)
    b1 = model.intercepts_[0].astype(np.float64)
    b2 = model.intercepts_[1].astype(np.float64)

    if W1.shape != (INPUT_SIZE, HIDDEN_SIZE):
        raise ValueError(f"Unexpected W1 shape: {W1.shape}")
    if b1.shape != (HIDDEN_SIZE,):
        raise ValueError(f"Unexpected b1 shape: {b1.shape}")

    binary_single_output = False

    if W2.shape == (HIDDEN_SIZE, NUM_CLASSES) and b2.shape == (NUM_CLASSES,):
        output_size_saved = NUM_CLASSES
    elif NUM_CLASSES == 2 and W2.shape == (HIDDEN_SIZE, 1) and b2.shape == (1,):
        binary_single_output = True
        output_size_saved = 1
    else:
        raise ValueError(
            f"Unexpected output layer shapes: W2={W2.shape}, b2={b2.shape}"
        )

    np.savez(
        MODEL_OUT,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        class_names=np.array(CLASS_NAMES, dtype=object),
        input_size=np.array([INPUT_SIZE], dtype=np.int64),
        hidden_size=np.array([HIDDEN_SIZE], dtype=np.int64),
        output_size=np.array([output_size_saved], dtype=np.int64),
        num_classes=np.array([NUM_CLASSES], dtype=np.int64),
        binary_single_output=np.array([int(binary_single_output)], dtype=np.int64),
        dataset_name=np.array([dataset_name], dtype=object),
        training_mode=np.array([training_mode], dtype=object),
        best_activation=np.array([best_summary["config"]["activation"]], dtype=object),
        best_alpha=np.array([best_summary["config"]["alpha"]], dtype=np.float64),
        best_learning_rate_init=np.array(
            [best_summary["config"]["learning_rate_init"]], dtype=np.float64
        ),
    )

    print(f"\nSaved model artifact: {MODEL_OUT}")
    print(f"Binary single output : {binary_single_output}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("\n==============================================================")
    print(" NNIA TRAINING - 16 -> 8 -> 2 OTT RECOMMENDATION MODEL ")
    print("==============================================================")

    validate_alignment()

    train_file, test_file, full_file, mode = get_dataset_files()

    if mode == "prepared_split":
        print("\nDataset mode            : prepared_split")
        print("Split policy            : use dataset-builder train/test files")
        print("Validation policy       : split validation only from train set")

        _, X_train_full, y_train_full = load_feature_label_dataframe(train_file)
        _, X_test, y_test = load_feature_label_dataframe(test_file)

        print("\n================ PREPARED LABEL DISTRIBUTION ================")
        print("Train:")
        for cls_name, cnt in class_distribution(y_train_full).items():
            print(f"{cls_name:<18s}: {cnt}")
        print("Test:")
        for cls_name, cnt in class_distribution(y_test).items():
            print(f"{cls_name:<18s}: {cnt}")

        unique_train, counts_train = np.unique(y_train_full, return_counts=True)
        min_class_count = int(np.min(counts_train))
        if min_class_count < 3:
            raise ValueError(
                "At least 3 samples per class are needed in training set for stable validation split.\n"
                f"Train class counts: {dict(zip(unique_train.tolist(), counts_train.tolist()))}"
            )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full,
            y_train_full,
            test_size=0.15,
            random_state=RANDOM_SEED,
            stratify=y_train_full,
        )

        dataset_name = f"{train_file.name} + {test_file.name}"

    else:
        print("\nDataset mode            : single_dataset")
        print("Split policy            : internal train/val/test split")
        print("Validation policy       : from internal training split")

        _, X, y = load_feature_label_dataframe(full_file)

        print("\n================ LABEL DISTRIBUTION ================")
        for cls_name, cnt in class_distribution(y).items():
            print(f"{cls_name:<18s}: {cnt}")

        unique, counts = np.unique(y, return_counts=True)
        min_class_count = int(np.min(counts))
        if min_class_count < 3:
            raise ValueError(
                "At least 3 samples per class are needed for stable stratified splitting.\n"
                f"Class counts: {dict(zip(unique.tolist(), counts.tolist()))}"
            )

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=0.1764705882,
            random_state=RANDOM_SEED,
            stratify=y_trainval,
        )

        dataset_name = full_file.name

    print("\n================ SPLIT SIZES ================")
    print(f"Train size : {len(y_train)}")
    print(f"Val size   : {len(y_val)}")
    print(f"Test size  : {len(y_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_bal, y_train_bal = oversample_training_set(X_train_scaled, y_train)

    print("\n================ BALANCED TRAIN DISTRIBUTION ================")
    for k, v in class_distribution(y_train_bal).items():
        print(f"{k:<18s}: {v}")

    best_model, best_summary, history = train_best_model(
        X_train_bal,
        y_train_bal,
        X_val_scaled,
        y_val,
    )

    print("\n============================================================")
    print("BEST MODEL SELECTED")
    print(json.dumps(best_summary, indent=2))

    train_metrics = evaluate_split(best_model, X_train_scaled, y_train, "train")
    val_metrics = evaluate_split(best_model, X_val_scaled, y_val, "val")
    test_metrics = evaluate_split(best_model, X_test_scaled, y_test, "test")

    save_model_artifact(
        model=best_model,
        scaler=scaler,
        best_summary=best_summary,
        dataset_name=dataset_name,
        training_mode=mode,
    )

    report = {
        "dataset_mode": mode,
        "dataset_name": dataset_name,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "output_size": NUM_CLASSES,
        "split_sizes": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "distributions": {
            "train": class_distribution(y_train),
            "val": class_distribution(y_val),
            "test": class_distribution(y_test),
            "train_balanced": class_distribution(y_train_bal),
        },
        "best_model": best_summary,
        "search_history": history,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "notes": [
            "Classifier shape is 16 -> 8 -> 2.",
            "Prepared train/test dataset files are preferred when available.",
            "Validation split is always derived from the training portion only.",
            "Oversampling is applied only to the training split.",
            "Selection score prioritizes validation accuracy and balanced class quality.",
            "Task is OTT-style binary recommendation prediction.",
        ],
    }

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved training report: {REPORT_OUT}")

    print("\n============================================================")
    print("FINAL SUMMARY")
    print("============================================================")
    print(f"Validation accuracy : {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy       : {test_metrics['accuracy']:.4f}")
    print(f"Test macro F1       : {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()