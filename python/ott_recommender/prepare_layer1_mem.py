"""
prepare_layer1_mem.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Description
-----------
This module prepares NNIA-compatible memory files for the first stage
of the MLP inference pipeline.

It loads the quantized model parameters and a batch of test samples,
applies the same feature standardization used during training, and
converts inputs to fixed-point representation.

The module also computes the expected output using NNIA-aligned
fixed-point arithmetic and generates memory files for hardware
simulation and verification.

Generated Artifacts
-------------------
- Input, weight, bias, and expected output memory files

"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# Python package path fix (IMPORTANT)
# =============================================================================
THIS_DIR = Path(__file__).resolve().parent      # Python/ott_recommender
PYTHON_ROOT = THIS_DIR.parent                  # Python/

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

# =============================================================================
# FIXED imports
# =============================================================================
from ott_recommender.feature_encoder import FEATURE_NAMES

from shared.fixed_point_utils import (
    add_bias,
    clamp_signed,
    dot_product_fixed,
    fixed_to_float,
    flatten_2d_row_major,
    float_to_fixed,
    relu,
    requantize,
    write_mem_file,
)

# =============================================================================
# Locked NNIA / MLP configuration
# =============================================================================
USE_FIXED_SEED = False
RANDOM_SEED = 7

M = 4
K = 16
N = 8

DATA_WIDTH = 16
FRAC_BITS = 8
ACC_WIDTH = 40

REQUANT_ROUNDING = "trunc"
INPUT_QUANT_ROUNDING = "trunc"

# =============================================================================
# Project paths (FIXED)
# =============================================================================
PROJECT_ROOT = THIS_DIR.parent.parent

DATASET_DIR = PROJECT_ROOT / "artifacts" / "datasets"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MEM_DIR = PROJECT_ROOT / "mem"
RUNS_DIR = ARTIFACTS_DIR / "runs"

TEST_CSV = DATASET_DIR / "ott_dataset_test.csv"
MODEL_QUANT_NPZ = ARTIFACTS_DIR / "model_quantized.npz"

INPUT_MEM = MEM_DIR / "input.mem"
WEIGHTS_MEM = MEM_DIR / "weights.mem"
BIAS_MEM = MEM_DIR / "bias.mem"
EXPECTED_OUTPUT_MEM = MEM_DIR / "expected_output_l1.mem"

LAYER1_BATCH_JSON = RUNS_DIR / "layer1_batch_info.json"
LAYER1_REF_NPZ = RUNS_DIR / "layer1_reference.npz"


# =============================================================================
# Validation helpers
# =============================================================================
def validate_required_files() -> None:
    missing = []

    if not TEST_CSV.exists():
        missing.append(str(TEST_CSV))
    if not MODEL_QUANT_NPZ.exists():
        missing.append(str(MODEL_QUANT_NPZ))

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(missing)
            + "\n\nRun create_dataset.py, train_mlp.py, and export_quantized_model.py first."
        )


def validate_model_shapes(
    W1_q: np.ndarray,
    b1_q: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> None:
    if W1_q.shape != (K, N):
        raise ValueError(f"W1_q shape mismatch: expected {(K, N)}, got {W1_q.shape}")
    if b1_q.shape != (N,):
        raise ValueError(f"b1_q shape mismatch: expected {(N,)}, got {b1_q.shape}")
    if scaler_mean.shape != (K,):
        raise ValueError(
            f"scaler_mean shape mismatch: expected {(K,)}, got {scaler_mean.shape}"
        )
    if scaler_scale.shape != (K,):
        raise ValueError(
            f"scaler_scale shape mismatch: expected {(K,)}, got {scaler_scale.shape}"
        )

    if not np.all(np.isfinite(W1_q)):
        raise ValueError("W1_q contains non-finite values")
    if not np.all(np.isfinite(b1_q)):
        raise ValueError("b1_q contains non-finite values")
    if not np.all(np.isfinite(scaler_mean)):
        raise ValueError("scaler_mean contains non-finite values")
    if not np.all(np.isfinite(scaler_scale)):
        raise ValueError("scaler_scale contains non-finite values")


def load_quantized_layer1_model() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(MODEL_QUANT_NPZ, allow_pickle=True)

    required_keys = {"W1_q", "b1_q", "scaler_mean", "scaler_scale"}
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"model_quantized.npz missing keys: {missing_keys}")

    W1_q = np.asarray(data["W1_q"], dtype=np.int32)
    b1_q = np.asarray(data["b1_q"], dtype=np.int32)
    scaler_mean = np.asarray(data["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(data["scaler_scale"], dtype=np.float32)

    validate_model_shapes(W1_q, b1_q, scaler_mean, scaler_scale)
    return W1_q, b1_q, scaler_mean, scaler_scale


# =============================================================================
# Dataset loading + scaling
# =============================================================================
def load_test_batch(
    csv_path: Path,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    batch_size: int = M,
) -> Tuple[List[List[int]], List[List[float]], List[List[float]], List[int], List[int]]:
    """
    Load a batch of test samples, standardize them using the saved scaler,
    and convert them to signed Q8.8.

    Returns
    -------
    input_q_2d:
        Standardized feature vectors in Q8.8, shape (M, K)
    raw_float_2d:
        Raw dataset feature values, shape (M, K)
    scaled_float_2d:
        Standardized floating-point feature values, shape (M, K)
    labels:
        Ground-truth labels for the selected rows
    selected_indices:
        Original row indices selected from the test dataset
    """
    df = pd.read_csv(csv_path)

    expected_cols = FEATURE_NAMES + ["label", "label_name"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Test CSV missing required columns: {missing}")

    if len(df) < batch_size:
        raise ValueError(f"Need at least {batch_size} test samples, found {len(df)}")

    feature_array = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    if not np.all(np.isfinite(feature_array)):
        raise ValueError("Test CSV contains non-finite feature values")

    if USE_FIXED_SEED:
        rng = random.Random(RANDOM_SEED)
    else:
        rng = random.Random()

    selected_indices = rng.sample(range(len(df)), batch_size)
    batch_df = df.iloc[selected_indices].copy().reset_index(drop=True)

    input_q_2d: List[List[int]] = []
    raw_float_2d: List[List[float]] = []
    scaled_float_2d: List[List[float]] = []
    labels: List[int] = []

    for _, row in batch_df.iterrows():
        raw_vec = [float(row[feat]) for feat in FEATURE_NAMES]
        raw_float_2d.append(raw_vec)

        scaled_vec: List[float] = []
        vec_q: List[int] = []

        for i, value_raw in enumerate(raw_vec):
            mean_i = float(scaler_mean[i])
            scale_i = float(scaler_scale[i])

            if abs(scale_i) < 1e-12:
                value_scaled = 0.0
            else:
                value_scaled = (value_raw - mean_i) / scale_i

            if not np.isfinite(value_scaled):
                raise ValueError(f"Non-finite scaled value at feature index {i}")

            scaled_vec.append(float(value_scaled))

            value_q = float_to_fixed(
                value_scaled,
                data_width=DATA_WIDTH,
                frac_bits=FRAC_BITS,
                rounding=INPUT_QUANT_ROUNDING,
                saturate=True,
            )
            value_q = clamp_signed(value_q, DATA_WIDTH)
            vec_q.append(int(value_q))

        if len(vec_q) != K:
            raise ValueError(f"Feature vector length mismatch: expected {K}, got {len(vec_q)}")

        scaled_float_2d.append(scaled_vec)
        input_q_2d.append(vec_q)
        labels.append(int(row["label"]))

    if len(input_q_2d) != M:
        raise ValueError(f"Batch row count mismatch: expected {M}, got {len(input_q_2d)}")

    return input_q_2d, raw_float_2d, scaled_float_2d, labels, selected_indices


# =============================================================================
# Layer-1 software reference model
# =============================================================================
def compute_layer1_expected_output(
    input_q_2d: List[List[int]],
    W1_q: np.ndarray,
    b1_q: np.ndarray,
) -> List[List[int]]:
    """
    Compute expected hidden-layer output in integer Q8.8 domain.

    For each output neuron:
        raw_acc = dot(input_q_row, W1_q_col)
        rq      = requantize(raw_acc, shift=FRAC_BITS)
        biased  = rq + b1_q[col]
        out     = ReLU(biased)
    """
    hidden_q_2d: List[List[int]] = [[0 for _ in range(N)] for _ in range(M)]

    for m in range(M):
        in_vec = input_q_2d[m]
        if len(in_vec) != K:
            raise ValueError(f"Input row {m} length mismatch: expected {K}, got {len(in_vec)}")

        for n in range(N):
            w_col = [int(W1_q[k][n]) for k in range(K)]

            raw_acc = dot_product_fixed(in_vec, w_col)
            raw_acc = clamp_signed(raw_acc, ACC_WIDTH)

            rq = requantize(
                raw_acc,
                shift=FRAC_BITS,
                out_width=ACC_WIDTH,
                rounding=REQUANT_ROUNDING,
                saturate=True,
            )

            biased = add_bias(rq, int(b1_q[n]))
            biased = clamp_signed(biased, ACC_WIDTH)

            out_val = relu(biased)
            out_val = clamp_signed(out_val, DATA_WIDTH)

            hidden_q_2d[m][n] = int(out_val)

    return hidden_q_2d


# =============================================================================
# Saving helpers
# =============================================================================
def save_run_artifacts(
    input_q_2d: List[List[int]],
    raw_float_2d: List[List[float]],
    scaled_float_2d: List[List[float]],
    hidden_q_2d: List[List[int]],
    labels: List[int],
    selected_indices: List[int],
) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    info: Dict[str, object] = {
        "stage": "layer1",
        "sampling_mode": "fixed" if USE_FIXED_SEED else "random",
        "random_seed": RANDOM_SEED if USE_FIXED_SEED else "random",
        "batch_size": M,
        "input_size": K,
        "output_size": N,
        "data_width": DATA_WIDTH,
        "frac_bits": FRAC_BITS,
        "input_quant_rounding": INPUT_QUANT_ROUNDING,
        "requant_rounding": REQUANT_ROUNDING,
        "labels": labels,
        "selected_indices": selected_indices,
        "notes": [
            "This batch info is saved for later layer-2 preparation.",
            "input_q_2d and hidden_q_2d are stored in layer1_reference.npz.",
            "When USE_FIXED_SEED=True, batch rows are selected deterministically.",
            "When USE_FIXED_SEED=False, a fresh random batch is selected each run.",
            "input_q_2d stores standardized feature values in Q8.8.",
            "expected_output_l1.mem stores the layer-1 golden output for RTL comparison.",
            "Layer-1 AI path uses trunc-style rounding to match RTL behavior.",
        ],
    }

    with open(LAYER1_BATCH_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    np.savez(
        LAYER1_REF_NPZ,
        input_q=np.array(input_q_2d, dtype=np.int32),
        input_raw_float=np.array(raw_float_2d, dtype=np.float32),
        input_scaled_float=np.array(scaled_float_2d, dtype=np.float32),
        hidden_q=np.array(hidden_q_2d, dtype=np.int32),
        labels=np.array(labels, dtype=np.int32),
        selected_indices=np.array(selected_indices, dtype=np.int32),
    )


def print_debug_preview(
    raw_float_2d: List[List[float]],
    scaled_float_2d: List[List[float]],
    input_q_2d: List[List[int]],
    hidden_q_2d: List[List[int]],
) -> None:
    print("\nDebug preview (sample 0):")
    print(f"  raw first 6     : {[round(v, 4) for v in raw_float_2d[0][:6]]}")
    print(f"  scaled first 6  : {[round(v, 4) for v in scaled_float_2d[0][:6]]}")
    print(f"  input_q first 6 : {input_q_2d[0][:6]}")
    print(f"  hidden_q        : {hidden_q_2d[0]}")
    print(
        f"  hidden_f        : "
        f"{[round(fixed_to_float(v, FRAC_BITS), 4) for v in hidden_q_2d[0]]}"
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    validate_required_files()
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    W1_q, b1_q, scaler_mean, scaler_scale = load_quantized_layer1_model()

    input_q_2d, raw_float_2d, scaled_float_2d, labels, selected_indices = load_test_batch(
        TEST_CSV,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        batch_size=M,
    )

    hidden_q_2d = compute_layer1_expected_output(input_q_2d, W1_q, b1_q)

    input_flat = flatten_2d_row_major(input_q_2d)
    weights_flat = flatten_2d_row_major(W1_q.tolist())
    bias_flat = [int(v) for v in b1_q.tolist()]
    expected_flat = flatten_2d_row_major(hidden_q_2d)

    write_mem_file(INPUT_MEM, input_flat, DATA_WIDTH, radix="hex")
    write_mem_file(WEIGHTS_MEM, weights_flat, DATA_WIDTH, radix="hex")
    write_mem_file(BIAS_MEM, bias_flat, DATA_WIDTH, radix="hex")
    write_mem_file(EXPECTED_OUTPUT_MEM, expected_flat, DATA_WIDTH, radix="hex")

    save_run_artifacts(
        input_q_2d=input_q_2d,
        raw_float_2d=raw_float_2d,
        scaled_float_2d=scaled_float_2d,
        hidden_q_2d=hidden_q_2d,
        labels=labels,
        selected_indices=selected_indices,
    )

    print("\n==================== PREPARE LAYER 1 MEM ====================\n")
    print(f"Test dataset      : {TEST_CSV}")
    print(f"Quant model       : {MODEL_QUANT_NPZ}")

    print("\nLocked shapes:")
    print(f"  input batch     : ({M}, {K})")
    print(f"  W1_q            : {tuple(W1_q.shape)}")
    print(f"  b1_q            : {tuple(b1_q.shape)}")
    print(f"  expected hidden : ({M}, {N})")

    print("\nSampling:")
    print(f"  mode            : {'fixed' if USE_FIXED_SEED else 'random'}")
    print(f"  seed            : {RANDOM_SEED if USE_FIXED_SEED else 'random'}")

    print("\nRounding:")
    print(f"  input quant     : {INPUT_QUANT_ROUNDING}")
    print(f"  requant         : {REQUANT_ROUNDING}")

    print("\nSelected dataset row indices:")
    print(f"  {selected_indices}")

    print_debug_preview(
        raw_float_2d=raw_float_2d,
        scaled_float_2d=scaled_float_2d,
        input_q_2d=input_q_2d,
        hidden_q_2d=hidden_q_2d,
    )

    print("\nGenerated mem files:")
    print(f" - {INPUT_MEM}")
    print(f" - {WEIGHTS_MEM}")
    print(f" - {BIAS_MEM}")
    print(f" - {EXPECTED_OUTPUT_MEM}")

    print("\nSaved run artifacts:")
    print(f" - {LAYER1_BATCH_JSON}")
    print(f" - {LAYER1_REF_NPZ}")

    print(f"\nBatch labels      : {labels}")
    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()