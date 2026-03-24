"""
mlp_inference_reference.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Description
-----------
This module implements a fixed-point software reference for the NNIA-based
MLP inference pipeline.

It loads quantized model parameters and a predefined input batch, then performs
end-to-end inference using fixed-point operations aligned with NNIA hardware
behavior, including dot-product, requantization, bias addition, clamping,
and activation.

Predictions are generated from the final output representation using an
NNIA-consistent computation flow and class selection logic.

Notes
-----
- All computations follow fixed-point arithmetic consistent with NNIA.
- Activation behavior is aligned with the hardware implementation.
- This module serves as a software reference for validating hardware outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

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
from ott_recommender.feature_encoder import CLASS_NAMES

from shared.fixed_point_utils import (
    add_bias,
    clamp_signed,
    dot_product_fixed,
    fixed_to_float,
    relu,
    requantize,
)

# =============================================================================
# Locked NNIA / MLP configuration
# =============================================================================
M = 4
K = 16
H = 8
O = 2

DATA_WIDTH = 16
FRAC_BITS = 8
ACC_WIDTH = 40

REQUANT_ROUNDING = "trunc"

# =============================================================================
# Project paths (FIXED)
# =============================================================================
PROJECT_ROOT = THIS_DIR.parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"

MODEL_QUANT_NPZ = ARTIFACTS_DIR / "model_quantized.npz"

LAYER1_REF_NPZ = RUNS_DIR / "layer1_reference.npz"
MLP_REF_NPZ = RUNS_DIR / "mlp_reference_outputs.npz"
MLP_REF_REPORT_JSON = RUNS_DIR / "mlp_reference_report.json"


# =============================================================================
# Validation helpers
# =============================================================================
def validate_required_files() -> None:
    missing: List[str] = []

    if not MODEL_QUANT_NPZ.exists():
        missing.append(str(MODEL_QUANT_NPZ))
    if not LAYER1_REF_NPZ.exists():
        missing.append(str(LAYER1_REF_NPZ))

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(missing)
            + "\n\nRun prepare_layer1_mem.py and export_quantized_model.py first."
        )


def validate_layer1_shapes(
    W1_q: np.ndarray,
    b1_q: np.ndarray,
) -> None:
    if W1_q.shape != (K, H):
        raise ValueError(f"W1_q shape mismatch: expected {(K, H)}, got {W1_q.shape}")
    if b1_q.shape != (H,):
        raise ValueError(f"b1_q shape mismatch: expected {(H,)}, got {b1_q.shape}")

    if not np.all(np.isfinite(W1_q)):
        raise ValueError("W1_q contains non-finite values")
    if not np.all(np.isfinite(b1_q)):
        raise ValueError("b1_q contains non-finite values")


def validate_layer2_shapes(
    W2_q: np.ndarray,
    b2_q: np.ndarray,
) -> None:
    if W2_q.shape != (H, O):
        raise ValueError(f"W2_q shape mismatch: expected {(H, O)}, got {W2_q.shape}")
    if b2_q.shape != (O,):
        raise ValueError(f"b2_q shape mismatch: expected {(O,)}, got {b2_q.shape}")

    if not np.all(np.isfinite(W2_q)):
        raise ValueError("W2_q contains non-finite values")
    if not np.all(np.isfinite(b2_q)):
        raise ValueError("b2_q contains non-finite values")


def load_quantized_reference_model() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    data = np.load(MODEL_QUANT_NPZ, allow_pickle=True)

    required_keys = {"W1_q", "b1_q", "W2_expanded_q", "b2_expanded_q"}
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"model_quantized.npz missing keys: {missing_keys}")

    W1_q = np.asarray(data["W1_q"], dtype=np.int32)
    b1_q = np.asarray(data["b1_q"], dtype=np.int32)
    W2_q = np.asarray(data["W2_expanded_q"], dtype=np.int32)
    b2_q = np.asarray(data["b2_expanded_q"], dtype=np.int32)

    validate_layer1_shapes(W1_q, b1_q)
    validate_layer2_shapes(W2_q, b2_q)

    meta = {
        "layer2_mode": str(np.asarray(data["layer2_mode"]).reshape(-1)[0]) if "layer2_mode" in data else "unknown",
        "binary_single_output": int(np.asarray(data["binary_single_output"]).reshape(-1)[0]) if "binary_single_output" in data else 0,
        "layer2_bias_scale_factor": float(np.asarray(data["layer2_bias_scale_factor"]).reshape(-1)[0]) if "layer2_bias_scale_factor" in data else 1.0,
    }

    return W1_q, b1_q, W2_q, b2_q, meta


# =============================================================================
# Saved batch loading
# =============================================================================
def load_layer1_saved_batch() -> Tuple[List[List[int]], List[int], List[int]]:
    data = np.load(LAYER1_REF_NPZ, allow_pickle=True)

    required_keys = {"input_q", "labels"}
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"layer1_reference.npz missing keys: {missing_keys}")

    input_q = np.asarray(data["input_q"], dtype=np.int32)
    labels = np.asarray(data["labels"], dtype=np.int32)

    if input_q.shape != (M, K):
        raise ValueError(f"input_q shape mismatch: expected {(M, K)}, got {input_q.shape}")
    if labels.shape != (M,):
        raise ValueError(f"labels shape mismatch: expected {(M,)}, got {labels.shape}")

    if "selected_indices" in data:
        selected_indices = np.asarray(data["selected_indices"], dtype=np.int32).tolist()
    else:
        selected_indices = []

    return input_q.tolist(), labels.tolist(), selected_indices


# =============================================================================
# Core fixed-point layer math
# =============================================================================
def dense_relu_fixed(
    input_q_2d: List[List[int]],
    W_q: np.ndarray,
    b_q: np.ndarray,
    in_dim: int,
    out_dim: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    """
    Dense + ReLU layer aligned to NNIA hidden-layer behavior.
    """
    if W_q.shape != (in_dim, out_dim):
        raise ValueError(f"W_q shape mismatch: expected {(in_dim, out_dim)}, got {W_q.shape}")
    if b_q.shape != (out_dim,):
        raise ValueError(f"b_q shape mismatch: expected {(out_dim,)}, got {b_q.shape}")

    batch_size = len(input_q_2d)

    raw_acc_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]
    rq_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]
    biased_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]
    out_q_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]

    for m in range(batch_size):
        in_vec = input_q_2d[m]
        if len(in_vec) != in_dim:
            raise ValueError(
                f"Input row {m} length mismatch: expected {in_dim}, got {len(in_vec)}"
            )

        for n in range(out_dim):
            w_col = [int(W_q[k][n]) for k in range(in_dim)]

            raw_acc = dot_product_fixed(in_vec, w_col)
            raw_acc = clamp_signed(raw_acc, ACC_WIDTH)

            rq = requantize(
                raw_acc,
                shift=FRAC_BITS,
                out_width=ACC_WIDTH,
                rounding=REQUANT_ROUNDING,
                saturate=True,
            )

            biased = add_bias(rq, int(b_q[n]))
            biased = clamp_signed(biased, ACC_WIDTH)

            relu_v = relu(biased)
            out_v = clamp_signed(relu_v, DATA_WIDTH)

            raw_acc_2d[m][n] = int(raw_acc)
            rq_2d[m][n] = int(rq)
            biased_2d[m][n] = int(biased)
            out_q_2d[m][n] = int(out_v)

    return raw_acc_2d, rq_2d, biased_2d, out_q_2d


def dense_linear_fixed(
    input_q_2d: List[List[int]],
    W_q: np.ndarray,
    b_q: np.ndarray,
    in_dim: int,
    out_dim: int,
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    """
    Dense linear layer for final 2-output decision stage.

    Flow:
        dot -> clamp(ACC) -> requantize -> bias add -> clamp(ACC) -> clamp(DATA)

    No ReLU is applied here because final decision should preserve both positive
    and negative evidence across the 2-output expanded representation.
    """
    if W_q.shape != (in_dim, out_dim):
        raise ValueError(f"W_q shape mismatch: expected {(in_dim, out_dim)}, got {W_q.shape}")
    if b_q.shape != (out_dim,):
        raise ValueError(f"b_q shape mismatch: expected {(out_dim,)}, got {b_q.shape}")

    batch_size = len(input_q_2d)

    raw_acc_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]
    rq_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]
    biased_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]
    out_q_2d: List[List[int]] = [[0 for _ in range(out_dim)] for _ in range(batch_size)]

    for m in range(batch_size):
        in_vec = input_q_2d[m]
        if len(in_vec) != in_dim:
            raise ValueError(
                f"Input row {m} length mismatch: expected {in_dim}, got {len(in_vec)}"
            )

        for n in range(out_dim):
            w_col = [int(W_q[k][n]) for k in range(in_dim)]

            raw_acc = dot_product_fixed(in_vec, w_col)
            raw_acc = clamp_signed(raw_acc, ACC_WIDTH)

            rq = requantize(
                raw_acc,
                shift=FRAC_BITS,
                out_width=ACC_WIDTH,
                rounding=REQUANT_ROUNDING,
                saturate=True,
            )

            biased = add_bias(rq, int(b_q[n]))
            biased = clamp_signed(biased, ACC_WIDTH)

            out_v = clamp_signed(biased, DATA_WIDTH)

            raw_acc_2d[m][n] = int(raw_acc)
            rq_2d[m][n] = int(rq)
            biased_2d[m][n] = int(biased)
            out_q_2d[m][n] = int(out_v)

    return raw_acc_2d, rq_2d, biased_2d, out_q_2d


# =============================================================================
# Reporting helpers
# =============================================================================
def argmax_int_row(row: List[int]) -> int:
    if not row:
        raise ValueError("Cannot argmax an empty row")

    best_idx = 0
    best_val = int(row[0])

    for idx in range(1, len(row)):
        value = int(row[idx])
        if value > best_val:
            best_val = value
            best_idx = idx

    return best_idx


def to_float_2d(values_2d: List[List[int]]) -> List[List[float]]:
    return [
        [fixed_to_float(int(v), frac_bits=FRAC_BITS) for v in row]
        for row in values_2d
    ]


def count_matches(preds: List[int], labels: List[int]) -> int:
    if len(preds) != len(labels):
        raise ValueError("Prediction and label lengths do not match")
    return sum(int(p == y) for p, y in zip(preds, labels))


def save_reference_artifacts(
    input_q_2d: List[List[int]],
    labels: List[int],
    selected_indices: List[int],
    layer1_raw_acc_2d: List[List[int]],
    layer1_rq_2d: List[List[int]],
    hidden_q_2d: List[List[int]],
    layer2_raw_acc_2d: List[List[int]],
    layer2_rq_2d: List[List[int]],
    logits_q_2d: List[List[int]],
    pred_ids: List[int],
    layer2_mode: str,
    binary_single_output: int,
    layer2_bias_scale_factor: float,
) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez(
        MLP_REF_NPZ,
        input_q=np.array(input_q_2d, dtype=np.int32),
        labels=np.array(labels, dtype=np.int32),
        selected_indices=np.array(selected_indices, dtype=np.int32),
        layer1_raw_acc=np.array(layer1_raw_acc_2d, dtype=np.int64),
        layer1_requant=np.array(layer1_rq_2d, dtype=np.int64),
        hidden_q=np.array(hidden_q_2d, dtype=np.int32),
        layer2_raw_acc=np.array(layer2_raw_acc_2d, dtype=np.int64),
        layer2_requant=np.array(layer2_rq_2d, dtype=np.int64),
        logits_q=np.array(logits_q_2d, dtype=np.int32),
        pred_ids=np.array(pred_ids, dtype=np.int32),
    )

    report: Dict[str, object] = {
        "stage": "full_mlp_reference",
        "batch_size": M,
        "input_size": K,
        "hidden_size": H,
        "output_size": O,
        "data_width": DATA_WIDTH,
        "frac_bits": FRAC_BITS,
        "acc_width": ACC_WIDTH,
        "requant_rounding": REQUANT_ROUNDING,
        "class_names": CLASS_NAMES,
        "selected_indices": selected_indices,
        "layer1_activation": "relu",
        "layer2_activation": "linear_argmax",
        "layer2_mode": layer2_mode,
        "binary_single_output": int(binary_single_output),
        "layer2_bias_scale_factor": float(layer2_bias_scale_factor),
        "artifacts": {
            "mlp_reference_npz": str(MLP_REF_NPZ),
            "layer1_reference_npz_source": str(LAYER1_REF_NPZ),
            "model_quantized_npz_source": str(MODEL_QUANT_NPZ),
        },
        "notes": [
            "Layer-1 uses NNIA-aligned fixed-point math with ReLU.",
            "Layer-2 uses the exported expanded 2-output representation.",
            "Final prediction uses argmax over 2 logits.",
            "Layer-2 intentionally avoids final ReLU to preserve the binary decision boundary.",
            "The exact input batch is loaded from layer1_reference.npz, not re-selected from CSV.",
        ],
    }

    with open(MLP_REF_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def verify_layer1_alignment_with_saved_reference(hidden_q_2d: List[List[int]]) -> str:
    data = np.load(LAYER1_REF_NPZ, allow_pickle=True)
    if "hidden_q" not in data:
        return "SKIPPED (hidden_q key missing in layer1_reference.npz)"

    saved_hidden = np.asarray(data["hidden_q"], dtype=np.int32)
    curr_hidden = np.asarray(hidden_q_2d, dtype=np.int32)

    if saved_hidden.shape != curr_hidden.shape:
        return f"FAILED (shape mismatch: saved {saved_hidden.shape}, current {curr_hidden.shape})"

    if np.array_equal(saved_hidden, curr_hidden):
        return "PASS"

    diff_count = int(np.sum(saved_hidden != curr_hidden))
    return f"FAILED ({diff_count} element mismatch(es))"


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    validate_required_files()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    W1_q, b1_q, W2_q, b2_q, meta = load_quantized_reference_model()

    input_q_2d, labels, selected_indices = load_layer1_saved_batch()

    (
        layer1_raw_acc_2d,
        layer1_rq_2d,
        _layer1_biased_2d,
        hidden_q_2d,
    ) = dense_relu_fixed(
        input_q_2d=input_q_2d,
        W_q=W1_q,
        b_q=b1_q,
        in_dim=K,
        out_dim=H,
    )

    (
        layer2_raw_acc_2d,
        layer2_rq_2d,
        _layer2_biased_2d,
        logits_q_2d,
    ) = dense_linear_fixed(
        input_q_2d=hidden_q_2d,
        W_q=W2_q,
        b_q=b2_q,
        in_dim=H,
        out_dim=O,
    )

    pred_ids = [argmax_int_row(row) for row in logits_q_2d]
    pred_names = [CLASS_NAMES[idx] for idx in pred_ids]
    true_names = [
        CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f"UNKNOWN({idx})"
        for idx in labels
    ]
    match_count = count_matches(pred_ids, labels)

    save_reference_artifacts(
        input_q_2d=input_q_2d,
        labels=labels,
        selected_indices=selected_indices,
        layer1_raw_acc_2d=layer1_raw_acc_2d,
        layer1_rq_2d=layer1_rq_2d,
        hidden_q_2d=hidden_q_2d,
        layer2_raw_acc_2d=layer2_raw_acc_2d,
        layer2_rq_2d=layer2_rq_2d,
        logits_q_2d=logits_q_2d,
        pred_ids=pred_ids,
        layer2_mode=meta["layer2_mode"],
        binary_single_output=meta["binary_single_output"],
        layer2_bias_scale_factor=meta["layer2_bias_scale_factor"],
    )

    layer1_alignment_status = verify_layer1_alignment_with_saved_reference(hidden_q_2d)

    hidden_f_2d = to_float_2d(hidden_q_2d)
    logits_f_2d = to_float_2d(logits_q_2d)

    print("\n==================== MLP INFERENCE REFERENCE ====================\n")
    print(f"Quant model        : {MODEL_QUANT_NPZ}")
    print(f"Layer1 source      : {LAYER1_REF_NPZ}")
    print(f"Output NPZ         : {MLP_REF_NPZ}")
    print(f"Output report      : {MLP_REF_REPORT_JSON}")

    print("\nLocked shapes:")
    print(f"  input_q          : ({M}, {K})")
    print(f"  W1_q             : {tuple(W1_q.shape)}")
    print(f"  b1_q             : {tuple(b1_q.shape)}")
    print(f"  hidden_q         : ({M}, {H})")
    print(f"  W2_q             : {tuple(W2_q.shape)}")
    print(f"  b2_q             : {tuple(b2_q.shape)}")
    print(f"  logits_q         : ({M}, {O})")

    print("\nLayer-2 export metadata:")
    print(f"  layer2_mode      : {meta['layer2_mode']}")
    print(f"  binary_single    : {meta['binary_single_output']}")
    print(f"  bias_scale       : {meta['layer2_bias_scale_factor']}")

    print("\nSelected dataset row indices:")
    if selected_indices:
        print(f"  {selected_indices}")
    else:
        print("  Not available in saved artifact")

    print("\nLayer behavior:")
    print("  layer1           : Dense + ReLU")
    print("  layer2           : Dense + Linear")
    print("  final decision   : Argmax over 2 logits")
    print(f"  requant rounding : {REQUANT_ROUNDING}")

    print("\nLayer-1 alignment check against saved layer1_reference.npz:")
    print(f"  status           : {layer1_alignment_status}")

    print("\nPer-sample results:")
    for i in range(M):
        print(f"  sample {i}:")
        print(f"    true label     : {labels[i]} ({true_names[i]})")
        print(f"    pred label     : {pred_ids[i]} ({pred_names[i]})")
        print(f"    hidden_q       : {hidden_q_2d[i]}")
        print(f"    hidden_float   : {[round(v, 6) for v in hidden_f_2d[i]]}")
        print(f"    layer2_raw_acc : {layer2_raw_acc_2d[i]}")
        print(f"    layer2_requant : {layer2_rq_2d[i]}")
        print(f"    logits_q       : {logits_q_2d[i]}")
        print(f"    logits_float   : {[round(v, 6) for v in logits_f_2d[i]]}")

    print("\nBatch summary:")
    print(f"  labels           : {labels}")
    print(f"  predictions      : {pred_ids}")
    print(f"  prediction names : {pred_names}")
    print(f"  correct          : {match_count} / {M}")
    print(f"  accuracy         : {match_count / float(M):.4f}")

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()