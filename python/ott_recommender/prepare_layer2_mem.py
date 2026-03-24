"""
prepare_layer2_mem.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Description
-----------
This module prepares NNIA-compatible memory files for the second stage
of the MLP inference pipeline.

It loads hidden-layer activations from previous stage artifacts and
maps them into the fixed NNIA input format using padding and embedding.

Quantized model parameters are similarly embedded to match NNIA
dimensions, allowing reuse of the same hardware without modification.

The module computes expected outputs using NNIA-aligned fixed-point
arithmetic and generates memory files for hardware simulation and
verification.

Notes
-----
- Outputs follow NNIA computation behavior with activation applied.
- Only a subset of output channels correspond to meaningful class scores.
- Remaining channels are structural padding required by NNIA dimensions.
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
# FIXED import
# =============================================================================
from shared.fixed_point_utils import (
    add_bias,
    clamp_signed,
    dot_product_fixed,
    fixed_to_float,
    flatten_2d_row_major,
    relu,
    requantize,
    write_mem_file,
)

# =============================================================================
# Locked NNIA / MLP configuration
# =============================================================================
M = 4
K_NNIA = 16
N_NNIA = 8

H_MLP = 8
O_MLP = 2

DATA_WIDTH = 16
FRAC_BITS = 8
ACC_WIDTH = 40

LAYER2_SHIFT = FRAC_BITS
REQUANT_ROUNDING = "trunc"

# =============================================================================
# Project paths (FIXED)
# =============================================================================
PROJECT_ROOT = THIS_DIR.parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
MEM_DIR = PROJECT_ROOT / "mem"

MODEL_QUANT_NPZ = ARTIFACTS_DIR / "model_quantized.npz"
MLP_REF_NPZ = RUNS_DIR / "mlp_reference_outputs.npz"
LAYER1_REF_NPZ = RUNS_DIR / "layer1_reference.npz"

INPUT_MEM = MEM_DIR / "input.mem"
WEIGHTS_MEM = MEM_DIR / "weights.mem"
BIAS_MEM = MEM_DIR / "bias.mem"
EXPECTED_OUTPUT_MEM = MEM_DIR / "expected_output_l2.mem"

LAYER2_BATCH_JSON = RUNS_DIR / "layer2_batch_info.json"
LAYER2_REF_NPZ = RUNS_DIR / "layer2_reference.npz"


# =============================================================================
# Validation helpers
# =============================================================================
def validate_required_files() -> None:
    missing: List[str] = []

    if not MODEL_QUANT_NPZ.exists():
        missing.append(str(MODEL_QUANT_NPZ))

    if not MLP_REF_NPZ.exists() and not LAYER1_REF_NPZ.exists():
        missing.append(
            f"Either {MLP_REF_NPZ} or {LAYER1_REF_NPZ} must exist to provide hidden_q"
        )

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(missing)
            + "\n\nRun export_quantized_model.py and layer-1/reference scripts first."
        )


def validate_model_shapes(W2_q: np.ndarray, b2_q: np.ndarray) -> None:
    if W2_q.shape != (H_MLP, O_MLP):
        raise ValueError(f"W2_q shape mismatch: expected {(H_MLP, O_MLP)}, got {W2_q.shape}")
    if b2_q.shape != (O_MLP,):
        raise ValueError(f"b2_q shape mismatch: expected {(O_MLP,)}, got {b2_q.shape}")

    if not np.all(np.isfinite(W2_q)):
        raise ValueError("W2_q contains non-finite values")
    if not np.all(np.isfinite(b2_q)):
        raise ValueError("b2_q contains non-finite values")


# =============================================================================
# Load hidden activations
# =============================================================================
def load_hidden_q() -> Tuple[List[List[int]], List[int], List[int], str]:
    candidates = [MLP_REF_NPZ, LAYER1_REF_NPZ]

    for path in candidates:
        if not path.exists():
            continue

        data = np.load(path, allow_pickle=True)
        if "hidden_q" not in data:
            continue

        hidden_q = np.asarray(data["hidden_q"], dtype=np.int32)
        if hidden_q.shape != (M, H_MLP):
            raise ValueError(
                f"{path.name}: hidden_q shape mismatch: expected {(M, H_MLP)}, got {hidden_q.shape}"
            )

        if "labels" in data:
            labels = np.asarray(data["labels"], dtype=np.int32)
            if labels.shape != (M,):
                raise ValueError(
                    f"{path.name}: labels shape mismatch: expected {(M,)}, got {labels.shape}"
                )
            label_list = labels.tolist()
        else:
            label_list = [-1] * M

        if "selected_indices" in data:
            selected_indices = np.asarray(data["selected_indices"], dtype=np.int32).tolist()
        else:
            selected_indices = []

        return hidden_q.tolist(), label_list, selected_indices, path.name

    raise FileNotFoundError("Could not find valid hidden_q in saved run artifacts.")


# =============================================================================
# Padding / embedding helpers
# =============================================================================
def pad_hidden_to_nnia_input(hidden_q_2d: List[List[int]]) -> List[List[int]]:
    input_padded_q_2d: List[List[int]] = []

    for row_idx, row in enumerate(hidden_q_2d):
        if len(row) != H_MLP:
            raise ValueError(
                f"Hidden row {row_idx} length mismatch: expected {H_MLP}, got {len(row)}"
            )

        padded = [int(v) for v in row] + [0] * (K_NNIA - H_MLP)
        input_padded_q_2d.append(padded)

    return input_padded_q_2d


def embed_w2_into_nnia_weights(W2_q: np.ndarray) -> List[List[int]]:
    W2_padded_q = [[0 for _ in range(N_NNIA)] for _ in range(K_NNIA)]

    for k in range(H_MLP):
        for n in range(O_MLP):
            W2_padded_q[k][n] = int(W2_q[k][n])

    return W2_padded_q


def embed_b2_into_nnia_bias(b2_q: np.ndarray) -> List[int]:
    bias_padded = [0 for _ in range(N_NNIA)]
    for n in range(O_MLP):
        bias_padded[n] = int(b2_q[n])
    return bias_padded


# =============================================================================
# RTL-aligned software reference
# =============================================================================
def compute_layer2_expected_output(
    input_padded_q_2d: List[List[int]],
    W2_padded_q_2d: List[List[int]],
    b2_padded_q: List[int],
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
    if len(input_padded_q_2d) != M:
        raise ValueError(f"Input batch row count mismatch: expected {M}, got {len(input_padded_q_2d)}")
    if len(W2_padded_q_2d) != K_NNIA:
        raise ValueError(f"Weight row count mismatch: expected {K_NNIA}, got {len(W2_padded_q_2d)}")
    if len(b2_padded_q) != N_NNIA:
        raise ValueError(f"Bias length mismatch: expected {N_NNIA}, got {len(b2_padded_q)}")

    raw_acc_2d = [[0 for _ in range(N_NNIA)] for _ in range(M)]
    rq_2d = [[0 for _ in range(N_NNIA)] for _ in range(M)]
    biased_2d = [[0 for _ in range(N_NNIA)] for _ in range(M)]
    output_q_2d = [[0 for _ in range(N_NNIA)] for _ in range(M)]

    for m in range(M):
        in_vec = input_padded_q_2d[m]
        if len(in_vec) != K_NNIA:
            raise ValueError(
                f"Input row {m} length mismatch: expected {K_NNIA}, got {len(in_vec)}"
            )

        for n in range(N_NNIA):
            w_col = [int(W2_padded_q_2d[k][n]) for k in range(K_NNIA)]

            raw_acc = dot_product_fixed(in_vec, w_col)
            raw_acc = clamp_signed(raw_acc, ACC_WIDTH)

            rq = requantize(
                raw_acc,
                shift=LAYER2_SHIFT,
                out_width=ACC_WIDTH,
                rounding=REQUANT_ROUNDING,
                saturate=True,
            )

            biased = add_bias(rq, int(b2_padded_q[n]))
            biased = clamp_signed(biased, ACC_WIDTH)

            out_val = relu(biased)
            out_val = clamp_signed(out_val, DATA_WIDTH)

            raw_acc_2d[m][n] = int(raw_acc)
            rq_2d[m][n] = int(rq)
            biased_2d[m][n] = int(biased)
            output_q_2d[m][n] = int(out_val)

    return raw_acc_2d, rq_2d, biased_2d, output_q_2d


def compute_pred_ids_from_first2(output_q_2d: List[List[int]]) -> List[int]:
    pred_ids: List[int] = []

    for row_idx, row in enumerate(output_q_2d):
        if len(row) != N_NNIA:
            raise ValueError(
                f"Output row {row_idx} length mismatch: expected {N_NNIA}, got {len(row)}"
            )

        score0 = int(row[0])
        score1 = int(row[1])

        pred_ids.append(1 if score1 > score0 else 0)

    return pred_ids


# =============================================================================
# Save helpers
# =============================================================================
def save_run_artifacts(
    hidden_q_2d: List[List[int]],
    input_padded_q_2d: List[List[int]],
    W2_padded_q_2d: List[List[int]],
    b2_padded_q: List[int],
    expected_q_2d: List[List[int]],
    labels: List[int],
    pred_ids: List[int],
    selected_indices: List[int],
    hidden_source_name: str,
    raw_acc_2d: List[List[int]],
    rq_2d: List[List[int]],
    biased_2d: List[List[int]],
    layer2_mode: str,
    binary_single_output: int,
) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    info: Dict[str, object] = {
        "stage": "layer2",
        "batch_size": M,
        "hidden_size_original": H_MLP,
        "output_size_original": O_MLP,
        "nnia_input_size": K_NNIA,
        "nnia_output_size": N_NNIA,
        "data_width": DATA_WIDTH,
        "frac_bits": FRAC_BITS,
        "acc_width": ACC_WIDTH,
        "layer2_shift": LAYER2_SHIFT,
        "requant_rounding": REQUANT_ROUNDING,
        "hidden_source": hidden_source_name,
        "layer2_behavior": "Dense + ReLU",
        "layer2_mode": layer2_mode,
        "binary_single_output": int(binary_single_output),
        "meaningful_output_columns": [0, 1],
        "dummy_output_columns": [2, 3, 4, 5, 6, 7],
        "labels": labels,
        "predictions_from_first2": pred_ids,
        "selected_indices": selected_indices,
        "notes": [
            "expected_output_l2.mem stores the layer-2 golden output for RTL comparison.",
            "Only output columns 0 and 1 are meaningful class-score columns.",
            "Remaining output columns are NNIA padding outputs.",
            "Layer-2 AI path uses trunc-style rounding to match RTL behavior.",
        ],
    }

    with open(LAYER2_BATCH_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    np.savez(
        LAYER2_REF_NPZ,
        hidden_q=np.array(hidden_q_2d, dtype=np.int32),
        input_padded_q=np.array(input_padded_q_2d, dtype=np.int32),
        W2_padded_q=np.array(W2_padded_q_2d, dtype=np.int32),
        b2_padded_q=np.array(b2_padded_q, dtype=np.int32),
        expected_output_q=np.array(expected_q_2d, dtype=np.int32),
        labels=np.array(labels, dtype=np.int32),
        pred_ids=np.array(pred_ids, dtype=np.int32),
        selected_indices=np.array(selected_indices, dtype=np.int32),
        raw_acc=np.array(raw_acc_2d, dtype=np.int64),
        requant=np.array(rq_2d, dtype=np.int64),
        biased=np.array(biased_2d, dtype=np.int64),
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    validate_required_files()
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(MODEL_QUANT_NPZ, allow_pickle=True)

    required_keys = {"W2_expanded_q", "b2_expanded_q"}
    missing_keys = [k for k in required_keys if k not in data]
    if missing_keys:
        raise ValueError(f"model_quantized.npz missing keys: {missing_keys}")

    W2_q = np.asarray(data["W2_expanded_q"], dtype=np.int32)
    b2_q = np.asarray(data["b2_expanded_q"], dtype=np.int32)
    validate_model_shapes(W2_q, b2_q)

    layer2_mode = str(np.asarray(data["layer2_mode"]).reshape(-1)[0]) if "layer2_mode" in data else "unknown"
    binary_single_output = int(np.asarray(data["binary_single_output"]).reshape(-1)[0]) if "binary_single_output" in data else 0

    hidden_q_2d, labels, selected_indices, hidden_source_name = load_hidden_q()

    input_padded_q_2d = pad_hidden_to_nnia_input(hidden_q_2d)
    W2_padded_q_2d = embed_w2_into_nnia_weights(W2_q)
    b2_padded_q = embed_b2_into_nnia_bias(b2_q)

    (
        raw_acc_2d,
        rq_2d,
        biased_2d,
        expected_q_2d,
    ) = compute_layer2_expected_output(
        input_padded_q_2d=input_padded_q_2d,
        W2_padded_q_2d=W2_padded_q_2d,
        b2_padded_q=b2_padded_q,
    )

    pred_ids = compute_pred_ids_from_first2(expected_q_2d)

    input_flat = flatten_2d_row_major(input_padded_q_2d)
    weights_flat = flatten_2d_row_major(W2_padded_q_2d)
    bias_flat = [int(v) for v in b2_padded_q]
    expected_flat = flatten_2d_row_major(expected_q_2d)

    write_mem_file(INPUT_MEM, input_flat, DATA_WIDTH, radix="hex")
    write_mem_file(WEIGHTS_MEM, weights_flat, DATA_WIDTH, radix="hex")
    write_mem_file(BIAS_MEM, bias_flat, DATA_WIDTH, radix="hex")
    write_mem_file(EXPECTED_OUTPUT_MEM, expected_flat, DATA_WIDTH, radix="hex")

    save_run_artifacts(
        hidden_q_2d=hidden_q_2d,
        input_padded_q_2d=input_padded_q_2d,
        W2_padded_q_2d=W2_padded_q_2d,
        b2_padded_q=b2_padded_q,
        expected_q_2d=expected_q_2d,
        labels=labels,
        pred_ids=pred_ids,
        selected_indices=selected_indices,
        hidden_source_name=hidden_source_name,
        raw_acc_2d=raw_acc_2d,
        rq_2d=rq_2d,
        biased_2d=biased_2d,
        layer2_mode=layer2_mode,
        binary_single_output=binary_single_output,
    )

    print("\n==================== PREPARE LAYER 2 MEM ====================\n")
    print(f"Quant model       : {MODEL_QUANT_NPZ}")
    print(f"Hidden source     : {hidden_source_name}")
    print(f"Layer2 shift      : {LAYER2_SHIFT}")
    print(f"Requant rounding  : {REQUANT_ROUNDING}")

    print("\nExport metadata:")
    print(f"  layer2_mode     : {layer2_mode}")
    print(f"  binary_single   : {binary_single_output}")

    print("\nEmbedded NNIA shapes:")
    print(f"  input_padded_q  : ({M}, {K_NNIA})")
    print(f"  W2_padded_q     : ({K_NNIA}, {N_NNIA})")
    print(f"  b2_padded_q     : ({N_NNIA},)")
    print(f"  expected_output : ({M}, {N_NNIA})")

    print("\nGenerated mem files:")
    print(f" - {INPUT_MEM}")
    print(f" - {WEIGHTS_MEM}")
    print(f" - {BIAS_MEM}")
    print(f" - {EXPECTED_OUTPUT_MEM}")

    print("\nSaved run artifacts:")
    print(f" - {LAYER2_BATCH_JSON}")
    print(f" - {LAYER2_REF_NPZ}")

    if selected_indices:
        print("\nDataset rows:")
        print(f"  {selected_indices}")

    print(f"\nLabels             : {labels}")
    print(f"Predictions(first2): {pred_ids}")

    print("\nLayer-2 debug (sample 0, class columns 0..1):")
    for n in range(O_MLP):
        print(
            f"  class {n}: "
            f"raw_acc={raw_acc_2d[0][n]}, "
            f"rq={rq_2d[0][n]}, "
            f"biased={biased_2d[0][n]}, "
            f"out={expected_q_2d[0][n]}, "
            f"out_float={fixed_to_float(expected_q_2d[0][n], FRAC_BITS):.6f}"
        )

    print("\nSample 0 preview:")
    print(f"  hidden_q[0]      : {hidden_q_2d[0]}")
    print(f"  input_padded[0]  : {input_padded_q_2d[0]}")
    print(f"  raw_acc[0][0:2]  : {raw_acc_2d[0][:O_MLP]}")
    print(f"  rq[0][0:2]       : {rq_2d[0][:O_MLP]}")
    print(f"  biased[0][0:2]   : {biased_2d[0][:O_MLP]}")
    print(f"  expected_out[0]  : {expected_q_2d[0]}")

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()