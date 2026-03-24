"""
generate_data.py

Author: Yuvraj Singh

Project: Neural Network Inference Accelerator (NNIA)

NNIA tiled data generator and Python golden-reference producer.

Purpose:
- Generate deterministic fixed-point test data for a tiled NNIA flow
- Create input / weight / bias memory files for RTL simulation
- Compute full raw accumulator reference
- Compute tile-step reference data for easier RTL debug
- Compute final post-processed expected output (requant + bias + ReLU)

Architecture intent:
- Matrix-style neural network inference data preparation
- Suitable for a 4x4 PE-array / tiled accelerator style flow
- Output-stationary friendly reference generation

Locked configuration in this file:
- M = 4   : activation rows
- K = 16  : input features
- N = 8   : output features
- TILE_M = 4
- TILE_K = 4
- TILE_N = 4

Generated files:
- mem/input.mem
- mem/weights.mem
- mem/bias.mem
- mem/raw_psum_ref.mem
- mem/tile_ref.mem
- mem/expected_output.mem
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# Project-root-aware import setup
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "Python"
MEM_DIR = PROJECT_ROOT / "mem"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from shared.fixed_point_utils import (
    add_bias,
    clamp_signed,
    fixed_to_float,
    float_to_fixed,
    flatten_2d_row_major,
    relu,
    requantize,
    write_mem_file,
)

# =============================================================================
# Locked NNIA tiled configuration
# =============================================================================
M = 4
K = 16
N = 8

TILE_M = 4
TILE_K = 4
TILE_N = 4

DATA_WIDTH = 16
FRAC_BITS = 8
ACC_WIDTH = 40

RANDOM_SEED = 7

INPUT_MIN_REAL = -1.0
INPUT_MAX_REAL = 1.0

WEIGHT_MIN_REAL = -1.0
WEIGHT_MAX_REAL = 1.0

BIAS_MIN_REAL = -0.5
BIAS_MAX_REAL = 0.5


# =============================================================================
# Validation helpers
# =============================================================================
def validate_config() -> None:
    if M <= 0 or K <= 0 or N <= 0:
        raise ValueError("M, K, N must all be > 0")

    if TILE_M <= 0 or TILE_K <= 0 or TILE_N <= 0:
        raise ValueError("TILE_M, TILE_K, TILE_N must all be > 0")

    if M % TILE_M != 0:
        raise ValueError(f"M={M} must be divisible by TILE_M={TILE_M}")

    if K % TILE_K != 0:
        raise ValueError(f"K={K} must be divisible by TILE_K={TILE_K}")

    if N % TILE_N != 0:
        raise ValueError(f"N={N} must be divisible by TILE_N={TILE_N}")

    if DATA_WIDTH <= 0 or ACC_WIDTH <= 0:
        raise ValueError("DATA_WIDTH and ACC_WIDTH must be > 0")

    if FRAC_BITS < 0:
        raise ValueError("FRAC_BITS must be >= 0")


def validate_shapes(
    input_q: List[List[int]],
    weight_q: List[List[int]],
    bias_q: List[int],
) -> None:
    if len(input_q) != M or any(len(row) != K for row in input_q):
        raise ValueError(f"input_q shape mismatch: expected ({M}, {K})")

    if len(weight_q) != K or any(len(row) != N for row in weight_q):
        raise ValueError(f"weight_q shape mismatch: expected ({K}, {N})")

    if len(bias_q) != N:
        raise ValueError(f"bias_q shape mismatch: expected ({N},)")


def validate_generated_files() -> None:
    expected_counts = {
        MEM_DIR / "input.mem": M * K,
        MEM_DIR / "weights.mem": K * N,
        MEM_DIR / "bias.mem": N,
        MEM_DIR / "raw_psum_ref.mem": M * N,
        MEM_DIR / "tile_ref.mem": (M // TILE_M) * (N // TILE_N) * (K // TILE_K) * TILE_M * TILE_N,
        MEM_DIR / "expected_output.mem": M * N,
    }

    for path, expected_lines in expected_counts.items():
        with open(path, "r", encoding="utf-8") as f:
            found_lines = sum(1 for line in f if line.strip())

        if found_lines != expected_lines:
            raise ValueError(
                f"{path} expected {expected_lines} lines but found {found_lines}"
            )


# =============================================================================
# Data generation helpers
# =============================================================================
def generate_real_matrix(
    rng: random.Random,
    rows: int,
    cols: int,
    min_val: float,
    max_val: float,
) -> List[List[float]]:
    return [
        [rng.uniform(min_val, max_val) for _ in range(cols)]
        for _ in range(rows)
    ]


def generate_real_vector(
    rng: random.Random,
    length: int,
    min_val: float,
    max_val: float,
) -> List[float]:
    return [rng.uniform(min_val, max_val) for _ in range(length)]


def real_matrix_to_fixed(
    real_mat: List[List[float]],
    data_width: int = DATA_WIDTH,
    frac_bits: int = FRAC_BITS,
) -> List[List[int]]:
    return [
        [
            float_to_fixed(v, data_width=data_width, frac_bits=frac_bits)
            for v in row
        ]
        for row in real_mat
    ]


def real_vector_to_fixed(
    real_vec: List[float],
    data_width: int = DATA_WIDTH,
    frac_bits: int = FRAC_BITS,
) -> List[int]:
    return [
        float_to_fixed(v, data_width=data_width, frac_bits=frac_bits)
        for v in real_vec
    ]


# =============================================================================
# Core tiled golden model
# =============================================================================
def compute_raw_psum_reference(
    input_q: List[List[int]],
    weight_q: List[List[int]],
) -> Tuple[List[List[int]], List[int]]:
    raw_psum_ref = [[0 for _ in range(N)] for _ in range(M)]
    tile_ref_flat: List[int] = []

    for tm in range(0, M, TILE_M):
        for tn in range(0, N, TILE_N):
            local_acc = [[0 for _ in range(TILE_N)] for _ in range(TILE_M)]

            for tk in range(0, K, TILE_K):
                for i in range(TILE_M):
                    for j in range(TILE_N):
                        acc_val = local_acc[i][j]

                        for kk in range(TILE_K):
                            a_val = input_q[tm + i][tk + kk]
                            w_val = weight_q[tk + kk][tn + j]
                            acc_val += a_val * w_val
                            acc_val = clamp_signed(acc_val, ACC_WIDTH)

                        local_acc[i][j] = acc_val

                for i in range(TILE_M):
                    for j in range(TILE_N):
                        tile_ref_flat.append(local_acc[i][j])

            for i in range(TILE_M):
                for j in range(TILE_N):
                    raw_psum_ref[tm + i][tn + j] = local_acc[i][j]

    return raw_psum_ref, tile_ref_flat


def postprocess_output_matrix(
    raw_psum_ref: List[List[int]],
    bias_q: List[int],
) -> List[List[int]]:
    out_q = [[0 for _ in range(N)] for _ in range(M)]

    for i in range(M):
        for j in range(N):
            val = raw_psum_ref[i][j]

            val = requantize(
                val,
                shift=FRAC_BITS,
                out_width=ACC_WIDTH,
                rounding="trunc",
                saturate=True,
            )

            val = add_bias(val, bias_q[j])
            val = clamp_signed(val, ACC_WIDTH)
            val = relu(val)
            val = clamp_signed(val, DATA_WIDTH)

            out_q[i][j] = val

    return out_q


# =============================================================================
# Debug helpers
# =============================================================================
def fixed_matrix_to_float_string(mat: List[List[int]], frac_bits: int = FRAC_BITS) -> str:
    rows = []
    for row in mat:
        row_f = [round(fixed_to_float(v, frac_bits), 4) for v in row]
        rows.append(str(row_f))
    return "\n".join(rows)


def fixed_vector_to_float_string(vec: List[int], frac_bits: int = FRAC_BITS) -> str:
    vals = [round(fixed_to_float(v, frac_bits), 4) for v in vec]
    return str(vals)


def print_config() -> None:
    print("\n==================== NNIA GENERATE DATA ====================\n")
    print("Configuration")
    print(f"M          : {M}")
    print(f"K          : {K}")
    print(f"N          : {N}")
    print(f"TILE_M     : {TILE_M}")
    print(f"TILE_K     : {TILE_K}")
    print(f"TILE_N     : {TILE_N}")
    print(f"DATA_WIDTH : {DATA_WIDTH}")
    print(f"FRAC_BITS  : {FRAC_BITS}")
    print(f"ACC_WIDTH  : {ACC_WIDTH}")
    print(f"SEED       : {RANDOM_SEED}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"MEM_DIR     : {MEM_DIR}")


def print_summary(
    input_q: List[List[int]],
    weight_q: List[List[int]],
    bias_q: List[int],
    raw_psum_ref: List[List[int]],
    expected_output_q: List[List[int]],
    tile_ref_flat: List[int],
) -> None:
    print("\nInput matrix shape      :", (len(input_q), len(input_q[0]) if input_q else 0))
    print("Weight matrix shape     :", (len(weight_q), len(weight_q[0]) if weight_q else 0))
    print("Bias vector shape       :", (len(bias_q),))
    print(
        "Raw psum matrix shape   :",
        (len(raw_psum_ref), len(raw_psum_ref[0]) if raw_psum_ref else 0),
    )
    print(
        "Expected output shape   :",
        (len(expected_output_q), len(expected_output_q[0]) if expected_output_q else 0),
    )
    print("Tile reference entries  :", len(tile_ref_flat))

    print("\nApprox input matrix (Q8.8 -> real):")
    print(fixed_matrix_to_float_string(input_q))

    print("\nApprox weight matrix (Q8.8 -> real):")
    print(fixed_matrix_to_float_string(weight_q))

    print("\nApprox bias vector (Q8.8 -> real):")
    print(fixed_vector_to_float_string(bias_q))

    print("\nApprox expected output (Q8.8 -> real):")
    print(fixed_matrix_to_float_string(expected_output_q))

    print(f"\nGenerated files in {MEM_DIR}")
    print(" - input.mem")
    print(" - weights.mem")
    print(" - bias.mem")
    print(" - raw_psum_ref.mem")
    print(" - tile_ref.mem")
    print(" - expected_output.mem")

    print("\nSTATUS: PASS")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    validate_config()
    MEM_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)

    input_real = generate_real_matrix(rng, M, K, INPUT_MIN_REAL, INPUT_MAX_REAL)
    weight_real = generate_real_matrix(rng, K, N, WEIGHT_MIN_REAL, WEIGHT_MAX_REAL)
    bias_real = generate_real_vector(rng, N, BIAS_MIN_REAL, BIAS_MAX_REAL)

    input_q = real_matrix_to_fixed(input_real)
    weight_q = real_matrix_to_fixed(weight_real)
    bias_q = real_vector_to_fixed(bias_real)

    validate_shapes(input_q, weight_q, bias_q)

    raw_psum_ref, tile_ref_flat = compute_raw_psum_reference(input_q, weight_q)
    expected_output_q = postprocess_output_matrix(raw_psum_ref, bias_q)

    input_flat = flatten_2d_row_major(input_q)
    weight_flat = flatten_2d_row_major(weight_q)
    bias_flat = bias_q[:]
    raw_psum_flat = flatten_2d_row_major(raw_psum_ref)
    expected_output_flat = flatten_2d_row_major(expected_output_q)

    write_mem_file(MEM_DIR / "input.mem", input_flat, DATA_WIDTH, radix="hex")
    write_mem_file(MEM_DIR / "weights.mem", weight_flat, DATA_WIDTH, radix="hex")
    write_mem_file(MEM_DIR / "bias.mem", bias_flat, DATA_WIDTH, radix="hex")
    write_mem_file(MEM_DIR / "raw_psum_ref.mem", raw_psum_flat, ACC_WIDTH, radix="hex")
    write_mem_file(MEM_DIR / "tile_ref.mem", tile_ref_flat, ACC_WIDTH, radix="hex")
    write_mem_file(
        MEM_DIR / "expected_output.mem",
        expected_output_flat,
        DATA_WIDTH,
        radix="hex",
    )

    validate_generated_files()

    print_config()
    print_summary(
        input_q=input_q,
        weight_q=weight_q,
        bias_q=bias_q,
        raw_psum_ref=raw_psum_ref,
        expected_output_q=expected_output_q,
        tile_ref_flat=tile_ref_flat,
    )


if __name__ == "__main__":
    main()