"""
tile_golden_model.py

Author: Yuvraj Singh

Project: Neural Network Inference Accelerator (NNIA)

NNIA tile-aware golden model and debug reference checker.

Purpose:
- Reconstruct NNIA matrices from generated .mem files
- Recompute tiled raw partial sums in the exact locked tile order
- Recompute final expected output using the same fixed-point post-processing flow
- Cross-check generated reference files for alignment and debug
- Provide a reusable Python-side tiled golden model for RTL bring-up

Expected flow relationship:
- generate_data.py writes deterministic .mem files
- this file reads those .mem files back
- this file recomputes the tiled flow independently
- results are compared against:
    - raw_psum_ref.mem
    - tile_ref.mem
    - expected_output.mem

Locked architecture assumptions:
- Matrix multiplication style NN inference
- Output-stationary friendly tiled accumulation
- Tile traversal order:
    [tile_m][tile_n][tile_k][local_m][local_n]

Files read from mem/:
- input.mem
- weights.mem
- bias.mem
- raw_psum_ref.mem
- tile_ref.mem
- expected_output.mem

This file does NOT generate random data.
It validates and explains already generated data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# =============================================================================
# Project-root-aware import and path setup
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
    read_mem_file,
    relu,
    requantize,
    reshape_1d_to_2d,
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


# =============================================================================
# Validation helpers
# =============================================================================
def validate_config() -> None:
    """Validate dimensions and tile compatibility."""
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


def validate_mem_files_exist() -> None:
    """Ensure all required memory files exist."""
    required_files = [
        MEM_DIR / "input.mem",
        MEM_DIR / "weights.mem",
        MEM_DIR / "bias.mem",
        MEM_DIR / "raw_psum_ref.mem",
        MEM_DIR / "tile_ref.mem",
        MEM_DIR / "expected_output.mem",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required .mem files are missing:\n" + "\n".join(missing)
        )


def validate_loaded_lengths(
    input_flat: List[int],
    weight_flat: List[int],
    bias_flat: List[int],
    raw_psum_ref_flat: List[int],
    tile_ref_flat: List[int],
    expected_output_flat: List[int],
) -> None:
    """Validate loaded file lengths against locked configuration."""
    expected_tile_entries = (
        (M // TILE_M)
        * (N // TILE_N)
        * (K // TILE_K)
        * TILE_M
        * TILE_N
    )

    checks = [
        ("input.mem", len(input_flat), M * K),
        ("weights.mem", len(weight_flat), K * N),
        ("bias.mem", len(bias_flat), N),
        ("raw_psum_ref.mem", len(raw_psum_ref_flat), M * N),
        ("tile_ref.mem", len(tile_ref_flat), expected_tile_entries),
        ("expected_output.mem", len(expected_output_flat), M * N),
    ]

    for name, found, expected in checks:
        if found != expected:
            raise ValueError(
                f"{name} length mismatch: found {found}, expected {expected}"
            )


# =============================================================================
# Load helpers
# =============================================================================
def load_mem_contents() -> Tuple[
    List[List[int]],
    List[List[int]],
    List[int],
    List[List[int]],
    List[int],
    List[List[int]],
]:
    """
    Read all generated mem files and reconstruct matrices/vectors.

    Returns:
    - input_q              : (M, K)
    - weight_q             : (K, N)
    - bias_q               : (N,)
    - raw_psum_ref_mem     : (M, N)
    - tile_ref_mem_flat    : flattened tile-step reference
    - expected_output_mem  : (M, N)
    """
    input_flat = read_mem_file(MEM_DIR / "input.mem", DATA_WIDTH, radix="hex")
    weight_flat = read_mem_file(MEM_DIR / "weights.mem", DATA_WIDTH, radix="hex")
    bias_flat = read_mem_file(MEM_DIR / "bias.mem", DATA_WIDTH, radix="hex")
    raw_psum_ref_flat = read_mem_file(
        MEM_DIR / "raw_psum_ref.mem",
        ACC_WIDTH,
        radix="hex",
    )
    tile_ref_flat = read_mem_file(MEM_DIR / "tile_ref.mem", ACC_WIDTH, radix="hex")
    expected_output_flat = read_mem_file(
        MEM_DIR / "expected_output.mem",
        DATA_WIDTH,
        radix="hex",
    )

    validate_loaded_lengths(
        input_flat=input_flat,
        weight_flat=weight_flat,
        bias_flat=bias_flat,
        raw_psum_ref_flat=raw_psum_ref_flat,
        tile_ref_flat=tile_ref_flat,
        expected_output_flat=expected_output_flat,
    )

    input_q = reshape_1d_to_2d(input_flat, M, K)
    weight_q = reshape_1d_to_2d(weight_flat, K, N)
    raw_psum_ref_mem = reshape_1d_to_2d(raw_psum_ref_flat, M, N)
    expected_output_mem = reshape_1d_to_2d(expected_output_flat, M, N)

    return (
        input_q,
        weight_q,
        bias_flat,
        raw_psum_ref_mem,
        tile_ref_flat,
        expected_output_mem,
    )


# =============================================================================
# Core tile-aware golden model
# =============================================================================
def compute_tiled_raw_psum(
    input_q: List[List[int]],
    weight_q: List[List[int]],
) -> Tuple[List[List[int]], List[int]]:
    """
    Recompute raw partial sums using the exact locked tile order.

    Returns:
    - raw_psum_calc : final accumulated matrix of shape (M, N)
    - tile_ref_calc : flattened intermediate tile references in order:
        [tile_m][tile_n][tile_k][local_m][local_n]
    """
    raw_psum_calc = [[0 for _ in range(N)] for _ in range(M)]
    tile_ref_calc: List[int] = []

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
                        tile_ref_calc.append(local_acc[i][j])

            for i in range(TILE_M):
                for j in range(TILE_N):
                    raw_psum_calc[tm + i][tn + j] = local_acc[i][j]

    return raw_psum_calc, tile_ref_calc


def postprocess_output_matrix(
    raw_psum: List[List[int]],
    bias_q: List[int],
) -> List[List[int]]:
    """
    Apply locked NNIA post-processing:
    raw accumulator -> requantize -> bias add -> ACC clamp -> ReLU -> DATA clamp
    """
    out_q = [[0 for _ in range(N)] for _ in range(M)]

    for i in range(M):
        for j in range(N):
            val = raw_psum[i][j]

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
# Comparison helpers
# =============================================================================
def first_mismatch_2d(
    a: List[List[int]],
    b: List[List[int]],
) -> Tuple[bool, str]:
    """Return first mismatch info for two equally shaped 2D matrices."""
    if len(a) != len(b):
        return False, f"row count mismatch: {len(a)} != {len(b)}"

    for r in range(len(a)):
        if len(a[r]) != len(b[r]):
            return False, f"col count mismatch at row {r}: {len(a[r])} != {len(b[r])}"

        for c in range(len(a[r])):
            if a[r][c] != b[r][c]:
                return (
                    False,
                    f"mismatch at [{r}][{c}] -> calc={a[r][c]}, ref={b[r][c]}",
                )

    return True, "match"


def first_mismatch_1d(
    a: List[int],
    b: List[int],
) -> Tuple[bool, str]:
    """Return first mismatch info for two equally sized 1D lists."""
    if len(a) != len(b):
        return False, f"length mismatch: {len(a)} != {len(b)}"

    for idx, (va, vb) in enumerate(zip(a, b)):
        if va != vb:
            return False, f"mismatch at index {idx} -> calc={va}, ref={vb}"

    return True, "match"


# =============================================================================
# Debug / display helpers
# =============================================================================
def fixed_matrix_to_float_string(
    mat: List[List[int]],
    frac_bits: int = FRAC_BITS,
) -> str:
    """Pretty-print a fixed-point matrix approximately as float."""
    rows = []
    for row in mat:
        row_f = [round(fixed_to_float(v, frac_bits), 4) for v in row]
        rows.append(str(row_f))
    return "\n".join(rows)


def fixed_vector_to_float_string(
    vec: List[int],
    frac_bits: int = FRAC_BITS,
) -> str:
    """Pretty-print a fixed-point vector approximately as float."""
    vals = [round(fixed_to_float(v, frac_bits), 4) for v in vec]
    return str(vals)


def print_config() -> None:
    print("\n================== NNIA TILE GOLDEN MODEL ==================\n")
    print("Configuration")
    print(f"M           : {M}")
    print(f"K           : {K}")
    print(f"N           : {N}")
    print(f"TILE_M      : {TILE_M}")
    print(f"TILE_K      : {TILE_K}")
    print(f"TILE_N      : {TILE_N}")
    print(f"DATA_WIDTH  : {DATA_WIDTH}")
    print(f"FRAC_BITS   : {FRAC_BITS}")
    print(f"ACC_WIDTH   : {ACC_WIDTH}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"MEM_DIR     : {MEM_DIR}")


def print_result_summary(
    input_q: List[List[int]],
    weight_q: List[List[int]],
    bias_q: List[int],
    raw_psum_calc: List[List[int]],
    expected_output_calc: List[List[int]],
    tile_ref_calc: List[int],
    raw_ok: bool,
    raw_msg: str,
    tile_ok: bool,
    tile_msg: str,
    out_ok: bool,
    out_msg: str,
) -> None:
    print("\nLoaded shapes")
    print("Input matrix shape      :", (len(input_q), len(input_q[0]) if input_q else 0))
    print("Weight matrix shape     :", (len(weight_q), len(weight_q[0]) if weight_q else 0))
    print("Bias vector shape       :", (len(bias_q),))
    print(
        "Raw psum matrix shape   :",
        (len(raw_psum_calc), len(raw_psum_calc[0]) if raw_psum_calc else 0),
    )
    print(
        "Expected output shape   :",
        (
            len(expected_output_calc),
            len(expected_output_calc[0]) if expected_output_calc else 0,
        ),
    )
    print("Tile reference entries  :", len(tile_ref_calc))

    print("\nApprox input matrix (Q8.8 -> real):")
    print(fixed_matrix_to_float_string(input_q))

    print("\nApprox weight matrix (Q8.8 -> real):")
    print(fixed_matrix_to_float_string(weight_q))

    print("\nApprox bias vector (Q8.8 -> real):")
    print(fixed_vector_to_float_string(bias_q))

    print("\nApprox recomputed output (Q8.8 -> real):")
    print(fixed_matrix_to_float_string(expected_output_calc))

    print("\nReference comparison")
    print(f"raw_psum_ref.mem    : {'MATCH' if raw_ok else 'MISMATCH'} ({raw_msg})")
    print(f"tile_ref.mem        : {'MATCH' if tile_ok else 'MISMATCH'} ({tile_msg})")
    print(f"expected_output.mem : {'MATCH' if out_ok else 'MISMATCH'} ({out_msg})")

    if raw_ok and tile_ok and out_ok:
        print("\nFINAL STATUS: PASS")
    else:
        print("\nFINAL STATUS: NEEDS REVIEW")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    validate_config()
    validate_mem_files_exist()

    (
        input_q,
        weight_q,
        bias_q,
        raw_psum_ref_mem,
        tile_ref_mem_flat,
        expected_output_mem,
    ) = load_mem_contents()

    raw_psum_calc, tile_ref_calc = compute_tiled_raw_psum(input_q, weight_q)
    expected_output_calc = postprocess_output_matrix(raw_psum_calc, bias_q)

    raw_ok, raw_msg = first_mismatch_2d(raw_psum_calc, raw_psum_ref_mem)
    tile_ok, tile_msg = first_mismatch_1d(tile_ref_calc, tile_ref_mem_flat)
    out_ok, out_msg = first_mismatch_2d(expected_output_calc, expected_output_mem)

    print_config()
    print_result_summary(
        input_q=input_q,
        weight_q=weight_q,
        bias_q=bias_q,
        raw_psum_calc=raw_psum_calc,
        expected_output_calc=expected_output_calc,
        tile_ref_calc=tile_ref_calc,
        raw_ok=raw_ok,
        raw_msg=raw_msg,
        tile_ok=tile_ok,
        tile_msg=tile_msg,
        out_ok=out_ok,
        out_msg=out_msg,
    )


if __name__ == "__main__":
    main()