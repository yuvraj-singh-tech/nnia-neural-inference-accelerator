"""
------------------------------------------------------------------------------
Module Name : fixed_point_utils.py
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Core fixed-point math and data handling utilities for NNIA.

Provides consistent implementations of encoding, arithmetic, requantization,
and memory formatting to ensure alignment between Python reference model and
hardware execution.

Key Points
----------
- Supports Q-format conversion (float ↔ fixed)
- Implements saturation, rounding, and requantization logic
- Provides two’s-complement and bit-width-safe operations
- Handles .mem file read/write in hardware-compatible formats
- Includes helpers for vector, matrix, and dot-product operations

Role in Flow
------------
- Serves as the common math foundation across all Python scripts
- Ensures exact HW–SW numerical alignment
- Used in data generation, reference inference, and comparison stages
------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable, List


# -----------------------------------------------------------------------------
# Integer range helpers
# -----------------------------------------------------------------------------
def signed_min(data_width: int) -> int:
    """Return minimum signed integer for the given width."""
    if data_width <= 0:
        raise ValueError("data_width must be > 0")
    return -(1 << (data_width - 1))


def signed_max(data_width: int) -> int:
    """Return maximum signed integer for the given width."""
    if data_width <= 0:
        raise ValueError("data_width must be > 0")
    return (1 << (data_width - 1)) - 1


def unsigned_max(data_width: int) -> int:
    """Return maximum unsigned integer for the given width."""
    if data_width <= 0:
        raise ValueError("data_width must be > 0")
    return (1 << data_width) - 1


def fits_in_signed(value: int, data_width: int) -> bool:
    """Check whether a value fits in signed data_width bits."""
    return signed_min(data_width) <= int(value) <= signed_max(data_width)


def clamp_signed(value: int, data_width: int) -> int:
    """Saturate a value to the signed range of data_width bits."""
    return max(signed_min(data_width), min(signed_max(data_width), int(value)))


def clamp_unsigned(value: int, data_width: int) -> int:
    """Clamp a value to the unsigned range of data_width bits."""
    return max(0, min(unsigned_max(data_width), int(value)))


# -----------------------------------------------------------------------------
# Two's complement helpers
# -----------------------------------------------------------------------------
def int_to_twos_complement(value: int, data_width: int) -> int:
    """
    Convert a signed integer to its unsigned two's-complement representation.
    """
    value = clamp_signed(value, data_width)
    return value & unsigned_max(data_width)


def twos_complement_to_int(value: int, data_width: int) -> int:
    """
    Convert an unsigned two's-complement value to a signed integer.
    """
    if data_width <= 0:
        raise ValueError("data_width must be > 0")

    mask = unsigned_max(data_width)
    value = int(value) & mask
    sign_bit = 1 << (data_width - 1)

    if value & sign_bit:
        return value - (1 << data_width)
    return value


# -----------------------------------------------------------------------------
# Fixed-point encode / decode
# -----------------------------------------------------------------------------
def float_to_fixed(
    value: float,
    data_width: int = 16,
    frac_bits: int = 8,
    rounding: str = "nearest",
    saturate: bool = True,
) -> int:
    """
    Convert a float to signed fixed-point integer representation.

    rounding:
    - 'nearest' : round to nearest integer
    - 'trunc'   : truncate toward zero
    """
    if frac_bits < 0:
        raise ValueError("frac_bits must be >= 0")

    scale = 1 << frac_bits

    if rounding == "nearest":
        scaled = int(round(value * scale))
    elif rounding == "trunc":
        scaled = int(value * scale)
    else:
        raise ValueError("rounding must be 'nearest' or 'trunc'")

    if saturate:
        return clamp_signed(scaled, data_width)

    if not fits_in_signed(scaled, data_width):
        raise OverflowError(
            f"Fixed-point overflow: value={value}, scaled={scaled}, "
            f"range=[{signed_min(data_width)}, {signed_max(data_width)}]"
        )

    return scaled


def fixed_to_float(value: int, frac_bits: int = 8) -> float:
    """Convert a signed fixed-point integer back to float."""
    if frac_bits < 0:
        raise ValueError("frac_bits must be >= 0")
    return int(value) / float(1 << frac_bits)


# -----------------------------------------------------------------------------
# Shift / requantization helpers
# -----------------------------------------------------------------------------
def arithmetic_right_shift(value: int, shift: int) -> int:
    """Arithmetic right shift for signed integers."""
    if shift < 0:
        raise ValueError("shift must be >= 0")
    return int(value) >> shift


def requantize(
    value: int,
    shift: int,
    out_width: int = 16,
    rounding: str = "trunc",
    saturate: bool = True,
) -> int:
    """
    Requantize a wide signed value by right shifting and clipping to out_width.

    Typical use:
        raw_acc -> shift by FRAC_BITS -> DATA_WIDTH result

    rounding:
    - 'trunc'   : arithmetic right shift
    - 'nearest' : symmetric round-to-nearest before shifting
    """
    if shift < 0:
        raise ValueError("shift must be >= 0")

    value = int(value)

    if rounding == "trunc":
        out = arithmetic_right_shift(value, shift)

    elif rounding == "nearest":
        if shift == 0:
            out = value
        else:
            # Symmetric round-to-nearest:
            # add +0.5 LSB for non-negative
            # add -0.5 LSB for negative, then shift arithmetically
            offset = 1 << (shift - 1)
            if value >= 0:
                out = (value + offset) >> shift
            else:
                out = -(((-value) + offset) >> shift)
    else:
        raise ValueError("rounding must be 'trunc' or 'nearest'")

    if saturate:
        return clamp_signed(out, out_width)

    if not fits_in_signed(out, out_width):
        raise OverflowError(
            f"Requantize overflow: value={value}, shifted={out}, "
            f"range=[{signed_min(out_width)}, {signed_max(out_width)}]"
        )

    return out


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
def relu(value: int) -> int:
    """Apply integer-domain ReLU."""
    return max(0, int(value))


def add_bias(value: int, bias: int) -> int:
    """Add bias in integer domain."""
    return int(value) + int(bias)


def clip_signed(value: int, data_width: int) -> int:
    """Alias for signed saturation."""
    return clamp_signed(value, data_width)


def fixed_mul(a: int, b: int) -> int:
    """
    Multiply two signed fixed-point stored integers.

    Result stays in widened scaled integer form.
    """
    return int(a) * int(b)


def fixed_mac(acc: int, a: int, b: int) -> int:
    """Multiply-accumulate in integer domain."""
    return int(acc) + (int(a) * int(b))


def dot_product_fixed(vec_a: Iterable[int], vec_b: Iterable[int]) -> int:
    """
    Compute a raw fixed-point dot product.
    """
    a_list = list(vec_a)
    b_list = list(vec_b)

    if len(a_list) != len(b_list):
        raise ValueError("dot_product_fixed inputs must have the same length")

    acc = 0
    for a, b in zip(a_list, b_list):
        acc += int(a) * int(b)
    return acc


# -----------------------------------------------------------------------------
# Memory formatting helpers
# -----------------------------------------------------------------------------
def int_to_hex(value: int, data_width: int, uppercase: bool = False) -> str:
    """
    Convert a signed integer to fixed-width two's-complement hex string.
    """
    raw = int_to_twos_complement(value, data_width)
    digits = (data_width + 3) // 4
    fmt = f"0{digits}{'X' if uppercase else 'x'}"
    return format(raw, fmt)


def hex_to_int(hex_str: str, data_width: int) -> int:
    """Parse a two's-complement hex string into a signed integer."""
    value = int(hex_str.strip(), 16)
    return twos_complement_to_int(value, data_width)


def int_to_mem_str(value: int, data_width: int, radix: str = "dec") -> str:
    """
    Format a value for .mem output.

    radix:
    - 'dec' : signed decimal
    - 'hex' : fixed-width two's-complement hex
    """
    value = int(value)

    if radix == "dec":
        return str(value)
    if radix == "hex":
        return int_to_hex(value, data_width)

    raise ValueError("radix must be 'dec' or 'hex'")


def write_mem_file(
    path: str,
    values: Iterable[int],
    data_width: int,
    radix: str = "dec",
) -> None:
    """Write one value per line to a .mem file."""
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(int_to_mem_str(int(value), data_width, radix))
            f.write("\n")


def read_mem_file(path: str, data_width: int, radix: str = "dec") -> List[int]:
    """Read a .mem file into a list of signed integers."""
    out: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if radix == "dec":
                out.append(int(line))
            elif radix == "hex":
                out.append(hex_to_int(line, data_width))
            else:
                raise ValueError("radix must be 'dec' or 'hex'")

    return out


# -----------------------------------------------------------------------------
# Matrix helpers
# -----------------------------------------------------------------------------
def flatten_2d_row_major(matrix: List[List[int]]) -> List[int]:
    """Flatten a 2D list in row-major order."""
    flat: List[int] = []
    for row in matrix:
        flat.extend(row)
    return flat


def reshape_1d_to_2d(values: List[int], rows: int, cols: int) -> List[List[int]]:
    """Reshape a flat row-major list into a 2D matrix."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be > 0")
    if len(values) != rows * cols:
        raise ValueError(
            f"Cannot reshape list of length {len(values)} into ({rows}, {cols})"
        )

    return [values[r * cols : (r + 1) * cols] for r in range(rows)]


# -----------------------------------------------------------------------------
# Small self-check
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_WIDTH = 16
    FRAC_BITS = 8

    a_real = 1.5
    b_real = -0.75

    a_fx = float_to_fixed(a_real, DATA_WIDTH, FRAC_BITS)
    b_fx = float_to_fixed(b_real, DATA_WIDTH, FRAC_BITS)

    raw_mul = fixed_mul(a_fx, b_fx)

    trunc_mul = requantize(
        raw_mul,
        shift=FRAC_BITS,
        out_width=DATA_WIDTH,
        rounding="trunc",
        saturate=True,
    )

    nearest_mul = requantize(
        raw_mul,
        shift=FRAC_BITS,
        out_width=DATA_WIDTH,
        rounding="nearest",
        saturate=True,
    )

    print("fixed_point_utils.py self-check")
    print(f"a_real      = {a_real}")
    print(f"b_real      = {b_real}")
    print(f"a_fixed     = {a_fx}")
    print(f"b_fixed     = {b_fx}")
    print(f"raw_mul     = {raw_mul}")
    print(f"trunc_mul   = {trunc_mul}")
    print(f"nearest_mul = {nearest_mul}")
    print(f"a_hex       = {int_to_hex(a_fx, DATA_WIDTH)}")
    print(f"b_hex       = {int_to_hex(b_fx, DATA_WIDTH)}")

    assert a_fx == 384
    assert b_fx == -192
    assert trunc_mul == -288
    assert nearest_mul == -288

    print("Self-check PASS")