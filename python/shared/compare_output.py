"""
------------------------------------------------------------------------------
Module Name : compare_output.py
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Output comparison and validation utility for NNIA.

Compares RTL-generated outputs against a golden reference model, providing
detailed mismatch analysis, summary statistics, and exportable reports for
verification and debugging.

Usage
-----
Core NNIA comparison:
    python Python/shared/compare_output.py --gold mem/expected_output.mem --tag core

Optional staged comparisons:
    python Python/shared/compare_output.py --gold mem/expected_output_l1.mem --tag l1
    python Python/shared/compare_output.py --gold mem/expected_output_l2.mem --tag l2

Key Points
----------
- Performs element-wise validation with error tracking
- Reports integer and approximate real-domain values
- Generates CSV report for offline analysis
- Returns exit status for automation workflows

Role in Flow
------------
- Used after Vivado simulation to validate correctness
- Bridges RTL results with Python reference model
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# =============================================================================
# Project-root-aware import and path setup
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "Python"
MEM_DIR = PROJECT_ROOT / "mem"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from shared.fixed_point_utils import (  # noqa: E402
    fixed_to_float,
    read_mem_file,
    reshape_1d_to_2d,
)


# =============================================================================
# NNIA locked output configuration
# =============================================================================
M = 4
N = 8
TOTAL_OUTPUTS = M * N

DATA_WIDTH = 16
FRAC_BITS = 8


# =============================================================================
# Argument parsing
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Compare NNIA RTL output against a golden reference."
    )

    parser.add_argument(
        "--gold",
        required=True,
        help="Path to the expected (golden) .mem file.",
    )

    parser.add_argument(
        "--rtl",
        default=str(MEM_DIR / "rtl_output.mem"),
        help=(
            "Path to the RTL output .mem file "
            f"(default: {MEM_DIR / 'rtl_output.mem'})"
        ),
    )

    parser.add_argument(
        "--tag",
        default="core",
        help="Short label for this comparison run, e.g. core, l1, l2.",
    )

    return parser


def resolve_path(path_str: str) -> Path:
    """
    Resolve a user-supplied path safely.

    Behavior:
    - Absolute paths are used as-is.
    - Relative paths are interpreted relative to the project root.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# =============================================================================
# Load outputs
# =============================================================================
def load_outputs(expected_file: Path, rtl_file: Path) -> Tuple[List[int], List[int]]:
    """Load and validate golden and RTL output vectors from .mem files."""
    if not expected_file.exists():
        raise FileNotFoundError(f"Golden file not found: {expected_file}")

    if not rtl_file.exists():
        raise FileNotFoundError(f"RTL file not found: {rtl_file}")

    expected = read_mem_file(expected_file, DATA_WIDTH, radix="hex")
    rtl = read_mem_file(rtl_file, DATA_WIDTH, radix="hex")

    if len(expected) != TOTAL_OUTPUTS:
        raise ValueError(
            f"Golden file size mismatch: found {len(expected)}, expected {TOTAL_OUTPUTS}"
        )

    if len(rtl) != TOTAL_OUTPUTS:
        raise ValueError(
            f"RTL file size mismatch: found {len(rtl)}, expected {TOTAL_OUTPUTS}"
        )

    return expected, rtl


# =============================================================================
# Comparison
# =============================================================================
def compare_outputs(
    expected: List[int],
    rtl: List[int],
) -> Tuple[List[Dict[str, object]], int, int]:
    """
    Compare RTL values against golden values element-by-element.

    Returns:
    - results: detailed comparison entries
    - matched: number of exact matches
    - max_abs_error: maximum absolute integer-domain error
    """
    results: List[Dict[str, object]] = []
    matched = 0
    max_abs_error = 0

    for idx, (exp_val, rtl_val) in enumerate(zip(expected, rtl)):
        row = idx // N
        col = idx % N

        delta = rtl_val - exp_val
        abs_error = abs(delta)
        status = "MATCH" if exp_val == rtl_val else "MISMATCH"

        if status == "MATCH":
            matched += 1

        if abs_error > max_abs_error:
            max_abs_error = abs_error

        results.append(
            {
                "idx": idx,
                "row": row,
                "col": col,
                "expected": exp_val,
                "rtl": rtl_val,
                "delta": delta,
                "status": status,
                "exp_real": fixed_to_float(exp_val, FRAC_BITS),
                "rtl_real": fixed_to_float(rtl_val, FRAC_BITS),
            }
        )

    return results, matched, max_abs_error


# =============================================================================
# Reporting
# =============================================================================
def print_header(compare_tag: str, expected_file: Path, rtl_file: Path) -> None:
    """Print report header information."""
    print("\n" + "=" * 72)
    print("NNIA OUTPUT COMPARISON REPORT")
    print("=" * 72)
    print(f"Tag          : {compare_tag}")
    print(f"Golden file  : {expected_file}")
    print(f"RTL file     : {rtl_file}")
    print(f"Shape        : {M} x {N}")
    print(f"Total values : {TOTAL_OUTPUTS}")
    print("Format       : HEX (Q8.8 signed)")
    print("=" * 72)


def print_results(results: List[Dict[str, object]]) -> None:
    """Print element-wise comparison results."""
    print("\nDetailed comparison\n" + "-" * 72)

    for r in results:
        print(
            f"[{r['status']}] idx={r['idx']:02d} "
            f"(r={r['row']}, c={r['col']}) | "
            f"exp={r['expected']:6d} | rtl={r['rtl']:6d} | "
            f"delta={r['delta']:6d}"
        )


def print_summary(matched: int, max_abs_error: int) -> None:
    """Print final comparison summary."""
    mismatched = TOTAL_OUTPUTS - matched

    print("\nSummary\n" + "-" * 72)
    print(f"Matched     : {matched}/{TOTAL_OUTPUTS}")
    print(f"Mismatched  : {mismatched}/{TOTAL_OUTPUTS}")
    print(f"Max error   : {max_abs_error}")

    if matched == TOTAL_OUTPUTS:
        print("\nFINAL STATUS: PASS")
    else:
        print("\nFINAL STATUS: FAIL")


def print_matrix_view(expected: List[int], rtl: List[int]) -> None:
    """Print approximate real-value matrix view for easier debug."""
    exp_2d = reshape_1d_to_2d(expected, M, N)
    rtl_2d = reshape_1d_to_2d(rtl, M, N)

    print("\nExpected (approx real):")
    for row in exp_2d:
        print([round(fixed_to_float(v, FRAC_BITS), 4) for v in row])

    print("\nRTL (approx real):")
    for row in rtl_2d:
        print([round(fixed_to_float(v, FRAC_BITS), 4) for v in row])


# =============================================================================
# CSV export
# =============================================================================
def write_csv(csv_report_file: Path, compare_tag: str, results: List[Dict[str, object]]) -> None:
    """Write detailed comparison results to CSV."""
    csv_report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_report_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "tag",
                "idx",
                "row",
                "col",
                "expected",
                "rtl",
                "delta",
                "expected_real",
                "rtl_real",
                "status",
            ]
        )

        for r in results:
            writer.writerow(
                [
                    compare_tag,
                    r["idx"],
                    r["row"],
                    r["col"],
                    r["expected"],
                    r["rtl"],
                    r["delta"],
                    f"{r['exp_real']:.6f}",
                    f"{r['rtl_real']:.6f}",
                    r["status"],
                ]
            )


# =============================================================================
# Main
# =============================================================================
def main() -> int:
    """Run the full comparison flow."""
    parser = build_arg_parser()
    args = parser.parse_args()

    compare_tag = args.tag.strip() if args.tag.strip() else "core"
    expected_file = resolve_path(args.gold)
    rtl_file = resolve_path(args.rtl)
    csv_report_file = MEM_DIR / f"comparison_report_{compare_tag}.csv"

    print_header(compare_tag, expected_file, rtl_file)

    expected, rtl = load_outputs(expected_file, rtl_file)
    results, matched, max_abs_error = compare_outputs(expected, rtl)

    print_results(results)
    print_summary(matched, max_abs_error)
    print_matrix_view(expected, rtl)

    write_csv(csv_report_file, compare_tag, results)
    print(f"\nCSV saved at: {csv_report_file}")

    return 0 if matched == TOTAL_OUTPUTS else 1


if __name__ == "__main__":
    sys.exit(main())