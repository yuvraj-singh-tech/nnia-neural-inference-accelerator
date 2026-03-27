"""
core_runner.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Description
-----------
This module executes the baseline NNIA verification pipeline.

It coordinates deterministic data generation, software reference validation,
Vivado-based hardware simulation, and final output comparison to ensure
end-to-end alignment between Python models and RTL implementation.

The runner manages execution flow, logging, error handling, and result
summarization, providing a single command interface for NNIA core verification.

Core Verification Flow
----------------------
1. Shared utility module check
2. Deterministic data generation
3. Software golden model validation
4. Vivado batch simulation
5. RTL vs expected output comparison

Notes
-----
- Uses fixed memory files for deterministic verification
- Ensures strict alignment between software reference and hardware output
- Designed for repeatable and automated validation of NNIA core functionality
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Project path configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "Python"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
MEM_DIR = PROJECT_ROOT / "mem"

VIVADO_BAT = Path(r"D:\VITIS_2022.1\Vivado\2022.1\bin\vivado.bat")
RUN_SIM_TCL = SCRIPTS_DIR / "run_vivado_sim.tcl"

CORE_LOG_FILE = LOGS_DIR / "core_runner.log"
VIVADO_LOG_FILE = LOGS_DIR / "vivado_batch.log"

GOLD_MEM = MEM_DIR / "expected_output.mem"
RTL_MEM = MEM_DIR / "rtl_output.mem"


# =============================================================================
# Console formatting helpers
# =============================================================================
def print_banner(title: str) -> None:
    """Print a major section banner."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_step(title: str) -> None:
    """Print a step header."""
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def append_log(text: str) -> None:
    """Append text to the persistent core runner log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with CORE_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


# =============================================================================
# Subprocess environment
# =============================================================================
def build_subprocess_env() -> dict[str, str]:
    """
    Build the environment for child processes.

    Why this is required:
    - The project uses a top-level Python/ directory with module folders
      such as shared/ and cores/.
    - Child processes launched with `python -m ...` need PYTHONPATH to include
      that top-level Python/ directory.
    """
    env = os.environ.copy()

    python_root_str = str(PYTHON_ROOT)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()

    if existing_pythonpath:
        env["PYTHONPATH"] = python_root_str + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = python_root_str

    return env


# =============================================================================
# Command execution
# =============================================================================
def run_command(
    cmd: list[str],
    step_title: str,
    save_stdout_to: Path | None = None,
) -> str:
    """
    Execute one pipeline step.

    Features
    --------
    - Uses project root as working directory
    - Passes PYTHONPATH for child Python module execution
    - Captures stdout and stderr
    - Prints output to terminal
    - Saves logs to core_runner.log
    - Optionally saves stdout to a dedicated output file
    """
    print_step(step_title)
    print("Command:")
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    process = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=build_subprocess_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout = process.stdout or ""
    stderr = process.stderr or ""

    if stdout.strip():
        print(stdout, end="" if stdout.endswith("\n") else "\n")

    if stderr.strip():
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)

    append_log(f"\n{'=' * 78}\n{step_title}\n{'=' * 78}")
    append_log("Command: " + " ".join(cmd))

    if stdout.strip():
        append_log(stdout)

    if stderr.strip():
        append_log("\n[STDERR]\n" + stderr)

    if save_stdout_to is not None:
        save_stdout_to.parent.mkdir(parents=True, exist_ok=True)
        save_stdout_to.write_text(stdout, encoding="utf-8")

    if process.returncode != 0:
        print("\nERROR : Step failed.")
        print(f"ERROR : Return code = {process.returncode}")
        sys.exit(process.returncode)

    print("STATUS: PASS")
    return stdout


# =============================================================================
# Output parsing helpers
# =============================================================================
def extract_metric(text: str, key: str) -> str | None:
    """Extract a HOST_* metric from Vivado batch output."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return None


def parse_compare_status(text: str) -> str:
    """Parse the final comparison status from comparator output."""
    upper = text.upper()

    if (
        "FINAL STATUS: PASS" in upper
        or "FINAL STATUS : PASS" in upper
        or "[FINAL PASS]" in upper
    ):
        return "PASS"

    if (
        "FINAL STATUS: FAIL" in upper
        or "FINAL STATUS : FAIL" in upper
        or "[FINAL FAIL]" in upper
    ):
        return "FAIL"

    return "UNKNOWN"


# =============================================================================
# Validation helpers
# =============================================================================
def validate_required_paths() -> None:
    """Validate critical external paths before starting the run."""
    missing: list[str] = []

    required_paths = [
        VIVADO_BAT,
        RUN_SIM_TCL,
    ]

    for path in required_paths:
        if not path.exists():
            missing.append(str(path))

    if missing:
        print_banner("NNIA CORE RUNNER - PATH VALIDATION FAILED")
        for item in missing:
            print(f"ERROR : Missing required path -> {item}")
        sys.exit(1)


# =============================================================================
# Main orchestration flow
# =============================================================================
def main() -> None:
    """Execute the complete NNIA core verification flow."""
    os.chdir(PROJECT_ROOT)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CORE_LOG_FILE.write_text("", encoding="utf-8")

    print_banner("NNIA CORE VERIFICATION RUNNER")
    print(f"Project root                 : {PROJECT_ROOT}")
    print(f"Python root                  : {PYTHON_ROOT}")
    print(f"Scripts directory            : {SCRIPTS_DIR}")
    print(f"Memory directory             : {MEM_DIR}")
    print(f"Logs directory               : {LOGS_DIR}")
    print(f"Vivado executable            : {VIVADO_BAT}")
    print(f"TCL script                   : {RUN_SIM_TCL}")
    print(f"Golden output file           : {GOLD_MEM}")
    print(f"RTL output file              : {RTL_MEM}")
    print(f"Core log file                : {CORE_LOG_FILE}")
    print(f"Vivado log file              : {VIVADO_LOG_FILE}")

    validate_required_paths()

    # -------------------------------------------------------------------------
    # STEP 1: Shared utility module check
    # -------------------------------------------------------------------------
    run_command(
        [sys.executable, "-m", "shared.fixed_point_utils"],
        "STEP 1 / 5 : SHARED UTILITY CHECK",
    )

    # -------------------------------------------------------------------------
    # STEP 2: Deterministic data generation
    # -------------------------------------------------------------------------
    run_command(
        [sys.executable, "-m", "cores.generate_data"],
        "STEP 2 / 5 : DATA GENERATION",
    )

    # -------------------------------------------------------------------------
    # STEP 3: Tile-aware golden model validation
    # -------------------------------------------------------------------------
    run_command(
        [sys.executable, "-m", "cores.tile_golden_model"],
        "STEP 3 / 5 : TILE GOLDEN MODEL VALIDATION",
    )

    # -------------------------------------------------------------------------
    # STEP 4: Vivado batch simulation
    # -------------------------------------------------------------------------
    run_command(
        [str(VIVADO_BAT), "-mode", "batch", "-source", str(RUN_SIM_TCL.resolve())],
        "STEP 4 / 5 : VIVADO BATCH SIMULATION",
        save_stdout_to=VIVADO_LOG_FILE,
    )

    sim_output = VIVADO_LOG_FILE.read_text(encoding="utf-8", errors="replace")

    latency_cycles = extract_metric(sim_output, "HOST_LATENCY_CYCLES")
    latency_time_us = extract_metric(sim_output, "HOST_LATENCY_TIME_US")
    total_macs = extract_metric(sim_output, "HOST_TOTAL_MACS")
    peak_macs_per_cycle = extract_metric(sim_output, "HOST_PEAK_MACS_PER_CYCLE")
    peak_throughput_mmacs = extract_metric(sim_output, "HOST_PEAK_THROUGHPUT_MMACS")
    effective_throughput_mmacs = extract_metric(
        sim_output, "HOST_EFFECTIVE_THROUGHPUT_MMACS"
    )

    # -------------------------------------------------------------------------
    # STEP 5: Explicit core output comparison
    # -------------------------------------------------------------------------
    compare_output = run_command(
        [
            sys.executable,
            "-m",
            "shared.compare_output",
            "--gold",
            str(GOLD_MEM.resolve()),
            "--rtl",
            str(RTL_MEM.resolve()),
            "--tag",
            "core",
        ],
        "STEP 5 / 5 : CORE OUTPUT COMPARISON",
    )

    final_status = parse_compare_status(compare_output)

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    print_banner("NNIA CORE RUN SUMMARY")

    if latency_cycles is not None:
        print(f"Inference latency (cycles)   : {latency_cycles}")
    if latency_time_us is not None:
        print(f"Inference latency (time)     : {latency_time_us} us")
    if total_macs is not None:
        print(f"Total MACs per inference     : {total_macs}")
    if peak_macs_per_cycle is not None:
        print(f"Peak MACs per cycle          : {peak_macs_per_cycle}")
    if peak_throughput_mmacs is not None:
        print(f"Peak throughput              : {peak_throughput_mmacs} MMAC/s")
    if effective_throughput_mmacs is not None:
        print(f"Effective throughput         : {effective_throughput_mmacs} MMAC/s")

    print(f"Final comparison status      : {final_status}")

    if final_status == "PASS":
        print("Run result                   : Core verification completed successfully.")
    elif final_status == "FAIL":
        print("Run result                   : Flow completed, but output comparison failed.")
        sys.exit(1)
    else:
        print("Run result                   : Flow completed, but final status was unclear.")
        sys.exit(1)


if __name__ == "__main__":
    main()
