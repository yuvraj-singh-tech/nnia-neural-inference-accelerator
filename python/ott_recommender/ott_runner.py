"""
ott_runner.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Description
-----------
This module orchestrates the complete NNIA-based recommendation pipeline.

It serves as the central execution script for the project by coordinating
dataset generation, feature validation, model training, quantized parameter
export, NNIA memory preparation, Vivado-based hardware simulation, output
comparison, and final result analysis.

The runner also manages path validation, logging, command execution,
error handling, and end-to-end status reporting, providing a single
entry point for software and hardware verification.

Pipeline Flow
-------------
1. Dataset preparation
2. Feature schema validation
3. Model training
4. Quantized model export
5. Layer-1 memory preparation
6. Vivado simulation for layer 1
7. Layer-1 output comparison
8. Software reference inference
9. Layer-2 memory preparation
10. Vivado simulation for layer 2
11. Layer-2 output comparison
12. Final output analysis
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# =============================================================================
# Project path configuration
# =============================================================================
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
PROJECT_ROOT = THIS_DIR.parent.parent

PYTHON_ROOT = PROJECT_ROOT / "Python"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
MEM_DIR = PROJECT_ROOT / "mem"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASETS_DIR = PROJECT_ROOT / "datasets"

VIVADO_BAT = Path(r"D:\VITIS_2022.1\Vivado\2022.1\bin\vivado.bat")
RUN_SIM_TCL = SCRIPTS_DIR / "run_vivado_sim.tcl"

OTT_LOG_FILE = LOGS_DIR / "ott_runner.log"
VIVADO_L1_LOG_FILE = LOGS_DIR / "vivado_layer1_batch.log"
VIVADO_L2_LOG_FILE = LOGS_DIR / "vivado_layer2_batch.log"

COMPARE_SCRIPT = PROJECT_ROOT / "Python" / "shared" / "compare_output.py"

L1_GOLD_MEM = MEM_DIR / "expected_output_l1.mem"
L2_GOLD_MEM = MEM_DIR / "expected_output_l2.mem"


# =============================================================================
# Console formatting helpers
# =============================================================================
def print_banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_step(title: str) -> None:
    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def append_log(text: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with OTT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


# =============================================================================
# Environment helpers
# =============================================================================
def build_python_env() -> dict[str, str]:
    env = os.environ.copy()

    python_root_str = str(PYTHON_ROOT)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()

    if existing_pythonpath:
        env["PYTHONPATH"] = python_root_str + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = python_root_str

    return env


def build_vivado_env() -> dict[str, str]:
    """
    Use a simpler environment for Vivado batch runs.

    Reason
    ------
    Vivado on Windows is usually more stable when launched through cmd /c
    with a clean working directory and without depending on Python module env.
    """
    env = os.environ.copy()
    return env


# =============================================================================
# Generic command execution
# =============================================================================
def run_command(
    cmd: list[str],
    step_title: str,
    save_stdout_to: Path | None = None,
) -> str:
    print_step(step_title)
    print("Command:")
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    process = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=build_python_env(),
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
        combined = stdout
        if stderr.strip():
            combined += ("\n" if combined and not combined.endswith("\n") else "") + "[STDERR]\n" + stderr
        save_stdout_to.write_text(combined, encoding="utf-8")

    if process.returncode != 0:
        print("\nERROR : Step failed.")
        print(f"ERROR : Return code = {process.returncode}")
        sys.exit(process.returncode)

    print("STATUS: PASS")
    return stdout


# =============================================================================
# Vivado execution
# =============================================================================
def latest_hs_err_log() -> Path | None:
    candidates = sorted(PROJECT_ROOT.glob("hs_err_pid*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def run_vivado_batch(step_title: str, save_log_to: Path) -> str:
    """
    Run Vivado batch safely on Windows.

    Key fix
    -------
    Launch vivado.bat through cmd.exe /c instead of calling the .bat directly.
    """
    print_step(step_title)

    cmd = [
        "cmd.exe",
        "/c",
        str(VIVADO_BAT),
        "-mode",
        "batch",
        "-source",
        str(RUN_SIM_TCL),
    ]

    print("Command:")
    print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd))
    print(f"Working directory:\n  {PROJECT_ROOT}")
    print()

    process = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=build_vivado_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )

    stdout = process.stdout or ""
    stderr = process.stderr or ""
    combined = stdout
    if stderr.strip():
        combined += ("\n" if combined and not combined.endswith("\n") else "") + "[STDERR]\n" + stderr

    if stdout.strip():
        print(stdout, end="" if stdout.endswith("\n") else "\n")

    if stderr.strip():
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)

    append_log(f"\n{'=' * 78}\n{step_title}\n{'=' * 78}")
    append_log("Command: " + " ".join(cmd))
    append_log(combined if combined.strip() else "[no output]")

    save_log_to.parent.mkdir(parents=True, exist_ok=True)
    save_log_to.write_text(combined, encoding="utf-8")

    if process.returncode != 0:
        hs_err = latest_hs_err_log()
        print("\nERROR : Vivado batch step failed.")
        print(f"ERROR : Return code = {process.returncode}")
        if hs_err is not None:
            print(f"ERROR : Crash log detected -> {hs_err}")
            append_log(f"\n[VIVADO CRASH LOG]\nDetected hs_err file: {hs_err}")
        sys.exit(process.returncode)

    print("STATUS: PASS")
    return combined


# =============================================================================
# Output parsing helpers
# =============================================================================
def extract_metric(text: str, key: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(key + "="):
            return line.split("=", 1)[1].strip()
    return None


def parse_compare_status(text: str) -> str:
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
    missing: list[str] = []

    required_paths = [
        VIVADO_BAT,
        RUN_SIM_TCL,
        COMPARE_SCRIPT,
    ]

    for path in required_paths:
        if not path.exists():
            missing.append(str(path))

    if missing:
        print_banner("NNIA OTT RUNNER - PATH VALIDATION FAILED")
        for item in missing:
            print(f"ERROR : Missing required path -> {item}")
        sys.exit(1)


# =============================================================================
# Main orchestration flow
# =============================================================================
def main() -> None:
    os.chdir(PROJECT_ROOT)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OTT_LOG_FILE.write_text("", encoding="utf-8")

    print_banner("NNIA OTT RECOMMENDATION RUNNER")
    print(f"Project root           : {PROJECT_ROOT}")
    print(f"Python root            : {PYTHON_ROOT}")
    print(f"Scripts directory      : {SCRIPTS_DIR}")
    print(f"Memory directory       : {MEM_DIR}")
    print(f"Artifacts directory    : {ARTIFACTS_DIR}")
    print(f"Datasets directory     : {DATASETS_DIR}")
    print(f"Logs directory         : {LOGS_DIR}")
    print(f"Vivado executable      : {VIVADO_BAT}")
    print(f"TCL script             : {RUN_SIM_TCL}")
    print(f"Compare script         : {COMPARE_SCRIPT}")
    print(f"L1 golden mem          : {L1_GOLD_MEM}")
    print(f"L2 golden mem          : {L2_GOLD_MEM}")
    print(f"OTT runner log         : {OTT_LOG_FILE}")
    print(f"Vivado layer-1 log     : {VIVADO_L1_LOG_FILE}")
    print(f"Vivado layer-2 log     : {VIVADO_L2_LOG_FILE}")

    validate_required_paths()

    run_command(
        [sys.executable, "-m", "ott_recommender.create_dataset"],
        "STEP 1 / 12 : DATASET PREPARATION",
    )

    run_command(
        [sys.executable, "-m", "ott_recommender.feature_encoder"],
        "STEP 2 / 12 : FEATURE ENCODER CHECK",
    )

    run_command(
        [sys.executable, "-m", "ott_recommender.train_mlp"],
        "STEP 3 / 12 : MODEL TRAINING",
    )

    run_command(
        [sys.executable, "-m", "ott_recommender.export_quantized_model"],
        "STEP 4 / 12 : QUANTIZED MODEL EXPORT",
    )

    run_command(
        [sys.executable, "-m", "ott_recommender.prepare_layer1_mem"],
        "STEP 5 / 12 : LAYER-1 MEMORY PREPARATION",
    )

    vivado_l1_output = run_vivado_batch(
        "STEP 6 / 12 : VIVADO BATCH SIMULATION (LAYER 1)",
        VIVADO_L1_LOG_FILE,
    )
    l1_latency_cycles = extract_metric(vivado_l1_output, "HOST_LATENCY_CYCLES")
    l1_latency_time_us = extract_metric(vivado_l1_output, "HOST_LATENCY_TIME_US")
    l1_throughput_mmacs = extract_metric(vivado_l1_output, "HOST_THROUGHPUT_MMACS")

    compare_l1_output = run_command(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--gold",
            str(L1_GOLD_MEM),
            "--tag",
            "l1",
        ],
        "STEP 7 / 12 : LAYER-1 OUTPUT COMPARISON",
    )
    l1_status = parse_compare_status(compare_l1_output)

    run_command(
        [sys.executable, "-m", "ott_recommender.mlp_inference_reference"],
        "STEP 8 / 12 : PYTHON MLP REFERENCE INFERENCE",
    )

    run_command(
        [sys.executable, "-m", "ott_recommender.prepare_layer2_mem"],
        "STEP 9 / 12 : LAYER-2 MEMORY PREPARATION",
    )

    vivado_l2_output = run_vivado_batch(
        "STEP 10 / 12 : VIVADO BATCH SIMULATION (LAYER 2)",
        VIVADO_L2_LOG_FILE,
    )
    l2_latency_cycles = extract_metric(vivado_l2_output, "HOST_LATENCY_CYCLES")
    l2_latency_time_us = extract_metric(vivado_l2_output, "HOST_LATENCY_TIME_US")
    l2_throughput_mmacs = extract_metric(vivado_l2_output, "HOST_THROUGHPUT_MMACS")

    compare_l2_output = run_command(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "--gold",
            str(L2_GOLD_MEM),
            "--tag",
            "l2",
        ],
        "STEP 11 / 12 : LAYER-2 OUTPUT COMPARISON",
    )
    l2_status = parse_compare_status(compare_l2_output)

    run_command(
        [sys.executable, "-m", "ott_recommender.mlp_output_analyzer"],
        "STEP 12 / 12 : FINAL OUTPUT ANALYSIS",
    )

    print_banner("NNIA OTT RUN SUMMARY")

    print("Layer-1 hardware pass")
    if l1_latency_cycles is not None:
        print(f"  Latency (cycles)      : {l1_latency_cycles}")
    if l1_latency_time_us is not None:
        print(f"  Latency (time)        : {l1_latency_time_us} us")
    if l1_throughput_mmacs is not None:
        print(f"  Throughput            : {l1_throughput_mmacs} MMAC/s")
    print(f"  Comparison status     : {l1_status}")

    print("\nLayer-2 hardware pass")
    if l2_latency_cycles is not None:
        print(f"  Latency (cycles)      : {l2_latency_cycles}")
    if l2_latency_time_us is not None:
        print(f"  Latency (time)        : {l2_latency_time_us} us")
    if l2_throughput_mmacs is not None:
        print(f"  Throughput            : {l2_throughput_mmacs} MMAC/s")
    print(f"  Comparison status     : {l2_status}")

    overall_pass = (l1_status == "PASS") and (l2_status == "PASS")

    print("\nOverall result")
    if overall_pass:
        print("  STATUS                : PASS")
        print("  Message               : OTT recommendation flow completed successfully.")
    else:
        print("  STATUS                : FAIL")
        print("  Message               : Flow completed, but one or more compare stages failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()