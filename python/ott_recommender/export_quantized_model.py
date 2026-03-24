"""
export_quantized_model.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Export trained MLP parameters into NNIA-aligned fixed-point representation.

Purpose
-------
- Load trained floating-point model parameters from train_mlp.py
- Validate alignment with the NNIA architecture (16 -> 8 -> 2)
- Quantize weights and biases to signed Q8.8 fixed-point format
- Preserve both original and hardware-aligned parameter forms
- Generate artifacts for downstream NNIA memory preparation and inference

Notes
-----
For binary classification models using a single output neuron, this module
constructs an equivalent two-output representation to match NNIA requirements.

Generated files
---------------
- artifacts/model_quantized.npz
- artifacts/quantization_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

# =============================================================================
# Python package path fix for direct script execution
# =============================================================================
THIS_DIR = Path(__file__).resolve().parent      # Python/ott_recommender
PYTHON_ROOT = THIS_DIR.parent                   # Python/

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from shared.fixed_point_utils import fixed_to_float, float_to_fixed


# =============================================================================
# Locked configuration
# =============================================================================
DATA_WIDTH = 16
FRAC_BITS = 8

INPUT_SIZE = 16
HIDDEN_SIZE = 8
NUM_CLASSES = 2

QUANT_ROUNDING = "nearest"

# Keep exporter honest by default.
# Bias scaling is intentionally disabled unless there is a deliberate,
# verified downstream decision to change it.
LAYER2_BIAS_SCALE_FACTOR = 1.00


# =============================================================================
# Project paths
# =============================================================================
PROJECT_ROOT = THIS_DIR.parent.parent

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_FLOAT_NPZ = ARTIFACTS_DIR / "trained_mlp_model.npz"
MODEL_QUANT_NPZ = ARTIFACTS_DIR / "model_quantized.npz"
QUANT_REPORT_JSON = ARTIFACTS_DIR / "quantization_report.json"


# =============================================================================
# Validation helpers
# =============================================================================
def validate_input_file() -> None:
    """Ensure the trained float model artifact exists."""
    if not MODEL_FLOAT_NPZ.exists():
        raise FileNotFoundError(
            f"Missing trained float model file:\n{MODEL_FLOAT_NPZ}\n\n"
            f"Run train_mlp.py first."
        )


def expected_float_keys() -> set[str]:
    """Return parameter keys required from train_mlp.py artifact."""
    return {"W1", "b1", "W2", "b2"}


def unexpected_layer_keys(keys: Iterable[str]) -> list[str]:
    """
    Detect unsupported deeper-layer parameter names.

    This keeps the exporter strict if software training is changed later.
    """
    extras: list[str] = []
    allowed = expected_float_keys()

    for key in keys:
        if key.startswith("W") or key.startswith("b") or key.startswith("B"):
            if key not in allowed and key not in {
                "best_activation",
                "best_alpha",
                "best_learning_rate_init",
                "binary_single_output",
                "class_names",
                "dataset_name",
                "feature_names",
                "hidden_size",
                "input_size",
                "num_classes",
                "output_size",
                "scaler_mean",
                "scaler_scale",
                "training_mode",
            }:
                extras.append(str(key))

    return sorted(extras)


def validate_finite_array(name: str, arr: np.ndarray) -> None:
    """Ensure parameter arrays contain only finite numeric values."""
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")


def validate_model_float_npz_keys(data: np.lib.npyio.NpzFile) -> None:
    """
    Validate that trained_mlp_model.npz is aligned to the locked exporter flow.
    """
    keys = set(data.files)
    required = expected_float_keys()
    missing = sorted(required - keys)
    extra_layers = unexpected_layer_keys(keys)

    if missing:
        raise ValueError(
            "trained_mlp_model.npz is missing required parameter keys:\n"
            + "\n".join(missing)
        )

    if extra_layers:
        raise ValueError(
            "trained_mlp_model.npz contains unsupported extra parameter keys:\n"
            + "\n".join(extra_layers)
            + "\n\nCurrent exporter supports only the hardware-aligned 2-layer model."
        )


def scalar_int_from_npz(data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in data:
        return int(default)
    value = data[key]
    return int(np.asarray(value).reshape(-1)[0])


def validate_shapes(
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> Tuple[bool, str]:
    """
    Validate the trained model shapes.

    Returns:
    - binary_single_output flag
    - descriptive mode string
    """
    if W1.shape != (INPUT_SIZE, HIDDEN_SIZE):
        raise ValueError(
            f"W1 shape mismatch: expected {(INPUT_SIZE, HIDDEN_SIZE)}, got {W1.shape}"
        )
    if b1.shape != (HIDDEN_SIZE,):
        raise ValueError(
            f"b1 shape mismatch: expected {(HIDDEN_SIZE,)}, got {b1.shape}"
        )

    if W2.shape == (HIDDEN_SIZE, NUM_CLASSES) and b2.shape == (NUM_CLASSES,):
        return False, "two_output"
    if W2.shape == (HIDDEN_SIZE, 1) and b2.shape == (1,):
        return True, "binary_single_output"

    raise ValueError(
        "Layer-2 shape mismatch.\n"
        f"Expected either W2={(HIDDEN_SIZE, NUM_CLASSES)}, b2={(NUM_CLASSES,)}\n"
        f"or W2={(HIDDEN_SIZE, 1)}, b2={(1,)}\n"
        f"but got W2={W2.shape}, b2={b2.shape}"
    )


def validate_scaler(
    scaler_mean: np.ndarray | None,
    scaler_scale: np.ndarray | None,
) -> None:
    """
    AI flow requires saved standardization parameters.
    """
    if scaler_mean is None or scaler_scale is None:
        raise ValueError(
            "Missing scaler_mean/scaler_scale in trained_mlp_model.npz.\n"
            "AI L1 preparation requires saved feature standardization artifacts."
        )

    if scaler_mean.shape != (INPUT_SIZE,):
        raise ValueError(
            f"scaler_mean shape mismatch: expected {(INPUT_SIZE,)}, got {scaler_mean.shape}"
        )

    if scaler_scale.shape != (INPUT_SIZE,):
        raise ValueError(
            f"scaler_scale shape mismatch: expected {(INPUT_SIZE,)}, got {scaler_scale.shape}"
        )

    validate_finite_array("scaler_mean", scaler_mean)
    validate_finite_array("scaler_scale", scaler_scale)

    if np.any(np.abs(scaler_scale) < 1e-12):
        raise ValueError(
            "scaler_scale contains zero or near-zero values; export aborted."
        )


# =============================================================================
# Quantization helpers
# =============================================================================
def quantize_array(
    arr_float: np.ndarray,
    data_width: int = DATA_WIDTH,
    frac_bits: int = FRAC_BITS,
) -> np.ndarray:
    """
    Quantize a float numpy array to signed fixed-point integer numpy array.

    Uses configured rounding and saturation via fixed_point_utils.float_to_fixed().
    Returns int32 for safe downstream Python handling.
    """
    flat_out = [
        float_to_fixed(
            float(v),
            data_width=data_width,
            frac_bits=frac_bits,
            rounding=QUANT_ROUNDING,
            saturate=True,
        )
        for v in arr_float.reshape(-1)
    ]
    return np.array(flat_out, dtype=np.int32).reshape(arr_float.shape)


def dequantize_array(
    arr_fixed: np.ndarray,
    frac_bits: int = FRAC_BITS,
) -> np.ndarray:
    """Convert a fixed-point integer array back to float for reporting."""
    flat_out = [fixed_to_float(int(v), frac_bits=frac_bits) for v in arr_fixed.reshape(-1)]
    return np.array(flat_out, dtype=np.float32).reshape(arr_fixed.shape)


def array_stats(arr: np.ndarray) -> Dict[str, float]:
    """Return basic array statistics for reporting."""
    arr = np.asarray(arr)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean_abs": float(np.mean(np.abs(arr))),
    }


def quant_error_stats(arr_float: np.ndarray, arr_dequant: np.ndarray) -> Dict[str, float]:
    """Return simple quantization error statistics."""
    err = np.asarray(arr_float, dtype=np.float32) - np.asarray(arr_dequant, dtype=np.float32)
    return {
        "mae": float(np.mean(np.abs(err))),
        "max_abs_error": float(np.max(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def saturation_count(
    arr_float: np.ndarray,
    data_width: int = DATA_WIDTH,
    frac_bits: int = FRAC_BITS,
) -> int:
    """Count how many values would saturate in signed fixed-point."""
    limit_pos = ((1 << (data_width - 1)) - 1) / float(1 << frac_bits)
    limit_neg = (-(1 << (data_width - 1))) / float(1 << frac_bits)
    arr_float = np.asarray(arr_float, dtype=np.float32)
    return int(np.sum((arr_float > limit_pos) | (arr_float < limit_neg)))


# =============================================================================
# Layer-2 expansion helpers
# =============================================================================
def expand_binary_single_output_to_two_output(
    W2_single: np.ndarray,
    b2_single: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand a binary single-logit layer into a two-logit representation.

    If original hidden output is h, sklearn binary output is:
        z = h @ W2_single + b2_single

    Expanded representation is:
        z0 = -z
        z1 = +z
    """
    if W2_single.shape != (HIDDEN_SIZE, 1):
        raise ValueError(f"Expected W2_single shape {(HIDDEN_SIZE, 1)}, got {W2_single.shape}")
    if b2_single.shape != (1,):
        raise ValueError(f"Expected b2_single shape {(1,)}, got {b2_single.shape}")

    W_pos = W2_single[:, 0]
    b_pos = float(b2_single[0])

    W2_two = np.stack([-W_pos, W_pos], axis=1).astype(np.float32)
    b2_two = np.array([-b_pos, b_pos], dtype=np.float32)
    return W2_two, b2_two


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    validate_input_file()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(MODEL_FLOAT_NPZ, allow_pickle=True)
    validate_model_float_npz_keys(data)

    W1_f = np.asarray(data["W1"], dtype=np.float32)
    b1_f = np.asarray(data["b1"], dtype=np.float32)
    W2_f = np.asarray(data["W2"], dtype=np.float32)
    b2_f = np.asarray(data["b2"], dtype=np.float32)

    binary_single_output_saved = scalar_int_from_npz(data, "binary_single_output", 0)
    num_classes_saved = scalar_int_from_npz(data, "num_classes", NUM_CLASSES)

    if num_classes_saved != NUM_CLASSES:
        raise ValueError(
            f"Saved model num_classes mismatch: expected {NUM_CLASSES}, got {num_classes_saved}"
        )

    binary_single_output_detected, layer2_mode = validate_shapes(W1_f, b1_f, W2_f, b2_f)

    if binary_single_output_saved not in (0, 1):
        raise ValueError(f"binary_single_output must be 0 or 1, got {binary_single_output_saved}")

    if binary_single_output_saved != int(binary_single_output_detected):
        raise ValueError(
            "Mismatch between saved binary_single_output flag and detected layer-2 shape.\n"
            f"Saved flag: {binary_single_output_saved}\n"
            f"Detected from shapes: {int(binary_single_output_detected)}"
        )

    validate_finite_array("W1", W1_f)
    validate_finite_array("b1", b1_f)
    validate_finite_array("W2", W2_f)
    validate_finite_array("b2", b2_f)

    scaler_mean = (
        np.asarray(data["scaler_mean"], dtype=np.float32)
        if "scaler_mean" in data
        else None
    )
    scaler_scale = (
        np.asarray(data["scaler_scale"], dtype=np.float32)
        if "scaler_scale" in data
        else None
    )
    validate_scaler(scaler_mean, scaler_scale)

    feature_names = data["feature_names"] if "feature_names" in data else None
    class_names = data["class_names"] if "class_names" in data else None

    # -------------------------------------------------------------------------
    # Preserve original layer-2 parameters and build downstream-safe 2-output form
    # -------------------------------------------------------------------------
    if binary_single_output_detected:
        W2_expanded_f, b2_expanded_f = expand_binary_single_output_to_two_output(W2_f, b2_f)
    else:
        W2_expanded_f = W2_f.astype(np.float32).copy()
        b2_expanded_f = b2_f.astype(np.float32).copy()

    b2_expanded_f_scaled = b2_expanded_f * np.float32(LAYER2_BIAS_SCALE_FACTOR)

    # -------------------------------------------------------------------------
    # Quantize parameters
    # -------------------------------------------------------------------------
    W1_q = quantize_array(W1_f)
    b1_q = quantize_array(b1_f)
    W2_q = quantize_array(W2_f)
    b2_q = quantize_array(b2_f)

    W2_expanded_q = quantize_array(W2_expanded_f)
    b2_expanded_q = quantize_array(b2_expanded_f_scaled)

    # -------------------------------------------------------------------------
    # Dequantize for error analysis
    # -------------------------------------------------------------------------
    W1_dq = dequantize_array(W1_q)
    b1_dq = dequantize_array(b1_q)
    W2_dq = dequantize_array(W2_q)
    b2_dq = dequantize_array(b2_q)

    W2_expanded_dq = dequantize_array(W2_expanded_q)
    b2_expanded_dq = dequantize_array(b2_expanded_q)

    # -------------------------------------------------------------------------
    # Save quantized model artifact
    # -------------------------------------------------------------------------
    npz_payload = {
        "W1_q": W1_q,
        "b1_q": b1_q,
        "W2_q": W2_q,
        "b2_q": b2_q,
        "W1_f": W1_f,
        "b1_f": b1_f,
        "W2_f": W2_f,
        "b2_f": b2_f,
        "W2_expanded_f": W2_expanded_f,
        "b2_expanded_f": b2_expanded_f,
        "b2_expanded_f_scaled": b2_expanded_f_scaled,
        "W2_expanded_q": W2_expanded_q,
        "b2_expanded_q": b2_expanded_q,
        "data_width": np.array(DATA_WIDTH, dtype=np.int32),
        "frac_bits": np.array(FRAC_BITS, dtype=np.int32),
        "input_size": np.array(INPUT_SIZE, dtype=np.int32),
        "hidden_size": np.array(HIDDEN_SIZE, dtype=np.int32),
        "num_classes": np.array(NUM_CLASSES, dtype=np.int32),
        "binary_single_output": np.array(int(binary_single_output_detected), dtype=np.int32),
        "layer2_mode": np.array(layer2_mode, dtype=object),
        "layer2_bias_scale_factor": np.array(LAYER2_BIAS_SCALE_FACTOR, dtype=np.float32),
        "quant_rounding": np.array(QUANT_ROUNDING, dtype=object),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
    }

    if feature_names is not None:
        npz_payload["feature_names"] = feature_names
    if class_names is not None:
        npz_payload["class_names"] = class_names

    np.savez(MODEL_QUANT_NPZ, **npz_payload)

    # -------------------------------------------------------------------------
    # Build quantization report
    # -------------------------------------------------------------------------
    report = {
        "data_width": DATA_WIDTH,
        "frac_bits": FRAC_BITS,
        "format": "signed Q8.8",
        "quant_rounding": QUANT_ROUNDING,
        "locked_architecture": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_classes": NUM_CLASSES,
            "logical_topology": "16 -> 8 -> 2",
        },
        "layer2_mode": layer2_mode,
        "binary_single_output": bool(binary_single_output_detected),
        "layer2_bias_scale_factor": LAYER2_BIAS_SCALE_FACTOR,
        "shapes": {
            "W1": list(W1_f.shape),
            "b1": list(b1_f.shape),
            "W2_original": list(W2_f.shape),
            "b2_original": list(b2_f.shape),
            "W2_expanded": list(W2_expanded_f.shape),
            "b2_expanded": list(b2_expanded_f.shape),
            "scaler_mean": list(scaler_mean.shape),
            "scaler_scale": list(scaler_scale.shape),
        },
        "float_stats": {
            "W1": array_stats(W1_f),
            "b1": array_stats(b1_f),
            "W2_original": array_stats(W2_f),
            "b2_original": array_stats(b2_f),
            "W2_expanded": array_stats(W2_expanded_f),
            "b2_expanded_original": array_stats(b2_expanded_f),
            "b2_expanded_scaled": array_stats(b2_expanded_f_scaled),
        },
        "quantized_int_stats": {
            "W1_q": array_stats(W1_q),
            "b1_q": array_stats(b1_q),
            "W2_q": array_stats(W2_q),
            "b2_q": array_stats(b2_q),
            "W2_expanded_q": array_stats(W2_expanded_q),
            "b2_expanded_q": array_stats(b2_expanded_q),
        },
        "dequantized_float_stats": {
            "W1_dq": array_stats(W1_dq),
            "b1_dq": array_stats(b1_dq),
            "W2_dq": array_stats(W2_dq),
            "b2_dq": array_stats(b2_dq),
            "W2_expanded_dq": array_stats(W2_expanded_dq),
            "b2_expanded_dq": array_stats(b2_expanded_dq),
        },
        "quantization_error": {
            "W1": quant_error_stats(W1_f, W1_dq),
            "b1": quant_error_stats(b1_f, b1_dq),
            "W2_original": quant_error_stats(W2_f, W2_dq),
            "b2_original": quant_error_stats(b2_f, b2_dq),
            "W2_expanded": quant_error_stats(W2_expanded_f, W2_expanded_dq),
            "b2_expanded_scaled": quant_error_stats(b2_expanded_f_scaled, b2_expanded_dq),
        },
        "saturation_count": {
            "W1": saturation_count(W1_f),
            "b1": saturation_count(b1_f),
            "W2_original": saturation_count(W2_f),
            "b2_original": saturation_count(b2_f),
            "W2_expanded": saturation_count(W2_expanded_f),
            "b2_expanded_scaled": saturation_count(b2_expanded_f_scaled),
        },
        "scaler_stats_present": {
            "scaler_mean": True,
            "scaler_scale": True,
        },
        "notes": [
            f"Quantization uses float_to_fixed(... rounding='{QUANT_ROUNDING}', saturate=True).",
            "All parameters are converted to signed 16-bit Q8.8 integer domain.",
            "Saved scaler artifacts are mandatory for AI L1 preparation.",
            "Original trained layer-2 parameters are preserved exactly.",
            "For binary single-output models, an expanded 2-output layer is also generated.",
            "Expanded layer uses class0=-z and class1=+z to preserve the decision boundary.",
            "Layer-2 bias scaling remains 1.0 to avoid unverified changes that could reduce accuracy.",
        ],
    }

    with open(QUANT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n==================== EXPORT QUANTIZED MODEL ====================\n")
    print(f"Input float model  : {MODEL_FLOAT_NPZ}")
    print(f"Output quant model : {MODEL_QUANT_NPZ}")
    print(f"Quant report       : {QUANT_REPORT_JSON}")
    print(f"Logical topology   : 16 -> 8 -> 2")
    print(f"Layer2 mode        : {layer2_mode}")
    print(f"Quant rounding     : {QUANT_ROUNDING}")
    print(f"Bias scale         : {LAYER2_BIAS_SCALE_FACTOR:.2f}")

    print("\nShapes:")
    print(f"  W1           : {W1_f.shape}")
    print(f"  b1           : {b1_f.shape}")
    print(f"  W2 original  : {W2_f.shape}")
    print(f"  b2 original  : {b2_f.shape}")
    print(f"  W2 expanded  : {W2_expanded_f.shape}")
    print(f"  b2 expanded  : {b2_expanded_f.shape}")
    print(f"  scaler_mean  : {scaler_mean.shape}")
    print(f"  scaler_scale : {scaler_scale.shape}")

    print("\nQuantization error summary:")
    for name in ["W1", "b1", "W2_original", "b2_original", "W2_expanded", "b2_expanded_scaled"]:
        err = report["quantization_error"][name]
        sat = report["saturation_count"][name]
        print(
            f"  {name:<18} -> "
            f"MAE={err['mae']:.8f}, "
            f"MaxAbs={err['max_abs_error']:.8f}, "
            f"RMSE={err['rmse']:.8f}, "
            f"Saturations={sat}"
        )

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()