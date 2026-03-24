"""
feature_encoder.py

Author: Yuvraj Singh
Project: Neural Network Inference Accelerator (NNIA)

Description
-----------
This module defines the shared feature schema used by the NNIA-based
recommendation pipeline.

It provides the fixed 16-feature input format, class metadata, default
feature values, and helper utilities for validation, encoding, and
conversion between dictionary and vector representations.

The module also supports feature construction from higher-level user
profiles and event statistics, ensuring that dataset generation,
model training, quantized export, and hardware inference all use the
same feature ordering and representation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence


# =============================================================================
# Project configuration
# =============================================================================
INPUT_SIZE = 16
NUM_CLASSES = 2

FEATURE_NAMES: List[str] = [
    "action_pref",           #  0
    "romance_pref",          #  1
    "comedy_pref",           #  2
    "thriller_pref",         #  3
    "avg_watch_time_norm",   #  4
    "weekend_watch_ratio",   #  5
    "prefers_new_releases",  #  6
    "skips_intro_ratio",     #  7
    "rating_generosity",     #  8
    "binge_watch_ratio",     #  9
    "night_watch_ratio",     # 10
    "mobile_watch_ratio",    # 11
    "short_content_pref",    # 12
    "rewatch_ratio",         # 13
    "exploration_score",     # 14
    "completion_ratio",      # 15
]

CLASS_NAMES: List[str] = [
    "Not Recommended",
    "Recommended",
]

CLASS_DISPLAY_TEXT: Dict[int, str] = {
    0: "This title is less aligned with your recent viewing pattern",
    1: "This title looks aligned with your recent viewing pattern",
}

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "action_pref": "Relative preference for action-oriented content.",
    "romance_pref": "Relative preference for romance-oriented content.",
    "comedy_pref": "Relative preference for comedy-oriented content.",
    "thriller_pref": "Relative preference for suspense/thriller content.",
    "avg_watch_time_norm": "Normalized viewing intensity or watch-duration proxy.",
    "weekend_watch_ratio": "Fraction of watch activity concentrated on weekends.",
    "prefers_new_releases": "Preference for more recently released content.",
    "skips_intro_ratio": "How often the user behaves like an intro-skipper.",
    "rating_generosity": "Normalized tendency to rate content more positively.",
    "binge_watch_ratio": "Fraction of behavior consistent with binge-like sessions.",
    "night_watch_ratio": "Fraction of watch activity during night hours.",
    "mobile_watch_ratio": "Fraction of watch activity from mobile/on-the-go behavior.",
    "short_content_pref": "Preference for shorter-form content.",
    "rewatch_ratio": "Fraction of behavior consistent with rewatches or repeats.",
    "exploration_score": "Tendency to explore broader content variety.",
    "completion_ratio": "Fraction of started content likely completed or engaged with.",
}

DEFAULT_FEATURE_DICT: Dict[str, float] = {
    "action_pref": 0.25,
    "romance_pref": 0.25,
    "comedy_pref": 0.25,
    "thriller_pref": 0.25,
    "avg_watch_time_norm": 0.50,
    "weekend_watch_ratio": 0.50,
    "prefers_new_releases": 0.50,
    "skips_intro_ratio": 0.50,
    "rating_generosity": 0.50,
    "binge_watch_ratio": 0.50,
    "night_watch_ratio": 0.50,
    "mobile_watch_ratio": 0.50,
    "short_content_pref": 0.50,
    "rewatch_ratio": 0.50,
    "exploration_score": 0.50,
    "completion_ratio": 0.50,
}

CLASS_NAME_TO_LABEL: Dict[str, int] = {
    name: idx for idx, name in enumerate(CLASS_NAMES)
}


# =============================================================================
# Small numeric helpers
# =============================================================================
def to_float(value: float) -> float:
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid numeric feature values")

    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Value must be finite, got {value}")
    return out


def clamp01(value: float) -> float:
    value = to_float(value)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    den = to_float(den)
    if abs(den) < 1e-12:
        return to_float(default)
    return to_float(num) / den


def normalize(value: float, min_val: float, max_val: float) -> float:
    value = to_float(value)
    min_val = to_float(min_val)
    max_val = to_float(max_val)

    if max_val <= min_val:
        return 0.0
    return clamp01((value - min_val) / (max_val - min_val))


def ratio_to_score(num: float, den: float, default: float = 0.0) -> float:
    return clamp01(safe_div(num, den, default=default))


# =============================================================================
# Genre helpers
# =============================================================================
def normalize_genre_preferences(
    action_score: float,
    romance_score: float,
    comedy_score: float,
    thriller_score: float,
    method: str = "l1",
) -> Dict[str, float]:
    scores = {
        "action_pref": max(0.0, to_float(action_score)),
        "romance_pref": max(0.0, to_float(romance_score)),
        "comedy_pref": max(0.0, to_float(comedy_score)),
        "thriller_pref": max(0.0, to_float(thriller_score)),
    }

    if method not in {"l1", "uniform_if_zero"}:
        raise ValueError(f"Unsupported normalization method: {method}")

    total = sum(scores.values())

    if total <= 1e-12:
        return {
            "action_pref": 0.25,
            "romance_pref": 0.25,
            "comedy_pref": 0.25,
            "thriller_pref": 0.25,
        }

    return {key: value / total for key, value in scores.items()}


# =============================================================================
# Metadata helpers
# =============================================================================
def get_feature_names() -> List[str]:
    return FEATURE_NAMES.copy()


def get_feature_descriptions() -> Dict[str, str]:
    return FEATURE_DESCRIPTIONS.copy()


def get_default_feature_dict() -> Dict[str, float]:
    return DEFAULT_FEATURE_DICT.copy()


def get_default_feature_vector() -> List[float]:
    return feature_dict_to_vector(DEFAULT_FEATURE_DICT, clamp_values=True)


def get_class_name_from_label(label: int) -> str:
    validate_label(label)
    return CLASS_NAMES[int(label)]


def get_label_from_class_name(name: str) -> int:
    if not isinstance(name, str):
        raise TypeError("Class name must be a string")

    key = name.strip()
    if key not in CLASS_NAME_TO_LABEL:
        raise ValueError(f"Unknown class name: {name}")

    return CLASS_NAME_TO_LABEL[key]


def get_display_text_from_label(label: int) -> str:
    validate_label(label)
    return CLASS_DISPLAY_TEXT[int(label)]


# =============================================================================
# Validation helpers
# =============================================================================
def validate_feature_dict(feature_dict: Mapping[str, float]) -> None:
    missing = [name for name in FEATURE_NAMES if name not in feature_dict]
    extra = [name for name in feature_dict.keys() if name not in FEATURE_NAMES]

    if missing:
        raise ValueError(f"Missing required features: {missing}")
    if extra:
        raise ValueError(f"Unexpected extra features: {extra}")

    for name in FEATURE_NAMES:
        _ = to_float(feature_dict[name])


def validate_feature_vector(vector: Sequence[float]) -> None:
    if len(vector) != INPUT_SIZE:
        raise ValueError(
            f"Feature vector length mismatch: expected {INPUT_SIZE}, got {len(vector)}"
        )

    for idx, value in enumerate(vector):
        try:
            _ = to_float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Feature index {idx} ('{FEATURE_NAMES[idx]}') must be finite and numeric-convertible"
            ) from exc


def validate_label(label: int) -> None:
    if isinstance(label, bool):
        raise TypeError("Label must be an integer class id, not bool")

    try:
        label_float = float(label)
        label_int = int(label)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Label must be int-convertible, got {type(label)}") from exc

    if label_float != float(label_int):
        raise TypeError(f"Label must be integral, got {label}")

    if not (0 <= label_int < NUM_CLASSES):
        raise ValueError(f"Label must be in [0, {NUM_CLASSES - 1}], got {label_int}")


# =============================================================================
# Core encoding helpers
# =============================================================================
def feature_dict_to_vector(
    feature_dict: Mapping[str, float],
    clamp_values: bool = True,
) -> List[float]:
    validate_feature_dict(feature_dict)

    vector: List[float] = []
    for name in FEATURE_NAMES:
        value = to_float(feature_dict[name])
        if clamp_values:
            value = clamp01(value)
        vector.append(value)

    return vector


def vector_to_feature_dict(vector: Sequence[float]) -> Dict[str, float]:
    validate_feature_vector(vector)
    return {FEATURE_NAMES[i]: to_float(vector[i]) for i in range(INPUT_SIZE)}


def build_feature_dict(
    *,
    action_pref: float,
    romance_pref: float,
    comedy_pref: float,
    thriller_pref: float,
    avg_watch_time_norm: float,
    weekend_watch_ratio: float,
    prefers_new_releases: float,
    skips_intro_ratio: float,
    rating_generosity: float,
    binge_watch_ratio: float,
    night_watch_ratio: float,
    mobile_watch_ratio: float,
    short_content_pref: float,
    rewatch_ratio: float,
    exploration_score: float,
    completion_ratio: float,
    clamp_values: bool = True,
) -> Dict[str, float]:
    feature_dict = {
        "action_pref": action_pref,
        "romance_pref": romance_pref,
        "comedy_pref": comedy_pref,
        "thriller_pref": thriller_pref,
        "avg_watch_time_norm": avg_watch_time_norm,
        "weekend_watch_ratio": weekend_watch_ratio,
        "prefers_new_releases": prefers_new_releases,
        "skips_intro_ratio": skips_intro_ratio,
        "rating_generosity": rating_generosity,
        "binge_watch_ratio": binge_watch_ratio,
        "night_watch_ratio": night_watch_ratio,
        "mobile_watch_ratio": mobile_watch_ratio,
        "short_content_pref": short_content_pref,
        "rewatch_ratio": rewatch_ratio,
        "exploration_score": exploration_score,
        "completion_ratio": completion_ratio,
    }

    if clamp_values:
        return {key: clamp01(value) for key, value in feature_dict.items()}
    return {key: to_float(value) for key, value in feature_dict.items()}


# =============================================================================
# Raw-profile encoders
# =============================================================================
def encode_user_profile(profile: Mapping[str, float]) -> List[float]:
    genre_prefs = normalize_genre_preferences(
        profile.get("genre_action_score", 0.0),
        profile.get("genre_romance_score", 0.0),
        profile.get("genre_comedy_score", 0.0),
        profile.get("genre_thriller_score", 0.0),
        method="l1",
    )

    feature_dict = build_feature_dict(
        action_pref=genre_prefs["action_pref"],
        romance_pref=genre_prefs["romance_pref"],
        comedy_pref=genre_prefs["comedy_pref"],
        thriller_pref=genre_prefs["thriller_pref"],
        avg_watch_time_norm=normalize(profile.get("avg_watch_minutes", 120.0), 0.0, 240.0),
        weekend_watch_ratio=clamp01(profile.get("weekend_sessions_ratio", 0.50)),
        prefers_new_releases=clamp01(profile.get("new_release_ratio", 0.50)),
        skips_intro_ratio=clamp01(profile.get("intro_skip_ratio", 0.50)),
        rating_generosity=normalize(profile.get("avg_rating_given", 3.0), 1.0, 5.0),
        binge_watch_ratio=clamp01(profile.get("binge_session_ratio", 0.50)),
        night_watch_ratio=clamp01(profile.get("night_watch_ratio", 0.50)),
        mobile_watch_ratio=clamp01(profile.get("mobile_device_ratio", 0.50)),
        short_content_pref=clamp01(profile.get("short_content_ratio", 0.50)),
        rewatch_ratio=clamp01(profile.get("rewatch_ratio", 0.50)),
        exploration_score=clamp01(profile.get("exploration_ratio", 0.50)),
        completion_ratio=clamp01(profile.get("completion_ratio", 0.50)),
        clamp_values=True,
    )

    return feature_dict_to_vector(feature_dict, clamp_values=True)


def encode_user_profile_from_event_stats(event_stats: Mapping[str, float]) -> List[float]:
    total_views = max(1.0, to_float(event_stats.get("total_views", 0.0)))
    total_sessions = max(1.0, to_float(event_stats.get("total_sessions", 0.0)))
    rated_items = max(0.0, to_float(event_stats.get("rated_items", 0.0)))
    avg_rating_given = to_float(event_stats.get("avg_rating_given", 3.0))
    avg_watch_minutes = to_float(event_stats.get("avg_watch_minutes", 120.0))

    genre_prefs = normalize_genre_preferences(
        event_stats.get("action_views", 0.0),
        event_stats.get("romance_views", 0.0),
        event_stats.get("comedy_views", 0.0),
        event_stats.get("thriller_views", 0.0),
        method="l1",
    )

    weekend_watch_ratio = ratio_to_score(
        event_stats.get("weekend_views", 0.0), total_views, default=0.50
    )
    prefers_new_releases = ratio_to_score(
        event_stats.get("new_release_views", 0.0), total_views, default=0.50
    )
    skips_intro_ratio = ratio_to_score(
        event_stats.get("intro_skips", 0.0), total_views, default=0.50
    )
    binge_watch_ratio = ratio_to_score(
        event_stats.get("binge_sessions", 0.0), total_sessions, default=0.50
    )
    night_watch_ratio = ratio_to_score(
        event_stats.get("night_views", 0.0), total_views, default=0.50
    )
    mobile_watch_ratio = ratio_to_score(
        event_stats.get("mobile_views", 0.0), total_views, default=0.50
    )
    short_content_pref = ratio_to_score(
        event_stats.get("short_content_views", 0.0), total_views, default=0.50
    )
    rewatch_ratio = ratio_to_score(
        event_stats.get("rewatch_views", 0.0), total_views, default=0.50
    )
    exploration_score = ratio_to_score(
        event_stats.get("exploratory_views", 0.0), total_views, default=0.50
    )
    completion_ratio = ratio_to_score(
        event_stats.get("completed_views", 0.0), total_views, default=0.50
    )

    if rated_items <= 0.0:
        rating_generosity = 0.50
    else:
        rating_generosity = normalize(avg_rating_given, 1.0, 5.0)

    feature_dict = build_feature_dict(
        action_pref=genre_prefs["action_pref"],
        romance_pref=genre_prefs["romance_pref"],
        comedy_pref=genre_prefs["comedy_pref"],
        thriller_pref=genre_prefs["thriller_pref"],
        avg_watch_time_norm=normalize(avg_watch_minutes, 0.0, 240.0),
        weekend_watch_ratio=weekend_watch_ratio,
        prefers_new_releases=prefers_new_releases,
        skips_intro_ratio=skips_intro_ratio,
        rating_generosity=rating_generosity,
        binge_watch_ratio=binge_watch_ratio,
        night_watch_ratio=night_watch_ratio,
        mobile_watch_ratio=mobile_watch_ratio,
        short_content_pref=short_content_pref,
        rewatch_ratio=rewatch_ratio,
        exploration_score=exploration_score,
        completion_ratio=completion_ratio,
        clamp_values=True,
    )

    return feature_dict_to_vector(feature_dict, clamp_values=True)


# =============================================================================
# Display helpers
# =============================================================================
def describe_feature_vector(vector: Sequence[float]) -> str:
    validate_feature_vector(vector)

    lines: List[str] = []
    for idx, (name, value) in enumerate(zip(FEATURE_NAMES, vector)):
        lines.append(f"{idx:02d} | {name:22s} : {to_float(value):.4f}")
    return "\n".join(lines)


def print_feature_schema() -> None:
    print("\n==================== FEATURE ENCODER SCHEMA ====================\n")
    print(f"INPUT_SIZE  : {INPUT_SIZE}")
    print(f"NUM_CLASSES : {NUM_CLASSES}")

    print("\nFEATURE ORDER:")
    for idx, name in enumerate(FEATURE_NAMES):
        print(f"  {idx:02d} -> {name:<22s} | {FEATURE_DESCRIPTIONS[name]}")

    print("\nOUTPUT CLASSES:")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"  {idx} -> {name} | {CLASS_DISPLAY_TEXT[idx]}")


# =============================================================================
# Self-check
# =============================================================================
if __name__ == "__main__":
    print_feature_schema()

    sample_profile = {
        "genre_action_score": 0.90,
        "genre_romance_score": 0.20,
        "genre_comedy_score": 0.75,
        "genre_thriller_score": 0.60,
        "avg_watch_minutes": 110.0,
        "weekend_sessions_ratio": 0.80,
        "new_release_ratio": 0.65,
        "intro_skip_ratio": 0.70,
        "avg_rating_given": 4.2,
        "binge_session_ratio": 0.85,
        "night_watch_ratio": 0.60,
        "mobile_device_ratio": 0.55,
        "short_content_ratio": 0.30,
        "rewatch_ratio": 0.25,
        "exploration_ratio": 0.40,
        "completion_ratio": 0.88,
    }

    vector = encode_user_profile(sample_profile)

    print("\nEncoded feature vector from raw profile:")
    print(describe_feature_vector(vector))

    sample_event_stats = {
        "action_views": 48,
        "romance_views": 9,
        "comedy_views": 31,
        "thriller_views": 22,
        "total_views": 110,
        "weekend_views": 72,
        "night_views": 49,
        "mobile_views": 58,
        "short_content_views": 20,
        "rewatch_views": 12,
        "completed_views": 93,
        "binge_sessions": 18,
        "total_sessions": 29,
        "intro_skips": 67,
        "rated_items": 40,
        "avg_rating_given": 4.1,
        "avg_watch_minutes": 104,
        "new_release_views": 36,
        "exploratory_views": 24,
    }

    vector2 = encode_user_profile_from_event_stats(sample_event_stats)

    print("\nEncoded feature vector from event stats:")
    print(describe_feature_vector(vector2))

    assert len(FEATURE_NAMES) == INPUT_SIZE
    assert len(CLASS_NAMES) == NUM_CLASSES
    assert len(vector) == INPUT_SIZE
    assert len(vector2) == INPUT_SIZE
    assert all(0.0 <= v <= 1.0 for v in vector)
    assert all(0.0 <= v <= 1.0 for v in vector2)

    default_vec = get_default_feature_vector()
    assert len(default_vec) == INPUT_SIZE
    assert all(0.0 <= v <= 1.0 for v in default_vec)

    validate_label(0)
    validate_label(1)
    assert get_class_name_from_label(1) == "Recommended"
    assert get_label_from_class_name("Not Recommended") == 0
    assert (
        get_display_text_from_label(0)
        == "This title is less aligned with your recent viewing pattern"
    )

    print("\nSelf-check PASS")