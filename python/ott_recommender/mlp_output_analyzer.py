"""
------------------------------------------------------------------------------
Module Name : mlp_output_analyzer.py
Author      : Yuvraj Singh
Project     : Neural Network Inference Accelerator (NNIA)

Description
-----------
Final analysis and presentation layer for the NNIA-based recommendation
pipeline.

This module reads the final NNIA / MLP inference artifacts, interprets the
binary recommendation decision, computes presentation-oriented confidence
metrics, maps suitable movie titles, and generates both terminal and HTML
reports.

The underlying inference and recommendation logic remains unchanged.
This module is responsible for report generation, viewer-facing formatting,
and final result presentation.

Purpose
-------
- Interpret final binary recommendation outputs
- Convert inference results into viewer-friendly summaries
- Map recommended titles for each evaluated profile
- Generate a structured JSON analysis report
- Generate an HTML display report
- Preserve technical reporting in terminal output

Inputs
------
Primary inference inputs:
- artifacts/runs/mlp_reference_outputs.npz
  or
- artifacts/runs/layer2_reference.npz

Optional recommendation pool input:
- artifacts/datasets/movie_reco_pools.json

Fallback MovieLens inputs:
- datasets/ml-25m/movies.csv
- datasets/ml-25m/ratings.csv

Generated Outputs
-----------------
- artifacts/runs/mlp_analysis_report.json
- artifacts/runs/mlp_analysis_report.html

Notes
-----
- This module does not change the locked NNIA inference logic
- Top-1 prediction is derived from the final two-class output
- Confidence percentage is computed from the binary output scores
- Movie title mapping is used only as a presentation layer
- Technical fields may appear in terminal output for validation purposes

Usage
-----
Run analyzer:
    python Python/ott_recommender/mlp_output_analyzer.py

Optional arguments:
    --top-k-movies <int>
        Number of mapped titles to show per evaluated viewer

    --min-rating-count <int>
        Minimum rating-count preference for fallback ranking
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# Python package path fix
# =============================================================================
THIS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = THIS_DIR.parent

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

# =============================================================================
# Imports
# =============================================================================
from ott_recommender.feature_encoder import CLASS_NAMES
from shared.fixed_point_utils import fixed_to_float

# =============================================================================
# Locked configuration
# =============================================================================
M = 4
O_MLP = 2
N_NNIA = 8
FRAC_BITS = 8

DEFAULT_TOP_K_MOVIES = 6
DEFAULT_MIN_RATING_COUNT = 20
RATINGS_CHUNK_SIZE = 50_000

# =============================================================================
# Presentation copy
# =============================================================================
DISPLAY_COPY = {
    0: {
        "headline": "😐 Not Your Vibe Right Now",
        "verdict_banner": "🤔 Better Picks Are Waiting",
        "tagline": "This one feels less aligned with your current watch mood.",
        "support_line": "A different title mix is likely to give you a better watch experience right now.",
        "movie_bucket_title": "Try These Instead",
        "result_status": "Better Alternative Available",
        "decision_badge": "😐 Not Recommended",
    },
    1: {
        "headline": "🔥 Recommended for You",
        "verdict_banner": "❤️ A Strong Match for Your Watchlist",
        "tagline": "This one lines up well with your current viewing taste.",
        "support_line": "This profile points toward content you are more likely to enjoy right now.",
        "movie_bucket_title": "Top Picks for You",
        "result_status": "Strong Watch Match",
        "decision_badge": "🔥 Recommended",
    },
}

# =============================================================================
# Project paths
# =============================================================================
PROJECT_ROOT = THIS_DIR.parent.parent

RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"
DATASET_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "datasets"
MOVIELENS_DIR = PROJECT_ROOT / "datasets" / "ml-25m"

MLP_REF_NPZ = RUNS_DIR / "mlp_reference_outputs.npz"
LAYER2_REF_NPZ = RUNS_DIR / "layer2_reference.npz"
ANALYSIS_REPORT_JSON = RUNS_DIR / "mlp_analysis_report.json"
ANALYSIS_REPORT_HTML = RUNS_DIR / "mlp_analysis_report.html"

MOVIE_RECO_POOLS_JSON = DATASET_ARTIFACTS_DIR / "movie_reco_pools.json"

MOVIES_CSV = MOVIELENS_DIR / "movies.csv"
RATINGS_CSV = MOVIELENS_DIR / "ratings.csv"

# =============================================================================
# Formatting helpers
# =============================================================================
REPORT_WIDTH = 108


def print_rule(char: str = "=", width: int = REPORT_WIDTH) -> None:
    print(char * width)


def print_title(title: str) -> None:
    print_rule("=")
    print(title.center(REPORT_WIDTH))
    print_rule("=")


def print_subtitle(title: str) -> None:
    print_rule("-")
    print(title.center(REPORT_WIDTH))
    print_rule("-")


def print_section(title: str) -> None:
    print()
    print_rule("-")
    print(f" {title}")
    print_rule("-")


def print_kv(key: str, value: object, key_width: int = 30) -> None:
    print(f"{key:<{key_width}} : {value}")


def print_banner_line(text: str) -> None:
    print_rule(".")
    print(text.center(REPORT_WIDTH))
    print_rule(".")


def viewer_header(viewer_idx: int) -> str:
    return f"VIEWER {viewer_idx + 1:02d}"


def profile_header(viewer_idx: int) -> str:
    return f"Profile {viewer_idx + 1:02d}"


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze NNIA recommendation outputs and generate presentation-ready reports."
    )
    parser.add_argument(
        "--top-k-movies",
        type=int,
        default=DEFAULT_TOP_K_MOVIES,
        help=f"Recommended movies per sample (default: {DEFAULT_TOP_K_MOVIES})",
    )
    parser.add_argument(
        "--min-rating-count",
        type=int,
        default=DEFAULT_MIN_RATING_COUNT,
        help=f"Minimum rating count preferred in fallback ranking (default: {DEFAULT_MIN_RATING_COUNT})",
    )
    return parser.parse_args()


# =============================================================================
# Validation
# =============================================================================
def validate_required_files() -> None:
    missing: List[str] = []

    if not MLP_REF_NPZ.exists() and not LAYER2_REF_NPZ.exists():
        missing.append(f"Either {MLP_REF_NPZ} or {LAYER2_REF_NPZ} must exist")

    if not MOVIE_RECO_POOLS_JSON.exists():
        if not MOVIES_CSV.exists():
            missing.append(str(MOVIES_CSV))
        if not RATINGS_CSV.exists():
            missing.append(str(RATINGS_CSV))

    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n"
            + "\n".join(missing)
            + "\n\nRun the inference flow first, and run create_dataset.py to generate recommendation pools."
        )


# =============================================================================
# Artifact loading
# =============================================================================
def load_inference_artifact() -> Tuple[List[List[int]], List[int], List[int], List[int], str]:
    if MLP_REF_NPZ.exists():
        data = np.load(MLP_REF_NPZ, allow_pickle=True)

        required = {"logits_q", "labels", "pred_ids"}
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"{MLP_REF_NPZ.name} missing keys: {missing}")

        logits_q = np.asarray(data["logits_q"], dtype=np.int32)
        labels = np.asarray(data["labels"], dtype=np.int32)
        pred_ids = np.asarray(data["pred_ids"], dtype=np.int32)

        if logits_q.shape != (M, O_MLP):
            raise ValueError(f"logits_q shape mismatch: expected {(M, O_MLP)}, got {logits_q.shape}")
        if labels.shape != (M,):
            raise ValueError(f"labels shape mismatch: expected {(M,)}, got {labels.shape}")
        if pred_ids.shape != (M,):
            raise ValueError(f"pred_ids shape mismatch: expected {(M,)}, got {pred_ids.shape}")

        selected_indices = (
            np.asarray(data["selected_indices"], dtype=np.int32).tolist()
            if "selected_indices" in data
            else []
        )

        return logits_q.tolist(), labels.tolist(), pred_ids.tolist(), selected_indices, MLP_REF_NPZ.name

    data = np.load(LAYER2_REF_NPZ, allow_pickle=True)

    required = {"expected_output_q", "labels", "pred_ids"}
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{LAYER2_REF_NPZ.name} missing keys: {missing}")

    expected_output_q = np.asarray(data["expected_output_q"], dtype=np.int32)
    labels = np.asarray(data["labels"], dtype=np.int32)
    pred_ids = np.asarray(data["pred_ids"], dtype=np.int32)

    if expected_output_q.shape != (M, N_NNIA):
        raise ValueError(
            f"expected_output_q shape mismatch: expected {(M, N_NNIA)}, got {expected_output_q.shape}"
        )
    if labels.shape != (M,):
        raise ValueError(f"labels shape mismatch: expected {(M,)}, got {labels.shape}")
    if pred_ids.shape != (M,):
        raise ValueError(f"pred_ids shape mismatch: expected {(M,)}, got {pred_ids.shape}")

    selected_indices = (
        np.asarray(data["selected_indices"], dtype=np.int32).tolist()
        if "selected_indices" in data
        else []
    )

    logits_q_2d = expected_output_q[:, :O_MLP].tolist()
    return logits_q_2d, labels.tolist(), pred_ids.tolist(), selected_indices, LAYER2_REF_NPZ.name


# =============================================================================
# Movie recommendation sources
# =============================================================================
def load_movie_recommendation_pool_from_json() -> Dict[str, List[Dict[str, object]]]:
    with open(MOVIE_RECO_POOLS_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"{MOVIE_RECO_POOLS_JSON} has invalid structure")

    normalized: Dict[str, List[Dict[str, object]]] = {}
    for key, value in raw.items():
        normalized[str(key)] = value if isinstance(value, list) else []

    return normalized


def load_movie_recommendation_pool_from_movielens() -> pd.DataFrame:
    movies_df = pd.read_csv(
        MOVIES_CSV,
        usecols=["movieId", "title", "genres"],
        dtype={"movieId": "int32", "title": "string", "genres": "string"},
        encoding="utf-8",
    )

    rating_sum: Dict[int, float] = {}
    rating_count: Dict[int, int] = {}

    for chunk in pd.read_csv(
        RATINGS_CSV,
        usecols=["movieId", "rating"],
        dtype={"movieId": "int32", "rating": "float32"},
        encoding="utf-8",
        low_memory=False,
        chunksize=RATINGS_CHUNK_SIZE,
    ):
        movie_ids = chunk["movieId"].to_numpy()
        ratings = chunk["rating"].to_numpy()

        for movie_id, rating in zip(movie_ids, ratings):
            mid = int(movie_id)
            r = float(rating)
            if mid in rating_sum:
                rating_sum[mid] += r
                rating_count[mid] += 1
            else:
                rating_sum[mid] = r
                rating_count[mid] = 1

    rating_stats = pd.DataFrame(
        {
            "movieId": list(rating_sum.keys()),
            "rating_mean": [rating_sum[mid] / rating_count[mid] for mid in rating_sum],
            "rating_count": [rating_count[mid] for mid in rating_sum],
        }
    )

    if not rating_stats.empty:
        rating_stats = rating_stats.astype(
            {"movieId": "int32", "rating_mean": "float32", "rating_count": "int32"}
        )

    merged = movies_df.merge(rating_stats, on="movieId", how="left")
    merged["rating_mean"] = merged["rating_mean"].fillna(0.0).astype("float32")
    merged["rating_count"] = merged["rating_count"].fillna(0).astype("int32")
    return merged


# =============================================================================
# Movie selection helpers
# =============================================================================
def normalize_movie_entry_from_json(item: Dict[str, object]) -> Dict[str, object]:
    return {
        "title": str(item.get("title", "")).strip(),
        "genres": str(item.get("genres", "")),
        "rating_mean": round(float(item.get("avg_rating", 0.0)), 3),
        "rating_count": int(item.get("rating_count", 0)),
    }


def get_rotated_movies_from_json_pool(
    movie_pool_json: Dict[str, List[Dict[str, object]]],
    predicted_label_name: str,
    sample_id: int,
    dataset_row_index: int | None,
    top_k: int,
    used_titles_global: Set[str],
) -> List[Dict[str, object]]:
    pool_raw = movie_pool_json.get(predicted_label_name, [])
    pool = [normalize_movie_entry_from_json(item) for item in pool_raw]

    pool = [item for item in pool if item["title"]]
    if not pool:
        return []

    rotation_seed = (sample_id * 7) + ((dataset_row_index or 0) % max(1, len(pool)))
    offset = rotation_seed % len(pool)

    rotated = pool[offset:] + pool[:offset]

    selected: List[Dict[str, object]] = []
    local_titles: Set[str] = set()

    for item in rotated:
        title = str(item["title"])
        if title in used_titles_global or title in local_titles:
            continue
        selected.append(item)
        local_titles.add(title)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for item in rotated:
            title = str(item["title"])
            if title in local_titles:
                continue
            selected.append(item)
            local_titles.add(title)
            if len(selected) >= top_k:
                break

    for item in selected:
        used_titles_global.add(str(item["title"]))

    return selected


def get_ranked_movies_from_df(
    movie_pool_df: pd.DataFrame,
    predicted_label_id: int,
    min_rating_count: int,
) -> List[Dict[str, object]]:
    if predicted_label_id == 1:
        filtered = movie_pool_df.loc[movie_pool_df["rating_mean"] >= 3.9].copy()
        strong = filtered.loc[filtered["rating_count"] >= min_rating_count].copy()
        ranked = (strong if len(strong) > 0 else filtered).sort_values(
            by=["rating_mean", "rating_count", "title"],
            ascending=[False, False, True],
        )
    else:
        filtered = movie_pool_df.loc[movie_pool_df["rating_mean"] <= 2.9].copy()
        strong = filtered.loc[filtered["rating_count"] >= min_rating_count].copy()
        ranked = (strong if len(strong) > 0 else filtered).sort_values(
            by=["rating_mean", "rating_count", "title"],
            ascending=[True, False, True],
        )

    out: List[Dict[str, object]] = []
    for _, row in ranked.iterrows():
        title = str(row["title"]).strip()
        if not title:
            continue
        out.append(
            {
                "title": title,
                "genres": str(row["genres"]),
                "rating_mean": round(float(row["rating_mean"]), 3),
                "rating_count": int(row["rating_count"]),
            }
        )
    return out


def get_rotated_movies_from_df(
    movie_pool_df: pd.DataFrame,
    predicted_label_id: int,
    sample_id: int,
    dataset_row_index: int | None,
    top_k: int,
    min_rating_count: int,
    used_titles_global: Set[str],
) -> List[Dict[str, object]]:
    ranked = get_ranked_movies_from_df(
        movie_pool_df=movie_pool_df,
        predicted_label_id=predicted_label_id,
        min_rating_count=min_rating_count,
    )

    if not ranked:
        return []

    rotation_seed = (sample_id * 7) + ((dataset_row_index or 0) % max(1, len(ranked)))
    offset = rotation_seed % len(ranked)
    rotated = ranked[offset:] + ranked[:offset]

    selected: List[Dict[str, object]] = []
    local_titles: Set[str] = set()

    for item in rotated:
        title = str(item["title"])
        if title in used_titles_global or title in local_titles:
            continue
        selected.append(item)
        local_titles.add(title)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for item in rotated:
            title = str(item["title"])
            if title in local_titles:
                continue
            selected.append(item)
            local_titles.add(title)
            if len(selected) >= top_k:
                break

    for item in selected:
        used_titles_global.add(str(item["title"]))

    return selected


# =============================================================================
# Analysis helpers
# =============================================================================
def safe_class_name(class_id: int) -> str:
    if 0 <= class_id < len(CLASS_NAMES):
        return str(CLASS_NAMES[class_id])
    return f"UNKNOWN({class_id})"


def to_float_row(row_q: List[int]) -> List[float]:
    return [fixed_to_float(int(v), frac_bits=FRAC_BITS) for v in row_q]


def count_matches(pred_ids: List[int], labels: List[int]) -> int:
    if len(pred_ids) != len(labels):
        raise ValueError("Prediction and label lengths do not match")
    return sum(int(p == y) for p, y in zip(pred_ids, labels))


def binary_match_percent(score0: float, score1: float) -> float:
    max_score = max(score0, score1)
    e0 = math.exp(score0 - max_score)
    e1 = math.exp(score1 - max_score)
    total = e0 + e1

    if total <= 0.0:
        return 50.0

    return round((e1 / total) * 100.0, 2)


def build_confidence_band(match_percent: float) -> str:
    if match_percent >= 90.0:
        return "Excellent Match"
    if match_percent >= 75.0:
        return "Strong Match"
    if match_percent >= 60.0:
        return "Good Match"
    if match_percent >= 45.0:
        return "Balanced Pick"
    return "Low Match"


def build_pattern_summary(pred_id: int, match_percent: float, score_gap: float) -> str:
    if pred_id == 1:
        if match_percent >= 95.0:
            return "A standout pick that strongly fits your current watch taste."
        if match_percent >= 85.0:
            return "This looks like a very comfortable match for your watch mood."
        if match_percent >= 70.0:
            return "A good fit with a solid chance of keeping you engaged."
        return "A decent match that still leans in your favor."
    else:
        if match_percent <= 10.0:
            return "This feels far less aligned with what you are likely to enjoy right now."
        if score_gap < 0.75:
            return "This one is close, but other picks seem more naturally aligned with your taste."
        return "This title appears less suited to your current viewing mood."


def build_support_summary(pred_id: int, match_percent: float) -> str:
    if pred_id == 1:
        if match_percent >= 90.0:
            return "🔥❤️ This one looks like a must-try pick for your current watchlist."
        if match_percent >= 75.0:
            return "❤️😄 A strong fit that feels comfortably aligned with your taste."
        return "🍿😄 A watchable pick with a positive recommendation leaning."
    else:
        if match_percent <= 15.0:
            return "😐🚫 This one is likely better skipped for now."
        return "🤔😐 Not a bad title, but other options look more suited to you right now."


def build_score_gap_text(score0: float, score1: float) -> str:
    gap = abs(score1 - score0)
    return f"{gap:.4f}"


def build_sample_analysis(
    logits_q_2d: List[List[int]],
    labels: List[int],
    pred_ids: List[int],
    selected_indices: List[int],
    movie_pool_json: Dict[str, List[Dict[str, object]]] | None,
    movie_pool_df: pd.DataFrame | None,
    top_k_movies: int,
    min_rating_count: int,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    used_titles_by_label: Dict[int, Set[str]] = {0: set(), 1: set()}

    for i in range(M):
        true_id = int(labels[i])
        saved_pred_id = int(pred_ids[i])

        logits_q = [int(v) for v in logits_q_2d[i]]
        if len(logits_q) != O_MLP:
            raise ValueError(f"Sample {i}: expected {O_MLP} logits, got {len(logits_q)}")

        logits_float_raw = to_float_row(logits_q)
        logits_float = [round(v, 6) for v in logits_float_raw]

        derived_pred_id = 1 if logits_q[1] > logits_q[0] else 0
        pred_id = derived_pred_id

        true_name = safe_class_name(true_id)
        pred_name = safe_class_name(pred_id)

        match_percent = binary_match_percent(logits_float_raw[0], logits_float_raw[1])
        confidence_band = build_confidence_band(match_percent)
        display_block = DISPLAY_COPY[pred_id]

        dataset_row_index = int(selected_indices[i]) if i < len(selected_indices) else None

        if movie_pool_json is not None:
            movies = get_rotated_movies_from_json_pool(
                movie_pool_json=movie_pool_json,
                predicted_label_name=pred_name,
                sample_id=i,
                dataset_row_index=dataset_row_index,
                top_k=top_k_movies,
                used_titles_global=used_titles_by_label[pred_id],
            )
        else:
            assert movie_pool_df is not None
            movies = get_rotated_movies_from_df(
                movie_pool_df=movie_pool_df,
                predicted_label_id=pred_id,
                sample_id=i,
                dataset_row_index=dataset_row_index,
                top_k=top_k_movies,
                min_rating_count=min_rating_count,
                used_titles_global=used_titles_by_label[pred_id],
            )

        score_gap_text = build_score_gap_text(logits_float_raw[0], logits_float_raw[1])
        pattern_summary = build_pattern_summary(
            pred_id,
            match_percent,
            abs(logits_float_raw[1] - logits_float_raw[0]),
        )
        support_summary = build_support_summary(pred_id, match_percent)

        results.append(
            {
                "sample_id": i,
                "dataset_row_index": dataset_row_index,
                "true_label_id": true_id,
                "true_label_name": true_name,
                "saved_pred_label_id": saved_pred_id,
                "derived_pred_label_id": pred_id,
                "prediction_consistent_with_saved": bool(saved_pred_id == pred_id),
                "pred_label_name": pred_name,
                "match": bool(true_id == pred_id),
                "headline": display_block["headline"],
                "verdict_banner": display_block["verdict_banner"],
                "tagline": display_block["tagline"],
                "support_line": display_block["support_line"],
                "movie_bucket_title": display_block["movie_bucket_title"],
                "result_status": display_block["result_status"],
                "decision_badge": display_block["decision_badge"],
                "pattern_summary": pattern_summary,
                "support_summary": support_summary,
                "class_scores_q": logits_q,
                "class_scores_float": logits_float,
                "score_gap_text": score_gap_text,
                "recommended_match_percent": match_percent,
                "confidence_band": confidence_band,
                "movies": movies,
            }
        )

    return results


# =============================================================================
# JSON report
# =============================================================================
def save_analysis_report(
    source_name: str,
    sample_results: List[Dict[str, object]],
    top_k_movies: int,
    min_rating_count: int,
    recommendation_source: str,
) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    labels = [int(item["true_label_id"]) for item in sample_results]
    predictions = [int(item["derived_pred_label_id"]) for item in sample_results]
    selected_indices = [
        item["dataset_row_index"] for item in sample_results if item["dataset_row_index"] is not None
    ]

    correct = count_matches(predictions, labels)
    accuracy = correct / float(len(sample_results))

    report: Dict[str, object] = {
        "stage": "mlp_output_analysis",
        "source_artifact": source_name,
        "batch_size": len(sample_results),
        "output_classes": list(CLASS_NAMES),
        "selected_indices": selected_indices,
        "labels": labels,
        "predictions": predictions,
        "correct": correct,
        "accuracy": round(accuracy, 6),
        "presentation_config": {
            "top_k_movies": top_k_movies,
            "min_rating_count": min_rating_count,
            "diverse_movie_mapping": True,
        },
        "recommendation_source": recommendation_source,
        "notes": [
            "Top-1 binary decision is derived from the 2 class scores.",
            "Recommended match percent is a stable softmax-like view over the binary logits.",
            "This analyzer is presentation and reporting logic, not the authoritative NNIA math reference.",
            "Movie mapping is used to make the final output more realistic and user-facing.",
            "Movie picks are rotated per viewer and duplicates are avoided within the same predicted label group when possible.",
        ],
        "samples": sample_results,
    }

    with open(ANALYSIS_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# =============================================================================
# HTML report
# =============================================================================
def generate_html_report(
    source_name: str,
    sample_results: List[Dict[str, object]],
    top_k_movies: int,
    min_rating_count: int,
    recommendation_source: str,
) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    labels = [int(item["true_label_id"]) for item in sample_results]
    pred_ids = [int(item["derived_pred_label_id"]) for item in sample_results]
    correct = count_matches(pred_ids, labels)
    accuracy = correct / float(len(sample_results))
    accuracy_text = f"{accuracy * 100:.2f}% across {len(sample_results)} evaluated viewers"

    def esc(text: object) -> str:
        return html.escape(str(text))

    def card_poster_gradient(pred_id: int, sample_id: int) -> str:
        rec_gradients = [
            "linear-gradient(135deg, rgba(255,94,98,0.95), rgba(255,153,102,0.82))",
            "linear-gradient(135deg, rgba(229,9,20,0.95), rgba(91,33,182,0.82))",
            "linear-gradient(135deg, rgba(255,120,73,0.95), rgba(255,198,93,0.82))",
            "linear-gradient(135deg, rgba(236,72,153,0.95), rgba(249,115,22,0.82))",
        ]
        not_rec_gradients = [
            "linear-gradient(135deg, rgba(71,85,105,0.95), rgba(30,41,59,0.84))",
            "linear-gradient(135deg, rgba(100,116,139,0.95), rgba(51,65,85,0.84))",
            "linear-gradient(135deg, rgba(82,82,91,0.95), rgba(39,39,42,0.84))",
            "linear-gradient(135deg, rgba(55,65,81,0.95), rgba(17,24,39,0.84))",
        ]
        pool = rec_gradients if pred_id == 1 else not_rec_gradients
        return pool[sample_id % len(pool)]

    cards_html: List[str] = []

    for item in sample_results:
        movie_rows: List[str] = []
        for idx, movie in enumerate(item["movies"], start=1):
            movie_rows.append(
                f"""
                <div class="movie-row">
                    <div class="movie-rank">{idx:02d}</div>
                    <div class="movie-main">
                        <div class="movie-title">{esc(movie["title"])}</div>
                        <div class="movie-genres">{esc(movie["genres"])}</div>
                    </div>
                    <div class="movie-metrics">
                        <div class="metric-chip">⭐ {esc(movie["rating_mean"])}</div>
                        <div class="metric-chip">👥 {esc(movie["rating_count"])}</div>
                    </div>
                </div>
                """
            )

        poster_style = card_poster_gradient(
            int(item["derived_pred_label_id"]),
            int(item["sample_id"]),
        )

        profile_num = f"{int(item['sample_id']) + 1:02d}"

        cards_html.append(
            f"""
            <section class="viewer-card">
                <div class="poster-glow"></div>

                <div class="card-shell">
                    <div class="fake-poster" style="background:{poster_style};">
                        <div class="poster-overlay"></div>
                        <div class="poster-chip">{esc(item["decision_badge"])}</div>

                        <div class="poster-number-block">
                            <div class="poster-profile-label">Profile</div>
                            <div class="poster-profile-number">{esc(profile_num)}</div>
                        </div>

                        <div class="poster-brand">WATCHLY</div>
                        <div class="poster-title">{esc(item["headline"])}</div>
                        <div class="poster-subtitle">{esc(item["tagline"])}</div>
                    </div>

                    <div class="card-content">
                        <div class="card-top">
                            <div>
                                <div class="eyebrow">{esc(profile_header(item["sample_id"]))}</div>
                                <h2>{esc(item["headline"])}</h2>
                                <p class="tagline">{esc(item["tagline"])}</p>
                            </div>
                            <div class="decision-pill">{esc(item["confidence_band"])}</div>
                        </div>

                        <div class="summary-panel">
                            <div class="summary-banner">{esc(item["verdict_banner"])}</div>
                            <p class="summary-text">{esc(item["support_summary"])}</p>
                        </div>

                        <div class="info-grid">
                            <div class="info-line">
                                <span class="label">Viewer Pick</span>
                                <span class="value">{esc(item["decision_badge"])}</span>
                            </div>
                            <div class="info-line">
                                <span class="label">Match Confidence</span>
                                <span class="value">{esc(item["recommended_match_percent"])}%</span>
                            </div>
                            <div class="info-line">
                                <span class="label">Viewer Summary</span>
                                <span class="value">{esc(item["pattern_summary"])}</span>
                            </div>
                            <div class="info-line">
                                <span class="label">Status</span>
                                <span class="value">{esc(item["result_status"])}</span>
                            </div>
                            
                            <div class="info-line">
                                <span class="label">Score Gap</span>
                                <span class="value">{esc(item["score_gap_text"])}</span>
                            </div>
                        </div>

                        <div class="movies-panel">
                            <div class="panel-title">{esc(item["movie_bucket_title"])}</div>
                            {''.join(movie_rows) if movie_rows else '<div class="empty-state">No mapped titles found for this result.</div>'}
                        </div>
                    </div>
                </div>
            </section>
            """
        )

    html_document = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>WATCHLY</title>
    <style>
        :root {{
            --bg: #080a0f;
            --bg-soft: #11141c;
            --panel: rgba(16, 20, 30, 0.90);
            --panel-strong: rgba(22, 27, 39, 0.96);
            --text: #f5f7fb;
            --muted: #b5bfd3;
            --subtle: #7f8ba3;
            --line: rgba(255,255,255,0.08);
            --accent: #e50914;
            --accent-soft: rgba(229, 9, 20, 0.18);
            --shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
            --radius: 22px;
        }}

        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            font-family: Inter, Segoe UI, Arial, sans-serif;
            background:
                radial-gradient(circle at top left, rgba(229, 9, 20, 0.15), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 140, 66, 0.10), transparent 24%),
                linear-gradient(180deg, #06080d 0%, #0b1017 100%);
            color: var(--text);
        }}

        .page {{
            width: min(1400px, calc(100% - 40px));
            margin: 28px auto 56px;
        }}

        .hero {{
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(229, 9, 20, 0.24), rgba(255, 128, 75, 0.12)),
                var(--panel);
            border: 1px solid var(--line);
            border-radius: 30px;
            box-shadow: var(--shadow);
            padding: 50px 44px 40px;
            margin-bottom: 28px;
        }}

        .hero::before {{
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(0deg, rgba(0, 0, 0, 0.20), rgba(0, 0, 0, 0.03));
            pointer-events: none;
        }}

        .hero::after {{
            content: "";
            position: absolute;
            inset: auto -80px -80px auto;
            width: 280px;
            height: 280px;
            background: radial-gradient(circle, rgba(229, 9, 20, 0.30), transparent 70%);
            pointer-events: none;
        }}

        .hero-content {{
            position: relative;
            z-index: 1;
        }}

        .brand-chip {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(255,255,255,0.05);
            color: #ffd7d9;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 16px;
            backdrop-filter: blur(6px);
        }}

        .hero h1 {{
            margin: 0;
            font-size: clamp(42px, 6vw, 74px);
            line-height: 0.95;
            letter-spacing: -0.05em;
        }}

        .hero h2 {{
            margin: 12px 0 0;
            font-size: clamp(18px, 2vw, 24px);
            font-weight: 600;
            color: #eef2fb;
            letter-spacing: -0.02em;
        }}

        .viewer-card {{
            position: relative;
            overflow: hidden;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 28px;
            box-shadow: var(--shadow);
            padding: 22px;
            margin-bottom: 22px;
        }}

        .viewer-card:hover {{
            transform: translateY(-2px);
            transition: transform 160ms ease;
        }}

        .poster-glow {{
            position: absolute;
            inset: auto -100px -100px auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(229, 9, 20, 0.12), transparent 70%);
            pointer-events: none;
        }}

        .card-shell {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 22px;
            align-items: stretch;
        }}

        .fake-poster {{
            position: relative;
            min-height: 420px;
            border-radius: 24px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 16px 40px rgba(0,0,0,0.28);
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            padding: 22px;
        }}

        .poster-overlay {{
            position: absolute;
            inset: 0;
            background:
                linear-gradient(180deg, rgba(0,0,0,0.08) 0%, rgba(0,0,0,0.18) 30%, rgba(0,0,0,0.72) 100%);
        }}

        .poster-chip,
        .poster-number-block,
        .poster-brand,
        .poster-title,
        .poster-subtitle {{
            position: relative;
            z-index: 1;
        }}

        .poster-chip {{
            align-self: flex-start;
            border-radius: 999px;
            padding: 8px 12px;
            background: rgba(255,255,255,0.14);
            border: 1px solid rgba(255,255,255,0.12);
            font-size: 12px;
            font-weight: 700;
            margin-bottom: 18px;
            backdrop-filter: blur(6px);
        }}

        .poster-number-block {{
            margin-top: auto;
            margin-bottom: 18px;
        }}

        .poster-profile-label {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.80);
            margin-bottom: 2px;
        }}

        .poster-profile-number {{
            font-size: 92px;
            line-height: 0.9;
            font-weight: 900;
            letter-spacing: -0.08em;
            color: rgba(255,255,255,0.96);
            font-style: italic;
            text-shadow: 0 10px 24px rgba(0,0,0,0.20);
        }}

        .poster-brand {{
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: rgba(255,255,255,0.82);
            margin-bottom: 10px;
        }}

        .poster-title {{
            font-size: 28px;
            font-weight: 800;
            line-height: 1.06;
            letter-spacing: -0.04em;
            margin-bottom: 8px;
            font-style: italic;
        }}

        .poster-subtitle {{
            color: rgba(255,255,255,0.88);
            font-size: 14px;
            line-height: 1.5;
            font-style: italic;
        }}

        .card-content {{
            min-width: 0;
        }}

        .card-top {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            margin-bottom: 18px;
        }}

        .eyebrow {{
            color: var(--subtle);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 8px;
        }}

        h2 {{
            margin: 0;
            font-size: 36px;
            line-height: 1.05;
            letter-spacing: -0.04em;
        }}

        .tagline {{
            margin: 10px 0 0;
            color: var(--muted);
            font-size: 15px;
            line-height: 1.65;
            max-width: 760px;
        }}

        .decision-pill {{
            padding: 10px 14px;
            border-radius: 999px;
            background: var(--accent-soft);
            border: 1px solid rgba(229, 9, 20, 0.28);
            font-size: 13px;
            font-weight: 700;
            white-space: nowrap;
        }}

        .summary-panel {{
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 18px;
        }}

        .summary-banner {{
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #ffd7d9;
            margin-bottom: 10px;
        }}

        .summary-text {{
            margin: 0;
            color: var(--muted);
            line-height: 1.75;
            font-size: 15px;
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px 18px;
            margin-bottom: 22px;
        }}

        .info-line {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 15px;
            min-height: 74px;
        }}

        .label {{
            color: var(--subtle);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }}

        .value {{
            color: var(--text);
            font-size: 15px;
            line-height: 1.55;
            word-break: break-word;
        }}

        .movies-panel {{
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px;
        }}

        .panel-title {{
            font-size: 14px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #ffd7d9;
            margin-bottom: 14px;
        }}

        .movie-row {{
            display: grid;
            grid-template-columns: 52px 1fr auto;
            gap: 14px;
            align-items: center;
            padding: 14px 0;
            border-top: 1px solid var(--line);
        }}

        .movie-row:first-of-type {{
            border-top: none;
            padding-top: 0;
        }}

        .movie-rank {{
            width: 40px;
            height: 40px;
            border-radius: 12px;
            background: rgba(255,255,255,0.06);
            display: grid;
            place-items: center;
            font-weight: 700;
            color: var(--text);
        }}

        .movie-main {{
            min-width: 0;
        }}

        .movie-title {{
            font-size: 16px;
            font-weight: 700;
            line-height: 1.4;
            margin-bottom: 4px;
        }}

        .movie-genres {{
            color: var(--muted);
            font-size: 13px;
            line-height: 1.5;
        }}

        .movie-metrics {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }}

        .metric-chip {{
            border: 1px solid var(--line);
            background: rgba(255,255,255,0.04);
            border-radius: 999px;
            padding: 8px 10px;
            font-size: 12px;
            color: var(--muted);
        }}

        .empty-state {{
            color: var(--muted);
            font-size: 14px;
            line-height: 1.6;
        }}

        .footer-note {{
            text-align: center;
            color: var(--subtle);
            font-size: 13px;
            margin-top: 26px;
            line-height: 1.7;
        }}

        @media (max-width: 1080px) {{
            .card-shell {{
                grid-template-columns: 1fr;
            }}

            .fake-poster {{
                min-height: 320px;
            }}

            .info-grid {{
                grid-template-columns: 1fr;
            }}

            .poster-profile-number {{
                font-size: 78px;
            }}
        }}

        @media (max-width: 720px) {{
            .page {{
                width: min(100% - 20px, 1400px);
                margin: 20px auto 40px;
            }}

            .hero {{
                padding: 30px 22px 28px;
            }}

            .card-top {{
                flex-direction: column;
            }}

            .movie-row {{
                grid-template-columns: 1fr;
                gap: 10px;
            }}

            .movie-metrics {{
                justify-content: flex-start;
            }}

            .poster-profile-number {{
                font-size: 64px;
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        <section class="hero">
            <div class="hero-content">
                <div class="brand-chip">🎬 WATCHLY</div>
                <h1>WATCHLY</h1>
                <h2>Your personalized watch picks for tonight.</h2>
            </div>
        </section>

        {''.join(cards_html)}

        <div class="footer-note">
            WATCHLY
        </div>
    </div>
</body>
</html>
"""

    with open(ANALYSIS_REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html_document)


# =============================================================================
# Console printing
# =============================================================================
def print_batch_summary(
    source_name: str,
    sample_results: List[Dict[str, object]],
    top_k_movies: int,
    min_rating_count: int,
    recommendation_source: str,
) -> None:
    labels = [int(item["true_label_id"]) for item in sample_results]
    pred_ids = [int(item["derived_pred_label_id"]) for item in sample_results]
    correct = count_matches(pred_ids, labels)
    accuracy = correct / float(len(sample_results))
    accuracy_text = f"{accuracy * 100:.2f}% across {len(sample_results)} evaluated viewers"

    label_names = [safe_class_name(x) for x in labels]
    pred_names = [safe_class_name(x) for x in pred_ids]
    selected_indices = [
        item["dataset_row_index"] for item in sample_results if item["dataset_row_index"] is not None
    ]

    print_section("RUN SUMMARY")
    print_kv("Source artifact", source_name)
    print_kv("Recommendation source", recommendation_source)
    print_kv("JSON report", ANALYSIS_REPORT_JSON)
    print_kv("HTML report", ANALYSIS_REPORT_HTML)
    print_kv("Top titles per viewer", top_k_movies)
    print_kv("Minimum rating count", min_rating_count)
    print()
    print_kv("Reference labels", labels)
    print_kv("Reference label names", label_names)
    print_kv("Predicted labels", pred_ids)
    print_kv("Predicted label names", pred_names)
    print_kv("Batch match accuracy", accuracy_text)

    if selected_indices:
        print_kv("Dataset rows", selected_indices)


def print_movies_table(movies: List[Dict[str, object]]) -> None:
    if not movies:
        print("No mapped titles found for this result.\n")
        return

    header = f"{'No.':<4} {'Title':<42} {'Rating':<8} {'Count':<8} Genres"
    print(header)
    print_rule("-")

    for idx, rec in enumerate(movies, start=1):
        title = str(rec["title"])
        if len(title) > 40:
            title = title[:37] + "..."

        print(
            f"{idx:<4} "
            f"{title:<42} "
            f"{rec['rating_mean']:<8} "
            f"{rec['rating_count']:<8} "
            f"{rec['genres']}"
        )
    print()


def print_sample_results(sample_results: List[Dict[str, object]]) -> None:
    print_section("VIEWER REPORTS")

    for item in sample_results:
        print_banner_line(viewer_header(item["sample_id"]))
        print(item["verdict_banner"].encode("ascii", "ignore").decode())
        print()

        if item["dataset_row_index"] is not None:
            print_kv("Dataset row", item["dataset_row_index"])

        print_kv("Viewer pick", item["decision_badge"])
        print_kv("Match confidence", f"{item['recommended_match_percent']}%")
        print_kv("Confidence band", item["confidence_band"])
        print_kv("Viewer summary", item["pattern_summary"])
        print_kv("Watch note", item["tagline"])
        print_kv("Support line", item["support_summary"])
        print_kv("Reference label", item["true_label_name"])
        print_kv("Predicted label", item["pred_label_name"])
        print_kv("Reference match", item["match"])
        print_kv("Stored prediction aligned", item["prediction_consistent_with_saved"])
        print_kv("Score gap", item["score_gap_text"])
        print_kv("Class scores (Q)", item["class_scores_q"])
        print_kv("Class scores (Float)", item["class_scores_float"])

        print()
        print(item["movie_bucket_title"])
        print_rule("-")
        print_movies_table(item["movies"])


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    validate_required_files()

    if args.top_k_movies < 1:
        raise ValueError("--top-k-movies must be at least 1")

    logits_q_2d, labels, pred_ids, selected_indices, source_name = load_inference_artifact()

    movie_pool_json: Dict[str, List[Dict[str, object]]] | None = None
    movie_pool_df: pd.DataFrame | None = None

    if MOVIE_RECO_POOLS_JSON.exists():
        movie_pool_json = load_movie_recommendation_pool_from_json()
        recommendation_source = str(MOVIE_RECO_POOLS_JSON)
    else:
        movie_pool_df = load_movie_recommendation_pool_from_movielens()
        recommendation_source = f"{MOVIES_CSV} + {RATINGS_CSV}"

    sample_results = build_sample_analysis(
        logits_q_2d=logits_q_2d,
        labels=labels,
        pred_ids=pred_ids,
        selected_indices=selected_indices,
        movie_pool_json=movie_pool_json,
        movie_pool_df=movie_pool_df,
        top_k_movies=args.top_k_movies,
        min_rating_count=args.min_rating_count,
    )

    save_analysis_report(
        source_name=source_name,
        sample_results=sample_results,
        top_k_movies=args.top_k_movies,
        min_rating_count=args.min_rating_count,
        recommendation_source=recommendation_source,
    )

    generate_html_report(
        source_name=source_name,
        sample_results=sample_results,
        top_k_movies=args.top_k_movies,
        min_rating_count=args.min_rating_count,
        recommendation_source=recommendation_source,
    )

    print()
    print_title("WATCHLY VIEWER REPORT")
    print_subtitle("NNIA Recommendation Summary")

    print_batch_summary(
        source_name=source_name,
        sample_results=sample_results,
        top_k_movies=args.top_k_movies,
        min_rating_count=args.min_rating_count,
        recommendation_source=recommendation_source,
    )

    print_sample_results(sample_results)

    print_section("EXECUTION STATUS")
    print("Report generation completed successfully.")
    print()


if __name__ == "__main__":
    main()