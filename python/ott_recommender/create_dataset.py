"""
create_dataset.py

Author: Yuvraj Singh

Project: Neural Network Inference Accelerator (NNIA)

MovieLens-based dataset builder for a two-class OTT-style recommendation task.

Purpose
-------
- Build dataset samples from user interaction history
- Generate feature vectors from history windows
- Derive binary labels from future user behavior
- Apply user-level train/test split to reduce data leakage
- Improve label quality by filtering ambiguous samples
- Maintain alignment with the NNIA training and inference pipeline

Label Mapping
-------------
0 -> Not Recommended
1 -> Recommended
"""


from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

# =============================================================================
# Python package path fix for direct script execution
# =============================================================================
THIS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = THIS_DIR.parent

if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from ott_recommender.feature_encoder import (
    FEATURE_NAMES,
    INPUT_SIZE,
    build_feature_dict,
    feature_dict_to_vector,
    normalize_genre_preferences,
    validate_feature_vector,
)

# =============================================================================
# Configuration
# =============================================================================
RANDOM_SEED = 7
TRAIN_RATIO = 0.80

# Window settings
HISTORY_WINDOW_LEN = 28
FUTURE_GAP_LEN = 4
FUTURE_WINDOW_LEN = 8
STRIDE = 4
MIN_USER_RATINGS = HISTORY_WINDOW_LEN + FUTURE_GAP_LEN + FUTURE_WINDOW_LEN

# Dataset size controls
MAX_SAMPLES_PER_USER = 8
MAX_USERS_TO_PROCESS = 7000
RATINGS_CHUNK_SIZE = 1_000_000

# Quality checks
MIN_FUTURE_EVENTS = 6
MIN_HISTORY_VARIANCE = 0.22
MIN_HISTORY_POS_NEG_MIX = 0.18

# Label purity thresholds
POS_MEAN_MIN = 3.95
NEG_MEAN_MAX = 2.85

POS_HIGH_RATIO_MIN = 0.70          # ratings >= 4.0
NEG_LOW_RATIO_MIN = 0.60           # ratings <= 2.5

POS_MED_RATIO_MIN = 0.85           # ratings >= 3.5
NEG_MED_RATIO_MIN = 0.70           # ratings <= 3.0

MIN_MEAN_MARGIN = 0.35
AMBIGUOUS_CENTER = 3.40
AMBIGUOUS_BAND = 0.30

# Mild balancing cap
MAX_SAMPLES_PER_CLASS = 4000

# Output controls
FORCE_REBUILD_OUTPUTS = True
MOVIE_POOL_MAX_PER_LABEL = 150
MOVIE_POOL_MIN_COUNT_SOFT = 50

# =============================================================================
# Project paths
# =============================================================================
PROJECT_ROOT = THIS_DIR.parent.parent

MOVIELENS_DIR = PROJECT_ROOT / "datasets" / "ml-25m"
MOVIES_CSV = MOVIELENS_DIR / "movies.csv"
RATINGS_CSV = MOVIELENS_DIR / "ratings.csv"

DATASET_DIR = PROJECT_ROOT / "artifacts" / "datasets"
FULL_DATASET_CSV = DATASET_DIR / "ott_dataset_full.csv"
TRAIN_DATASET_CSV = DATASET_DIR / "ott_dataset_train.csv"
TEST_DATASET_CSV = DATASET_DIR / "ott_dataset_test.csv"
METADATA_JSON = DATASET_DIR / "dataset_metadata.json"
MOVIE_RECO_POOLS_JSON = DATASET_DIR / "movie_reco_pools.json"

CLASS_NAMES = ["Not Recommended", "Recommended"]
NUM_CLASSES = 2

DISPLAY_MESSAGES = {
    0: "This title is less aligned with your recent taste pattern",
    1: "This title is strongly aligned with your recent taste pattern",
}

# Time-of-day proxies
NIGHT_START_HOUR = 20
NIGHT_END_HOUR = 5

COMMUTE_MORNING_START = 7
COMMUTE_MORNING_END = 10
COMMUTE_EVENING_START = 17
COMMUTE_EVENING_END = 20

# Gap thresholds
BINGE_GAP_SEC = 3 * 3600
FAST_RATE_GAP_SEC = 30 * 60

# Genre support for feature construction
TARGET_GENRES = ["Action", "Romance", "Comedy", "Thriller"]
LIKED_RATING_THRESHOLD = 3.5
RECENCY_GAMMA_HISTORY = 0.93


# =============================================================================
# Small helpers
# =============================================================================
def clamp01(value: float) -> float:
    value = float(value)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    den = float(den)
    if abs(den) < 1e-12:
        return float(default)
    return float(num) / den


def normalize(value: float, min_val: float, max_val: float) -> float:
    min_val = float(min_val)
    max_val = float(max_val)
    if max_val <= min_val:
        return 0.0
    return clamp01((float(value) - min_val) / (max_val - min_val))


def extract_year_from_title(title: str) -> Optional[int]:
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    return int(match.group(1)) if match else None


def is_night_hour(hour: int) -> bool:
    return (hour >= NIGHT_START_HOUR) or (hour <= NIGHT_END_HOUR)


def is_commute_hour(hour: int) -> bool:
    return (
        COMMUTE_MORNING_START <= hour < COMMUTE_MORNING_END
        or COMMUTE_EVENING_START <= hour < COMMUTE_EVENING_END
    )


def entropy_normalized(probabilities: Sequence[float]) -> float:
    probs = [float(p) for p in probabilities if p > 0.0]
    if not probs:
        return 0.0

    h_val = -sum(p * math.log(p, 2) for p in probs)
    h_max = math.log(len(probabilities), 2) if len(probabilities) > 1 else 1.0
    if h_max <= 0.0:
        return 0.0

    return clamp01(h_val / h_max)


def recency_weights(length: int, gamma: float) -> List[float]:
    if length <= 0:
        return []

    raw = [gamma ** (length - 1 - i) for i in range(length)]
    total = sum(raw)
    if total <= 0.0:
        return [1.0 / length] * length

    return [value / total for value in raw]


def history_rating_weight(rating: float) -> float:
    rating = float(rating)
    return clamp01((rating - 2.5) / 2.5)


def delete_existing_generated_outputs() -> None:
    for path in [
        FULL_DATASET_CSV,
        TRAIN_DATASET_CSV,
        TEST_DATASET_CSV,
        METADATA_JSON,
        MOVIE_RECO_POOLS_JSON,
    ]:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


# =============================================================================
# Label logic
# =============================================================================
def future_window_to_binary_label(future_df: pd.DataFrame) -> Optional[int]:
    """
    Convert a future window into a cleaner binary recommendation label.

    Strategy
    --------
    - Keep only high-confidence positive or negative future windows
    - Reject ambiguous middle windows instead of forcing a label
    - Prefer purity over raw dataset size
    """
    if future_df is None or future_df.empty:
        return None

    ratings = future_df["rating"].astype(float).tolist()
    if len(ratings) < MIN_FUTURE_EVENTS:
        return None

    count = len(ratings)
    mean_rating = sum(ratings) / count

    ratio_ge_40 = safe_div(sum(1 for r in ratings if r >= 4.0), count, 0.0)
    ratio_ge_35 = safe_div(sum(1 for r in ratings if r >= 3.5), count, 0.0)
    ratio_le_25 = safe_div(sum(1 for r in ratings if r <= 2.5), count, 0.0)
    ratio_le_30 = safe_div(sum(1 for r in ratings if r <= 3.0), count, 0.0)

    mean_margin_from_center = abs(mean_rating - AMBIGUOUS_CENTER)

    if mean_margin_from_center < AMBIGUOUS_BAND:
        return None

    if (
        mean_rating >= POS_MEAN_MIN
        and ratio_ge_40 >= POS_HIGH_RATIO_MIN
        and ratio_ge_35 >= POS_MED_RATIO_MIN
        and ratio_le_25 <= 0.10
    ):
        return 1

    if (
        mean_rating <= NEG_MEAN_MAX
        and ratio_le_25 >= NEG_LOW_RATIO_MIN
        and ratio_le_30 >= NEG_MED_RATIO_MIN
        and ratio_ge_40 <= 0.10
    ):
        return 0

    if (
        mean_rating >= (AMBIGUOUS_CENTER + MIN_MEAN_MARGIN)
        and ratio_ge_40 >= 0.60
        and ratio_le_30 <= 0.20
    ):
        return 1

    if (
        mean_rating <= (AMBIGUOUS_CENTER - MIN_MEAN_MARGIN)
        and ratio_le_25 >= 0.50
        and ratio_ge_35 <= 0.20
    ):
        return 0

    return None


# =============================================================================
# Loading
# =============================================================================
def validate_movielens_files() -> None:
    missing = []

    if not MOVIES_CSV.exists():
        missing.append(str(MOVIES_CSV))
    if not RATINGS_CSV.exists():
        missing.append(str(RATINGS_CSV))

    if missing:
        raise FileNotFoundError(
            "Missing required MovieLens file(s):\n"
            + "\n".join(missing)
            + "\n\nExpected extracted dataset under:\n"
            + f"{MOVIELENS_DIR}"
        )


def load_movies_df() -> pd.DataFrame:
    movies_df = pd.read_csv(
        MOVIES_CSV,
        usecols=["movieId", "title", "genres"],
        dtype={
            "movieId": "int32",
            "title": "string",
            "genres": "string",
        },
    )

    if "movieId" not in movies_df.columns or "genres" not in movies_df.columns:
        raise ValueError("movies.csv is missing required columns")

    movies_df = movies_df.drop_duplicates(subset=["movieId"]).copy()
    movies_df["genres"] = movies_df["genres"].fillna("(no genres listed)")
    movies_df["year"] = movies_df["title"].apply(extract_year_from_title)
    return movies_df


def get_selected_user_ids() -> Optional[set[int]]:
    if MAX_USERS_TO_PROCESS is None:
        return None

    selected: List[int] = []
    seen: set[int] = set()

    for chunk in pd.read_csv(
        RATINGS_CSV,
        usecols=["userId"],
        dtype={"userId": "int32"},
        chunksize=RATINGS_CHUNK_SIZE,
    ):
        for uid in chunk["userId"].drop_duplicates().tolist():
            uid = int(uid)
            if uid not in seen:
                seen.add(uid)
                selected.append(uid)
                if len(selected) >= MAX_USERS_TO_PROCESS:
                    return set(selected)

    return set(selected)


def load_ratings_for_selected_users(selected_users: Optional[set[int]]) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []

    dtype_map = {
        "userId": "int32",
        "movieId": "int32",
        "rating": "float32",
        "timestamp": "int64",
    }

    for chunk in pd.read_csv(
        RATINGS_CSV,
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype=dtype_map,
        chunksize=RATINGS_CHUNK_SIZE,
    ):
        if selected_users is not None:
            chunk = chunk[chunk["userId"].isin(selected_users)]

        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        raise RuntimeError(
            "No rating rows found after filtering. "
            "Check MAX_USERS_TO_PROCESS and ratings.csv."
        )

    return pd.concat(chunks, ignore_index=True)


def load_movielens() -> pd.DataFrame:
    validate_movielens_files()

    movies_df = load_movies_df()
    selected_users = get_selected_user_ids()
    ratings_df = load_ratings_for_selected_users(selected_users)

    if not {"userId", "movieId", "rating", "timestamp"}.issubset(ratings_df.columns):
        raise ValueError("ratings.csv is missing required columns")

    merged = ratings_df.merge(movies_df, on="movieId", how="left", copy=False)
    merged = merged.dropna(subset=["title", "genres"]).copy()

    ts = pd.to_datetime(merged["timestamp"], unit="s", utc=True)
    merged["hour"] = ts.dt.hour.astype("int8")
    merged["dayofweek"] = ts.dt.dayofweek.astype("int8")
    merged["is_weekend"] = merged["dayofweek"].isin([5, 6]).astype("int8")
    del ts

    merged = merged.sort_values(["userId", "timestamp"]).reset_index(drop=True)
    return merged


# =============================================================================
# Genre helpers
# =============================================================================
def split_genres(genres_str: str) -> List[str]:
    if not isinstance(genres_str, str) or not genres_str.strip():
        return []
    if genres_str == "(no genres listed)":
        return []
    return [genre.strip() for genre in genres_str.split("|") if genre.strip()]


def compute_target_genre_scores_history(
    df: pd.DataFrame,
    recency_gamma: float,
) -> Dict[str, float]:
    scores = {genre: 0.0 for genre in TARGET_GENRES}

    if len(df) == 0:
        return scores

    weights = recency_weights(len(df), recency_gamma)

    for idx, row in enumerate(df.itertuples(index=False)):
        row_genres = split_genres(row.genres)
        rating = float(row.rating)
        base = history_rating_weight(rating)

        if base <= 0.0:
            continue

        for genre in TARGET_GENRES:
            if genre in row_genres:
                scores[genre] += base * weights[idx]

    return scores


# =============================================================================
# Feature engineering
# =============================================================================
def exploration_score_from_history(history_df: pd.DataFrame) -> float:
    genre_counter: Counter[str] = Counter()

    for row in history_df.itertuples(index=False):
        for genre in split_genres(row.genres):
            genre_counter[genre] += 1

    total_genre_hits = sum(genre_counter.values())
    if total_genre_hits <= 0:
        return 0.0

    genre_probs = [count / total_genre_hits for count in genre_counter.values()]
    return entropy_normalized(genre_probs)


def engineer_features_from_history(history_df: pd.DataFrame) -> List[float]:
    if len(history_df) == 0:
        raise ValueError("history_df must not be empty")

    history_df = history_df.sort_values("timestamp").reset_index(drop=True)

    ratings = history_df["rating"].astype(float).tolist()
    timestamps = history_df["timestamp"].astype(int).tolist()
    hours = history_df["hour"].astype(int).tolist()
    recency_w = recency_weights(len(history_df), RECENCY_GAMMA_HISTORY)

    genre_scores = compute_target_genre_scores_history(
        history_df,
        recency_gamma=RECENCY_GAMMA_HISTORY,
    )

    genre_prefs = normalize_genre_preferences(
        genre_scores["Action"],
        genre_scores["Romance"],
        genre_scores["Comedy"],
        genre_scores["Thriller"],
        method="l1",
    )

    action_pref = genre_prefs["action_pref"]
    romance_pref = genre_prefs["romance_pref"]
    comedy_pref = genre_prefs["comedy_pref"]
    thriller_pref = genre_prefs["thriller_pref"]

    t_min = int(history_df["timestamp"].min())
    t_max = int(history_df["timestamp"].max())
    active_days = max(1.0, (t_max - t_min) / 86400.0)
    ratings_per_day = len(history_df) / active_days
    avg_watch_time_norm = normalize(ratings_per_day, 0.5, 8.0)

    weekend_watch_ratio = float(history_df["is_weekend"].mean())

    years_series = history_df["year"]
    valid_year_indices = [i for i, y in enumerate(years_series.tolist()) if pd.notna(y)]
    if valid_year_indices:
        weighted_year_sum = 0.0
        weighted_year_den = 0.0
        years_list = years_series.tolist()

        for i in valid_year_indices:
            weighted_year_sum += float(years_list[i]) * recency_w[i]
            weighted_year_den += recency_w[i]

        mean_year = safe_div(weighted_year_sum, weighted_year_den, default=1995.0)
        prefers_new_releases = normalize(mean_year, 1970.0, 2020.0)
    else:
        prefers_new_releases = 0.50

    fast_gaps = 0
    binge_gaps = 0
    total_gaps = 0

    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        total_gaps += 1
        if gap <= FAST_RATE_GAP_SEC:
            fast_gaps += 1
        if gap <= BINGE_GAP_SEC:
            binge_gaps += 1

    skips_intro_ratio = safe_div(fast_gaps, total_gaps, 0.0)
    binge_watch_ratio = safe_div(binge_gaps, total_gaps, 0.0)

    weighted_rating = sum(float(r) * w for r, w in zip(ratings, recency_w))
    rating_generosity = normalize(weighted_rating, 0.5, 5.0)

    night_count = sum(1 for hour in hours if is_night_hour(hour))
    night_watch_ratio = safe_div(night_count, len(history_df), 0.0)

    commute_count = sum(1 for hour in hours if is_commute_hour(hour))
    mobile_watch_ratio = safe_div(commute_count, len(history_df), 0.0)

    exploration_score = exploration_score_from_history(history_df)

    short_content_pref = clamp01(
        0.25 * comedy_pref
        + 0.20 * mobile_watch_ratio
        + 0.15 * skips_intro_ratio
        + 0.10 * weekend_watch_ratio
        + 0.30 * (1.0 - avg_watch_time_norm)
    )

    dominant_target_pref = max(action_pref, romance_pref, comedy_pref, thriller_pref)
    rewatch_ratio = clamp01(
        0.35 * (1.0 - prefers_new_releases)
        + 0.35 * dominant_target_pref
        + 0.15 * rating_generosity
        + 0.15 * (1.0 - exploration_score)
    )

    liked_ratio = safe_div(
        sum(1 for r in ratings if r >= LIKED_RATING_THRESHOLD),
        len(ratings),
        0.5,
    )
    completion_ratio = clamp01(0.75 * liked_ratio + 0.25 * (1.0 - skips_intro_ratio))

    feature_dict = build_feature_dict(
        action_pref=action_pref,
        romance_pref=romance_pref,
        comedy_pref=comedy_pref,
        thriller_pref=thriller_pref,
        avg_watch_time_norm=avg_watch_time_norm,
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

    vector = feature_dict_to_vector(feature_dict, clamp_values=True)
    validate_feature_vector(vector)
    return vector


# =============================================================================
# Dataset construction
# =============================================================================
def dominant_label_for_user_samples(user_samples: Sequence[Tuple[List[float], int]]) -> int:
    counter: Counter[int] = Counter(label for _, label in user_samples)
    return counter.most_common(1)[0][0]


def history_window_quality_ok(history_df: pd.DataFrame) -> bool:
    ratings = history_df["rating"].astype(float).tolist()
    if len(ratings) < HISTORY_WINDOW_LEN:
        return False

    rating_std = float(pd.Series(ratings).std()) if len(ratings) > 1 else 0.0
    if rating_std < MIN_HISTORY_VARIANCE:
        return False

    pos_ratio = safe_div(sum(1 for r in ratings if r >= 4.0), len(ratings), 0.0)
    neg_ratio = safe_div(sum(1 for r in ratings if r <= 2.5), len(ratings), 0.0)

    if (pos_ratio + neg_ratio) < MIN_HISTORY_POS_NEG_MIX:
        return False

    return True


def apply_optional_class_cap(
    samples: Sequence[Tuple[int, List[float], int]],
    max_samples_per_class: int,
    seed: int,
) -> List[Tuple[int, List[float], int]]:
    if max_samples_per_class <= 0:
        return list(samples)

    by_class: Dict[int, List[Tuple[int, List[float], int]]] = defaultdict(list)
    for sample in samples:
        by_class[int(sample[2])].append(sample)

    rng = random.Random(seed)
    capped: List[Tuple[int, List[float], int]] = []

    available_counts = [len(v) for v in by_class.values() if len(v) > 0]
    minority_count = min(available_counts, default=0)
    if minority_count > 0:
        effective_cap = min(max_samples_per_class, minority_count)
    else:
        effective_cap = max_samples_per_class

    for label in range(NUM_CLASSES):
        group = by_class.get(label, [])
        rng.shuffle(group)
        capped.extend(group[:effective_cap])

    rng.shuffle(capped)
    return capped


def build_samples_from_movielens(
    merged_df: pd.DataFrame,
) -> Tuple[List[Tuple[int, List[float], int]], Dict[str, object], Dict[int, int]]:
    samples: List[Tuple[int, List[float], int]] = []

    counters: Dict[str, object] = {
        "users_total": 0,
        "users_used": 0,
        "windows_total": 0,
        "windows_skipped_short_user": 0,
        "windows_skipped_no_future_window": 0,
        "windows_skipped_future_label_none": 0,
        "windows_skipped_history_low_quality": 0,
        "windows_skipped_user_cap": 0,
        "samples_kept_before_class_cap": 0,
        "samples_kept_final": 0,
        "user_sample_count_min": 0,
        "user_sample_count_max": 0,
        "user_sample_count_mean": 0.0,
    }

    per_user_sample_counts: List[int] = []
    user_ids = sorted(merged_df["userId"].unique().tolist())
    counters["users_total"] = len(user_ids)

    grouped = merged_df.groupby("userId", sort=False)
    total_needed = HISTORY_WINDOW_LEN + FUTURE_GAP_LEN + FUTURE_WINDOW_LEN

    for user_id in user_ids:
        if user_id not in grouped.groups:
            continue

        user_df = grouped.get_group(user_id).sort_values("timestamp").reset_index(drop=True)

        if len(user_df) < total_needed:
            counters["windows_skipped_short_user"] += 1
            continue

        counters["users_used"] += 1
        kept_for_user = 0
        max_start = len(user_df) - total_needed

        for start in range(0, max_start + 1, STRIDE):
            counters["windows_total"] += 1

            if kept_for_user >= MAX_SAMPLES_PER_USER:
                counters["windows_skipped_user_cap"] += 1
                continue

            history_start = start
            history_end = history_start + HISTORY_WINDOW_LEN
            future_start = history_end + FUTURE_GAP_LEN
            future_end = future_start + FUTURE_WINDOW_LEN

            if future_end > len(user_df):
                counters["windows_skipped_no_future_window"] += 1
                continue

            history_df = user_df.iloc[history_start:history_end]
            future_df = user_df.iloc[future_start:future_end]

            if not history_window_quality_ok(history_df):
                counters["windows_skipped_history_low_quality"] += 1
                continue

            label = future_window_to_binary_label(future_df)
            if label is None:
                counters["windows_skipped_future_label_none"] += 1
                continue

            vector = engineer_features_from_history(history_df)
            samples.append((int(user_id), vector, int(label)))
            kept_for_user += 1

        if kept_for_user > 0:
            per_user_sample_counts.append(kept_for_user)

    counters["samples_kept_before_class_cap"] = len(samples)

    samples = apply_optional_class_cap(
        samples=samples,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS,
        seed=RANDOM_SEED,
    )

    counters["samples_kept_final"] = len(samples)

    if per_user_sample_counts:
        counters["user_sample_count_min"] = int(min(per_user_sample_counts))
        counters["user_sample_count_max"] = int(max(per_user_sample_counts))
        counters["user_sample_count_mean"] = round(
            sum(per_user_sample_counts) / len(per_user_sample_counts),
            6,
        )

    per_user_final_samples: Dict[int, List[Tuple[List[float], int]]] = defaultdict(list)
    for user_id, vector, label in samples:
        per_user_final_samples[int(user_id)].append((vector, int(label)))

    user_primary_label = {
        user_id: dominant_label_for_user_samples(user_samples)
        for user_id, user_samples in per_user_final_samples.items()
        if user_samples
    }

    return samples, counters, user_primary_label


def split_dataset_by_user_balanced(
    dataset: Sequence[Tuple[int, List[float], int]],
    user_primary_label: Dict[int, int],
    train_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[List[float], int]], List[Tuple[List[float], int]], List[int], List[int]]:
    user_to_samples: Dict[int, List[Tuple[List[float], int]]] = defaultdict(list)
    for user_id, vector, label in dataset:
        user_to_samples[int(user_id)].append((vector, label))

    label_to_users: Dict[int, List[int]] = defaultdict(list)
    for user_id in sorted(user_to_samples.keys()):
        label = user_primary_label.get(user_id)
        if label is None:
            counter = Counter(lbl for _, lbl in user_to_samples[user_id])
            label = counter.most_common(1)[0][0]
        label_to_users[int(label)].append(user_id)

    rng = random.Random(seed)
    train_users: List[int] = []
    test_users: List[int] = []

    for label in range(NUM_CLASSES):
        users = label_to_users.get(label, [])
        rng.shuffle(users)

        if len(users) == 0:
            continue

        if len(users) == 1:
            train_count = 1
        else:
            train_count = int(round(len(users) * train_ratio))
            train_count = max(1, min(train_count, len(users) - 1))

        train_users.extend(users[:train_count])
        test_users.extend(users[train_count:])

    if not train_users or not test_users:
        all_users = sorted(user_to_samples.keys())
        rng.shuffle(all_users)

        split_idx = max(1, int(len(all_users) * train_ratio))
        split_idx = min(split_idx, len(all_users) - 1) if len(all_users) > 1 else len(all_users)

        train_users = all_users[:split_idx]
        test_users = all_users[split_idx:] if len(all_users) > 1 else []

    train_set: List[Tuple[List[float], int]] = []
    test_set: List[Tuple[List[float], int]] = []

    for uid in train_users:
        train_set.extend(user_to_samples[uid])
    for uid in test_users:
        test_set.extend(user_to_samples[uid])

    rng.shuffle(train_set)
    rng.shuffle(test_set)

    return train_set, test_set, sorted(train_users), sorted(test_users)


# =============================================================================
# Recommendation pools
# =============================================================================
def build_movie_recommendation_pools(
    merged_df: pd.DataFrame,
) -> Dict[str, List[Dict[str, object]]]:
    movie_stats = (
        merged_df.groupby(["movieId", "title", "genres", "year"], dropna=False)["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_rating", "count": "rating_count"})
    )

    if movie_stats.empty:
        return {name: [] for name in CLASS_NAMES}

    movie_stats = movie_stats[movie_stats["rating_count"] >= MOVIE_POOL_MIN_COUNT_SOFT].copy()

    pools: Dict[str, List[Dict[str, object]]] = {name: [] for name in CLASS_NAMES}

    pos_df = movie_stats[movie_stats["avg_rating"] >= 3.9].copy()
    pos_df = pos_df.sort_values(by=["avg_rating", "rating_count"], ascending=[False, False])

    neg_df = movie_stats[movie_stats["avg_rating"] <= 2.9].copy()
    neg_df = neg_df.sort_values(by=["avg_rating", "rating_count"], ascending=[True, False])

    for label_name, label_df in [("Recommended", pos_df), ("Not Recommended", neg_df)]:
        reco_list: List[Dict[str, object]] = []
        for _, row in label_df.head(MOVIE_POOL_MAX_PER_LABEL).iterrows():
            reco_list.append(
                {
                    "movieId": int(row["movieId"]),
                    "title": str(row["title"]),
                    "genres": str(row["genres"]),
                    "year": None if pd.isna(row["year"]) else int(row["year"]),
                    "avg_rating": round(float(row["avg_rating"]), 4),
                    "rating_count": int(row["rating_count"]),
                }
            )
        pools[label_name] = reco_list

    return pools


# =============================================================================
# File writing
# =============================================================================
def write_dataset_csv(path: Path, dataset: Sequence[Tuple[List[float], int]]) -> None:
    header = FEATURE_NAMES + ["label", "label_name", "display_text"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for vector, label in dataset:
            validate_feature_vector(vector)
            row = [f"{float(value):.6f}" for value in vector]
            row.append(int(label))
            row.append(CLASS_NAMES[int(label)])
            row.append(DISPLAY_MESSAGES[int(label)])
            writer.writerow(row)


def compute_label_distribution(dataset: Sequence[Tuple[List[float], int]]) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in dataset:
        counts[CLASS_NAMES[int(label)]] += 1
    return counts


def summarize_distribution_quality(
    train_set: Sequence[Tuple[List[float], int]],
    test_set: Sequence[Tuple[List[float], int]],
) -> Dict[str, object]:
    train_dist = compute_label_distribution(train_set)
    test_dist = compute_label_distribution(test_set)

    train_total = max(1, len(train_set))
    test_total = max(1, len(test_set))

    train_ratio = {key: round(value / float(train_total), 6) for key, value in train_dist.items()}
    test_ratio = {key: round(value / float(test_total), 6) for key, value in test_dist.items()}
    abs_gap = {key: round(abs(train_ratio[key] - test_ratio[key]), 6) for key in CLASS_NAMES}

    return {
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "abs_ratio_gap": abs_gap,
    }


def write_metadata_json(
    full_set: Sequence[Tuple[List[float], int]],
    train_set: Sequence[Tuple[List[float], int]],
    test_set: Sequence[Tuple[List[float], int]],
    build_counters: Dict[str, object],
    train_users: Sequence[int],
    test_users: Sequence[int],
) -> None:
    metadata = {
        "project": "OTT Recommendation Dataset",
        "random_seed": RANDOM_SEED,
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "display_messages": DISPLAY_MESSAGES,
        "movielens_dir": str(MOVIELENS_DIR),
        "history_window_len": HISTORY_WINDOW_LEN,
        "future_gap_len": FUTURE_GAP_LEN,
        "future_window_len": FUTURE_WINDOW_LEN,
        "stride": STRIDE,
        "min_user_ratings": MIN_USER_RATINGS,
        "train_ratio": TRAIN_RATIO,
        "max_samples_per_user": MAX_SAMPLES_PER_USER,
        "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
        "max_users_to_process": MAX_USERS_TO_PROCESS,
        "ratings_chunk_size": RATINGS_CHUNK_SIZE,
        "split_strategy": "user_level_balanced",
        "label_strategy": "strict_future_window_binary_recommendation",
        "full_samples": len(full_set),
        "train_samples": len(train_set),
        "test_samples": len(test_set),
        "train_users": len(train_users),
        "test_users": len(test_users),
        "full_distribution": compute_label_distribution(full_set),
        "train_distribution": compute_label_distribution(train_set),
        "test_distribution": compute_label_distribution(test_set),
        "distribution_quality": summarize_distribution_quality(train_set, test_set),
        "notes": [
            "Features are created only from history windows.",
            "Labels are created only from future-window behavior.",
            "Ambiguous future windows are discarded to improve label purity.",
            "User-level train/test split is used to reduce leakage.",
            "The 16-feature input format from feature_encoder.py is preserved.",
            "Dataset is intentionally stricter to improve downstream trainability.",
        ],
        "build_counters": build_counters,
    }

    with open(METADATA_JSON, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def write_movie_reco_pools_json(movie_pools: Dict[str, List[Dict[str, object]]]) -> None:
    with open(MOVIE_RECO_POOLS_JSON, "w", encoding="utf-8") as handle:
        json.dump(movie_pools, handle, indent=2)


# =============================================================================
# Console summary
# =============================================================================
def print_summary(
    merged_df: pd.DataFrame,
    full_set: Sequence[Tuple[List[float], int]],
    train_set: Sequence[Tuple[List[float], int]],
    test_set: Sequence[Tuple[List[float], int]],
    build_counters: Dict[str, object],
    train_users: Sequence[int],
    test_users: Sequence[int],
    movie_pools: Dict[str, List[Dict[str, object]]],
) -> None:
    distribution_quality = summarize_distribution_quality(train_set, test_set)

    print("\n==============================================================")
    print("                 DATASET CREATION SUMMARY                     ")
    print("==============================================================\n")

    print(f"MovieLens dir              : {MOVIELENS_DIR}")
    print(f"Users in ratings           : {merged_df['userId'].nunique()}")
    print(f"Movies in merge            : {merged_df['movieId'].nunique()}")
    print(f"Raw rating rows            : {len(merged_df)}")

    print("\n---------------------- Window Settings -----------------------")
    print(f"History window length      : {HISTORY_WINDOW_LEN}")
    print(f"Future gap length          : {FUTURE_GAP_LEN}")
    print(f"Future window length       : {FUTURE_WINDOW_LEN}")
    print(f"Stride                     : {STRIDE}")
    print(f"Min user ratings           : {MIN_USER_RATINGS}")
    print(f"Max samples per user       : {MAX_SAMPLES_PER_USER}")
    print(f"Max samples per class      : {MAX_SAMPLES_PER_CLASS}")
    print(f"Max users to process       : {MAX_USERS_TO_PROCESS}")

    print("\n----------------------- Label Mapping ------------------------")
    for idx, name in enumerate(CLASS_NAMES):
        print(f"{idx} -> {name:16s} | {DISPLAY_MESSAGES[idx]}")

    print("\n----------------------- Split Summary ------------------------")
    print(f"Users in train split       : {len(train_users)}")
    print(f"Users in test split        : {len(test_users)}")
    print(f"Full samples               : {len(full_set)}")
    print(f"Train samples              : {len(train_set)}")
    print(f"Test samples               : {len(test_set)}")

    print("\n----------------------- Build Counters -----------------------")
    for key, value in build_counters.items():
        print(f"{key:36s} : {value}")

    print("\n--------------------- Class Distribution ---------------------")
    print("Full:")
    for name, count in compute_label_distribution(full_set).items():
        print(f"  {name:16s} : {count}")

    print("Train:")
    for name, count in compute_label_distribution(train_set).items():
        print(f"  {name:16s} : {count}")

    print("Test:")
    for name, count in compute_label_distribution(test_set).items():
        print(f"  {name:16s} : {count}")

    print("\n--------------------- Train/Test Gap -------------------------")
    for name, gap in distribution_quality["abs_ratio_gap"].items():
        print(f"  {name:16s} : {gap:.6f}")

    print("\n------------------- Recommendation Pools ---------------------")
    for label_name in CLASS_NAMES:
        print(f"  {label_name:16s} : {len(movie_pools.get(label_name, []))} titles")

    print("\n---------------------- Generated Files -----------------------")
    print(f" - {FULL_DATASET_CSV}")
    print(f" - {TRAIN_DATASET_CSV}")
    print(f" - {TEST_DATASET_CSV}")
    print(f" - {METADATA_JSON}")
    print(f" - {MOVIE_RECO_POOLS_JSON}")

    print("\nSTATUS: PASS\n")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    random.seed(RANDOM_SEED)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if FORCE_REBUILD_OUTPUTS:
        delete_existing_generated_outputs()

    merged_df = load_movielens()

    samples_with_users, build_counters, user_primary_label = build_samples_from_movielens(merged_df)
    if not samples_with_users:
        raise RuntimeError(
            "No samples were generated.\n"
            "Try relaxing thresholds or increasing MAX_USERS_TO_PROCESS."
        )

    full_set = [(vector, label) for _, vector, label in samples_with_users]

    train_set, test_set, train_users, test_users = split_dataset_by_user_balanced(
        dataset=samples_with_users,
        user_primary_label=user_primary_label,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED,
    )

    if not train_set or not test_set:
        raise RuntimeError(
            "Train/test split produced an empty set. "
            "Relax thresholds or adjust TRAIN_RATIO."
        )

    selected_users = set(train_users) | set(test_users)
    pool_df = merged_df[merged_df["userId"].isin(selected_users)].copy()
    movie_pools = build_movie_recommendation_pools(pool_df)

    write_dataset_csv(FULL_DATASET_CSV, full_set)
    write_dataset_csv(TRAIN_DATASET_CSV, train_set)
    write_dataset_csv(TEST_DATASET_CSV, test_set)
    write_metadata_json(
        full_set=full_set,
        train_set=train_set,
        test_set=test_set,
        build_counters=build_counters,
        train_users=train_users,
        test_users=test_users,
    )
    write_movie_reco_pools_json(movie_pools)

    print_summary(
        merged_df=merged_df,
        full_set=full_set,
        train_set=train_set,
        test_set=test_set,
        build_counters=build_counters,
        train_users=train_users,
        test_users=test_users,
        movie_pools=movie_pools,
    )


if __name__ == "__main__":
    main()