import pandas as pd
import numpy as np

# -------------------------------
# Drift Classification Thresholds
# Based on standard PSI interpretation:
#   < 0.1  → negligible drift
#   < 0.2  → minor drift
#   >= 0.2 → significant drift
# We scale these up since our score combines mean_diff + psi
# -------------------------------
DRIFT_LOW_THRESHOLD = 0.2
DRIFT_HIGH_THRESHOLD = 1.0


# -------------------------------
# Split Data
# -------------------------------
def split_data(data: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data.sample(frac=train_frac, random_state=42)
    new = data.drop(train.index).reset_index(drop=True)
    train = train.reset_index(drop=True)
    return train, new


# -------------------------------
# Convert Datatypes Safely
# -------------------------------
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    for col in data.columns:
        if "time" in col.lower() or "date" in col.lower():
            # BUG FIX: NaT → int64 produced large negative sentinel values
            parsed = pd.to_datetime(data[col], errors="coerce")
            data[col] = parsed.fillna(pd.Timestamp("1970-01-01")).astype("int64") // 10**9

        elif data[col].dtype == "object":
            data[col] = data[col].astype("category")

    return data


# -------------------------------
# Validate Input
# -------------------------------
def validate_data(data: pd.DataFrame) -> tuple[bool, str]:
    if data.empty:
        return False, "Uploaded file is empty."
    if len(data.columns) < 1:
        return False, "Dataset must have at least one column."
    if len(data) < 20:
        return False, f"Dataset too small ({len(data)} rows). Need at least 20 rows for meaningful drift detection."
    return True, ""


# -------------------------------
# Simulate Drift (for demo purposes)
# Applies deterministic, reproducible drift to numeric and categorical columns
# -------------------------------
def simulate_drift(new: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    new = new.copy()

    for col in new.columns:
        if pd.api.types.is_numeric_dtype(new[col]):
            # Shift mean and add noise to simulate real-world distribution shift
            scale_factor = rng.uniform(1.05, 1.25)
            noise = rng.normal(0, new[col].std() * 0.1, size=len(new))
            new[col] = new[col] * scale_factor + noise

        elif pd.api.types.is_categorical_dtype(new[col]):
            # Simulate categorical drift by biasing toward a subset of categories
            categories = new[col].cat.categories.tolist()
            if len(categories) > 1:
                # Over-sample a random subset to create imbalance
                biased = rng.choice(categories[:max(1, len(categories) // 2)], size=len(new))
                new[col] = pd.Categorical(biased, categories=categories)

    return new


# -------------------------------
# PSI Calculation (Numeric)
# BUG FIX: duplicate bin edges caused histogram mismatch
# -------------------------------
def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # BUG FIX: use np.unique to eliminate duplicate breakpoints from low-cardinality cols
    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))

    if len(breakpoints) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    # Avoid division by zero with smoothing
    expected_perc = (expected_counts + 1e-6) / (len(expected) + 1e-6 * len(expected_counts))
    actual_perc = (actual_counts + 1e-6) / (len(actual) + 1e-6 * len(actual_counts))

    psi = float(np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc)))
    return round(psi, 6)


# -------------------------------
# Categorical Drift (Total Variation Distance)
# -------------------------------
def categorical_drift(train: pd.Series, new: pd.Series) -> float:
    train_dist = train.value_counts(normalize=True)
    new_dist = new.value_counts(normalize=True)

    all_categories = set(train_dist.index).union(set(new_dist.index))

    tvd = sum(
        abs(train_dist.get(cat, 0.0) - new_dist.get(cat, 0.0))
        for cat in all_categories
    )
    # TVD is already in [0, 1]; normalise to 0.5 max for fair comparison with PSI
    return round(tvd / 2, 6)


# -------------------------------
# Per-Feature Drift
# -------------------------------
def calculate_feature_drift(train: pd.DataFrame, new: pd.DataFrame) -> dict:
    drift_results = {}

    for col in train.columns:
        if col not in new.columns:
            continue

        if pd.api.types.is_numeric_dtype(train[col]):
            mean_diff = abs(train[col].mean() - new[col].mean())
            std_diff = abs(train[col].std() - new[col].std())
            psi = calculate_psi(train[col], new[col])
            drift_score = psi + (mean_diff / (train[col].std() + 1e-6)) * 0.3
            drift_type = "numeric"

        else:
            cat_drift = categorical_drift(train[col], new[col])
            mean_diff = cat_drift
            std_diff = 0.0
            psi = 0.0
            drift_score = cat_drift
            drift_type = "categorical"

        drift_results[col] = {
            "type": drift_type,
            "mean_diff": round(float(mean_diff), 4),
            "std_diff": round(float(std_diff), 4),
            "psi": round(float(psi), 4),
            "drift_score": round(float(drift_score), 4),
            "severity": classify_drift(drift_score),
        }

    return drift_results


# -------------------------------
# Overall Drift Score
# -------------------------------
def calculate_overall_drift(drift_results: dict) -> float:
    if not drift_results:
        return 0.0
    scores = [v["drift_score"] for v in drift_results.values()]
    return round(float(np.mean(scores)), 4)


# -------------------------------
# Classification
# -------------------------------
def classify_drift(score: float) -> str:
    if score < DRIFT_LOW_THRESHOLD:
        return "Low Drift"
    elif score < DRIFT_HIGH_THRESHOLD:
        return "Medium Drift"
    else:
        return "High Drift"


# -------------------------------
# Top Feature by Drift Score
# -------------------------------
def get_top_feature(drift_results: dict) -> str | None:
    if not drift_results:
        return None
    return max(drift_results, key=lambda col: drift_results[col]["drift_score"])


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def detect_drift(data: pd.DataFrame) -> dict:
    is_valid, error_msg = validate_data(data)
    if not is_valid:
        raise ValueError(error_msg)

    data = preprocess_data(data)
    train, new = split_data(data)
    new = simulate_drift(new)

    drift_results = calculate_feature_drift(train, new)
    overall_score = calculate_overall_drift(drift_results)
    status = classify_drift(overall_score)
    top_feature = get_top_feature(drift_results)

    return {
        "overall_score": overall_score,
        "status": status,
        "feature_drift": drift_results,
        "top_feature": top_feature,
        "train": train,
        "new": new,
    }