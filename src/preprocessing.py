"""
Feature engineering pipeline for the Rossmann sales forecasting model.

Defines `build_preprocessor()` which returns a scikit-learn ColumnTransformer
that:
    - One-hot encodes low-cardinality categoricals (StoreType, Assortment,
      StateHoliday).
    - Passes through numeric features.
    - Drops everything else (Date, Customers, Open, Sales, etc.).

Why a ColumnTransformer instead of inline pandas operations?
    1. The fitted encoders travel WITH the model when we save it to MLflow,
       so deployment doesn't need a separate preprocessing step.
    2. Train/test consistency is guaranteed: encoders are fit on train only.
    3. The whole pipeline becomes a single picklable object, which is
       reproducibility-friendly (a hash uniquely identifies it).

Also defines `FEATURE_COLUMNS` and `TARGET_COLUMN` as the single source of
truth for what goes into the model — both train.py and monitor.py import
these so we never have a drift between training-time and inference-time
feature lists.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ---------------------------------------------------------------------------
# Column inventory — single source of truth
# ---------------------------------------------------------------------------
TARGET_COLUMN = "Sales"

# Numeric features pass through as-is (will be scaled inside specific models
# that need it, e.g. LinearRegression — handled in train.py).
NUMERIC_FEATURES = [
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend",
]

# Low-cardinality categoricals — one-hot encoded.
CATEGORICAL_FEATURES = [
    "StoreType",     # 4 values (a, b, c, d)
    "Assortment",    # 3 values (a, b, c)
    "StateHoliday",  # 4 values (0, a, b, c)
]

# Columns deliberately EXCLUDED and why:
#   Date           - already decomposed into Year/Month/Day/WeekOfYear/IsWeekend
#   Sales          - the target
#   Customers      - LEAKAGE: not available at prediction time
#   Open           - constant 1 after cleaning
#   Store          - 1115 unique values; too high cardinality for one-hot
#                    with linear models. Tree models handle it natively, but
#                    we keep the feature list uniform across models for the
#                    baseline. (We could re-introduce via target encoding
#                    in a later iteration.)
#   PromoInterval  - free-text-ish field, low marginal value vs complexity

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Construct the ColumnTransformer used by every model in this project.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer. Fit it on training data via Pipeline.fit().
    """
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",  # explicitly drop anything not listed above
        verbose_feature_names_out=False,
    )
