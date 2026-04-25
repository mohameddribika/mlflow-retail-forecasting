"""
Data loader for the Rossmann Store Sales forecasting dataset (Kaggle).

Source: https://www.kaggle.com/competitions/rossmann-store-sales

Dataset characteristics:
    - 1,115 stores in Germany
    - Daily sales records 2013-01-01 to 2015-07-31
    - ~1.0M rows in train.csv
    - Target: `Sales` (daily sales in EUR per store)
    - Key features: Store, DayOfWeek, Date, Customers, Open, Promo,
      StateHoliday, SchoolHoliday, plus per-store metadata in store.csv

Prerequisites (one-time setup on the host machine):
    1. Create a Kaggle API token at https://www.kaggle.com/settings/account
       (downloads kaggle.json).
    2. Place it at ~/.kaggle/kaggle.json with `chmod 600`.
    3. Accept the Rossmann competition rules at
       https://www.kaggle.com/competitions/rossmann-store-sales/rules

Run:
    python src/data_loader.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import RAW_DIR, PROCESSED_DIR  # noqa: E402

KAGGLE_COMPETITION = "rossmann-store-sales"
EXPECTED_FILES = ("train.csv", "store.csv")


# ---------------------------------------------------------------------------
# Download — multi-strategy
# ---------------------------------------------------------------------------
def _files_present() -> bool:
    return all((RAW_DIR / f).exists() for f in EXPECTED_FILES)


def _try_kagglehub() -> bool:
    """Use the modern kagglehub library (supports new KGAT_xxx tokens via
    KAGGLE_API_TOKEN env var). Returns True on success."""
    if not os.getenv("KAGGLE_API_TOKEN"):
        return False
    try:
        import kagglehub
    except ImportError:
        return False

    print("[data_loader] Trying kagglehub (new KAGGLE_API_TOKEN style) ...")
    try:
        # Downloads to kagglehub's cache dir, returns the path.
        cache_path = kagglehub.competition_download(KAGGLE_COMPETITION)
    except Exception as e:
        print(f"[data_loader] kagglehub failed: {e}")
        return False

    # Copy the relevant files into our project's data/raw/.
    cache_dir = Path(cache_path)
    for fname in EXPECTED_FILES:
        src = cache_dir / fname
        if src.exists():
            shutil.copy(src, RAW_DIR / fname)
    return _files_present()


def _try_kaggle_cli() -> bool:
    """Use the legacy kaggle CLI (needs ~/.kaggle/kaggle.json)."""
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists():
        return False

    print("[data_loader] Trying kaggle CLI (legacy kaggle.json style) ...")
    cmd = [
        "kaggle", "competitions", "download",
        "-c", KAGGLE_COMPETITION,
        "-p", str(RAW_DIR),
        "--force",
    ]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[data_loader] kaggle CLI failed: {e}")
        return False

    for zpath in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(RAW_DIR)
        zpath.unlink()
    return _files_present()


def download_rossmann() -> None:
    """Try every available auth strategy. If all fail, print a clear
    instruction block telling the user to download manually."""
    if _files_present():
        print(f"[data_loader] Rossmann files already cached in {RAW_DIR}")
        return

    if _try_kagglehub():
        print(f"[data_loader] Downloaded via kagglehub -> {RAW_DIR}")
        return
    if _try_kaggle_cli():
        print(f"[data_loader] Downloaded via kaggle CLI -> {RAW_DIR}")
        return

    raise RuntimeError(
        "\n\nCould not download Rossmann automatically. Manual fallback:\n"
        "  1. Visit https://www.kaggle.com/competitions/"
        "rossmann-store-sales/data\n"
        "  2. Click 'I Understand and Accept' at the top of the page.\n"
        "  3. Click 'Download All' at the bottom.\n"
        f"  4. Move + unzip into {RAW_DIR}:\n"
        f"       mv ~/Downloads/rossmann-store-sales.zip {RAW_DIR}/\n"
        f"       cd {RAW_DIR} && unzip rossmann-store-sales.zip "
        "&& rm rossmann-store-sales.zip\n"
        "  5. Re-run: python src/data_loader.py\n"
    )


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    """Load train.csv joined with store.csv. Returns the full raw frame."""
    train = pd.read_csv(
        RAW_DIR / "train.csv",
        parse_dates=["Date"],
        low_memory=False,
    )
    stores = pd.read_csv(RAW_DIR / "store.csv", low_memory=False)
    df = train.merge(stores, on="Store", how="left")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning steps for Rossmann.

    Decisions made and why:
      - Drop rows where Open == 0 (Sales is always 0 by definition; keeping
        them would teach the model the trivial "closed -> 0" rule and
        inflate metrics dishonestly).
      - Drop rows where Sales <= 0 (a handful of erroneous records).
      - Convert StateHoliday to a clean string ('0', 'a', 'b', 'c'); the raw
        column has mixed int/string types.
      - Fill missing CompetitionDistance with the median (most common
        Rossmann preprocessing recipe).
      - Fill missing Promo2 fields with 0 / sensible defaults.
    """
    df = df.copy()

    # Drop closed-store and zero-sales rows (industry-standard for Rossmann).
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Mixed-type column -> string
    df["StateHoliday"] = df["StateHoliday"].astype(str).replace("0.0", "0")

    # Fill missing competition info
    median_comp = df["CompetitionDistance"].median()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_comp)
    df["CompetitionOpenSinceMonth"] = (
        df["CompetitionOpenSinceMonth"].fillna(0).astype(int)
    )
    df["CompetitionOpenSinceYear"] = (
        df["CompetitionOpenSinceYear"].fillna(0).astype(int)
    )

    # Fill Promo2 fields
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    # Date features (handy for everyone downstream — train, monitor, etc.)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)

    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "sales.parquet"
    df.to_parquet(out, index=False)
    return out


def load_processed() -> pd.DataFrame:
    """One-call helper: download if needed, clean, cache, and return."""
    cached = PROCESSED_DIR / "sales.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    download_rossmann()
    df = clean(load_raw())
    save_processed(df)
    return df


# ---------------------------------------------------------------------------
# EDA summary
# ---------------------------------------------------------------------------
def quick_eda(df: pd.DataFrame) -> None:
    """Print a compact EDA summary suitable for the project report."""
    print("\n" + "=" * 70)
    print("ROSSMANN STORE SALES — DATASET SUMMARY")
    print("=" * 70)
    print(f"Shape:              {df.shape[0]:,} rows  x  {df.shape[1]} cols")
    print(f"Date range:         {df['Date'].min().date()}  ->  "
          f"{df['Date'].max().date()}")
    print(f"Stores:             {df['Store'].nunique():,}")
    print(f"Missing values:     {df.isna().sum().sum()}")

    print("\n" + "-" * 70)
    print("TARGET (Sales) STATISTICS")
    print("-" * 70)
    print(df["Sales"].describe().round(2).to_string())

    print("\n" + "-" * 70)
    print("MEAN SALES BY DAY OF WEEK (1=Mon ... 7=Sun)")
    print("-" * 70)
    print(df.groupby("DayOfWeek")["Sales"].mean().round(1).to_string())

    print("\n" + "-" * 70)
    print("PROMOTION LIFT (mean Sales)")
    print("-" * 70)
    print(df.groupby("Promo")["Sales"].mean().round(1).to_string())

    print("\n" + "-" * 70)
    print("STORE TYPE DISTRIBUTION")
    print("-" * 70)
    print(df["StoreType"].value_counts().to_string())


def main() -> None:
    download_rossmann()
    print("[data_loader] Loading + cleaning ...")
    df = clean(load_raw())
    out = save_processed(df)
    print(f"[data_loader] Processed dataset cached at: {out}")
    quick_eda(df)


if __name__ == "__main__":
    main()
