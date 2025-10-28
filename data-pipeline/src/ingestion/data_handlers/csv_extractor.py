from pathlib import Path
import pandas as pd


def read_raw_listings(csv_path: Path) -> pd.DataFrame:
    """
    Read the raw MLOPs project CSV.
    - Treat empty strings as empty, not NaN for categorical text fields.
    - Keep original column names; downstream transform will normalize.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(
        csv_path,
        dtype=str,
        keep_default_na=False,
        na_values=["NA", "N/A", "null", "None"],
        encoding="utf-8",
    )
    return df
