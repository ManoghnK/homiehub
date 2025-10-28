import pandas as pd
import numpy as np
from dateutil import parser


def _to_lower_strip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"": np.nan})


def _parse_money(s: pd.Series) -> pd.Series:
    cleaned = s.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_bool(s: pd.Series) -> pd.Series:
    trues = {"yes", "y", "true", "1", "included", "inclusion", "needed", "required"}
    falses = {"no", "n", "false", "0", "not included", "none"}
    val = _to_lower_strip(s)
    return val.map(lambda x: True if x in trues else (False if x in falses else np.nan))


def _parse_date(s: pd.Series) -> pd.Series:
    def parse_one(x):
        try:
            return parser.parse(str(x), fuzzy=True).date().isoformat()
        except Exception:
            return np.nan
    return s.apply(parse_one)


def _parse_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def transform_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "timestamp": "timestamp",
        "requirement": "requirement",
        "accom_type": "accom_type",
        "gender": "gender",
        "food_pref": "food_pref",
        "furnished": "furnished",
        "red_eye": "red_eye",
        "area": "area",
        "move_in_date": "move_in_date",
        "rent_amount": "rent_amount",
        "lease_duration": "lease_duration",
        "utilities_included": "utilities_included",
        "bathroom_type": "bathroom_type",
        "distance_to_campus": "distance_to_campus",
        "people_count": "people_count",
        "description_summary": "description_summary",
        "contact": "contact",
        "heat_available": "heat_available",
        "water_available": "water_available",
        "laundry_available": "laundry_available",
        "other_details": "other_details",
    }
    df = df.rename(columns=rename_map)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    if "timestamp" in df:
        df["timestamp_iso"] = _parse_date(df["timestamp"]) 
    if "rent_amount" in df:
        df["rent_amount_num"] = _parse_money(df["rent_amount"]) 
    if "lease_duration" in df:
        dur = df["lease_duration"].astype(str).str.extract(r"(\d+)", expand=False)
        df["lease_duration_months"] = _parse_int(dur)
    for bcol in ["utilities_included", "furnished", "red_eye", "heat_available", "water_available", "laundry_available"]:
        if bcol in df:
            df[bcol + "_bool"] = _parse_bool(df[bcol])
    cat_cols = ["requirement", "accom_type", "gender", "food_pref", "bathroom_type", "area"]
    for c in cat_cols:
        if c in df:
            df[c + "_norm"] = _to_lower_strip(df[c])
    text_cols = ["description_summary", "other_details"]
    for c in text_cols:
        if c in df:
            df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if "people_count" in df:
        pc = df["people_count"].astype(str).str.extract(r"(\d+)", expand=False)
        df["people_count_num"] = _parse_int(pc)
    if "distance_to_campus" in df:
        miles = df["distance_to_campus"].astype(str).str.replace("miles", "", regex=False).str.replace("mile", "", regex=False).str.strip()
        df["distance_to_campus_miles"] = pd.to_numeric(miles, errors="coerce")
    if "move_in_date" in df:
        df["move_in_date_iso"] = _parse_date(df["move_in_date"]) 
    base_cols = [
        "timestamp", "timestamp_iso", "requirement_norm", "accom_type_norm", "gender_norm", "food_pref_norm",
        "furnished_bool", "red_eye_bool", "area_norm", "move_in_date", "move_in_date_iso", "rent_amount", "rent_amount_num",
        "lease_duration", "lease_duration_months", "utilities_included_bool", "bathroom_type_norm", "distance_to_campus_miles",
        "people_count_num", "description_summary", "contact", "heat_available_bool", "water_available_bool", "laundry_available_bool",
        "other_details"
    ]
    present_cols = [c for c in base_cols if c in df.columns]
    return df[present_cols]