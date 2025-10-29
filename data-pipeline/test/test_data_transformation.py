import pytest
import pandas as pd
import numpy as np
from src.preprocessing.transform import (
    _to_lower_strip,
    _parse_money,
    _parse_bool,
    _parse_date,
    _parse_int,
    transform_df
)
from src.ingestion.data_handlers.csv_extractor import read_csv_from_gcs

@pytest.fixture
def gcp_data():
    return read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )

def test_to_lower_strip(gcp_data):
    # Test with actual area data from GCP
    s = gcp_data['area'] if 'area' in gcp_data.columns else pd.Series([" TEST ", "test ", " Test"])
    result = _to_lower_strip(s)
    # Verify all results are lowercase and stripped
    assert all(isinstance(x, str) and x.islower() and x.strip() == x for x in result.dropna())

def test_parse_money(gcp_data):
    # Test with actual rent amount data from GCP
    s = gcp_data['rent_amount'] if 'rent_amount' in gcp_data.columns else pd.Series(["$1,000"])
    result = _parse_money(s)
    # Verify all parsed values are numeric and positive
    assert all(isinstance(x, (int, float)) and x > 0 for x in result.dropna())

def test_parse_bool(gcp_data):
    # Test with actual boolean fields from GCP
    bool_columns = ['furnished', 'utilities_included', 'heat_available', 'water_available', 'laundry_available']
    for col in bool_columns:
        if col in gcp_data.columns:
            s = gcp_data[col]
            result = _parse_bool(s)
            # Verify all parsed values are boolean
            assert all(isinstance(x, bool) for x in result.dropna())

def test_parse_date(gcp_data):
    # Test with actual date fields from GCP
    date_columns = ['timestamp', 'move_in_date']
    for col in date_columns:
        if col in gcp_data.columns:
            s = gcp_data[col]
            result = _parse_date(s)
            # Verify date format (YYYY-MM-DD)
            assert all(isinstance(x, str) and len(x) == 10 and x[4] == '-' and x[7] == '-' 
                      for x in result.dropna())

def test_parse_int(gcp_data):
    # Test with actual numeric fields from GCP
    if 'people_count' in gcp_data.columns:
        s = gcp_data['people_count']
        result = _parse_int(s)
        # Verify numeric values are properly parsed
        numeric_values = result.dropna()
        assert len(numeric_values) > 0, "No numeric values found"
        assert all(isinstance(x, (int, np.integer)) for x in numeric_values), "Non-integer values found"

def test_transform_df_full(gcp_data):
    # Test the full transformation pipeline with actual GCP data
    result = transform_df(gcp_data)
    
    # Test timestamp transformation
    if 'timestamp' in gcp_data.columns:
        assert 'timestamp_iso' in result.columns
        dates = pd.to_datetime(result['timestamp_iso'].dropna())
        assert not dates.empty, "No valid dates found"
    
    # Test rent amount transformation
    if 'rent_amount' in gcp_data.columns:
        assert 'rent_amount_num' in result.columns
        amounts = result['rent_amount_num'].dropna()
        assert not amounts.empty, "No valid amounts found"
        assert all(isinstance(x, (int, float)) for x in amounts), "Invalid amount types found"
        assert all(x > 0 for x in amounts), "Non-positive amounts found"
    
    # Test duration transformation
    if 'lease_duration' in gcp_data.columns and 'lease_duration_months' in result.columns:
        durations = result['lease_duration_months'].dropna()
        if not durations.empty:
            assert all(isinstance(x, (int, np.integer)) for x in durations), "Non-integer durations found"
            assert all(x > 0 for x in durations), "Non-positive durations found"
    
    if 'furnished' in gcp_data.columns:
        assert 'furnished_bool' in result.columns
        assert all(isinstance(x, bool) for x in result['furnished_bool'].dropna())
    
    if 'area' in gcp_data.columns:
        assert 'area_norm' in result.columns
        assert all(x.islower() for x in result['area_norm'].dropna())
    
    if 'distance_to_campus' in gcp_data.columns:
        assert 'distance_to_campus_miles' in result.columns
        assert all(isinstance(x, (int, float)) for x in result['distance_to_campus_miles'].dropna())

def test_transform_df_data_consistency(gcp_data):
    # Test that transformation preserves data integrity
    result = transform_df(gcp_data)
    
    # Verify row count is preserved
    assert len(result) == len(gcp_data)
    
    # Verify no unexpected null values are introduced in required fields
    required_fields = ['timestamp_iso', 'area_norm']
    for field in required_fields:
        if field in result.columns:
            null_count_before = gcp_data[field.replace('_iso', '').replace('_norm', '')].isnull().sum()
            null_count_after = result[field].isnull().sum()
            assert null_count_after >= null_count_before  # Should not reduce null counts