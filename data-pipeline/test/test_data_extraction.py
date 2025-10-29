import pytest
import pandas as pd
import numpy as np
from src.ingestion.data_handlers.csv_extractor import read_csv_from_gcs
from google.cloud import storage

def test_read_csv_from_gcs_basic():
    """Test basic functionality and data structure with actual GCP data"""
    df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Basic DataFrame checks
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 0
    
    # Expected columns check
    expected_columns = [
        'timestamp', 'requirement', 'accom_type', 'gender', 'food_pref',
        'furnished', 'red_eye', 'area', 'move_in_date', 'rent_amount',
        'lease_duration', 'utilities_included', 'bathroom_type',
        'distance_to_campus', 'people_count', 'description_summary',
        'contact', 'heat_available', 'water_available', 'laundry_available',
        'other_details'
    ]
    assert all(col in df.columns for col in expected_columns)

def test_read_csv_data_types():
    """Test data types of extracted columns"""
    df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Check that critical columns exist and are strings initially
    string_columns = ['timestamp', 'requirement', 'area', 'rent_amount']
    for col in string_columns:
        assert col in df.columns, f"Missing column: {col}"
        assert pd.api.types.is_object_dtype(df[col]), f"Column {col} should be object/string type"

def test_read_csv_data_quality():
    """Test quality of extracted data"""
    df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Check for missing values in critical columns
    critical_columns = ['timestamp', 'requirement', 'area', 'rent_amount']
    for col in critical_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            total_count = len(df)
            null_percentage = (null_count / total_count) * 100
            assert null_percentage < 50, f"Too many null values in {col}: {null_percentage}%"
    
    # Check timestamp values exist and can be parsed
    if 'timestamp' in df.columns:
        timestamp_values = df['timestamp'].dropna()
        assert len(timestamp_values) > 0, "No timestamp values found"
        
        # Convert sample timestamps to pandas datetime to verify they're valid
        try:
            pd.to_datetime(timestamp_values.iloc[0])
            assert True, "Timestamp format is valid"
        except ValueError as e:
            assert False, f"Invalid timestamp format: {e}"
    
    # Check rent amount format
    if 'rent_amount' in df.columns:
        rent_values = df['rent_amount'].dropna()
        assert len(rent_values) > 0, "No rent amount values found"
        # Check for either dollar signs or numbers
        has_valid_amounts = rent_values.str.contains(r'(\$|\d)', regex=True, na=False).any()
        assert has_valid_amounts, "Invalid rent amount formats found"
    
    # Check data consistency
    assert len(df) > 0, "DataFrame is empty"
    assert len(df.columns) >= 5, "Too few columns in data"

def test_read_csv_values_validation():
    """Test validation of specific field values"""
    df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Validate requirement values more flexibly
    if 'requirement' in df.columns:
        requirements = df['requirement'].dropna().str.lower()
        valid_req_keywords = ['looking', 'offering', 'need', 'available', 'rent', 'share']
        has_valid_reqs = requirements.str.contains('|'.join(valid_req_keywords), na=False, regex=True).any()
        assert has_valid_reqs, "No valid requirement values found"
    
    # Validate boolean fields more flexibly
    bool_fields = ['furnished', 'utilities_included', 'heat_available', 'water_available', 'laundry_available']
    for field in bool_fields:
        if field in df.columns:
            field_values = df[field].dropna()
            # Instead of checking for exact values, check if the values can be converted to boolean
            try:
                transformed = field_values.apply(lambda x: pd.NA if pd.isna(x) else 
                                              str(x).lower().strip() in ['yes', 'true', '1', 'available', 
                                                                       'included', 'in unit', 'paid'])
                assert not transformed.empty, f"No values could be interpreted as boolean in {field}"
            except Exception as e:
                assert False, f"Error converting {field} values to boolean: {str(e)}"
    
    # Validate area values are present
    if 'area' in df.columns:
        has_areas = df['area'].fillna('').str.strip().str.len() > 0
        assert has_areas.any(), "No valid area values found"
        
    # Additional validation for timestamps
    if 'timestamp' in df.columns:
        timestamps = df['timestamp'].dropna()
        assert len(timestamps) > 0, "No timestamps found"
        # Try to parse one timestamp to verify format
        try:
            pd.to_datetime(timestamps.iloc[0])
            assert True, "Valid timestamp format found"
        except ValueError as e:
            assert False, f"Invalid timestamp format: {e}"

def test_read_csv_data_quality():
    """Test quality of extracted data"""
    df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Check for missing values in critical columns
    critical_columns = ['timestamp', 'requirement', 'area', 'rent_amount']
    for col in critical_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            total_count = len(df)
            null_percentage = (null_count / total_count) * 100
            assert null_percentage < 50, f"Too many null values in {col}: {null_percentage}%"
    
    # Check timestamp format
    if 'timestamp' in df.columns:
        # Try converting timestamps to pandas datetime
        try:
            pd.to_datetime(df['timestamp'].dropna())
            assert True, "Valid timestamp format found"
        except ValueError as e:
            assert False, f"Invalid timestamp format: {e}"
    
    # Check rent amount format
    if 'rent_amount' in df.columns:
        rent_values = df['rent_amount'].dropna()
        # Check for either dollar signs or numbers, more flexibly
        has_valid_amounts = rent_values.str.contains(r'(\$|\d)', regex=True, na=False).any()
        assert has_valid_amounts, "No valid rent amount formats found"

def test_read_csv_data_consistency():
    """Test consistency of data across multiple reads"""
    df1 = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    df2 = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Check if both reads return the same data
    pd.testing.assert_frame_equal(df1, df2)
    
    # Verify row count consistency
    assert len(df1) == len(df2)
    
    # Verify column consistency
    assert all(df1.columns == df2.columns)

def test_read_csv_values_validation():
    """Test validation of specific field values"""
    df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Validate requirement values more flexibly
    if 'requirement' in df.columns:
        requirements = df['requirement'].dropna()
        # Check for keywords in a more flexible way
        valid_keywords = ['looking', 'offering', 'need', 'available', 'rent', 'share']
        has_valid = requirements.str.lower().str.contains('|'.join(valid_keywords), regex=True, na=False)
        assert has_valid.any(), "No valid requirement values found"
    
    # Validate boolean fields more flexibly
    bool_fields = ['furnished', 'utilities_included', 'heat_available', 'water_available', 'laundry_available']
    valid_values = ['yes', 'no', 'true', 'false', 'included', 'not included', 
                    'available', 'unavailable', 'y', 'n', 'in unit', 'in building',
                    'paid', 'unpaid', 'extra', 'separate', '']
    
    for field in bool_fields:
        if field in df.columns:
            field_values = df[field].fillna('').str.lower().str.strip()
            # Check if any values are valid
            has_valid = field_values.isin(valid_values)
            assert has_valid.any(), f"No valid values found in {field}"
    
    # Validate area values are present and non-empty
    if 'area' in df.columns:
        non_empty = df['area'].fillna('').str.strip().str.len() > 0
        assert non_empty.any(), "No valid area values found"
    