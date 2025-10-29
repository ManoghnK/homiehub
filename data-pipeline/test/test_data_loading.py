import pytest
import pandas as pd
from unittest.mock import patch, Mock
from datetime import datetime
from src.load.upload_cleaned_df_to_gcp import upload_df_to_gcp
from src.ingestion.data_handlers.csv_extractor import read_csv_from_gcs
from src.preprocessing.transform import transform_df
from google.cloud import storage
import os

@pytest.fixture
def gcp_data():
    # Get real data from GCP
    df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json"
    )
    return transform_df(df)  # Return transformed data ready for upload

@pytest.fixture
def mock_storage_client():
    with patch('google.cloud.storage.Client') as mock_client:
        yield mock_client

def test_upload_df_to_gcp_basic(gcp_data):
    # Test basic functionality with actual data
    test_filename = "test_upload_real_data.csv"
    
    # Upload to the same folder structure as the source data
    final_path = upload_df_to_gcp(
        df=gcp_data,
        filename=test_filename,
        bucket_name="homiehub",
        service_account_key_path="./GCP_Account_Key.json",
        folder="raw"  # Use the same folder as source data
    )
    
    # Basic path assertions
    assert final_path is not None
    assert "raw" in final_path
    assert test_filename in final_path
    
    # Verify data was uploaded
    uploaded_df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename=test_filename,  # Let the reader add the path
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Compare the DataFrames
    assert len(uploaded_df) == len(gcp_data)
    assert set(uploaded_df.columns) == set(gcp_data.columns)
    
    # Compare a few key values
    if 'timestamp' in gcp_data.columns:
        assert all(uploaded_df['timestamp'] == gcp_data['timestamp'])
    if 'rent_amount' in gcp_data.columns:
        assert all(uploaded_df['rent_amount'] == gcp_data['rent_amount'])

def test_upload_df_to_gcp_with_transformed_data(gcp_data):
    # Test uploading transformed data
    test_filename = "processed.csv"
    final_path = upload_df_to_gcp(
        df=gcp_data,
        filename=test_filename,
        bucket_name="homiehub",
        service_account_key_path="./GCP_Account_Key.json",
        folder="raw"  # Match the folder structure expected by read_csv_from_gcs
    )
    
    # Verify the data was uploaded successfully
    assert final_path is not None
    
    # Verify structure before upload
    numeric_cols_before = gcp_data.select_dtypes(include=['int64', 'float64']).columns
    bool_cols_before = gcp_data.select_dtypes(include=['bool']).columns
    
    # Get basic statistics before upload
    numeric_stats_before = {col: gcp_data[col].mean() for col in numeric_cols_before}
    bool_stats_before = {col: gcp_data[col].sum() for col in bool_cols_before}
    
    # Read back the uploaded file using the correct path
    uploaded_df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename=test_filename,  # The read_csv_from_gcs will add the date folder
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Basic validation
    assert len(uploaded_df) == len(gcp_data), "Row count mismatch"
    assert set(gcp_data.columns) == set(uploaded_df.columns), "Column mismatch"
    
    # Compare numeric columns statistics
    for col in numeric_cols_before:
        assert abs(uploaded_df[col].mean() - numeric_stats_before[col]) < 0.01, f"Statistics mismatch for {col}"
        
    # Compare boolean columns
    for col in bool_cols_before:
        bool_val_after = uploaded_df[col].astype('bool').sum()
        assert abs(bool_val_after - bool_stats_before[col]) < 0.01, f"Boolean counts mismatch for {col}"

def test_upload_df_to_gcp_folder_structure(gcp_data):
    # Test folder structure with real data
    today = datetime.now().strftime('%Y-%m-%d')
    custom_folder = "test_folder"
    test_filename = "folder_structure_test.csv"
    
    final_path = upload_df_to_gcp(
        df=gcp_data,
        filename=test_filename,
        bucket_name="homiehub",
        service_account_key_path="./GCP_Account_Key.json",
        folder=custom_folder
    )
    
    assert final_path == f"{custom_folder}/{today}/{test_filename}"
    
    # Verify the file exists in GCP
    uploaded_df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename=final_path,
        service_account_key_path="./GCP_Account_Key.json"
    )
    assert not uploaded_df.empty

def test_upload_df_to_gcp_data_validation(gcp_data):
    # Test data validation during upload
    test_filename = "validation_test.csv"
    
    # Test with actual data
    final_path = upload_df_to_gcp(
        df=gcp_data,
        filename=test_filename,
        bucket_name="homiehub",
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Read back the uploaded file
    uploaded_df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename=final_path,
        service_account_key_path="./GCP_Account_Key.json"
    )
    
    # Validate numeric columns
    numeric_cols = ['rent_amount_num', 'lease_duration_months', 'distance_to_campus_miles']
    for col in numeric_cols:
        if col in gcp_data.columns:
            assert uploaded_df[col].dtype in ['int64', 'float64', 'Int64']
    
    # Validate boolean columns
    bool_cols = ['furnished_bool', 'utilities_included_bool', 'heat_available_bool']
    for col in bool_cols:
        if col in gcp_data.columns:
            assert str(uploaded_df[col].dtype) == 'boolean'
    
    # Verify content type was set to 'text/csv'
    storage_client = storage.Client()
    bucket = storage_client.bucket("homiehub")
    blob = bucket.get_blob(final_path)
    assert blob.content_type == 'text/csv'

def test_upload_df_to_gcp_custom_folder(gcp_data):
    # Test with custom folder
    custom_folder = "test_custom_folder"
    filename = "test.csv"
    today = datetime.now().strftime('%Y-%m-%d')
    expected_path = f"{custom_folder}/{today}/{filename}"
    
    result = upload_df_to_gcp(
        df=gcp_data,
        filename=filename,
        bucket_name="homiehub",
        service_account_key_path="./GCP_Account_Key.json",
        folder=custom_folder
    )
    
    assert result == expected_path
    
    # Verify file exists in the custom folder
    uploaded_df = read_csv_from_gcs(
        bucket_name="homiehub",
        filename=result,
        service_account_key_path="./GCP_Account_Key.json"
    )
    assert not uploaded_df.empty

@pytest.mark.parametrize("df,filename,bucket_name,key_path", [
    (None, "test.csv", "homiehub", "./GCP_Account_Key.json"),
    (pd.DataFrame(), None, "homiehub", "./GCP_Account_Key.json"),
    (pd.DataFrame(), "test.csv", None, "./GCP_Account_Key.json"),
    (pd.DataFrame(), "test.csv", "homiehub", None)
])
def test_upload_df_to_gcp_invalid_inputs(df, filename, bucket_name, key_path):
    # Test handling of invalid inputs
    with pytest.raises(Exception):
        upload_df_to_gcp(
            df=df,
            filename=filename,
            bucket_name=bucket_name,
            service_account_key_path=key_path
        )