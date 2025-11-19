# test/conftest.py
import pytest

@pytest.fixture
def bucket_config():
    """Bucket configuration - just filename since csv_extractor adds the path"""
    return {
        "bucket_name": "homiehubbucket",
        "service_account_key_path": "./GCP_Account_Key.json",
        "raw_file_path": "homiehub_listings.csv",  # JUST the filename
        "today": "2025-11-18"
    }