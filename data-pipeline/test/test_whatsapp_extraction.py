import pytest
from src.ingestion.data_handlers.csv_extractor import read_csv_from_gcs
import pandas as pd

def test_read_csv_from_gcs():
    df =  read_csv_from_gcs(bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) > 0
    