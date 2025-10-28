import pytest
from src.ingestion.data_handlers.csv_extractor import read_csv_from_gcs
from src.preprocessing.transform import transform_df
import pandas as pd

def test_transform_df():
    df =  read_csv_from_gcs(bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json")
    transformed_df = transform_df(df)
    assert isinstance(transformed_df, pd.DataFrame)
    assert not transformed_df.empty
    assert len(df) > 0