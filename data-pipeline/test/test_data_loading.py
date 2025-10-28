import pytest
from src.ingestion.data_handlers.csv_extractor import read_csv_from_gcs
from src.preprocessing.transform import transform_df
from src.load.upload_cleaned_df_to_gcp import upload_df_to_gcs
import pandas as pd

def test_loading_transformed_data():
    df =  read_csv_from_gcs(bucket_name="homiehub",
        filename="homiehub_listings.csv",
        service_account_key_path="./GCP_Account_Key.json")
    transformed_df = transform_df(df)
    final_path = upload_df_to_gcs(
        df=transformed_df,
        filename="homiehub_listings_processed.csv",
        bucket_name="homiehub",
        service_account_key_path="./GCP_Account_Key.json",
    )
    assert final_path is not None