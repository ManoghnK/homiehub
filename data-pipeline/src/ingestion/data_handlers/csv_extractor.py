from google.cloud import storage
import pandas as pd
import io
import os
from datetime import datetime

def read_csv_from_gcs(bucket_name, filename, service_account_key_path):
    """
    Read CSV from GCS using today's date as folder
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path
    
    # Get today's date dynamically
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Construct blob path with today's date
    blob_name = f"raw/{today}/{filename}"
    
    print(f"Reading from: gs://{bucket_name}/{blob_name}")
    
    # Initialize client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download as string
    csv_data = blob.download_as_text()
    
    # Convert to DataFrame
    df = pd.read_csv(io.StringIO(csv_data))
    return df