from google.cloud import storage
from datetime import datetime
import os

def upload_csv_with_date_folder(local_file_path, bucket_name, service_account_key_path):
    """
    Upload a CSV file to GCS bucket organized by today's date
    
    Args:
        local_file_path: Path to local CSV file
        bucket_name: Name of GCS bucket (e.g., 'homiehub')
        service_account_key_path: Path to your service account JSON key
    """
    # Set credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key_path
    
    # Get today's date in YYYY-MM-DD format
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get the filename from the path
    filename = os.path.basename(local_file_path)
    
    # Create destination path: YYYY-MM-DD/filename.csv
    destination_blob_name = f"raw/{today}/{filename}"
    
    # Initialize storage client
    storage_client = storage.Client()
    
    # Get bucket
    bucket = storage_client.bucket(bucket_name)
    
    # Create blob and upload
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    
    return destination_blob_name

# Usage
if __name__ == "__main__":
    upload_csv_with_date_folder(
        local_file_path="./data/raw/homiehub_listings.csv",
        bucket_name="homiehubbucket",
        service_account_key_path="./GCP_Account_Key.json"
    )