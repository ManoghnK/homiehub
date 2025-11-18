# HomieHub Data Pipeline DAG with GCS Integration
# =====================================================

import sys
import pandas as pd
import io
import os
from datetime import datetime
from pathlib import Path
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.utils.email import send_email
from google.cloud import storage
import tempfile
from google.cloud import storage
from datetime import datetime

# Add project root to sys.path to allow imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.preprocessing.transform import transform_df
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# ---------- Configuration ----------
GCS_BUCKET = "homiehubbucket"
GCP_KEY_PATH = "./GCP_Account_Key.json"  # Update this path as needed
RAW_FILENAME = "homiehub_listings.csv"
PROCESSED_FILENAME = "homiehub_listings_processed.csv"

# ========== STEP 1: Load Raw Data from GCS ==========
def load_raw_listings_task(**kwargs):
    """Load raw listings from GCS"""
    logger.info("Starting to load raw listings from GCS")
    
    # Set credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_KEY_PATH
    
    # Get today's date for folder structure
    today = datetime.now().strftime('%Y-%m-%d')
    prefix = f"raw/{today}/"
    
    # Initialize storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    
    # List blobs with the prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    csv_files = [blob for blob in blobs if blob.name.endswith('.csv')]
    
    if not csv_files:
        # Try without date folder as fallback
        logger.warning(f"No files found in {prefix}, trying raw/ directly")
        blobs = list(bucket.list_blobs(prefix="raw/"))
        csv_files = [blob for blob in blobs if blob.name.endswith('.csv')]
        
        if not csv_files:
            raise ValueError(f"No CSV files found in gs://{GCS_BUCKET}/raw/")
    
    # Get the most recent file (or specific file if you prefer)
    latest_blob = sorted(csv_files, key=lambda b: b.time_created)[-1]
    
    # Download as string and convert to DataFrame
    csv_string = latest_blob.download_as_text()
    df = pd.read_csv(io.StringIO(csv_string))
    
    logger.info(f"Loaded {len(df)} rows from gs://{GCS_BUCKET}/{latest_blob.name}")
    
    # Save to temp location for next task
    temp_file = Path("/tmp") / f"raw_df_temp.csv"
    df.to_csv(temp_file, index=False)
    
    return str(temp_file)

# ========== STEP 2: Transform Data ==========
def transform_listings_task(**kwargs):
    """Apply preprocessing and transformation to the raw dataset"""
    ti = kwargs['ti']
    raw_file = ti.xcom_pull(task_ids='load_raw_listings')
    
    logger.info(f"Loading raw data from {raw_file}")
    df = pd.read_csv(raw_file)
    
    logger.info(f"Transforming {len(df)} rows")
    transformed_df = transform_df(df)
    
    # Save transformed data to temp file
    temp_file = Path("/tmp") / f"transformed_df_temp.csv"
    transformed_df.to_csv(temp_file, index=False)
    
    logger.info(f"Transformation complete: {len(transformed_df)} rows, {len(transformed_df.columns)} columns")
    
    return str(temp_file)

# ========== STEP 3: Save Processed Data to GCS ==========
def save_processed_listings_task(**kwargs):
    """Upload the processed data to GCS"""
    ti = kwargs['ti']
    transformed_file = ti.xcom_pull(task_ids='transform_listings')
    
    logger.info(f"Loading transformed data from {transformed_file}")
    df = pd.read_csv(transformed_file)
    
    # Set credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_KEY_PATH
    
    # Create destination path with timestamp
    today = datetime.now().strftime('%Y-%m-%d')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Use the processed filename format
    filename = f"{PROCESSED_FILENAME.replace('.csv', '')}_{timestamp}.csv"
    destination_blob_name = f"processed/{today}/{filename}"
    
    # Initialize storage client and upload
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(destination_blob_name)
    
    # Convert DataFrame to CSV and upload
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    blob.upload_from_string(csv_string, content_type='text/csv')
    
    out_path = f"gs://{GCS_BUCKET}/{destination_blob_name}"
    logger.info(f"Wrote {len(df)} rows to {out_path}")
    print(f"Wrote {len(df)} rows to {out_path}")
    
    return out_path

# ========== STEP 4: Finalize ETL ==========
def finalize_etl_task(**kwargs):
    """Marks the ETL pipeline completion in logs"""
    logger.info("ETL DAG completed successfully")
    print("ETL DAG completed successfully.")

# ========== STEP 5: Push Summary to XCom ==========
def push_summary_task(**kwargs):
    """Generates a summary of processed rows and ETL status"""
    ti = kwargs['ti']
    
    try:
        # Get the GCS path from the save task
        gcs_path = ti.xcom_pull(task_ids='save_processed_listings')
        
        # Get row count from transform task
        transformed_file = ti.xcom_pull(task_ids='transform_listings')
        df = pd.read_csv(transformed_file)
        
        summary = {
            "rows_processed": len(df),
            "status": "success",
            "gcs_path": gcs_path,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"ETL Summary: {summary}")
        
    except Exception as e:
        summary = {
            "rows_processed": 0,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        logger.error(f"ETL Failed: {summary}")
    
    ti.xcom_push(key="etl_summary", value=summary)
    return summary
# ========== STEP 6: Send Email with GCS Attachment ==========
def send_email_task(**kwargs):
    """Sends an email with ETL summary and attaches the processed file from GCS"""
    import tempfile
    from google.cloud import storage
    import os
    
    ti = kwargs['ti']
    summary = ti.xcom_pull(task_ids='push_summary', key='etl_summary')
    
    # Prepare email content
    if summary is None:
        subject = "ETL Summary Missing"
        html = "<p>No summary found from ETL DAG.</p>"
        files = []
    elif summary["status"] == "success":
        subject = "ETL Completed Successfully"
        html = f"""
        <h2>ETL Pipeline Success</h2>
        <p><strong>Rows Processed:</strong> {summary['rows_processed']}</p>
        <p><strong>Data Location:</strong> {summary.get('gcs_path', 'N/A')}</p>
        <p><strong>Completed At:</strong> {summary.get('timestamp', 'N/A')}</p>
        <hr>
        <p><em>The processed CSV file is attached to this email.</em></p>
        """
        
        # Download file from GCS to attach
        files = []
        if 'gcs_path' in summary:
            try:
                # Parse the GCS path
                gcs_path = summary['gcs_path']
                
                # Extract bucket and blob path
                if gcs_path.startswith('gs://'):
                    path_parts = gcs_path[5:].split('/', 1)
                    bucket_name = path_parts[0]
                    blob_path = path_parts[1] if len(path_parts) > 1 else ''
                    
                    # Set credentials
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_KEY_PATH
                    
                    # Download the file from GCS
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    
                    # Create temp file to store the download
                    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                        blob.download_to_file(tmp_file)
                        temp_filepath = tmp_file.name
                    
                    files = [temp_filepath]
                    logger.info(f"Downloaded file from GCS for attachment: {gcs_path}")
                    
            except Exception as e:
                logger.warning(f"Could not download GCS file for attachment: {e}")
                html += f"<p><small>Note: Could not attach file from GCS: {e}</small></p>"
    else:
        subject = "ETL Failed"
        html = f"""
        <h2>ETL Pipeline Failed</h2>
        <p><strong>Error:</strong> {summary.get('error', 'Unknown')}</p>
        <p><strong>Time:</strong> {summary.get('timestamp', 'N/A')}</p>
        """
        files = []
    
    # Send email with attachment
    try:
        send_email(
            to="samhita.kolluri@gmail.com",
            subject=subject,
            html_content=html,
            files=files  # Attach the downloaded file
        )
        logger.info(f"Email sent successfully with attachment: {subject}")
        print(f"Email sent: {subject}")
        
        # Clean up temp file if created
        if files:
            for filepath in files:
                try:
                    os.remove(filepath)
                    logger.info(f"Cleaned up temp file: {filepath}")
                except:
                    pass
        
    except Exception as e:
        # Clean up temp file even if email fails
        if files:
            for filepath in files:
                try:
                    os.remove(filepath)
                except:
                    pass
        
        logger.error(f"Failed to send email: {str(e)}")
        raise Exception(f"Email notification failed: {str(e)}")

# ========== Alternative: Simpler version with just attachment ==========
def send_email_with_attachment_simple(**kwargs):
    """Simpler version - downloads and attaches the GCS file"""
    from google.cloud import storage
    import tempfile
    import os
    
    ti = kwargs['ti']
    summary = ti.xcom_pull(task_ids='push_summary', key='etl_summary')
    
    if not summary or summary["status"] != "success":
        # Skip attachment if ETL failed
        subject = "ETL Failed or No Summary"
        html = f"<p>ETL failed or no summary available.</p>"
        send_email(
            to="samhita.kolluri@gmail.com",
            subject=subject,
            html_content=html
        )
        return
    
    # Parse GCS path from summary
    gcs_path = summary.get('gcs_path', '')
    
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    # Extract bucket and blob names
    path_without_prefix = gcs_path[5:]
    bucket_name, blob_path = path_without_prefix.split('/', 1)
    
    # Set up GCS client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_KEY_PATH
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Download to temp file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp:
        blob.download_to_file(tmp)
        temp_path = tmp.name
    
    try:
        # Send email with attachment
        send_email(
            to="samhita.kolluri@gmail.com",
            subject=f"ETL Success - {summary['rows_processed']} rows processed",
            html_content=f"""
            <h2>ETL Pipeline Completed</h2>
            <p>Successfully processed <strong>{summary['rows_processed']}</strong> rows.</p>
            <p>File location: <code>{gcs_path}</code></p>
            <p>See attached CSV file for processed data.</p>
            """,
            files=[temp_path]  # Attach the file
        )
        print(f"Email sent with attachment from {gcs_path}")
        
    finally:
        # Always clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ========== STEP 7: Send Logs Email with Stats ==========
def send_logs_email_task(**kwargs):
    """Sends email with processing statistics"""
    from google.cloud import storage
    import os
    
    dag_run = kwargs.get('dag_run')
    run_id = dag_run.run_id if dag_run else "manual_run"
    
    ti = kwargs['ti']
    summary = ti.xcom_pull(task_ids='push_summary', key='etl_summary')
    
    subject = f"ETL Run Report - {run_id}"
    
    # Try to get file size from GCS
    file_size_str = "N/A"
    if summary and 'gcs_path' in summary:
        try:
            gcs_path = summary['gcs_path']
            if gcs_path.startswith('gs://'):
                path_without_prefix = gcs_path[5:]
                bucket_name, blob_path = path_without_prefix.split('/', 1)
                
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCP_KEY_PATH
                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                blob.reload()  # Get latest metadata
                
                # Format file size
                size_bytes = blob.size
                if size_bytes < 1024:
                    file_size_str = f"{size_bytes} bytes"
                elif size_bytes < 1024 * 1024:
                    file_size_str = f"{size_bytes / 1024:.2f} KB"
                else:
                    file_size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
                    
        except Exception as e:
            logger.warning(f"Could not get file size: {e}")
    
    if summary and summary["status"] == "success":
        body = f"""
        <h3>ETL Pipeline Report</h3>
        <table border="1" cellpadding="8" style="border-collapse: collapse;">
            <tr style="background-color: #f0f0f0;">
                <td><strong>Metric</strong></td>
                <td><strong>Value</strong></td>
            </tr>
            <tr>
                <td>Run ID</td>
                <td>{run_id}</td>
            </tr>
            <tr>
                <td>Status</td>
                <td style="color: green;">Success</td>
            </tr>
            <tr>
                <td>Rows Processed</td>
                <td>{summary.get('rows_processed', 'N/A')}</td>
            </tr>
            <tr>
                <td>File Size</td>
                <td>{file_size_str}</td>
            </tr>
            <tr>
                <td>GCS Location</td>
                <td><code>{summary.get('gcs_path', 'N/A')}</code></td>
            </tr>
            <tr>
                <td>Completed At</td>
                <td>{summary.get('timestamp', 'N/A')}</td>
            </tr>
        </table>
        """
    else:
        body = f"""
        <h3>📊 ETL Pipeline Report</h3>
        <table border="1" cellpadding="8" style="border-collapse: collapse;">
            <tr style="background-color: #f0f0f0;">
                <td><strong>Metric</strong></td>
                <td><strong>Value</strong></td>
            </tr>
            <tr>
                <td>Run ID</td>
                <td>{run_id}</td>
            </tr>
            <tr>
                <td>Status</td>
                <td style="color: red;">Failed</td>
            </tr>
            <tr>
                <td>Error</td>
                <td>{summary.get('error', 'Unknown') if summary else 'No summary available'}</td>
            </tr>
        </table>
        """
    
    try:
        send_email(
            to="samhita.kolluri@gmail.com",
            subject=subject,
            html_content=body
        )
        logger.info(f"✅ Report email sent for run: {run_id}")
        
    except Exception as e:
        logger.error(f"Failed to send report email: {str(e)}")
        raise Exception(f"Report email failed: {str(e)}")

# ---------- DAG Default Args ----------
default_args = {
    'owner': 'homiehub-team',
    'depends_on_past': False,
    'retries': 1,
}

# ---------- DAG Definition ----------
with DAG(
    dag_id='homiehub_data_pipeline',
    default_args=default_args,
    description='ETL DAG with email notification',
    start_date=datetime.today(),
    schedule=None,
    catchup=False,
    max_active_runs=1,
) as dag:
    
    # ---------- Define Tasks ----------
    load_raw = PythonOperator(
        task_id='load_raw_listings',
        python_callable=load_raw_listings_task,
    )
    
    transform = PythonOperator(
        task_id='transform_listings',
        python_callable=transform_listings_task,
    )
    
    save_processed = PythonOperator(
        task_id='save_processed_listings',
        python_callable=save_processed_listings_task,
    )
    
    finalize = PythonOperator(
        task_id='finalize_etl',
        python_callable=finalize_etl_task,
    )
    
    push_summary = PythonOperator(
        task_id="push_summary",
        python_callable=push_summary_task,
    )
    
    notify = PythonOperator(
        task_id="send_email",
        python_callable=send_email_task,
    )
    
    send_logs_email = PythonOperator(
        task_id="send_logs_email",
        python_callable=send_logs_email_task,
    )
    
    # ---------- Task Dependencies ----------
    load_raw >> transform >> save_processed >> finalize >> push_summary >> notify >> send_logs_email