# HomieHub ETL Data Pipeline

## 1. Project Overview
HomieHub is an automated, end-to-end data pipeline built to process unstructured housing-related data (such as WhatsApp listings) into structured, verified, and normalized datasets ready for downstream analysis. 
The project demonstrates modern **MLOps practices** including modular code organization, Airflow-based orchestration, Dockerized reproducibility, and automated logging and notifications.

---

## 2. DAG Overview and Architecture
The central orchestrator is the **Airflow DAG**: `homiehub_data_pipeline.py`.

- **Tasks:** Extract → Transform → Save → Push Summary → Email Notification
- **Key Outputs:** Processed CSVs stored under `/data/processed/`
- **Scheduling:** Daily (configurable), with retries and email alerts
- **Visualization:**

![](assets/2_homiehub_data_pipeline-graph.png)

This DAG manages data flow dependencies, ensuring reliable task execution and logging at every stage.

---

## 3. Gantt Chart and Workflow Visualization
The pipeline’s runtime flow is visualized below, showing the sequential and parallel dependencies of tasks:

![](assets/1_grant_chart.png)

---

## 4. Data Acquisition and Ingestion

The pipeline begins by extracting unstructured housing data from WhatsApp chat exports using an NLP-based extraction module:

- **File:** `src/extraction/whatsapp_data_extraction.py`
- **Description:** Parses WhatsApp text files and converts conversational posts into structured CSVs (`structured_listings_nlp.csv`).
- **Output Directory:** `/data/raw/`

Once extracted, the data is ingested via:
- **File:** `src/ingestion/data_handlers/csv_extractor.py`
- **Purpose:** Safely reads CSVs with custom NA handling and validates file availability before transformation.
- **Output:** A `pandas` DataFrame ready for downstream preprocessing.

This ensures a reproducible and traceable data ingestion process from raw text to structured CSVs.

---

## 5. Data Preprocessing and Transformation

The pre-processing step standardizes and cleans the structured CSV:

- **File:** `src/preprocessing/transform.py`
- **Tasks:**
  - Normalizes categorical features (`gender`, `accom_type`, `food_pref`, `area`).
  - Converts monetary and numeric fields (`rent_amount_num`, `lease_duration_months`, `distance_to_campus_miles`).
  - Parses boolean indicators (`furnished_bool`, `utilities_included_bool`, etc.).
  - Generates standardized ISO date formats and derived columns.
- **Output Directory:** `/data/processed/`

All transformations are modular and deterministic to guarantee reproducibility across pipeline runs.

---


## 6. Logging and Monitoring

Airflow’s native logging system is enhanced with custom notifications and log bundling.

- Each task logs execution status and runtime metadata (start time, end time, duration).
- A summary of the run, including **rows processed and validation statistics**, is sent via email at the end of each run.
- Failure and success states are captured separately, with corresponding log attachments.
- Logs for all tasks are bundled and sent automatically to the pipeline administrator.

### Example Notifications and Logs:

#### Successful Completion Email
![ETL Completion](assets/3_email_logging1.jpg)
> “ETL Completed Successfully – Processed 25 Rows.”

####  Failure Alert Email
![Airflow Failure Alert](assets/4_email_log2.jpg)
> Email Operator alerts triggered on failed runs with detailed exception trace.

####  ETL Log Attachments
![ETL Logs](assets/5_email_log3.jpg)
> All task logs (attempt logs per operator) automatically attached for audit and troubleshooting.

---


Airflow’s native logging system is enhanced with custom notifications and log bundling:

- Each task logs execution status and time to `logs/`.
- A summary of the run (rows processed, start/end time) is sent via email.
- Task logs are bundled and attached to the email for reference.
- Failure alerts are automatically triggered with detailed error information.

**Example Success Notification:**
> “ETL Completed Successfully – Processed 25 Rows.”

**Example Failure Notification:**
> “Airflow Alert – Task Failure (SMTP Host Unreachable).”

---

## 7. Reproducibility and Containerization

Reproducibility is ensured through containerized deployment using Docker.

- **Files:**
  - `docker-compose.yaml`: Launches Airflow services.
  - `setup.sh`: Automates environment setup.
  - `requirements.txt`: Captures dependency versions.

All configurations and data paths are abstracted through `src/utils/io.py`, maintaining consistent directory resolution across environments.

---

## 8. Testing and Validation

Automated tests ensure correctness of core data processing functions.

- **Framework:** `pytest`
- **Test Modules:**
  - `test/test_transform.py` – validates transformations (dates, numerics, booleans).
  - `test/test_schema.py` – ensures schema consistency across pipeline runs.

**Run Command:**
```bash
pytest -v
```
Testing confirms stable schema and value integrity between raw and processed datasets.

*(If tests are not yet implemented, placeholders have been added to be integrated in the next iteration.)*

---

## 9. Data Versioning with DVC

To maintain data lineage and reproducibility, **DVC (Data Version Control)** will be used to track all dataset versions.

- **Setup:**
```bash
dvc init
dvc add data/raw data/processed
dvc remote add -d homiehub_drive <remote-url>
```
- **Tracked Artifacts:** Raw and processed CSVs
- **Configuration Files:** `.dvc`, `dvc.yaml`
- **Purpose:** Ensures consistent snapshots of datasets tied to specific DAG runs or Git commits.

---

## 10. Metrics and Tracking

At the end of each pipeline run, a summary is generated and emailed automatically.

**Metrics Captured:**
- Number of rows processed
- Start and end timestamps
- Success/Failure status
- Run ID and log file bundle

Future enhancements will include integration with **MLflow** to track additional metrics such as record distribution, error rates, and transformation latency.

---

## 11. Error Handling and Notifications

The DAG is configured with built-in retries and detailed email notifications.

- **Retries:** 2 attempts per task with a 3-minute delay.
- **Failure Alerts:** Triggered via `EmailOperator` upon task failure.
- **Success Alerts:** Summary email confirms successful ETL completion.
- **Log Bundling:** All per-task logs are zipped and emailed post-run.

**Example Notifications:**
- **Success Email:** “ETL Completed Successfully – Processed 25 Rows.”
- **Failure Email:** “Airflow Alert – Task Failure (SMTP Host Unreachable).”

---


## 14. Schema Validation and Statistics Generation

HomieHub’s pipeline integrates automated **schema and data statistics generation** directly within the ETL workflow.

While external libraries such as *Great Expectations* or *TensorFlow Data Validation (TFDV)* can be integrated, the current implementation already validates data quality through in-pipeline checks and automated summaries.

### Key Implementations:
- **Schema Validation:**
  - During ingestion and transformation, the pipeline verifies the presence of all mandatory columns:
    `['timestamp', 'rent_amount', 'gender', 'accom_type', 'food_pref', 'area']`
  - Missing or unexpected columns trigger logged warnings and are flagged in the email summary.
  - Datatype consistency is validated after transformation (numeric, boolean, categorical fields).

- **Statistics Generation:**
  - The pipeline computes key data statistics per run and embeds them in the **ETL Summary Email**:
    - Total rows processed
    - Null value counts per column
    - Percentage of missing values
    - Average, minimum, and maximum rent amount
  - These metrics are written to the Airflow logs and attached to the summary email.

- **Quality Assurance via Logs:**
  - Each task logs schema shape, column counts, and transformation summaries.
  - Logs are automatically bundled and sent to the pipeline administrator after every DAG run.

### Example of Logged Statistics (from ETL summary email):
```
ETL Summary:
- Rows processed: 25
- Columns validated: 12
- Missing values: 0.8%
- Average Rent: $1,320.50
- Outlier threshold breaches: 0
```

These built-in validations and metrics provide automated, reproducible **data quality monitoring** at every pipeline run, ensuring that each dataset version meets the required schema integrity and consistency standards.

---

## 12. Future Enhancements


- Add DVC tracking and remote configuration for dataset versioning.
- Integrate `pytest` validation in CI/CD workflow.
- Incorporate MLflow for run-level metric tracking.
- Implement lightweight bias monitoring on demographic distributions.
- Automate deployment on a cloud-hosted Airflow instance (e.g., Composer, MWAA).

---

## 13. Conclusion

HomieHub’s MLOps pipeline demonstrates a **fully automated, modular, and reproducible ETL workflow** built with Apache Airflow. 
The system successfully processes raw, unstructured text into normalized, verified datasets ready for machine learning and analytical modeling.

This implementation aligns with industry MLOps best practices and covers every critical pipeline component from **data ingestion to orchestration and monitoring**.


## **Reproducing This Project from GitHub**

1. **Clone the repository** and navigate into it:

   ```bash
   git clone <your_repo_url>.git
   cd homiehub
   ```

2. **Run the setup script** to install dependencies and prepare the environment:

   ```bash
   bash setup.sh
   ```

3. **Follow the printed next steps**:

   ```bash
   echo "Next steps:"
   echo "1. Ensure your WhatsApp export or structured housing CSV is placed under: data/raw/"
   echo "2. Initialize Airflow (one-time setup): docker-compose up airflow-init"
   echo "3. Start Airflow services: docker-compose up -d"
   echo "4. Open Airflow UI: http://localhost:8080"
   ```

4. **Add your input data**:  
   Place your file in the correct folder:
   ```bash
   cp <your_whatsapp_export_or_structured_listings.csv> data/raw/
   ```

   - If you’re using a **WhatsApp chat export**, the extraction module will parse it automatically.  
   - If you already have a **structured CSV**, the pipeline will skip extraction and proceed to ingestion and transformation.

5. **Trigger the DAG**:

   - Open the Airflow UI at [http://localhost:8080](http://localhost:8080)  
   - Navigate to **`homiehub_data_pipeline`**  
   - Toggle it **ON** and click **Trigger DAG**

6. **Monitor pipeline progress**:

   - Task logs are stored under `logs/` and viewable in the Airflow UI  
   - Success/failure summary emails (if configured) will be sent automatically to the alert email configured in your `.env` file

7. **Verify output artifacts**:

   - Extracted structured file: `data/raw/structured_listings_nlp.csv`  
   - Processed final file: `data/processed/listings_processed.csv`  
   - DAG run logs: `logs/dag_id=homiehub_data_pipeline/`  

8. **DVC data versioning** (if enabled in the project):

   - Pull tracked dataset versions before re-running the DAG:
     ```bash
     dvc pull
     ```
   - Push updates after successful runs:
     ```bash
     dvc add data/raw data/processed
     dvc push
     ```

9. **Reproduce a clean deterministic run**:
   ```bash
   docker-compose down -v    # stop containers and remove Airflow state
   rm -rf logs/* data/processed/*
   docker-compose up -d --build
   # Trigger the DAG again in the Airflow UI → outputs should match the previous run
   ```

---
