
#  **HomieHub ETL Data Pipeline**

---

## **1. Project Overview**
HomieHub is an automated, end-to-end ETL pipeline designed to process **unstructured housing-related data (such as WhatsApp listings)** into **structured, verified, and normalized datasets** ready for analysis.  

This project demonstrates **modern MLOps practices**, including:
- Modular and reusable code structure
- **Airflow-based orchestration**
- **Dockerized reproducibility**
- Automated **logging, versioning, and testing**

---

## Key Features

* Modular ETL code: `extraction`, `ingestion`, `preprocessing`, and `utils`
* Reproducible workflow using **Docker** and **Airflow**
* Logging and alert system for monitoring pipeline execution
* Data versioning using **DVC**
* SMTP email notifications for ETL summary and logs
* Supports anomaly detection and schema validation in preprocessing

---

## Folder Structure

```text
.
├── __init__.py
├── assets/
│   ├── 1_grant_chart.png
│   └── 2_homiehub_data_pipeline-graph.png
├── config/
├── dags/
│   ├── __init__.py
│   ├── airflow_etl.py       # old (deprecated)
│   └── homiehub_data_pipeline.py
├── data/
│   ├── features/
│   ├── processed/
│   └── raw/
├── docker-compose.yaml       # DO NOT PUSH TO GIT (use locally)
├── docs/
├── logs/
├── plugins/
├── readme.md
├── requirements.txt
├── setup.sh
├── src/
│   ├── __init__.py
│   ├── extraction/
│   │   ├── __init__.py
│   │   └── whatsapp_data_extraction.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── data_handlers/
│   │       ├── __init__.py
│   │       └── csv_extractor.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── transform.py
│   └── utils/
│       ├── __init__.py
│       └── io.py
├── test/
│   └── __init__.py
│   └── __test_data_loading.py
│   └── __init__.py
│   └── __init__.py
└── working_data/
    └── __init__.py
```

---


## **2. Pipeline Architecture**


### **2.1 End-to-End Pipeline Flow**
The following flowchart illustrates the **complete ETL pipeline**, detailing how data moves from unstructured sources (e.g., WhatsApp exports or Facebook Marketplace listings) through Airflow-orchestrated tasks into cleaned, versioned outputs:

![](assets/0_flowchart_datapipeline.png)

**Flow Overview:**
- **Stage 1 – Data Acquisition:** Collects housing data from WhatsApp, Facebook Marketplace, or synthetic test datasets.  
- **Stage 2 – Data Cleaning & Preprocessing:** Handles missing values, removes duplicates, and performs outlier treatment.  
- **Stage 3 – Data Transformation:** Applies feature engineering, categorical normalization, and value conversions.  
- **Stage 4 – Processed Storage:** Stores validated outputs in `data/processed/`, logs in `logs/`, and triggers summary emails.  

All stages are orchestrated by the Airflow DAG: `homiehub_data_pipeline.py`.

---


### **2.1 DAG Overview**
The entire workflow is orchestrated through **Apache Airflow** using the DAG file:  
`dags/homiehub_data_pipeline.py`.

**Tasks:**  
`Extract → Ingest → Transform → Save → Push Summary → Email Notification`

**Key Features:**
- Automatic scheduling (daily or manual)
- Retries and alert notifications on failure
- Modular task definitions for reproducibility

**Visualization:**
![](assets/2_homiehub_data_pipeline-graph.png)

---

### **2.2 Workflow Timeline**
The runtime dependencies and task sequence are represented below:

![](assets/1_grant_chart.png)

This Gantt chart visualizes the task execution order and timing dependencies within the Airflow DAG.

**homiehub_data_pipeline DAG** consists of the following steps:

1. **Load Raw Listings** – reads raw CSV into a temporary file
2. **Transform Listings** – data cleaning, preprocessing, and feature engineering
3. **Save Processed Listings** – outputs cleaned CSV to `data/processed/`
4. **Finalize ETL** – prints ETL completion in logs
5. **Push Summary** – generates ETL summary for XCom and notifications
6. **Send Summary Email** – sends success/failure summary via SMTP
7. **Send Logs Email** – emails logs for the current DAG run

---

## **3. Data Acquisition and Ingestion**

### **Extraction**
- **Script:** `src/extraction/whatsapp_data_extraction.py`
- **Purpose:** Converts unstructured WhatsApp chat exports into structured CSVs.  
- **Output:** `data/raw/structured_listings_nlp.csv`

### **Ingestion**
- **Script:** `src/ingestion/data_handlers/csv_extractor.py`
- **Purpose:** Reads the structured CSV safely into a pandas DataFrame.  
- **Validations:**  
  - Confirms file presence and schema consistency  
  - Applies safe NA handling and type inference  

**Result:** A traceable, clean ingestion process from raw text → structured CSV → pandas DataFrame.

---

## **4. Data Preprocessing and Transformation**

### **Transformation**
- **Script:** `src/preprocessing/transform.py`
- **Core Tasks:**
  - Normalizes categorical fields (`gender`, `accom_type`, `food_pref`, `area`)
  - Converts monetary fields to numeric (`rent_amount_num`)
  - Parses boolean flags (`furnished_bool`, `utilities_included_bool`)
  - Parses dates and derives ISO timestamps (`timestamp_iso`)
  - Computes helper fields like `lease_duration_months`

**Output:**  
A fully cleaned, analysis-ready dataset stored in `/data/processed/`.

**Reproducibility:**  
Every transformation step is deterministic — identical input yields identical output.

---

## **5. Error Handling and Resilience**

- **Retries:** Configured per task (2 attempts, 3-minute delay)
- **Failure Alerts:** Triggered via Airflow `EmailOperator`
- **Graceful Degradation:**  
  - If extraction fails, the DAG halts with descriptive logs  
  - Empty data files still maintain headers for downstream compatibility
- **Success Alerts:** Summaries with record counts sent via email upon completion

---

## **6. Logging and Monitoring**

The pipeline leverages **Airflow’s built-in logging and monitoring**, enhanced with email notifications.

### **Logging Features**
- Each task logs start/end timestamps and row counts.
- Logs are saved under `logs/` and attached to summary emails.
- Task-level logs can also be viewed directly in the Airflow UI.

### **Monitoring**
- Airflow tracks task status and run duration.
- Completion summaries and statistics are emailed to administrators.

**Example Logs & Notifications:**
| Type | Image | Description |
|------|--------|-------------|
|  Success Email | ![](assets/3_email_logging1.jpg) | “ETL Completed Successfully – Processed 25 Rows.” |
|  Failure Alert | ![](assets/4_email_log2.jpg) | Detailed trace for failure cases |
|  Log Bundle | ![](assets/5_email_log3.jpg) | Attached `.log` files per task for auditing |

# SMTP Email Configuration

Airflow `smtp_default` connection in Airflow UI. Go to admin, connections and add `smtp_default` connection. Don’t use your Gmail login password you need an App Password.

| Field       | Value                                                                             |
| ----------- | --------------------------------------------------------------------------------- |
| Conn Id     | smtp_default                                                                      |
| Conn Type   | SMTP                                                                              |
| Host        | smtp.gmail.com                                                                    |
| Port        | 587                                                                               |
| Login       | (`your_email@gmail.com`) |
| Password    | `your_app_password`                                                     |
| Timeout     | 30                                                                                |
| Retry Limit | 5                                                                                 |
| TLS         | false                                                                             |
| SSL         | true                                                                              |
| Auth Type   | basic                                                                             |
---

## **7. Schema Validation and Data Quality**

HomieHub validates schema integrity and computes summary statistics automatically during ETL.

### **Schema Checks**
- Validates presence of all mandatory columns:  
  `['timestamp', 'rent_amount', 'gender', 'accom_type', 'food_pref', 'area']`
- Ensures data type consistency for numeric, boolean, and categorical columns.
- Logs missing or unexpected columns as warnings.

### **Statistics Generation**
Included automatically in the ETL summary email:
```
ETL Summary:
- Rows processed: 25
- Columns validated: 12
- Missing values: 0.8%
- Average Rent: $1,320.50
- Outlier threshold breaches: 0
```

These validations ensure high data quality, reproducibility, and schema stability across runs.

---

## **8. Testing and Validation**

Automated tests ensure **robustness**, **reproducibility**, and **edge-case resilience** across the pipeline.

### **Framework**
All tests are written in **pytest** and stored in `/tests/`.

### **Modules Covered**
| Test Script | Description |
|--------------|--------------|
| `test_data_extraction.py` | Validates extraction from GCS; checks schema, nulls, and data consistency. |
| `test_data_loading.py` | Tests ingestion, GCS upload/download consistency, and data structure preservation. |
| `test_data_transformation.py` | Unit-tests each preprocessing function (`_parse_money`, `_parse_bool`, `_parse_date`, etc.) and ensures correct column creation. |

### **Execution**
```bash
pytest -v
```

### **Key Assertions**
- Schema consistency between raw and processed datasets  
- Correct parsing of numerics, booleans, and dates  
- Deterministic outputs across repeated runs  
- Robust handling of malformed or missing input data

---

## **9. Data Versioning and Reproducibility**

### **Data Version Control (DVC)**
Used to version and reproduce datasets tied to specific DAG runs.

**Setup:**
```bash
dvc init
dvc add data/raw data/processed
dvc remote add -d homiehub_remote <remote-url>
```

**Purpose:**
- Maintain dataset lineage
- Guarantee reproducible pipeline states
- Sync exact dataset versions across collaborators

**Tracked Artifacts:** `data/raw/` and `data/processed/`

---

## **10. Metrics and Run Tracking**

At the end of each DAG run, the ETL summary captures:
- Rows processed  
- Null percentages  
- Start/end timestamps  
- Success/failure status  
- Log bundle path  

All metrics are included in the **email notification** for full traceability.

---

## **11. Reproducing the Project (from GitHub)**

1. **Clone the repository:**
   ```bash
   git clone <your_repo_url>.git
   cd homiehub
   ```

2. **Run the setup script:**
   ```bash
   bash setup.sh
   ```

3. **Follow setup prompts:**
   ```bash
   echo "Next steps:"
   echo "1. Place your WhatsApp export or structured CSV under: data/raw/"
   echo "2. Initialize Airflow: docker-compose up airflow-init"
   echo "3. Start Airflow services: docker-compose up -d"
   echo "4. Access Airflow UI: http://localhost:8080"
   ```

4. **Add your input data:**
   ```bash
   cp <your_file>.csv data/raw/
   ```

5. **Trigger the DAG in Airflow UI:**
   - DAG ID: `homiehub_data_pipeline`
   - Toggle ON → **Trigger DAG**

6. **Monitor progress and review emails/logs.**

7. **DVC versioning:**
   ```bash
   dvc pull   # Get tracked data versions
   dvc push   # Push updates after runs
   ```

8. **Re-run deterministically:**
   ```bash
   docker-compose down -v
   rm -rf logs/* data/processed/*
   docker-compose up -d --build
   ```

---

## **12. Future Enhancements**
- Integrate DVC remotes for cloud-based data lineage.
- Add CI/CD for automated testing.
- Introduce lightweight drift monitoring on rent/area features.
- Extend Airflow metrics with MLflow for run tracking.

---

## **13. Conclusion**
HomieHub’s MLOps pipeline demonstrates a fully automated, modular, and reproducible ETL workflow built with Apache Airflow. The system successfully processes raw, unstructured text into normalized, verified datasets ready for machine learning and analytical modeling.
