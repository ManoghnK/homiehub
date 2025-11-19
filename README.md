# HomieHub: AI-Powered Roommate Matching Platform

### Overview
HomieHub is an MLOps-driven platform designed to streamline roommate matching for university students. It aggregates and parses unstructured listings from WhatsApp groups, social media platforms, kaggle, and synthetic data data using a conversational WhatsApp bot for efficient lead capture. A web platform, secured with .edu authentication, enables verified students to post and edit listings seamlessly. The system employs AI-driven matching to deliver personalized, context-aware recommendations, leveraging advanced algorithms to enhance relevance. Data is managed in a scalable storage solution, ensuring accessibility and performance. The platform prioritizes reliability and security through automated deployment pipelines, real-time monitoring, and robust privacy measures, including anonymization of WhatsApp data and encryption

### Project Structure
```text
homiehub/
├── LICENSE
├── README.md                         # Main project documentation
├── requirements.txt                  # Project-wide dependencies
└── data-pipeline/                    # Data Pipeline Module
    ├── ETL_SCRIPT.sh                 # Manual ETL execution script
    ├── README.md                     # Pipeline-specific documentation
    ├── airflow_setup.sh              # Airflow environment setup script
    ├── requirement.txt               # Pipeline-specific dependencies
    ├── __init__.py
    ├── assets/                       # Documentation assets
    │   ├── 0_flowchart_datapipeline.png
    │   ├── 1_grant_chart.png
    │   ├── 2_homiehub_data_pipeline-graph.png
    │   ├── 3_email_errorlog.png
    │   ├── 3_email_logs.png
    │   ├── 3_email_notify.png
    │   ├── 3_email_successlog.png
    │   └── 4_dag_report.png
    ├── dags/                         # Airflow DAG definitions
    │   ├── __init__.py
    │   └── homiehub_data_pipeline.py
    ├── data/                         # Data storage (DVC tracked)
    │   ├── processed/
    │   │   └── homiehub_listings_processed.csv.dvc
    │   └── raw/
    │       └── homiehub_listings.csv.dvc
│   ├── docs/                         # Pipeline documentation
│   │   ├── scripts_guide.md               # scripts guide
│   │   ├── setup_guide.md                # setup guide
    ├── pipelines/                    # Core pipeline logic
    │   ├── __init__.py
    │   └── etl.py
    ├── plugins/                      # Airflow plugins
    ├── src/                          # Source code modules
    │   ├── __init__.py
    │   ├── extraction/               # Data extraction
    │   │   ├── __init__.py
    │   │   ├── upload_script_to_GCP.py
    │   │   └── whatsapp_data_extraction.py
    │   ├── ingestion/                # Data ingestion
    │   │   ├── __init__.py
    │   │   └── data_handlers/
    │   │       ├── __init__.py
    │   │       └── csv_extractor.py
    │   ├── load/                     # Data loading to GCS
    │   │   ├── __init__.py
    │   │   └── upload_cleaned_df_to_gcp.py
    │   ├── preprocessing/            # Data transformation
    │   │   ├── __init__.py
    │   │   └── transform.py
    │   │   └── bias.py
    │   └── utils/                    # Utility functions
    │       ├── __init__.py
    │       ├── io.py
    │       └── logger.py
    ├── test/                         # Test suite
    │   ├── __init__.py
    │   ├── test_data_extraction.py
    │   ├── test_data_loading.py
    │   └── test_data_transformation.py
    └── working_data/                 # Temporary processing directory
        └── __init__.py
```

## Architecture
![](data-pipeline/assets/0_flowchart_datapipeline.png)

## Features

### [Data Pipeline](data-pipeline/readme.md) (Implemented)
- **Multi-Source Aggregation**: WhatsApp, Facebook Marketplace, Kaggle datasets, synthetic data
- **Automated ETL**: Apache Airflow orchestration with Docker containerization
- **NLP Processing**: spaCy-based extraction from unstructured text
- **Cloud Native**: Google Cloud Storage integration
- **Data Quality**: Schema validation, DVC versioning
- **Monitoring**: Email notifications, comprehensive logging

### [AI Model Pipeline](model-pipeline/readme.md) (Implemented)

The `model-pipeline/` module handles AI-driven roommate matching using semantic search, embeddings, and preference-aware ranking. This pipeline is designed to be fully MLOps-compliant, supporting modular model updates, batch and real-time inference, and scalable deployment.

**Model Pipeline Capabilities**
1. Semantic Embedding Models
- Sentence Transformer–based vectorization
- Room & user representation learning
- Cosine similarity + hybrid BM25 reranking
2. LLM Agent for Context-Aware Scoring
- Evaluates roommate compatibility using LLM-based reasoning
- Tools-enabled evaluation (budget alignment, commute, noise tolerance, habits)
- Supports OpenAI API + local LLM deployments
3. Recommendation Engine
- Top-K candidate retrieval using vector search
- LLM reranker for personalized scoring
- Real-time recommendation function via Cloud Functions
4. MLOps Features
- Dockerized microservices for all three components
- GitHub Actions-based CI/CD 
- Configurable deployment targets: Cloud Run / GKE
- Monitoring & metrics hooks (Prometheus-ready)


### Web Platform (Planned)
- .edu authentication via Auth0
- Verified student listings
- Real-time matching interface
- Privacy-first design

## Documentation

- [**Data Pipeline Documentation**](./data-pipeline/README.md) - Complete data and ETL pipeline guide
- [**Setup Guide**](./data-pipeline/docs/setup_guide.md) - Installation and configuration
- [**Scripts Usage**](./data-pipeline/docs/scripts_usage.md) - Automation scripts guide
- [**Model Pipeline Documentation**](./model-pipeline/README.md) - Complete model pipeline guide

## Quick Start

### Prerequisites
- **Python 3.11** (required for Apache Airflow 3.1.1)
- Docker & Docker Compose (20.10+ / 2.0+)
- Google Cloud Platform account
- Git

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/homiehub/homiehub.git
   cd homiehub
   ```

2. **Set Up Python Environment**
   ```bash
   # Ensure Python 3.11 is installed
   python3.11 --version
   
   # Create virtual environment with Python 3.11
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Verify Python version
   python --version  # Should show 3.11.x
   ```

3. **Install Dependencies**
   ```bash
   # Install project requirements
   pip install -r requirements.txt
   
   # For data pipeline specifically
   cd data-pipeline
   pip install -r requirement.txt
   ```

4. **Set Up Data Pipeline**
   ```bash
   cd data-pipeline
   
   # Run setup script
   chmod +x airflow_setup.sh
   ./airflow_setup.sh
   
   # Configure GCP credentials
   cp /path/to/your-service-account-key.json GCP_Account_Key.json
   
   # Set environment variables
   echo "GCP_PROJECT_ID=your-project-id" >> .env
   echo "GCP_BUCKET_NAME=homiehub-data-bucket" >> .env
   ```

5. **Initialize Airflow**
   ```bash
   # From data-pipeline directory
   docker-compose up airflow-init
   docker-compose up -d
   ```

6. **Access Airflow UI**
   - Navigate to http://localhost:8080
   - Username: `airflow2`, Password: `airflow2`
   - Enable the `homiehub_data_pipeline` DAG

Deploy to GCP:
* Push to GKE/Cloud Run via GitHub Actions (deployment/workflows/deploy.yml).

## Contributing

This is an academic MLOps project — contributions are welcome!

**Workflow:**

1. Fork the repo & create a branch

   ```bash
   git checkout -b feature/YourFeature
   ```
2. Make changes and run tests

   ```bash
   cd data-pipeline && python -m pytest test/
   ```
3. Commit & push

   ```bash
   git commit -m "Add YourFeature"
   git push origin feature/YourFeature
   ```

**Standards:** Follow PEP 8, write tests, update docs, and ensure all tests pass.

## License

Licensed under the [MIT License](./LICENSE).

## Acknowledgments

MLOps course mentors, Apache Airflow, GCP, and the open-source community.

## Contact

For questions or collaboration:
- Create an [issue](https://github.com/homiehub/homiehub/issues)
- Review [data pipeline documentation](./data-pipeline/README.md)
- Review [model pipeline documentation](./model-pipeline/README.md)

---
> For comprehensive details about the implemented ETL system, refer to the [Data Pipeline Documentation](/data-pipeline/README.md), the [Setup Guide](/data-pipeline/docs/setup_guide.md), and the [Scripts Usage Guide](/data-pipeline/docs/scripts_usage.md).
