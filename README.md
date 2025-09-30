# HomieHub: AI-Powered Roommate Matching Platform

### Overview
HomieHub is an MLOps-driven platform designed to streamline roommate matching for university students. It aggregates and parses unstructured listings from WhatsApp groups, social media platforms, kaggle, and synthetic data data using a conversational WhatsApp bot for efficient lead capture. A web platform, secured with .edu authentication, enables verified students to post and edit listings seamlessly. The system employs AI-driven matching to deliver personalized, context-aware recommendations, leveraging advanced algorithms to enhance relevance. Data is managed in a scalable storage solution, ensuring accessibility and performance. The platform prioritizes reliability and security through automated deployment pipelines, real-time monitoring, and robust privacy measures, including anonymization of WhatsApp data and encryption

### Installation

1. Clone the Repository:
```
git clone https://github.com/homiehub
cd homiehub
```
2. Install Dependencies:
```
pip install -r requirements.txt
```

3. Set Up Environment:

* Configure GCP credentials (gcloud auth application-default login).
* Set environment variables in .env (GCP_PROJECT_ID, AUTH0_CLIENT_ID).

### Usage

Run WhatsApp Bot Locally:
```
python src/bot/app.py
```

Launch Web Platform:
```
cd src/web
npm install
npm start
```

Train Models:
```
python src/models/train.py --model distilbert
python src/models/train.py --model sentence-bert
```

Deploy to GCP:
* Push to GKE/Cloud Run via GitHub Actions (deployment/workflows/deploy.yml).

### Project Structure
```
homiehub/
├── data/
│   ├── raw/
│   │   ├── whatsapp_exports/     # Unstructured WhatsApp chat exports
│   │   ├── public_datasets/      # Public dataset files (CSV, JSON)
│   │   └── synthetic_data/       # Synthetic listing files
│   ├── processed/                # Processed listings and embeddings
│   └── version_control/          # Metadata for data versioning (DVC)
├── src/
│   ├── ingestion/
│   │   ├── bot/                  # Conversational bot for lead capture
│   │   └── data_handlers/        # Scripts for data ingestion
│   ├── preprocessing/
│   │   ├── text_processing/      # Text cleaning and segmentation logic
│   │   └── feature_extraction/   # Attribute extraction (location, budget)
│   ├── matching/
│   │   ├── algorithm/            # AI-driven matching logic
│   │   └── recommendation/       # Recommendation generation scripts
│   ├── web/
│   │   ├── frontend/             # Web interface for posting/editing
│   │   └── auth/                 # Authentication module (.edu via Auth0)
│   └── utils/                    # Helper scripts (geocoding, logging)
├── deployment/
│   ├── deployment_configs/       # Infrastructure setup files (Terraform)
│   ├── workflows/                # Automation scripts (CI/CD pipelines)
│   └── container_configs/        # Containerization files (Docker)
├── notebooks/
│   ├── exploration/              # EDA and prototyping notebooks
│   ├── modeling/                 # Model development and testing
│   └── evaluation/               # Performance evaluation scripts
├── docs/
│   ├── api_specs.md              # API documentation
│   ├── architecture.md           # System architecture and diagrams
│   └── usage_guide.md            # User and developer instructions
├── tests/
│   ├── unit_tests/               # Unit tests for components
│   ├── integration_tests/        # Integration tests for pipelines
│   └── performance_tests/        # Performance and scalability tests
├── requirements.txt              
├── .gitignore                    
├── LICENSE                       
└── README.md                     
```
