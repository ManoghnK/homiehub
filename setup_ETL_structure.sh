#!/bin/bash

# ================================
# HomieHub ETL Pipeline Setup
# ================================

# Root directory
PROJECT_ROOT="HomieHub_ETL"
mkdir -p $PROJECT_ROOT

# --------------------------------
# Folder Structure
# --------------------------------
mkdir -p $PROJECT_ROOT/{dags,data/{raw,processed,features},scripts/{stage1_acquisition,stage2_cleaning,stage3_feature},tests,logs}

# --------------------------------
# Stage 1: Data Acquisition
# --------------------------------
touch $PROJECT_ROOT/scripts/stage1_acquisition/{download_data.py,unzip_data.py,data_loader.py}

# --------------------------------
# Stage 2: Data Cleaning & Preprocessing
# --------------------------------
touch $PROJECT_ROOT/scripts/stage2_cleaning/{missing_values_handler.py,duplicate_handler.py,listing_status_handler.py,anomaly_code_handler.py,cleaning_description.py,removing_invalid_prices.py,outlier_treatment.py}

# --------------------------------
# Stage 3: Feature Engineering
# --------------------------------
touch $PROJECT_ROOT/scripts/stage3_feature/{location_features.py,price_features.py,room_characteristics.py,temporal_features.py,listing_quality_features.py,roommate_matching_features.py,scaler.py,encoder.py}

# --------------------------------
# Other Essential Files
# --------------------------------
touch $PROJECT_ROOT/{dvc.yaml,requirements.txt,README.md}
touch $PROJECT_ROOT/dags/homiehub_dag.py
touch $PROJECT_ROOT/tests/test_data_cleaning.py
touch $PROJECT_ROOT/logs/pipeline.log

# --------------------------------
# Git & DVC Initialization
# --------------------------------
cd $PROJECT_ROOT
git init -q
dvc init -q

echo "✅ HomieHub ETL structure created successfully!"
echo "📁 Directory: $(pwd)"
tree -L 3