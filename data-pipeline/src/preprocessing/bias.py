import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from google.cloud import storage
import io
import os
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

class BiasAnalyzer:
    def __init__(self, data_path, bucket_name=None, service_account_key=None):
        """
        Initialize BiasAnalyzer with either local file or GCS path.
        
        Args:
            data_path: Either local file path or GCS path
            bucket_name: GCS bucket name (optional, for cloud storage)
            service_account_key: Path to GCP service account JSON (optional, for cloud storage)
        """
        self.setup_logging()
        
        # Determine if using cloud or local storage
        self.use_cloud = bucket_name is not None
        
        if self.use_cloud:
            # Set up GCS client
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_key
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(bucket_name)
            self.bucket_name = bucket_name
            self.data = self._load_data_from_gcs(data_path)
        else:
            # Load from local file
            self.data = self._load_data(data_path)
        
        self._preprocess_data()
    
    def _load_data(self, data_path):
        """Load data from local file."""
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            logging.error(f"Error loading data from file: {str(e)}")
            raise
    
    def _load_data_from_gcs(self, data_path):
        """Load data from Google Cloud Storage."""
        try:
            logging.info(f"Loading data from gs://{self.bucket_name}/{data_path}")
            blob = self.bucket.blob(data_path)
            content = blob.download_as_text()
            return pd.read_csv(io.StringIO(content))
        except Exception as e:
            logging.error(f"Error loading data from GCS: {str(e)}")
            raise
    
    def _save_file(self, content, filename, file_type='csv'):
        """Save file either locally or to GCS."""
        if self.use_cloud:
            # Save to GCS
            today = datetime.now().strftime('%Y-%m-%d')
            
            if file_type == 'png':
                blob_path = f'bias_analysis/{today}/figures/{filename}'
                blob = self.bucket.blob(blob_path)
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                blob.upload_from_string(buffer.getvalue(), content_type='image/png')
                logging.info(f"Saved to gs://{self.bucket_name}/{blob_path}")
            elif file_type == 'csv':
                blob_path = f'bias_analysis/{today}/data/{filename}'
                blob = self.bucket.blob(blob_path)
                csv_buffer = io.StringIO()
                content.to_csv(csv_buffer, index=False)
                blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
                logging.info(f"Saved to gs://{self.bucket_name}/{blob_path}")
        else:
            # Save locally
            if file_type == 'png':
                plt.savefig(filename)
            elif file_type == 'csv':
                content.to_csv(filename, index=False)
            logging.info(f"Saved locally: {filename}")
    
    def _preprocess_data(self):
        """Preprocess the data before analysis."""
        # Clean rent amount: remove $ and convert to numeric
        self.data['rent_amount'] = self.data['rent_amount'].replace(r'[\$,]', '', regex=True)
        self.data['rent_amount'] = pd.to_numeric(self.data['rent_amount'], errors='coerce')
        
        # Clean sensitive features
        for feature in ['gender', 'area', 'food_pref']:
            if feature in self.data.columns:
                self.data[feature] = self.data[feature].fillna('not_specified')
        
        logging.info("Data preprocessing completed successfully")
    
    def setup_logging(self):
        """Set up logging for bias analysis documentation."""
        logging.basicConfig(
            filename=f'bias_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def analyze_data_distribution(self, sensitive_features):
        """Analyze distribution of data across sensitive features."""
        logging.info("Starting data distribution analysis")
        distribution_stats = {}
        
        for feature in sensitive_features:
            distribution = self.data[feature].value_counts(normalize=True)
            distribution_stats[feature] = distribution
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            distribution.plot(kind='bar')
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
            
            # Save figure (locally or to GCS)
            self._save_file(None, f'distribution_{feature}.png', 'png')
            plt.close()
            
            # Log distribution statistics
            logging.info(f"\nDistribution for {feature}:\n{distribution}")
            
            # Check for severe imbalance
            if (distribution > 0.7).any():
                logging.warning(f"Severe imbalance detected in {feature}")
        
        return distribution_stats
    
    def slice_performance_analysis(self, feature, target_col):
        """Analyze model performance across different slices of data."""
        logging.info(f"Analyzing performance across {feature} slices")
        
        slice_metrics = {}
        for value in self.data[feature].unique():
            slice_data = self.data[self.data[feature] == value]
            
            # Calculate relevant metrics for each slice
            metrics = {
                'size': len(slice_data),
                'mean_target': slice_data[target_col].mean(),
                'std_target': slice_data[target_col].std()
            }
            slice_metrics[value] = metrics
            
            logging.info(f"\nMetrics for {feature}={value}:\n{metrics}")
        
        return slice_metrics
    
    def detect_bias(self, sensitive_features, target_col):
        """Detect bias across sensitive features."""
        logging.info("Starting bias detection")
        bias_metrics = {}
        
        for feature in sensitive_features:
            # Calculate demographic parity
            feature_values = self.data[feature].unique()
            base_rate = self.data[target_col].mean()
            
            feature_rates = {}
            for value in feature_values:
                slice_rate = self.data[self.data[feature] == value][target_col].mean()
                disparity = abs(slice_rate - base_rate)
                feature_rates[value] = {
                    'rate': slice_rate,
                    'disparity': disparity
                }
            
            bias_metrics[feature] = feature_rates
            
            # Log findings
            logging.info(f"\nBias metrics for {feature}:\n{feature_rates}")
            
            # Visualize disparities
            plt.figure(figsize=(10, 6))
            disparities = [v['disparity'] for v in feature_rates.values()]
            plt.bar(feature_rates.keys(), disparities)
            plt.title(f'Disparity Analysis for {feature}')
            plt.tight_layout()
            
            # Save figure
            self._save_file(None, f'disparity_{feature}.png', 'png')
            plt.close()
        
        return bias_metrics
    
    def mitigate_bias(self, sensitive_features, target_col):
        """Implement bias mitigation strategies."""
        logging.info("Starting bias mitigation")
        
        # 1. Resampling strategy
        for feature in sensitive_features:
            feature_dist = self.data[feature].value_counts()
            max_size = feature_dist.max()
            
            balanced_data = pd.DataFrame()
            for value in feature_dist.index:
                slice_data = self.data[self.data[feature] == value]
                # Oversample minority classes
                if len(slice_data) < max_size:
                    resampled = slice_data.sample(n=max_size, replace=True)
                    balanced_data = pd.concat([balanced_data, resampled])
                else:
                    balanced_data = pd.concat([balanced_data, slice_data])
            
            self.data = balanced_data
            logging.info(f"Resampled data for {feature}")
        
        # Save mitigated data
        timestamp = datetime.now().strftime('%H%M%S')
        self._save_file(self.data, f'mitigated_data_{timestamp}.csv', 'csv')
        
        return self.data
    
    def generate_report(self):
        """Generate comprehensive bias analysis report."""
        logging.info("Generating final report")
        
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_size': len(self.data),
            'features_analyzed': list(self.data.columns),
            'bias_detection_summary': {},
            'mitigation_actions': {},
            'recommendations': []
        }
        
        # Add visualizations and metrics to report
        logging.info("Report generated successfully")
        
        return report


def main():
    # Configuration - can be either local or cloud
    USE_CLOUD = True  # Set to False for local file processing
    
    if USE_CLOUD:
        # Cloud configuration
        BUCKET_NAME = "homiehubbucket"
        SERVICE_ACCOUNT_KEY = "./GCP_Account_Key.json"
        DATA_PATH = "cleaned/2024-11-03/homiehub_listings.csv"  # Path within bucket
        
        analyzer = BiasAnalyzer(
            data_path=DATA_PATH,
            bucket_name=BUCKET_NAME,
            service_account_key=SERVICE_ACCOUNT_KEY
        )
    else:
        # Local configuration
        DATA_PATH = "../working_data/homiehub_listings.csv"
        
        analyzer = BiasAnalyzer(data_path=DATA_PATH)
    
    try:
        # Define sensitive features
        sensitive_features = ['gender', 'area', 'food_pref']
        target_col = 'rent_amount'
        
        # Run analysis
        distribution_stats = analyzer.analyze_data_distribution(sensitive_features)
        logging.info("Data distribution analysis completed")
        
        for feature in sensitive_features:
            slice_metrics = analyzer.slice_performance_analysis(feature, target_col)
            logging.info(f"Slice analysis completed for {feature}")
        
        bias_metrics = analyzer.detect_bias(sensitive_features, target_col)
        logging.info("Bias detection completed")
        
        # Mitigate bias if necessary
        if any([any(v['disparity'] > 0.1 for v in m.values()) for m in bias_metrics.values()]):
            mitigated_data = analyzer.mitigate_bias(sensitive_features, target_col)
            logging.info("Bias mitigation completed")
        
        # Generate final report
        report = analyzer.generate_report()
        logging.info("Report generation completed")
        
    except Exception as e:
        logging.error(f"Error in bias analysis pipeline: {str(e)}")
        raise
    else:
        logging.info("Bias analysis completed successfully")


if __name__ == "__main__":
    main()