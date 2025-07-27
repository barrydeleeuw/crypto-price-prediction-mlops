"""
Configuration management for the crypto price prediction pipeline.
Centralized configuration handling with environment variable support.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Kaggle Dataset Configuration
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "sudalairajkumar/cryptocurrencypricehistory")
CRYPTO_SYMBOL = os.getenv("CRYPTO_SYMBOL", "BTC")
CRYPTO_FILE = os.getenv("CRYPTO_FILE", "coin_Bitcoin.csv")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "CryptoPricePredictor")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "crypto_price_prediction")

# API Configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Data Configuration
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.6"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.2"))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.2"))

# Feature Engineering Configuration
ROLLING_WINDOWS = [5, 10, 20]  # Default rolling windows for technical indicators
FEATURE_GROUPS = ['price', 'volume', 'marketcap', 'time']

# Model Parameters
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2

# Data Processing Parameters
TIME_SERIES_SPLITS = 5  # For cross-validation
CONFIDENCE_THRESHOLD = 0.8

# File paths
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
MODEL_ARTIFACTS_PATH = MODELS_DIR / MODEL_NAME

# Ensure all paths exist
RAW_DATA_PATH.mkdir(exist_ok=True)
PROCESSED_DATA_PATH.mkdir(exist_ok=True)
MODEL_ARTIFACTS_PATH.mkdir(exist_ok=True)

# Data validation parameters
MAX_PRICE = 100000  # Maximum reasonable Bitcoin price
MIN_PRICE = 0.01    # Minimum reasonable Bitcoin price
MAX_VOLUME = 1e12   # Maximum reasonable volume
MIN_VOLUME = 0      # Minimum volume

# Feature engineering parameters
TECHNICAL_INDICATORS = [
    'sma_5', 'sma_10', 'sma_20',
    'ema_5', 'ema_10', 'ema_20',
    'rsi_14', 'macd', 'bollinger_bands'
]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def get_model_path(version: str = None) -> Path:
    """Get the path for a specific model version."""
    if version is None:
        version = MODEL_VERSION
    return MODEL_ARTIFACTS_PATH / version

def get_data_path(data_type: str = "raw") -> Path:
    """Get the path for data files."""
    if data_type == "raw":
        return RAW_DATA_PATH
    elif data_type == "processed":
        return PROCESSED_DATA_PATH
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def validate_config() -> bool:
    """Validate the configuration parameters."""
    try:
        # Validate splits sum to 1
        total_split = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
        if abs(total_split - 1.0) > 0.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Validate positive values
        if any(split <= 0 for split in [TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT]):
            raise ValueError("All data splits must be positive")
        
        # Validate model parameters
        if N_ESTIMATORS <= 0:
            raise ValueError("N_ESTIMATORS must be positive")
        
        if MAX_DEPTH <= 0:
            raise ValueError("MAX_DEPTH must be positive")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    print("Configuration validation:", validate_config())
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Model name: {MODEL_NAME}")
    print(f"API host: {API_HOST}:{API_PORT}") 