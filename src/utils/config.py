"""
Configuration management for the crypto price prediction pipeline.

This module provides centralized configuration handling with environment variable support.
All pipeline settings, paths, and parameters are defined here for easy maintenance.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

# Define project directory structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Create necessary directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA SOURCE CONFIGURATION
# =============================================================================

# Kaggle dataset settings
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "sudalairajkumar/cryptocurrencypricehistory")
CRYPTO_SYMBOL = os.getenv("CRYPTO_SYMBOL", "BTC")
CRYPTO_FILE = os.getenv("CRYPTO_FILE", "coin_Bitcoin.csv")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model metadata
MODEL_NAME = os.getenv("MODEL_NAME", "CryptoPricePredictor")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")

# Random Forest hyperparameters
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

# Experiment tracking settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "crypto_price_prediction")

# =============================================================================
# API CONFIGURATION
# =============================================================================

# FastAPI server settings
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

# Data split ratios (must sum to 1.0)
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.6"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.2"))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.2"))

# Cross-validation settings
TIME_SERIES_SPLITS = 5  # Number of splits for time series CV

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Rolling window sizes for technical indicators
ROLLING_WINDOWS = [5, 10, 20]

# Feature categories for organization
FEATURE_GROUPS = ['price', 'volume', 'marketcap', 'time']

# Technical indicators to calculate
TECHNICAL_INDICATORS = [
    'sma_5', 'sma_10', 'sma_20',      # Simple Moving Averages
    'ema_5', 'ema_10', 'ema_20',      # Exponential Moving Averages
    'rsi_14', 'macd', 'bollinger_bands'  # Advanced indicators
]

# =============================================================================
# DATA VALIDATION PARAMETERS
# =============================================================================

# Price validation bounds (in USD)
MAX_PRICE = 100000  # Maximum reasonable Bitcoin price
MIN_PRICE = 0.01    # Minimum reasonable Bitcoin price

# Volume validation bounds
MAX_VOLUME = 1e12   # Maximum reasonable volume
MIN_VOLUME = 0      # Minimum volume

# =============================================================================
# MODEL DEPLOYMENT PARAMETERS
# =============================================================================

CONFIDENCE_THRESHOLD = 0.8  # Minimum confidence for predictions

# =============================================================================
# FILE PATH CONFIGURATION
# =============================================================================

# Data storage paths
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
MODEL_ARTIFACTS_PATH = MODELS_DIR / MODEL_NAME

# Create all necessary directories
RAW_DATA_PATH.mkdir(exist_ok=True)
PROCESSED_DATA_PATH.mkdir(exist_ok=True)
MODEL_ARTIFACTS_PATH.mkdir(exist_ok=True)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_path(version: str = None) -> Path:
    """
    Get the file path for a specific model version.
    
    Args:
        version: Model version string. If None, uses default MODEL_VERSION.
        
    Returns:
        Path object pointing to the model directory.
    """
    if version is None:
        version = MODEL_VERSION
    return MODEL_ARTIFACTS_PATH / version

def get_data_path(data_type: str = "raw") -> Path:
    """
    Get the file path for data storage.
    
    Args:
        data_type: Type of data ('raw' or 'processed').
        
    Returns:
        Path object for the specified data type.
        
    Raises:
        ValueError: If data_type is not recognized.
    """
    if data_type == "raw":
        return RAW_DATA_PATH
    elif data_type == "processed":
        return PROCESSED_DATA_PATH
    else:
        raise ValueError(f"Unknown data type: {data_type}. Use 'raw' or 'processed'.")

def validate_config() -> bool:
    """
    Validate all configuration parameters for consistency.
    
    Performs checks on:
    - Data splits sum to 1.0
    - All splits are positive
    - Model parameters are valid
    
    Returns:
        True if all validations pass, False otherwise.
    """
    try:
        # Validate data splits sum to 1.0
        total_split = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
        if abs(total_split - 1.0) > 0.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Validate all splits are positive
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

# =============================================================================
# CONFIGURATION TESTING
# =============================================================================

if __name__ == "__main__":
    """Test configuration when run as a script."""
    print("=" * 50)
    print("CONFIGURATION VALIDATION")
    print("=" * 50)
    
    print(f"Configuration valid: {validate_config()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Model name: {MODEL_NAME}")
    print(f"API endpoint: {API_HOST}:{API_PORT}")
    print(f"Data splits: Train={TRAIN_SPLIT}, Val={VAL_SPLIT}, Test={TEST_SPLIT}")
    print("=" * 50) 