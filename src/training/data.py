"""
Data processing pipeline for cryptocurrency price prediction.

This module provides a comprehensive data processing pipeline including:
- Data loading from Kaggle datasets
- Data validation and cleaning
- Feature engineering and preprocessing
- Time series data splitting
- Pipeline orchestration
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple, Dict, Optional, List
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import kagglehub

from ..utils.config import (
    KAGGLE_DATASET, CRYPTO_FILE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    RAW_DATA_PATH, PROCESSED_DATA_PATH, RANDOM_STATE
)
from ..utils.data_validation import DataValidator, clean_dataframe
from .features import FeatureEngineer

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DATA LOADER CLASS
# =============================================================================

class DataLoader:
    """
    Data loader for cryptocurrency price data from Kaggle.
    
    This class handles downloading and loading cryptocurrency datasets from Kaggle,
    with built-in error handling and logging.
    """
    
    def __init__(self, dataset_name: str = None, crypto_file: str = None):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Kaggle dataset name (e.g., 'sudalairajkumar/cryptocurrencypricehistory')
            crypto_file: Specific crypto file to load (e.g., 'coin_Bitcoin.csv')
        """
        self.dataset_name = dataset_name or KAGGLE_DATASET
        self.crypto_file = crypto_file or CRYPTO_FILE
        self.validator = DataValidator()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load cryptocurrency data from Kaggle.
        
        Downloads the dataset from Kaggle and loads the specified cryptocurrency file.
        
        Returns:
            Loaded dataframe with cryptocurrency price data
            
        Raises:
            FileNotFoundError: If the specified crypto file is not found
            Exception: For other loading errors
        """
        logger.info(f"Loading data from Kaggle dataset: {self.dataset_name}")
        
        try:
            # Download dataset from Kaggle
            dataset_path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded to: {dataset_path}")
            
            # List available files in the dataset
            files = os.listdir(dataset_path)
            logger.info(f"Available files: {files}")
            
            # Verify target file exists
            if self.crypto_file not in files:
                raise FileNotFoundError(f"File {self.crypto_file} not found in dataset")
            
            # Load the specific crypto file
            file_path = os.path.join(dataset_path, self.crypto_file)
            df = pd.read_csv(file_path)
            
            logger.info(f"Successfully loaded {len(df)} records from {self.crypto_file}")
            logger.info(f"Data shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the loaded data.
        
        Performs comprehensive data validation and applies cleaning operations
        to ensure data quality for machine learning.
        
        Args:
            df: Raw dataframe from Kaggle
            
        Returns:
            Cleaned and validated dataframe ready for processing
        """
        logger.info("Starting data validation and cleaning")
        
        # Perform comprehensive validation
        validation_results = self.validator.validate_dataframe(df)
        self.validator.print_validation_report()
        
        # Proceed with cleaning even if validation fails (with warning)
        if not validation_results.get('overall_valid', False):
            logger.warning("Data validation failed, but proceeding with cleaning")
        
        # Apply data cleaning operations
        df_clean = clean_dataframe(df)
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean

# =============================================================================
# DATA PREPROCESSOR CLASS
# =============================================================================

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data preprocessor for machine learning pipeline.
    
    This transformer handles data type conversion, missing value imputation,
    and feature scaling for the machine learning pipeline.
    """
    
    def __init__(self, scale_features: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            scale_features: Whether to scale numeric features
        """
        self.scale_features = scale_features
        self.preprocessing_pipeline = None
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """
        Fit the preprocessor on the training data.
        
        Args:
            X: Training data
            y: Target values (not used)
            
        Returns:
            self: The fitted preprocessor
        """
        logger.info("Fitting data preprocessor")
        
        # Prepare data for preprocessing
        X_prepared = self._prepare_for_preprocessing(X)
        
        # Create and fit preprocessing pipeline
        self.preprocessing_pipeline = self._create_preprocessing_pipeline(X_prepared)
        self.preprocessing_pipeline.fit(X_prepared)
        
        # Store feature names
        self.feature_names_ = list(X_prepared.columns)
        
        logger.info(f"Preprocessor fitted with {len(self.feature_names_)} features")
        return self
    
    def transform(self, X):
        """
        Transform the input data using the fitted preprocessor.
        
        Args:
            X: Input data to transform
            
        Returns:
            Transformed data
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Prepare data for transformation
        X_prepared = self._prepare_for_preprocessing(X)
        
        # Apply preprocessing pipeline
        X_transformed = self.preprocessing_pipeline.transform(X_prepared)
        
        # Convert back to dataframe with feature names
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        result_df = pd.DataFrame(X_transformed, columns=self.feature_names_, index=X.index)
        
        logger.info(f"Data transformation complete. Shape: {result_df.shape}")
        return result_df
    
    def _prepare_for_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for preprocessing by converting types and handling missing values.
        
        Args:
            X: Input dataframe
            
        Returns:
            Prepared dataframe ready for preprocessing
        """
        df = X.copy()
        
        # Convert date columns to numeric (days since epoch)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Date'] = (df['Date'] - pd.Timestamp('1970-01-01')).dt.days
        
        # Convert categorical columns to numeric
        categorical_columns = ['Name', 'Symbol']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # Ensure all columns are numeric
        df = df.select_dtypes(include=[np.number])
        
        return df
    
    def _create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Create a preprocessing pipeline for the data.
        
        Args:
            X: Input dataframe
            
        Returns:
            Fitted preprocessing pipeline
        """
        # Define preprocessing steps
        preprocessing_steps = []
        
        # Add imputation step
        imputer = SimpleImputer(strategy='median')
        preprocessing_steps.append(('imputer', imputer))
        
        # Add scaling step if requested
        if self.scale_features:
            scaler = StandardScaler()
            preprocessing_steps.append(('scaler', scaler))
        
        # Create pipeline
        pipeline = Pipeline(preprocessing_steps)
        
        return pipeline

# =============================================================================
# DATA SPLITTER CLASS
# =============================================================================

class DataSplitter:
    """
    Time series data splitter for cryptocurrency price prediction.
    
    This class handles temporal data splitting while preserving the time order,
    which is crucial for time series prediction tasks.
    """
    
    def __init__(self, train_split: float = None, val_split: float = None, test_split: float = None):
        """
        Initialize the data splitter.
        
        Args:
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            test_split: Proportion of data for testing
        """
        self.train_split = train_split or TRAIN_SPLIT
        self.val_split = val_split or VAL_SPLIT
        self.test_split = test_split or TEST_SPLIT
        
        # Validate splits
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 0.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    def split_data(self, df: pd.DataFrame, target_column: str = 'Close') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into train, validation, and test sets while preserving temporal order.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Splitting data into train/validation/test sets")
        
        # Ensure data is sorted by date
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        
        # Calculate split indices
        total_rows = len(df)
        train_end = int(total_rows * self.train_split)
        val_end = int(total_rows * (self.train_split + self.val_split))
        
        # Split the data
        train_data = df.iloc[:train_end]
        val_data = df.iloc[train_end:val_end]
        test_data = df.iloc[val_end:]
        
        # Separate features and target
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        
        X_val = val_data.drop(columns=[target_column])
        y_val = val_data[target_column]
        
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        
        # Log split information
        logger.info(f"Data split complete:")
        logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/total_rows:.1%})")
        logger.info(f"  Validation: {len(X_val)} samples ({len(X_val)/total_rows:.1%})")
        logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/total_rows:.1%})")
        
        return X_train, y_train, X_val, y_val, X_test, y_test

# =============================================================================
# DATA PIPELINE CLASS
# =============================================================================

class DataPipeline:
    """
    Complete data processing pipeline for cryptocurrency price prediction.
    
    This class orchestrates the entire data processing workflow:
    1. Data loading from Kaggle
    2. Data validation and cleaning
    3. Feature engineering
    4. Data preprocessing
    5. Time series splitting
    """
    
    def __init__(self, 
                 dataset_name: str = None,
                 crypto_file: str = None,
                 add_technical_indicators: bool = True,
                 scale_features: bool = True):
        """
        Initialize the data pipeline.
        
        Args:
            dataset_name: Kaggle dataset name
            crypto_file: Specific crypto file to load
            add_technical_indicators: Whether to add technical indicators
            scale_features: Whether to scale features
        """
        self.data_loader = DataLoader(dataset_name, crypto_file)
        self.feature_engineer = FeatureEngineer(add_technical_indicators=add_technical_indicators)
        self.preprocessor = DataPreprocessor(scale_features=scale_features)
        self.data_splitter = DataSplitter()
        
        # Store processed data
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names = []
    
    def run_pipeline(self, target_column: str = 'Close') -> Dict:
        """
        Run the complete data processing pipeline.
        
        Args:
            target_column: Name of the target column for prediction
            
        Returns:
            Dictionary with pipeline summary information
        """
        logger.info("Starting complete data processing pipeline")
        
        # =====================================================================
        # STEP 1: LOAD AND VALIDATE DATA
        # =====================================================================
        
        logger.info("Step 1: Loading and validating data")
        raw_data = self.data_loader.load_data()
        clean_data = self.data_loader.validate_and_clean(raw_data)
        
        # =====================================================================
        # STEP 2: FEATURE ENGINEERING
        # =====================================================================
        
        logger.info("Step 2: Feature engineering")
        engineered_data = self.feature_engineer.transform(clean_data)
        
        # =====================================================================
        # STEP 3: DATA SPLITTING
        # =====================================================================
        
        logger.info("Step 3: Splitting data")
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
            self.data_splitter.split_data(engineered_data, target_column)
        
        # =====================================================================
        # STEP 4: DATA PREPROCESSING
        # =====================================================================
        
        logger.info("Step 4: Preprocessing data")
        self.preprocessor.fit(self.X_train)
        
        self.X_train = self.preprocessor.transform(self.X_train)
        self.X_val = self.preprocessor.transform(self.X_val)
        self.X_test = self.preprocessor.transform(self.X_test)
        
        # Store feature names
        self.feature_names = list(self.X_train.columns)
        
        # =====================================================================
        # STEP 5: CREATE SUMMARY
        # =====================================================================
        
        logger.info("Step 5: Creating pipeline summary")
        summary = self._create_pipeline_summary(engineered_data)
        
        logger.info("Data processing pipeline completed successfully")
        return summary
    
    def _create_pipeline_summary(self, engineered_data: pd.DataFrame) -> Dict:
        """
        Create a comprehensive summary of the data processing pipeline.
        
        Args:
            engineered_data: Data after feature engineering
            
        Returns:
            Dictionary with pipeline summary information
        """
        summary = {
            'pipeline_steps': {
                'data_loading': 'Completed',
                'data_validation': 'Completed',
                'feature_engineering': 'Completed',
                'data_splitting': 'Completed',
                'data_preprocessing': 'Completed'
            },
            'data_info': {
                'original_shape': len(engineered_data),
                'final_features': len(self.feature_names),
                'train_samples': len(self.X_train),
                'validation_samples': len(self.X_val),
                'test_samples': len(self.X_test)
            },
            'feature_info': {
                'total_features': len(self.feature_names),
                'feature_names': self.feature_names[:10] + ['...'] if len(self.feature_names) > 10 else self.feature_names
            }
        }
        
        return summary
    
    def get_processed_data(self) -> Tuple:
        """
        Get the processed data for model training.
        
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if self.X_train is None:
            raise ValueError("Pipeline must be run before getting processed data")
        
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names.copy() 