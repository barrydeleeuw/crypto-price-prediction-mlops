"""
Data processing pipeline for cryptocurrency price prediction.
Handles data loading, validation, preprocessing, and splitting.
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

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for cryptocurrency price data from Kaggle."""
    
    def __init__(self, dataset_name: str = None, crypto_file: str = None):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Kaggle dataset name
            crypto_file: Specific crypto file to load
        """
        self.dataset_name = dataset_name or KAGGLE_DATASET
        self.crypto_file = crypto_file or CRYPTO_FILE
        self.validator = DataValidator()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load cryptocurrency data from Kaggle.
        
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading data from Kaggle dataset: {self.dataset_name}")
        
        try:
            # Download dataset from Kaggle
            dataset_path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Dataset downloaded to: {dataset_path}")
            
            # List available files
            files = os.listdir(dataset_path)
            logger.info(f"Available files: {files}")
            
            # Check if target file exists
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
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned and validated dataframe
        """
        logger.info("Starting data validation and cleaning")
        
        # Validate data
        validation_results = self.validator.validate_dataframe(df)
        self.validator.print_validation_report()
        
        if not validation_results.get('overall_valid', False):
            logger.warning("Data validation failed, but proceeding with cleaning")
        
        # Clean data
        df_clean = clean_dataframe(df)
        
        # Save cleaned data
        output_path = PROCESSED_DATA_PATH / f"cleaned_{self.crypto_file}"
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to: {output_path}")
        
        return df_clean

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Data preprocessing pipeline for cryptocurrency price data.
    Handles data type conversion, missing value imputation, and scaling.
    """
    
    def __init__(self, scale_features: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            scale_features: Whether to scale numerical features
        """
        self.scale_features = scale_features
        self.preprocessor = None
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """
        Fit the preprocessor on the training data.
        
        Args:
            X: Training features
            y: Target variable (unused)
            
        Returns:
            Self
        """
        logger.info("Fitting data preprocessor")
        
        # Prepare data for preprocessing
        X_processed = self._prepare_for_preprocessing(X)
        
        # Create preprocessing pipeline
        self.preprocessor = self._create_preprocessing_pipeline(X_processed)
        
        # Fit the preprocessor
        self.preprocessor.fit(X_processed)
        
        # Store feature names
        self.feature_names_ = X_processed.columns.tolist()
        
        logger.info(f"Preprocessor fitted with {len(self.feature_names_)} features")
        return self
    
    def transform(self, X):
        """
        Transform the input data.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        logger.info("Transforming data")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before transforming")
        
        # Prepare data for preprocessing
        X_processed = self._prepare_for_preprocessing(X)
        
        # Transform data
        X_transformed = self.preprocessor.transform(X_processed)
        
        # Convert back to dataframe
        if hasattr(X_transformed, 'toarray'):
            X_transformed = X_transformed.toarray()
        
        result_df = pd.DataFrame(X_transformed, columns=self.feature_names_)
        
        logger.info(f"Data transformation complete. Shape: {result_df.shape}")
        return result_df
    
    def _prepare_for_preprocessing(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for preprocessing by handling data types."""
        X_processed = X.copy()
        
        # Convert datetime columns to numeric features
        datetime_columns = X_processed.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            X_processed[f'{col}_year'] = X_processed[col].dt.year
            X_processed[f'{col}_month'] = X_processed[col].dt.month
            X_processed[f'{col}_day'] = X_processed[col].dt.day
            X_processed[f'{col}_dayofweek'] = X_processed[col].dt.dayofweek
            X_processed = X_processed.drop(col, axis=1)
        
        # Convert categorical columns to numeric
        categorical_columns = X_processed.select_dtypes(include=['object', 'string']).columns
        for col in categorical_columns:
            X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Ensure all columns are numeric
        numeric_columns = X_processed.select_dtypes(include=[np.number]).columns
        X_processed = X_processed[numeric_columns]
        
        return X_processed
    
    def _create_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """Create the preprocessing pipeline."""
        # Define numeric features
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create numeric transformer
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler() if self.scale_features else 'passthrough')
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='drop'
        )
        
        return preprocessor

class DataSplitter:
    """Data splitter for time series data."""
    
    def __init__(self, train_split: float = None, val_split: float = None, test_split: float = None):
        """
        Initialize the data splitter.
        
        Args:
            train_split: Training set proportion
            val_split: Validation set proportion
            test_split: Test set proportion
        """
        self.train_split = train_split or TRAIN_SPLIT
        self.val_split = val_split or VAL_SPLIT
        self.test_split = test_split or TEST_SPLIT
        
        # Validate splits
        total = self.train_split + self.val_split + self.test_split
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total}")
    
    def split_data(self, df: pd.DataFrame, target_column: str = 'Close') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into train, validation, and test sets respecting temporal order.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Splitting data into train/validation/test sets")
        
        # Sort by date to maintain temporal order
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date').reset_index(drop=True)
        else:
            df_sorted = df.copy()
        
        # Calculate split indices
        n_samples = len(df_sorted)
        train_end = int(n_samples * self.train_split)
        val_end = train_end + int(n_samples * self.val_split)
        
        # Split the data
        train_df = df_sorted[:train_end]
        val_df = df_sorted[train_end:val_end]
        test_df = df_sorted[val_end:]
        
        # Separate features and target
        X_train = train_df.drop(target_column, axis=1)
        y_train = train_df[target_column]
        X_val = val_df.drop(target_column, axis=1)
        y_val = val_df[target_column]
        X_test = test_df.drop(target_column, axis=1)
        y_test = test_df[target_column]
        
        logger.info(f"Data split complete:")
        logger.info(f"  Train set: {len(X_train)} samples")
        logger.info(f"  Validation set: {len(X_val)} samples")
        logger.info(f"  Test set: {len(X_test)} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test

class DataPipeline:
    """
    Complete data processing pipeline.
    Combines loading, validation, feature engineering, and preprocessing.
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
        self.loader = DataLoader(dataset_name, crypto_file)
        self.feature_engineer = FeatureEngineer(add_technical_indicators=add_technical_indicators)
        self.preprocessor = DataPreprocessor(scale_features=scale_features)
        self.splitter = DataSplitter()
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
    
    def run_pipeline(self, target_column: str = 'Close') -> Dict:
        """
        Run the complete data processing pipeline.
        
        Args:
            target_column: Name of the target column
            
        Returns:
            Dictionary with processed data and metadata
        """
        logger.info("Starting complete data processing pipeline")
        
        # Step 1: Load and validate data
        logger.info("Step 1: Loading and validating data")
        raw_data = self.loader.load_data()
        clean_data = self.loader.validate_and_clean(raw_data)
        
        # Step 2: Feature engineering
        logger.info("Step 2: Feature engineering")
        engineered_data = self.feature_engineer.transform(clean_data)
        
        # Step 3: Split data
        logger.info("Step 3: Splitting data")
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
            self.splitter.split_data(engineered_data, target_column)
        
        # Step 4: Preprocessing
        logger.info("Step 4: Preprocessing")
        self.preprocessor.fit(self.X_train)
        self.X_train_processed = self.preprocessor.transform(self.X_train)
        self.X_val_processed = self.preprocessor.transform(self.X_val)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Store feature names
        self.feature_names = self.preprocessor.feature_names_
        
        # Create summary
        summary = self._create_pipeline_summary(engineered_data)
        
        logger.info("Data processing pipeline complete")
        return summary
    
    def _create_pipeline_summary(self, engineered_data: pd.DataFrame) -> Dict:
        """Create a summary of the pipeline results."""
        from .features import create_feature_summary
        
        feature_summary = create_feature_summary(engineered_data)
        
        summary = {
            'data_info': {
                'original_shape': len(engineered_data),
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'total_features': len(self.feature_names)
            },
            'feature_summary': feature_summary,
            'target_info': {
                'train_mean': self.y_train.mean(),
                'train_std': self.y_train.std(),
                'val_mean': self.y_val.mean(),
                'val_std': self.y_val.std(),
                'test_mean': self.y_test.mean(),
                'test_std': self.y_test.std()
            }
        }
        
        return summary
    
    def get_processed_data(self) -> Tuple:
        """Get the processed data for training."""
        return (
            self.X_train_processed, self.y_train,
            self.X_val_processed, self.y_val,
            self.X_test_processed, self.y_test
        )
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names."""
        return self.feature_names 