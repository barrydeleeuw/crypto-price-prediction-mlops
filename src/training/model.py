"""
Model training and evaluation for cryptocurrency price prediction.

This module provides comprehensive model training capabilities including:
- Model creation and training
- Cross-validation with time series splits
- Hyperparameter tuning with grid search
- Model evaluation and metrics calculation
- Model persistence and loading
- Feature importance analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path

from ..utils.config import (
    RANDOM_STATE, N_ESTIMATORS, MAX_DEPTH, MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF,
    TIME_SERIES_SPLITS, MODEL_ARTIFACTS_PATH, MODEL_NAME, MODEL_VERSION
)
from .features import get_feature_importance

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL TRAINER CLASS
# =============================================================================

class ModelTrainer:
    """
    Model trainer for cryptocurrency price prediction.
    
    This class handles the complete model training workflow including:
    - Model creation and configuration
    - Training with optional cross-validation
    - Hyperparameter tuning with grid search
    - Model evaluation and metrics calculation
    - Feature importance analysis
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 random_state: int = None,
                 n_estimators: int = None,
                 max_depth: int = None):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train ('random_forest' supported)
            random_state: Random state for reproducibility
            n_estimators: Number of estimators for Random Forest
            max_depth: Maximum depth for Random Forest
        """
        self.model_type = model_type
        self.random_state = random_state or RANDOM_STATE
        self.n_estimators = n_estimators or N_ESTIMATORS
        self.max_depth = max_depth or MAX_DEPTH
        
        # Model and training state
        self.model = None
        self.best_params = None
        self.cv_scores = {}
        self.feature_importance = None
        
    def create_model(self) -> RandomForestRegressor:
        """
        Create the model instance with specified parameters.
        
        Returns:
            Configured model instance
            
        Raises:
            ValueError: If model_type is not supported
        """
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=MIN_SAMPLES_SPLIT,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=self.random_state,
                n_jobs=-1  # Use all available CPU cores
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def train_model(self, 
                   X_train: pd.DataFrame, 
                   y_train: pd.Series,
                   X_val: pd.DataFrame = None,
                   y_val: pd.Series = None,
                   perform_cv: bool = True,
                   perform_grid_search: bool = False) -> Dict:
        """
        Train the model with optional cross-validation and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            perform_cv: Whether to perform cross-validation
            perform_grid_search: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with comprehensive training results
        """
        logger.info("Starting model training process")
        
        # =====================================================================
        # STEP 1: CREATE MODEL
        # =====================================================================
        
        self.model = self.create_model()
        logger.info(f"Created {self.model_type} model with parameters: {self.model.get_params()}")
        
        # =====================================================================
        # STEP 2: HYPERPARAMETER TUNING (OPTIONAL)
        # =====================================================================
        
        if perform_grid_search:
            logger.info("Performing hyperparameter tuning with grid search")
            best_model, best_params = self._perform_grid_search(X_train, y_train)
            self.model = best_model
            self.best_params = best_params
            logger.info(f"Best parameters found: {best_params}")
        
        # =====================================================================
        # STEP 3: CROSS-VALIDATION (OPTIONAL)
        # =====================================================================
        
        if perform_cv:
            logger.info("Performing cross-validation")
            self.cv_scores = self._perform_cross_validation(X_train, y_train)
            logger.info(f"Cross-validation scores: {self.cv_scores}")
        
        # =====================================================================
        # STEP 4: MODEL TRAINING
        # =====================================================================
        
        logger.info("Training final model on full training set")
        self.model.fit(X_train, y_train)
        
        # =====================================================================
        # STEP 5: FEATURE IMPORTANCE
        # =====================================================================
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = get_feature_importance(self.model, X_train.columns)
            logger.info("Feature importance calculated")
        
        # =====================================================================
        # STEP 6: VALIDATION EVALUATION (IF PROVIDED)
        # =====================================================================
        
        validation_results = {}
        if X_val is not None and y_val is not None:
            logger.info("Evaluating model on validation set")
            validation_results = self._evaluate_model(X_val, y_val, 'validation')
        
        # =====================================================================
        # STEP 7: CREATE TRAINING SUMMARY
        # =====================================================================
        
        training_summary = {
            'model_type': self.model_type,
            'model_params': self.model.get_params(),
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'validation_results': validation_results,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
        }
        
        logger.info("Model training completed successfully")
        return training_summary
    
    def _perform_grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[RandomForestRegressor, Dict]:
        """
        Perform hyperparameter tuning using grid search with time series cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Tuple of (best model, best parameters)
        """
        # Define parameter grid for Random Forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create base model
        base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _perform_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with cross-validation scores
        """
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
        
        # Perform cross-validation with multiple metrics
        cv_results = {}
        
        # Mean Squared Error
        mse_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
        cv_results['mse'] = -mse_scores  # Convert back to positive
        cv_results['mse_mean'] = -mse_scores.mean()
        cv_results['mse_std'] = mse_scores.std()
        
        # Mean Absolute Error
        mae_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
        cv_results['mae'] = -mae_scores  # Convert back to positive
        cv_results['mae_mean'] = -mae_scores.mean()
        cv_results['mae_std'] = mae_scores.std()
        
        # R-squared
        r2_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='r2')
        cv_results['r2'] = r2_scores
        cv_results['r2_mean'] = r2_scores.mean()
        cv_results['r2_std'] = r2_scores.std()
        
        return cv_results
    
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict:
        """
        Evaluate model performance on a specific dataset.
        
        Args:
            X: Features
            y: Targets
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y - y_pred) / y)) * 100  # Mean Absolute Percentage Error
        
        results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred.tolist(),
            'actuals': y.tolist()
        }
        
        logger.info(f"{dataset_name.capitalize()} metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dataframe with feature importance scores
        """
        if self.feature_importance is not None:
            return self.feature_importance.copy()
        else:
            logger.warning("Feature importance not available. Train the model first.")
            return pd.DataFrame()

# =============================================================================
# MODEL EVALUATOR CLASS
# =============================================================================

class ModelEvaluator:
    """
    Model evaluator for comprehensive performance assessment.
    
    This class provides detailed evaluation of model performance across
    multiple datasets (train, validation, test) with comprehensive metrics.
    """
    
    def __init__(self, model_trainer: ModelTrainer):
        """
        Initialize the model evaluator.
        
        Args:
            model_trainer: Trained model trainer instance
        """
        self.model_trainer = model_trainer
        self.evaluation_results = {}
    
    def evaluate_model(self, 
                      X_train: pd.DataFrame, 
                      y_train: pd.Series,
                      X_val: pd.DataFrame, 
                      y_val: pd.Series,
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on all datasets.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation")
        
        # Evaluate on all datasets
        self.evaluation_results['train'] = self.model_trainer._evaluate_model(X_train, y_train, 'train')
        self.evaluation_results['validation'] = self.model_trainer._evaluate_model(X_val, y_val, 'validation')
        self.evaluation_results['test'] = self.model_trainer._evaluate_model(X_test, y_test, 'test')
        
        # Create prediction comparisons
        self.evaluation_results['train_comparison'] = self._create_prediction_comparison(X_train, y_train)
        self.evaluation_results['validation_comparison'] = self._create_prediction_comparison(X_val, y_val)
        self.evaluation_results['test_comparison'] = self._create_prediction_comparison(X_test, y_test)
        
        # Calculate performance summary
        self.evaluation_results['summary'] = self._create_performance_summary()
        
        logger.info("Model evaluation completed")
        return self.evaluation_results
    
    def _create_prediction_comparison(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Create a comparison dataframe of actual vs predicted values.
        
        Args:
            X: Features
            y: Actual targets
            
        Returns:
            Dataframe with actual and predicted values
        """
        y_pred = self.model_trainer.predict(X)
        
        comparison_df = pd.DataFrame({
            'actual': y,
            'predicted': y_pred,
            'error': y - y_pred,
            'abs_error': np.abs(y - y_pred),
            'pct_error': np.abs((y - y_pred) / y) * 100
        })
        
        return comparison_df
    
    def _create_performance_summary(self) -> Dict:
        """
        Create a summary of model performance across all datasets.
        
        Returns:
            Dictionary with performance summary
        """
        summary = {}
        
        for dataset_name, results in self.evaluation_results.items():
            if dataset_name not in ['summary', 'train_comparison', 'validation_comparison', 'test_comparison']:
                summary[dataset_name] = {
                    'mae': results['mae'],
                    'rmse': results['rmse'],
                    'r2': results['r2'],
                    'mape': results['mape']
                }
        
        return summary
    
    def print_evaluation_summary(self):
        """Print a formatted evaluation summary to console."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Print metrics for each dataset
        for dataset_name, results in self.evaluation_results['summary'].items():
            print(f"\n{dataset_name.upper()} SET:")
            print(f"  MAE:  {results['mae']:.2f}")
            print(f"  RMSE: {results['rmse']:.2f}")
            print(f"  R²:   {results['r2']:.3f}")
            print(f"  MAPE: {results['mape']:.2f}%")
        
        # Print feature importance (top 10)
        feature_importance = self.model_trainer.get_feature_importance()
        if not feature_importance.empty:
            print(f"\nTOP 10 FEATURE IMPORTANCE:")
            top_features = feature_importance.head(10)
            for _, row in top_features.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        print("="*60)

# =============================================================================
# MODEL PERSISTENCE CLASS
# =============================================================================

class ModelPersistence:
    """
    Model persistence utilities for saving and loading trained models.
    
    This class handles model serialization, metadata storage, and model loading
    with comprehensive error handling and validation.
    """
    
    @staticmethod
    def save_model(model_trainer: ModelTrainer, 
                  evaluation_results: Dict,
                  model_path: str = None) -> str:
        """
        Save the trained model and associated metadata.
        
        Args:
            model_trainer: Trained model trainer instance
            evaluation_results: Model evaluation results
            model_path: Path to save the model (optional)
            
        Returns:
            Path where the model was saved
        """
        if model_trainer.model is None:
            raise ValueError("No trained model to save")
        
        # Create model directory
        if model_path is None:
            model_path = MODEL_ARTIFACTS_PATH / MODEL_VERSION
        else:
            model_path = Path(model_path)
        
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / "model.joblib"
        joblib.dump(model_trainer.model, model_file)
        logger.info(f"Model saved to: {model_file}")
        
        # Save metadata
        metadata = {
            'model_type': model_trainer.model_type,
            'model_params': model_trainer.model.get_params(),
            'best_params': model_trainer.best_params,
            'cv_scores': model_trainer.cv_scores,
            'evaluation_results': evaluation_results,
            'feature_importance': model_trainer.feature_importance.to_dict('records') if model_trainer.feature_importance is not None else None,
            'model_version': MODEL_VERSION,
            'save_timestamp': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to: {metadata_file}")
        
        # Save feature importance separately
        if model_trainer.feature_importance is not None:
            importance_file = model_path / "feature_importance.csv"
            model_trainer.feature_importance.to_csv(importance_file, index=False)
            logger.info(f"Feature importance saved to: {importance_file}")
        
        return str(model_path)
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[RandomForestRegressor, Dict]:
        """
        Load a trained model and its metadata.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Tuple of (loaded model, metadata)
        """
        model_path = Path(model_path)
        
        # Load model
        model_file = model_path / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model = joblib.load(model_file)
        logger.info(f"Model loaded from: {model_file}")
        
        # Load metadata
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded from: {metadata_file}")
        else:
            metadata = {}
            logger.warning(f"Metadata file not found: {metadata_file}")
        
        return model, metadata 