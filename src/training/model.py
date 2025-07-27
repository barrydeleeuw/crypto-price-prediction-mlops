"""
Model training and evaluation for cryptocurrency price prediction.
Comprehensive model training with cross-validation and evaluation metrics.
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

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Model trainer for cryptocurrency price prediction.
    Handles model training, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 random_state: int = None,
                 n_estimators: int = None,
                 max_depth: int = None):
        """
        Initialize the model trainer.
        
        Args:
            model_type: Type of model to train
            random_state: Random state for reproducibility
            n_estimators: Number of estimators for Random Forest
            max_depth: Maximum depth for Random Forest
        """
        self.model_type = model_type
        self.random_state = random_state or RANDOM_STATE
        self.n_estimators = n_estimators or N_ESTIMATORS
        self.max_depth = max_depth or MAX_DEPTH
        
        self.model = None
        self.best_params = None
        self.cv_scores = {}
        self.feature_importance = None
        
    def create_model(self) -> RandomForestRegressor:
        """
        Create the model instance.
        
        Returns:
            Model instance
        """
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=MIN_SAMPLES_SPLIT,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=self.random_state,
                n_jobs=-1
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
            X_val: Validation features
            y_val: Validation targets
            perform_cv: Whether to perform cross-validation
            perform_grid_search: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training")
        
        # Create model
        self.model = self.create_model()
        
        # Perform hyperparameter tuning if requested
        if perform_grid_search:
            logger.info("Performing hyperparameter tuning")
            self.model, self.best_params = self._perform_grid_search(X_train, y_train)
        
        # Perform cross-validation if requested
        if perform_cv:
            logger.info("Performing cross-validation")
            self.cv_scores = self._perform_cross_validation(X_train, y_train)
        
        # Train final model
        logger.info("Training final model")
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            logger.info("Evaluating on validation set")
            val_metrics = self._evaluate_model(X_val, y_val, 'validation')
        
        # Create training summary
        training_summary = {
            'model_type': self.model_type,
            'model_params': self.model.get_params(),
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'val_metrics': val_metrics,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None
        }
        
        logger.info("Model training complete")
        return training_summary
    
    def _perform_grid_search(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Tuple of (best_model, best_params)
        """
        # Define parameter grid
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
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _perform_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Perform time series cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with CV scores
        """
        # Create time series cross-validation
        tscv = TimeSeriesSplit(n_splits=TIME_SERIES_SPLITS)
        
        # Perform cross-validation with multiple metrics
        cv_scores = {}
        
        # MAE
        mae_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=tscv, scoring='neg_mean_absolute_error'
        )
        cv_scores['mae'] = {
            'scores': -mae_scores.tolist(),
            'mean': -mae_scores.mean(),
            'std': mae_scores.std()
        }
        
        # RMSE
        rmse_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=tscv, scoring='neg_mean_squared_error'
        )
        cv_scores['rmse'] = {
            'scores': np.sqrt(-rmse_scores).tolist(),
            'mean': np.sqrt(-rmse_scores.mean()),
            'std': np.sqrt(-rmse_scores).std()
        }
        
        # R²
        r2_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=tscv, scoring='r2'
        )
        cv_scores['r2'] = {
            'scores': r2_scores.tolist(),
            'mean': r2_scores.mean(),
            'std': r2_scores.std()
        }
        
        logger.info(f"CV MAE: {cv_scores['mae']['mean']:.4f} (+/- {cv_scores['mae']['std'] * 2:.4f})")
        logger.info(f"CV RMSE: {cv_scores['rmse']['mean']:.4f} (+/- {cv_scores['rmse']['std'] * 2:.4f})")
        logger.info(f"CV R²: {cv_scores['r2']['mean']:.4f} (+/- {cv_scores['r2']['std'] * 2:.4f})")
        
        return cv_scores
    
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> Dict:
        """
        Evaluate the model on a dataset.
        
        Args:
            X: Features
            y: Targets
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Calculate percentage error
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred.tolist(),
            'actuals': y.tolist()
        }
        
        logger.info(f"{dataset_name.title()} Metrics:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dataframe with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained to get feature importance")
        
        return self.feature_importance

class ModelEvaluator:
    """
    Model evaluator for comprehensive model assessment.
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
        Comprehensive model evaluation on all datasets.
        
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
        
        # Add cross-validation results
        self.evaluation_results['cross_validation'] = self.model_trainer.cv_scores
        
        # Create detailed comparison
        comparison_df = self._create_prediction_comparison(X_test, y_test)
        self.evaluation_results['detailed_comparison'] = comparison_df.to_dict('records')
        
        # Add feature importance
        if self.model_trainer.feature_importance is not None:
            self.evaluation_results['feature_importance'] = self.model_trainer.feature_importance.to_dict('records')
        
        logger.info("Model evaluation complete")
        return self.evaluation_results
    
    def _create_prediction_comparison(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Create detailed prediction comparison.
        
        Args:
            X: Features
            y: Actual targets
            
        Returns:
            Dataframe with prediction comparison
        """
        y_pred = self.model_trainer.predict(X)
        
        comparison_df = pd.DataFrame({
            'actual': y,
            'predicted': y_pred,
            'absolute_error': np.abs(y - y_pred),
            'percentage_error': np.abs((y - y_pred) / y) * 100
        })
        
        return comparison_df
    
    def print_evaluation_summary(self):
        """Print a comprehensive evaluation summary."""
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        # Print metrics for each dataset
        for dataset, metrics in self.evaluation_results.items():
            if dataset in ['train', 'validation', 'test']:
                print(f"\n{dataset.upper()} SET:")
                print(f"  MAE: ${metrics['mae']:,.2f}")
                print(f"  RMSE: ${metrics['rmse']:,.2f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print(f"  MAPE: {metrics['mape']:.2f}%")
        
        # Print cross-validation results
        if 'cross_validation' in self.evaluation_results:
            cv = self.evaluation_results['cross_validation']
            print(f"\nCROSS-VALIDATION:")
            print(f"  MAE: {cv['mae']['mean']:.4f} (+/- {cv['mae']['std'] * 2:.4f})")
            print(f"  RMSE: {cv['rmse']['mean']:.4f} (+/- {cv['rmse']['std'] * 2:.4f})")
            print(f"  R²: {cv['r2']['mean']:.4f} (+/- {cv['r2']['std'] * 2:.4f})")
        
        # Print top features
        if 'feature_importance' in self.evaluation_results:
            print(f"\nTOP 10 FEATURES:")
            top_features = self.evaluation_results['feature_importance'][:10]
            for i, feature in enumerate(top_features, 1):
                print(f"  {i:2d}. {feature['feature']}: {feature['importance']:.4f}")
        
        print("="*60)

class ModelPersistence:
    """
    Model persistence utilities for saving and loading models.
    """
    
    @staticmethod
    def save_model(model_trainer: ModelTrainer, 
                  evaluation_results: Dict,
                  model_path: str = None) -> str:
        """
        Save the trained model and metadata.
        
        Args:
            model_trainer: Trained model trainer
            evaluation_results: Evaluation results
            model_path: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if model_path is None:
            model_path = MODEL_ARTIFACTS_PATH / MODEL_VERSION
        
        # Create directory if it doesn't exist
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        model_file = model_path / f"{MODEL_NAME}.pkl"
        joblib.dump(model_trainer.model, model_file)
        
        # Save evaluation results
        eval_file = model_path / "evaluation_results.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # Save feature importance
        if model_trainer.feature_importance is not None:
            importance_file = model_path / "feature_importance.csv"
            model_trainer.feature_importance.to_csv(importance_file, index=False)
        
        # Save model metadata
        metadata = {
            'model_name': MODEL_NAME,
            'model_version': MODEL_VERSION,
            'model_type': model_trainer.model_type,
            'model_params': model_trainer.model.get_params(),
            'training_date': pd.Timestamp.now().isoformat(),
            'feature_count': len(model_trainer.feature_importance) if model_trainer.feature_importance is not None else 0
        }
        
        metadata_file = model_path / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to: {model_path}")
        return str(model_path)
    
    @staticmethod
    def load_model(model_path: str) -> Tuple[RandomForestRegressor, Dict]:
        """
        Load a saved model and metadata.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Tuple of (model, metadata)
        """
        model_path = Path(model_path)
        
        # Load the model
        model_file = model_path / f"{MODEL_NAME}.pkl"
        model = joblib.load(model_file)
        
        # Load metadata
        metadata_file = model_path / "model_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model loaded from: {model_path}")
        return model, metadata 