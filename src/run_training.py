#!/usr/bin/env python3
"""
Training script for cryptocurrency price prediction pipeline.
Runs the complete ML pipeline including data processing, model training, and MLflow logging.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from utils.config import (
    validate_config, LOG_LEVEL, LOG_FORMAT, MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME, MODEL_NAME, MODEL_VERSION
)
from training.data import DataPipeline
from training.model import ModelTrainer, ModelEvaluator, ModelPersistence

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    logger.info("Starting training pipeline for cryptocurrency price prediction")
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    try:
        # Initialize data pipeline
        logger.info("Initializing data pipeline")
        data_pipeline = DataPipeline(
            add_technical_indicators=True,
            scale_features=True
        )
        
        # Run data processing pipeline
        logger.info("Running data processing pipeline")
        data_summary = data_pipeline.run_pipeline(target_column='Close')
        
        # Get processed data
        X_train, y_train, X_val, y_val, X_test, y_test = data_pipeline.get_processed_data()
        feature_names = data_pipeline.get_feature_names()
        
        # Initialize model trainer
        logger.info("Initializing model trainer")
        model_trainer = ModelTrainer(
            model_type='random_forest',
            n_estimators=100,
            max_depth=10
        )
        
        # Start MLflow run
        with mlflow.start_run() as run:
            logger.info(f"MLflow run started: {run.info.run_id}")
            
            # Log data summary
            mlflow.log_dict(data_summary, "data_summary.json")
            
            # Log feature names
            mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")
            
            # Train model
            logger.info("Training model")
            training_summary = model_trainer.train_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                perform_cv=True,
                perform_grid_search=False  # Set to True for hyperparameter tuning
            )
            
            # Log training parameters
            mlflow.log_params(training_summary['model_params'])
            if training_summary['best_params']:
                mlflow.log_params(training_summary['best_params'])
            
            # Log cross-validation metrics
            cv_scores = training_summary['cv_scores']
            mlflow.log_metric("cv_mae_mean", cv_scores['mae']['mean'])
            mlflow.log_metric("cv_mae_std", cv_scores['mae']['std'])
            mlflow.log_metric("cv_rmse_mean", cv_scores['rmse']['mean'])
            mlflow.log_metric("cv_rmse_std", cv_scores['rmse']['std'])
            mlflow.log_metric("cv_r2_mean", cv_scores['r2']['mean'])
            mlflow.log_metric("cv_r2_std", cv_scores['r2']['std'])
            
            # Log validation metrics
            val_metrics = training_summary['val_metrics']
            mlflow.log_metric("val_mae", val_metrics['mae'])
            mlflow.log_metric("val_rmse", val_metrics['rmse'])
            mlflow.log_metric("val_r2", val_metrics['r2'])
            mlflow.log_metric("val_mape", val_metrics['mape'])
            
            # Evaluate model comprehensively
            logger.info("Evaluating model")
            evaluator = ModelEvaluator(model_trainer)
            evaluation_results = evaluator.evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test
            )
            
            # Log test metrics
            test_metrics = evaluation_results['test']
            mlflow.log_metric("test_mae", test_metrics['mae'])
            mlflow.log_metric("test_rmse", test_metrics['rmse'])
            mlflow.log_metric("test_r2", test_metrics['r2'])
            mlflow.log_metric("test_mape", test_metrics['mape'])
            
            # Log feature importance
            if training_summary['feature_importance']:
                feature_importance_df = model_trainer.get_feature_importance()
                feature_importance_df.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
            
            # Log evaluation results
            mlflow.log_dict(evaluation_results, "evaluation_results.json")
            
            # Create and log the full pipeline
            from sklearn.pipeline import Pipeline
            from training.features import FeatureEngineer
            from training.data import DataPreprocessor
            
            # Create preprocessing pipeline
            feature_engineer = FeatureEngineer(add_technical_indicators=True)
            preprocessor = DataPreprocessor(scale_features=True)
            
            # Fit preprocessor on training data
            preprocessor.fit(X_train)
            
            # Create full pipeline
            full_pipeline = Pipeline([
                ('feature_engineering', feature_engineer),
                ('preprocessing', preprocessor),
                ('model', model_trainer.model)
            ])
            
            # Log the full pipeline
            mlflow.sklearn.log_model(
                full_pipeline,
                "CryptoPricePredictor_FullPipeline",
                signature=infer_signature(X_train.head(5), full_pipeline.predict(X_train.head(5))),
                input_example=X_train.head(1),
                registered_model_name="CryptoPricePredictor_FullPipeline"
            )
            
            # Log just the model
            mlflow.sklearn.log_model(
                model_trainer.model,
                "CryptoPricePredictor",
                signature=infer_signature(X_train, model_trainer.predict(X_train)),
                input_example=X_train.head(1),
                registered_model_name="CryptoPricePredictor"
            )
            
            # Save model locally
            logger.info("Saving model locally")
            model_path = ModelPersistence.save_model(model_trainer, evaluation_results)
            
            # Print evaluation summary
            evaluator.print_evaluation_summary()
            
            # Print training summary
            print("\n" + "="*60)
            print("TRAINING PIPELINE SUMMARY")
            print("="*60)
            
            print(f"\nModel Information:")
            print(f"  Model Type: {training_summary['model_type']}")
            print(f"  Model Version: {MODEL_VERSION}")
            print(f"  MLflow Run ID: {run.info.run_id}")
            print(f"  Model Path: {model_path}")
            
            print(f"\nCross-Validation Results:")
            print(f"  MAE: {cv_scores['mae']['mean']:.4f} (+/- {cv_scores['mae']['std'] * 2:.4f})")
            print(f"  RMSE: {cv_scores['rmse']['mean']:.4f} (+/- {cv_scores['rmse']['std'] * 2:.4f})")
            print(f"  R²: {cv_scores['r2']['mean']:.4f} (+/- {cv_scores['r2']['std'] * 2:.4f})")
            
            print(f"\nValidation Results:")
            print(f"  MAE: ${val_metrics['mae']:,.2f}")
            print(f"  RMSE: ${val_metrics['rmse']:,.2f}")
            print(f"  R²: {val_metrics['r2']:.4f}")
            print(f"  MAPE: {val_metrics['mape']:.2f}%")
            
            print(f"\nTest Results:")
            print(f"  MAE: ${test_metrics['mae']:,.2f}")
            print(f"  RMSE: ${test_metrics['rmse']:,.2f}")
            print(f"  R²: {test_metrics['r2']:.4f}")
            print(f"  MAPE: {test_metrics['mape']:.2f}%")
            
            print("="*60)
            
            logger.info("Training pipeline completed successfully")
            
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 