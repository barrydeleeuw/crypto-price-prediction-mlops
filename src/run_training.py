#!/usr/bin/env python3
"""
Training script for cryptocurrency price prediction pipeline.

This script orchestrates the complete machine learning workflow:
1. Data processing and feature engineering
2. Model training with cross-validation
3. Model evaluation and performance assessment
4. MLflow experiment tracking and model logging
5. Model persistence and metadata storage

Usage:
    python src/run_training.py
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

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
from training.features import FeatureEngineer

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def print_training_summary(training_summary: dict, evaluation_results: dict):
    """
    Print a formatted summary of the training results.
    
    Args:
        training_summary: Dictionary containing training summary
        evaluation_results: Dictionary containing evaluation results
    """
    print("\n" + "="*70)
    print("MODEL TRAINING SUMMARY")
    print("="*70)
    
    # Print model information
    print("\n🤖 MODEL INFORMATION:")
    print(f"  📊 Model Type: {training_summary.get('model_type', 'Unknown')}")
    print(f"  🔧 Parameters: {len(training_summary.get('model_params', {}))} parameters")
    
    # Print cross-validation results
    cv_scores = training_summary.get('cv_scores', {})
    if cv_scores:
        print(f"\n📈 CROSS-VALIDATION RESULTS:")
        print(f"  📊 MAE: {cv_scores.get('mae_mean', 0):.2f} (±{cv_scores.get('mae_std', 0):.2f})")
        print(f"  📊 RMSE: {cv_scores.get('mse_mean', 0):.2f} (±{cv_scores.get('mse_std', 0):.2f})")
        print(f"  📊 R²: {cv_scores.get('r2_mean', 0):.3f} (±{cv_scores.get('r2_std', 0):.3f})")
    
    # Print evaluation results
    if evaluation_results and 'summary' in evaluation_results:
        print(f"\n🎯 MODEL PERFORMANCE:")
        for dataset_name, metrics in evaluation_results['summary'].items():
            print(f"  📊 {dataset_name.upper()} SET:")
            print(f"    MAE: {metrics.get('mae', 0):.2f}")
            print(f"    RMSE: {metrics.get('rmse', 0):.2f}")
            print(f"    R²: {metrics.get('r2', 0):.3f}")
            print(f"    MAPE: {metrics.get('mape', 0):.2f}%")
    
    # Print feature importance (top 5)
    feature_importance = training_summary.get('feature_importance', [])
    if feature_importance:
        print(f"\n🔍 TOP 5 FEATURE IMPORTANCE:")
        for i, feature in enumerate(feature_importance[:5], 1):
            print(f"  {i}. {feature.get('feature', 'Unknown')}: {feature.get('importance', 0):.4f}")
    
    print("="*70)

def log_mlflow_artifacts(mlflow_run, training_summary: dict, evaluation_results: dict, 
                        data_summary: dict, feature_names: list):
    """
    Log all artifacts and metrics to MLflow.
    
    Args:
        mlflow_run: Active MLflow run
        training_summary: Training summary dictionary
        evaluation_results: Evaluation results dictionary
        data_summary: Data processing summary
        feature_names: List of feature names
    """
    logger.info("Logging artifacts and metrics to MLflow")
    
    # Log data summary
    mlflow.log_dict(data_summary, "data_summary.json")
    mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")
    
    # Log training parameters
    mlflow.log_params(training_summary['model_params'])
    if training_summary.get('best_params'):
        mlflow.log_params(training_summary['best_params'])
    
    # Log cross-validation metrics
    cv_scores = training_summary.get('cv_scores', {})
    if cv_scores:
        mlflow.log_metric("cv_mae_mean", cv_scores.get('mae_mean', 0))
        mlflow.log_metric("cv_mae_std", cv_scores.get('mae_std', 0))
        mlflow.log_metric("cv_mse_mean", cv_scores.get('mse_mean', 0))
        mlflow.log_metric("cv_mse_std", cv_scores.get('mse_std', 0))
        mlflow.log_metric("cv_r2_mean", cv_scores.get('r2_mean', 0))
        mlflow.log_metric("cv_r2_std", cv_scores.get('r2_std', 0))
    
    # Log evaluation metrics
    if evaluation_results and 'summary' in evaluation_results:
        for dataset_name, metrics in evaluation_results['summary'].items():
            prefix = f"{dataset_name}_"
            mlflow.log_metric(f"{prefix}mae", metrics.get('mae', 0))
            mlflow.log_metric(f"{prefix}rmse", metrics.get('rmse', 0))
            mlflow.log_metric(f"{prefix}r2", metrics.get('r2', 0))
            mlflow.log_metric(f"{prefix}mape", metrics.get('mape', 0))
    
    # Log feature importance
    if training_summary.get('feature_importance'):
        mlflow.log_dict(training_summary['feature_importance'], "feature_importance.json")
    
    # Log evaluation results
    mlflow.log_dict(evaluation_results, "evaluation_results.json")
    
    logger.info("✅ All artifacts and metrics logged to MLflow")

def log_mlflow_models(mlflow_run, model_trainer, X_train, y_train, feature_engineer, preprocessor):
    """
    Log trained models to MLflow model registry.
    
    Args:
        mlflow_run: Active MLflow run
        model_trainer: Trained model trainer instance
        X_train: Training features
        y_train: Training targets
        feature_engineer: Feature engineering transformer
        preprocessor: Data preprocessing transformer
    """
    logger.info("Logging models to MLflow model registry")
    
    # Create full pipeline (feature engineering + preprocessing + model)
    from sklearn.pipeline import Pipeline
    
    full_pipeline = Pipeline([
        ('feature_engineering', feature_engineer),
        ('preprocessing', preprocessor),
        ('model', model_trainer.model)
    ])
    
    # Log full pipeline
    mlflow.sklearn.log_model(
        full_pipeline,
        "CryptoPricePredictor_FullPipeline",
        signature=infer_signature(X_train.head(5), full_pipeline.predict(X_train.head(5))),
        input_example=X_train.head(1),
        registered_model_name="CryptoPricePredictor_FullPipeline"
    )
    
    # Log individual model
    mlflow.sklearn.log_model(
        model_trainer.model,
        "CryptoPricePredictor",
        signature=infer_signature(X_train, model_trainer.predict(X_train)),
        input_example=X_train.head(1),
        registered_model_name="CryptoPricePredictor"
    )
    
    logger.info("✅ Models logged to MLflow model registry")

def main():
    """
    Main training function that orchestrates the complete ML pipeline.
    
    This function:
    1. Validates configuration and sets up MLflow
    2. Processes data through the ETL pipeline
    3. Trains the model with cross-validation
    4. Evaluates model performance
    5. Logs everything to MLflow
    6. Saves the trained model
    """
    logger.info("🚀 Starting training pipeline for cryptocurrency price prediction")
    
    # =====================================================================
    # STEP 1: CONFIGURATION VALIDATION
    # =====================================================================
    
    logger.info("Step 1: Validating configuration")
    if not validate_config():
        logger.error("❌ Configuration validation failed")
        sys.exit(1)
    logger.info("✅ Configuration validation passed")
    
    # =====================================================================
    # STEP 2: SETUP MLFLOW
    # =====================================================================
    
    logger.info("Step 2: Setting up MLflow")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"✅ MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    
    # =====================================================================
    # STEP 3: DATA PROCESSING
    # =====================================================================
    
    logger.info("Step 3: Processing data")
    try:
        # Initialize data pipeline
        data_pipeline = DataPipeline(
            add_technical_indicators=True,  # Include technical indicators
            scale_features=True             # Scale features for ML
        )
        
        # Run data processing pipeline
        data_summary = data_pipeline.run_pipeline(target_column='Close')
        
        # Get processed data
        X_train, y_train, X_val, y_val, X_test, y_test = data_pipeline.get_processed_data()
        feature_names = data_pipeline.get_feature_names()
        
        logger.info(f"✅ Data processing completed. Features: {len(feature_names)}")
        
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        sys.exit(1)
    
    # =====================================================================
    # STEP 4: MODEL TRAINING
    # =====================================================================
    
    logger.info("Step 4: Training model")
    try:
        # Initialize model trainer
        model_trainer = ModelTrainer(
            model_type='random_forest',
            n_estimators=100,
            max_depth=10
        )
        
        # Train model
        training_summary = model_trainer.train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            perform_cv=True,           # Perform cross-validation
            perform_grid_search=False  # Set to True for hyperparameter tuning
        )
        
        logger.info("✅ Model training completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}")
        sys.exit(1)
    
    # =====================================================================
    # STEP 5: MODEL EVALUATION
    # =====================================================================
    
    logger.info("Step 5: Evaluating model")
    try:
        # Initialize model evaluator
        model_evaluator = ModelEvaluator(model_trainer)
        
        # Evaluate model on all datasets
        evaluation_results = model_evaluator.evaluate_model(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Print evaluation summary
        model_evaluator.print_evaluation_summary()
        
        logger.info("✅ Model evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Model evaluation failed: {e}")
        sys.exit(1)
    
    # =====================================================================
    # STEP 6: MLFLOW LOGGING
    # =====================================================================
    
    logger.info("Step 6: Logging to MLflow")
    try:
        with mlflow.start_run() as run:
            logger.info(f"MLflow run started: {run.info.run_id}")
            
            # Log artifacts and metrics
            log_mlflow_artifacts(run, training_summary, evaluation_results, data_summary, feature_names)
            
            # Log models
            feature_engineer = FeatureEngineer(add_technical_indicators=True)
            preprocessor = data_pipeline.preprocessor
            log_mlflow_models(run, model_trainer, X_train, y_train, feature_engineer, preprocessor)
            
            logger.info(f"✅ MLflow run completed: {run.info.run_id}")
            
    except Exception as e:
        logger.error(f"❌ MLflow logging failed: {e}")
        sys.exit(1)
    
    # =====================================================================
    # STEP 7: MODEL PERSISTENCE
    # =====================================================================
    
    logger.info("Step 7: Saving model")
    try:
        # Save model and metadata
        model_path = ModelPersistence.save_model(model_trainer, evaluation_results)
        logger.info(f"✅ Model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Model saving failed: {e}")
        sys.exit(1)
    
    # =====================================================================
    # STEP 8: DISPLAY RESULTS
    # =====================================================================
    
    logger.info("Step 8: Displaying training results")
    print_training_summary(training_summary, evaluation_results)
    
    # =====================================================================
    # STEP 9: COMPLETION
    # =====================================================================
    
    logger.info("🎉 Training pipeline completed successfully!")
    logger.info("📁 Model is ready for deployment")
    logger.info("💡 Next step: Run 'python src/app.py' to start the API server")

if __name__ == "__main__":
    main() 