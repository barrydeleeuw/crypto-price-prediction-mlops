#!/usr/bin/env python3
"""
ETL (Extract, Transform, Load) script for cryptocurrency price prediction pipeline.

This script orchestrates the complete data processing workflow:
1. Extract data from Kaggle datasets
2. Transform data through validation, cleaning, and feature engineering
3. Load processed data into the pipeline for model training

Usage:
    python src/run_etl.py
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from utils.config import validate_config, LOG_LEVEL, LOG_FORMAT, PROCESSED_DATA_PATH
from training.data import DataPipeline

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def print_etl_summary(summary: dict):
    """
    Print a formatted summary of the ETL pipeline results.
    
    Args:
        summary: Dictionary containing ETL pipeline summary
    """
    print("\n" + "="*70)
    print("ETL PIPELINE SUMMARY")
    print("="*70)
    
    # Print pipeline status
    print("\n📋 PIPELINE STATUS:")
    for step, status in summary.get('pipeline_steps', {}).items():
        status_icon = "✅" if status == "Completed" else "❌"
        print(f"  {status_icon} {step.replace('_', ' ').title()}: {status}")
    
    # Print data information
    data_info = summary.get('data_info', {})
    print(f"\n📊 DATA INFORMATION:")
    print(f"  📈 Original dataset size: {data_info.get('original_shape', 0):,} records")
    print(f"  🎯 Final features: {data_info.get('final_features', 0)}")
    print(f"  📚 Training samples: {data_info.get('train_samples', 0):,}")
    print(f"  🔍 Validation samples: {data_info.get('validation_samples', 0):,}")
    print(f"  🧪 Test samples: {data_info.get('test_samples', 0):,}")
    
    # Print feature information
    feature_info = summary.get('feature_info', {})
    print(f"\n🔧 FEATURE INFORMATION:")
    print(f"  📊 Total features: {feature_info.get('total_features', 0)}")
    
    # Show sample feature names
    feature_names = feature_info.get('feature_names', [])
    if feature_names:
        print(f"  📝 Sample features: {', '.join(feature_names[:5])}")
        if len(feature_names) > 5:
            print(f"    ... and {len(feature_names) - 5} more features")
    
    print("="*70)

def save_etl_summary(summary: dict, output_path: Path):
    """
    Save the ETL summary to a JSON file.
    
    Args:
        summary: Dictionary containing ETL pipeline summary
        output_path: Path where to save the summary file
    """
    # Add timestamp to summary
    summary['etl_timestamp'] = datetime.now().isoformat()
    summary['etl_version'] = '1.0.0'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save summary to JSON file
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"ETL summary saved to: {output_path}")

def main():
    """
    Main ETL function that orchestrates the complete data processing pipeline.
    
    This function:
    1. Validates configuration settings
    2. Initializes the data pipeline
    3. Runs the complete ETL process
    4. Prints and saves the results summary
    """
    logger.info("🚀 Starting ETL pipeline for cryptocurrency price prediction")
    
    # =====================================================================
    # STEP 1: CONFIGURATION VALIDATION
    # =====================================================================
    
    logger.info("Step 1: Validating configuration")
    if not validate_config():
        logger.error("❌ Configuration validation failed")
        sys.exit(1)
    logger.info("✅ Configuration validation passed")
    
    # =====================================================================
    # STEP 2: INITIALIZE DATA PIPELINE
    # =====================================================================
    
    logger.info("Step 2: Initializing data pipeline")
    try:
        data_pipeline = DataPipeline(
            add_technical_indicators=True,  # Include technical indicators
            scale_features=True             # Scale features for ML
        )
        logger.info("✅ Data pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize data pipeline: {e}")
        sys.exit(1)
    
    # =====================================================================
    # STEP 3: RUN ETL PIPELINE
    # =====================================================================
    
    logger.info("Step 3: Running complete data processing pipeline")
    try:
        # Run the complete data processing pipeline
        summary = data_pipeline.run_pipeline(target_column='Close')
        logger.info("✅ Data processing pipeline completed successfully")
    except Exception as e:
        logger.error(f"❌ Data processing pipeline failed: {e}")
        sys.exit(1)
    
    # =====================================================================
    # STEP 4: DISPLAY RESULTS
    # =====================================================================
    
    logger.info("Step 4: Displaying ETL results")
    print_etl_summary(summary)
    
    # =====================================================================
    # STEP 5: SAVE SUMMARY
    # =====================================================================
    
    logger.info("Step 5: Saving ETL summary")
    try:
        summary_file = PROCESSED_DATA_PATH / "etl_summary.json"
        save_etl_summary(summary, summary_file)
        logger.info("✅ ETL summary saved successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save ETL summary: {e}")
    
    # =====================================================================
    # STEP 6: COMPLETION
    # =====================================================================
    
    logger.info("🎉 ETL pipeline completed successfully!")
    logger.info("📁 Processed data is ready for model training")
    logger.info("💡 Next step: Run 'python src/run_training.py' to train the model")

if __name__ == "__main__":
    main() 