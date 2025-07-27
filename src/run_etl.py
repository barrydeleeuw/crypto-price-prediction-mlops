#!/usr/bin/env python3
"""
ETL script for cryptocurrency price prediction pipeline.
Extracts data from Kaggle, validates, cleans, and prepares it for training.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from utils.config import validate_config, LOG_LEVEL, LOG_FORMAT
from training.data import DataPipeline
from utils.data_validation import DataValidator

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

def main():
    """Main ETL function."""
    logger.info("Starting ETL pipeline for cryptocurrency price prediction")
    
    # Validate configuration
    if not validate_config():
        logger.error("Configuration validation failed")
        sys.exit(1)
    
    try:
        # Initialize data pipeline
        data_pipeline = DataPipeline(
            add_technical_indicators=True,
            scale_features=True
        )
        
        # Run the complete data processing pipeline
        logger.info("Running data processing pipeline")
        summary = data_pipeline.run_pipeline(target_column='Close')
        
        # Print summary
        print("\n" + "="*60)
        print("ETL PIPELINE SUMMARY")
        print("="*60)
        
        print(f"\nData Information:")
        print(f"  Original dataset size: {summary['data_info']['original_shape']:,} records")
        print(f"  Training samples: {summary['data_info']['train_samples']:,}")
        print(f"  Validation samples: {summary['data_info']['val_samples']:,}")
        print(f"  Test samples: {summary['data_info']['test_samples']:,}")
        print(f"  Total features: {summary['data_info']['total_features']}")
        
        print(f"\nFeature Summary:")
        for group, count in summary['feature_summary']['feature_groups'].items():
            print(f"  {group.replace('_', ' ').title()}: {count} features")
        
        print(f"\nTarget Variable Statistics:")
        print(f"  Training - Mean: ${summary['target_info']['train_mean']:,.2f}, Std: ${summary['target_info']['train_std']:,.2f}")
        print(f"  Validation - Mean: ${summary['target_info']['val_mean']:,.2f}, Std: ${summary['target_info']['val_std']:,.2f}")
        print(f"  Test - Mean: ${summary['target_info']['test_mean']:,.2f}, Std: ${summary['target_info']['test_std']:,.2f}")
        
        print("="*60)
        
        logger.info("ETL pipeline completed successfully")
        
        # Save processed data summary
        from utils.config import PROCESSED_DATA_PATH
        summary_file = PROCESSED_DATA_PATH / "etl_summary.json"
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"ETL summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"ETL pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 