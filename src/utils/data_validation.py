"""
Data validation and quality checks for the crypto price prediction pipeline.

This module provides comprehensive validation functions for financial time series data,
ensuring data quality and consistency before processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from .config import (
    MAX_PRICE, MIN_PRICE, MAX_VOLUME, MIN_VOLUME,
    LOG_LEVEL, LOG_FORMAT
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

# Required columns for cryptocurrency price data
REQUIRED_COLUMNS = [
    'SNo', 'Name', 'Symbol', 'Date', 
    'High', 'Low', 'Open', 'Close', 
    'Volume', 'Marketcap'
]

# Expected data types for each column
EXPECTED_DATA_TYPES = {
    'SNo': 'int64',
    'Name': 'object',
    'Symbol': 'object',
    'Date': 'object',  # Will be converted to datetime
    'High': 'float64',
    'Low': 'float64',
    'Open': 'float64',
    'Close': 'float64',
    'Volume': 'float64',
    'Marketcap': 'float64'
}

# Critical columns that cannot have missing values
CRITICAL_COLUMNS = ['Date', 'High', 'Low', 'Open', 'Close']

# Minimum dataset size for meaningful analysis
MIN_DATASET_SIZE = 10

# Maximum allowed gap between consecutive dates (in days)
MAX_DATE_GAP_DAYS = 30

# =============================================================================
# DATA VALIDATOR CLASS
# =============================================================================

class DataValidator:
    """
    Comprehensive data validator for cryptocurrency price data.
    
    This class performs multiple validation checks to ensure data quality:
    - Structure validation (columns, data types)
    - Data quality checks (missing values, duplicates)
    - Business logic validation (price consistency, temporal order)
    - Range validation (reasonable price/volume bounds)
    """
    
    def __init__(self):
        """Initialize the validator with empty results tracking."""
        self.validation_results = {}
        self.issues_found = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Perform comprehensive validation of the input dataframe.
        
        Args:
            df: Input dataframe to validate
            
        Returns:
            Dictionary with validation results for each check
        """
        logger.info("Starting comprehensive data validation")
        
        # Reset validation state
        self.validation_results = {}
        self.issues_found = []
        
        # =====================================================================
        # STRUCTURE VALIDATION
        # =====================================================================
        
        # Check basic structure
        self.validation_results['has_required_columns'] = self._validate_required_columns(df)
        self.validation_results['has_data'] = self._validate_data_presence(df)
        self.validation_results['correct_data_types'] = self._validate_data_types(df)
        
        # =====================================================================
        # DATA QUALITY VALIDATION
        # =====================================================================
        
        # Check for data quality issues
        self.validation_results['no_missing_values'] = self._validate_missing_values(df)
        self.validation_results['no_duplicates'] = self._validate_duplicates(df)
        
        # =====================================================================
        # VALUE RANGE VALIDATION
        # =====================================================================
        
        # Validate price and volume ranges
        self.validation_results['valid_prices'] = self._validate_price_data(df)
        self.validation_results['valid_volumes'] = self._validate_volume_data(df)
        
        # =====================================================================
        # TEMPORAL VALIDATION
        # =====================================================================
        
        # Validate date-related aspects
        self.validation_results['valid_dates'] = self._validate_date_data(df)
        self.validation_results['temporal_consistency'] = self._validate_temporal_consistency(df)
        
        # =====================================================================
        # BUSINESS LOGIC VALIDATION
        # =====================================================================
        
        # Validate business rules
        self.validation_results['price_consistency'] = self._validate_price_consistency(df)
        self.validation_results['volume_consistency'] = self._validate_volume_consistency(df)
        
        # =====================================================================
        # FINAL SUMMARY
        # =====================================================================
        
        # Determine overall validation result
        all_valid = all(self.validation_results.values())
        self.validation_results['overall_valid'] = all_valid
        
        # Log results
        logger.info(f"Validation complete. Overall valid: {all_valid}")
        if self.issues_found:
            logger.warning(f"Found {len(self.issues_found)} issues: {self.issues_found}")
        
        return self.validation_results
    
    # =====================================================================
    # STRUCTURE VALIDATION METHODS
    # =====================================================================
    
    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required columns are present in the dataframe.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if all required columns are present, False otherwise
        """
        missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
        
        if missing_columns:
            self.issues_found.append(f"Missing required columns: {missing_columns}")
            return False
        
        return True
    
    def _validate_data_presence(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains sufficient data.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if dataframe has sufficient data, False otherwise
        """
        if df.empty:
            self.issues_found.append("Dataframe is empty")
            return False
        
        if len(df) < MIN_DATASET_SIZE:
            self.issues_found.append(f"Dataset too small: {len(df)} rows (minimum: {MIN_DATASET_SIZE})")
            return False
        
        return True
    
    def _validate_data_types(self, df: pd.DataFrame) -> bool:
        """
        Validate that columns have the expected data types.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if all columns have correct types, False otherwise
        """
        issues = []
        
        for col, expected_type in EXPECTED_DATA_TYPES.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    issues.append(f"Column {col}: expected {expected_type}, got {actual_type}")
        
        if issues:
            self.issues_found.extend(issues)
            return False
        
        return True
    
    # =====================================================================
    # DATA QUALITY VALIDATION METHODS
    # =====================================================================
    
    def _validate_missing_values(self, df: pd.DataFrame) -> bool:
        """
        Validate that critical columns have no missing values.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if no missing values in critical columns, False otherwise
        """
        missing_counts = df[CRITICAL_COLUMNS].isnull().sum()
        
        if missing_counts.sum() > 0:
            missing_info = missing_counts[missing_counts > 0].to_dict()
            self.issues_found.append(f"Missing values found: {missing_info}")
            return False
        
        return True
    
    def _validate_duplicates(self, df: pd.DataFrame) -> bool:
        """
        Validate that there are no duplicate records.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if no duplicates found, False otherwise
        """
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            self.issues_found.append(f"Found {duplicates} duplicate rows")
            return False
        
        return True
    
    # =====================================================================
    # VALUE RANGE VALIDATION METHODS
    # =====================================================================
    
    def _validate_price_data(self, df: pd.DataFrame) -> bool:
        """
        Validate price data for reasonable ranges and consistency.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if price data is valid, False otherwise
        """
        price_columns = ['High', 'Low', 'Open', 'Close']
        issues = []
        
        for col in price_columns:
            if col in df.columns:
                # Check for negative prices
                negative_prices = (df[col] < 0).sum()
                if negative_prices > 0:
                    issues.append(f"Column {col}: {negative_prices} negative prices")
                
                # Check for unreasonable prices
                too_high = (df[col] > MAX_PRICE).sum()
                too_low = (df[col] < MIN_PRICE).sum()
                
                if too_high > 0:
                    issues.append(f"Column {col}: {too_high} prices above {MAX_PRICE}")
                if too_low > 0:
                    issues.append(f"Column {col}: {too_low} prices below {MIN_PRICE}")
        
        if issues:
            self.issues_found.extend(issues)
            return False
        
        return True
    
    def _validate_volume_data(self, df: pd.DataFrame) -> bool:
        """
        Validate volume data for reasonable ranges.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if volume data is valid, False otherwise
        """
        if 'Volume' not in df.columns:
            return True
        
        # Check for negative volumes
        negative_volumes = (df['Volume'] < 0).sum()
        if negative_volumes > 0:
            self.issues_found.append(f"Found {negative_volumes} negative volumes")
            return False
        
        # Check for unreasonable volumes
        too_high = (df['Volume'] > MAX_VOLUME).sum()
        if too_high > 0:
            self.issues_found.append(f"Found {too_high} volumes above {MAX_VOLUME}")
            return False
        
        return True
    
    # =====================================================================
    # TEMPORAL VALIDATION METHODS
    # =====================================================================
    
    def _validate_date_data(self, df: pd.DataFrame) -> bool:
        """
        Validate date data for consistency and reasonableness.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if date data is valid, False otherwise
        """
        if 'Date' not in df.columns:
            return False
        
        try:
            # Convert to datetime
            dates = pd.to_datetime(df['Date'], errors='coerce')
            
            # Check for invalid dates
            invalid_dates = dates.isnull().sum()
            if invalid_dates > 0:
                self.issues_found.append(f"Found {invalid_dates} invalid dates")
                return False
            
            # Check for future dates
            future_dates = (dates > pd.Timestamp.now()).sum()
            if future_dates > 0:
                self.issues_found.append(f"Found {future_dates} future dates")
                return False
            
            # Check for very old dates (before 2010 for crypto)
            old_dates = (dates < pd.Timestamp('2010-01-01')).sum()
            if old_dates > 0:
                self.issues_found.append(f"Found {old_dates} dates before 2010")
                return False
            
            return True
            
        except Exception as e:
            self.issues_found.append(f"Date validation error: {e}")
            return False
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> bool:
        """
        Validate temporal consistency of the data.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if temporal data is consistent, False otherwise
        """
        if 'Date' not in df.columns:
            return False
        
        try:
            dates = pd.to_datetime(df['Date'])
            sorted_dates = dates.sort_values()
            
            # Check if dates are already sorted
            if not dates.equals(sorted_dates):
                self.issues_found.append("Dates are not in chronological order")
                return False
            
            # Check for reasonable gaps (not more than MAX_DATE_GAP_DAYS)
            date_diffs = dates.diff().dropna()
            large_gaps = (date_diffs > timedelta(days=MAX_DATE_GAP_DAYS)).sum()
            
            if large_gaps > 0:
                self.issues_found.append(f"Found {large_gaps} gaps larger than {MAX_DATE_GAP_DAYS} days")
                return False
            
            return True
            
        except Exception as e:
            self.issues_found.append(f"Temporal validation error: {e}")
            return False
    
    # =====================================================================
    # BUSINESS LOGIC VALIDATION METHODS
    # =====================================================================
    
    def _validate_price_consistency(self, df: pd.DataFrame) -> bool:
        """
        Validate price consistency (High >= Low, etc.).
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if price relationships are consistent, False otherwise
        """
        issues = []
        
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            # High should be >= Low
            high_low_violations = (df['High'] < df['Low']).sum()
            if high_low_violations > 0:
                issues.append(f"High < Low in {high_low_violations} rows")
            
            # High should be >= Open and Close
            high_open_violations = (df['High'] < df['Open']).sum()
            high_close_violations = (df['High'] < df['Close']).sum()
            
            if high_open_violations > 0:
                issues.append(f"High < Open in {high_open_violations} rows")
            if high_close_violations > 0:
                issues.append(f"High < Close in {high_close_violations} rows")
            
            # Low should be <= Open and Close
            low_open_violations = (df['Low'] > df['Open']).sum()
            low_close_violations = (df['Low'] > df['Close']).sum()
            
            if low_open_violations > 0:
                issues.append(f"Low > Open in {low_open_violations} rows")
            if low_close_violations > 0:
                issues.append(f"Low > Close in {low_close_violations} rows")
        
        if issues:
            self.issues_found.extend(issues)
            return False
        
        return True
    
    def _validate_volume_consistency(self, df: pd.DataFrame) -> bool:
        """
        Validate volume consistency.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if volume data is consistent, False otherwise
        """
        if 'Volume' not in df.columns:
            return True
        
        # Volume should generally be positive (0 is acceptable for some cases)
        negative_volumes = (df['Volume'] < 0).sum()
        if negative_volumes > 0:
            self.issues_found.append(f"Found {negative_volumes} negative volumes")
            return False
        
        return True
    
    # =====================================================================
    # REPORTING METHODS
    # =====================================================================
    
    def get_validation_summary(self) -> Dict:
        """
        Get a summary of validation results.
        
        Returns:
            Dictionary with validation summary information
        """
        return {
            'overall_valid': self.validation_results.get('overall_valid', False),
            'validation_results': self.validation_results,
            'issues_found': self.issues_found,
            'total_issues': len(self.issues_found)
        }
    
    def print_validation_report(self):
        """Print a detailed validation report to console."""
        print("\n" + "="*50)
        print("DATA VALIDATION REPORT")
        print("="*50)
        
        # Print individual validation results
        for check, result in self.validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{check.replace('_', ' ').title()}: {status}")
        
        # Print overall result
        overall_status = "✅ PASS" if self.validation_results.get('overall_valid', False) else "❌ FAIL"
        print(f"\nOverall Validation: {overall_status}")
        
        # Print issues if any found
        if self.issues_found:
            print(f"\nIssues Found ({len(self.issues_found)}):")
            for i, issue in enumerate(self.issues_found, 1):
                print(f"  {i}. {issue}")
        
        print("="*50)

# =============================================================================
# DATA CLEANING FUNCTION
# =============================================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataframe based on validation results.
    
    This function applies common data cleaning operations:
    - Converts dates to datetime
    - Removes duplicates
    - Fixes negative values
    - Sorts by date
    - Resets indices
    
    Args:
        df: Input dataframe to clean
        
    Returns:
        Cleaned dataframe
    """
    logger.info("Starting data cleaning process")
    
    df_clean = df.copy()
    
    # Convert dates to datetime and remove invalid dates
    if 'Date' in df_clean.columns:
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['Date'])
    
    # Remove duplicate records
    df_clean = df_clean.drop_duplicates()
    
    # Fix negative prices by taking absolute values
    price_columns = ['High', 'Low', 'Open', 'Close']
    for col in price_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].abs()
    
    # Fix negative volumes
    if 'Volume' in df_clean.columns:
        df_clean['Volume'] = df_clean['Volume'].abs()
    
    # Sort by date to maintain temporal order
    if 'Date' in df_clean.columns:
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
    
    # Reset serial numbers to be sequential
    if 'SNo' in df_clean.columns:
        df_clean['SNo'] = range(1, len(df_clean) + 1)
    
    logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")
    return df_clean 