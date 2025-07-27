"""
Feature engineering for cryptocurrency price prediction.
Comprehensive feature creation including technical indicators and time-based features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.config import ROLLING_WINDOWS, TECHNICAL_INDICATORS

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for cryptocurrency price data.
    Creates technical indicators, time-based features, and statistical features.
    """
    
    def __init__(self, rolling_windows: List[int] = None, add_technical_indicators: bool = True):
        """
        Initialize the feature engineer.
        
        Args:
            rolling_windows: List of window sizes for rolling statistics
            add_technical_indicators: Whether to add technical indicators
        """
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS
        self.add_technical_indicators = add_technical_indicators
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """Fit the feature engineer (no-op for this transformer)."""
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding engineered features.
        
        Args:
            X: Input dataframe with OHLCV data
            
        Returns:
            Dataframe with additional engineered features
        """
        logger.info("Starting feature engineering")
        
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise ValueError("Input must be a pandas DataFrame")
        
        # Add basic price-based features
        df = self._add_price_features(df)
        
        # Add volume-based features
        df = self._add_volume_features(df)
        
        # Add market cap features
        df = self._add_marketcap_features(df)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Add technical indicators
        if self.add_technical_indicators:
            df = self._add_technical_indicators(df)
        
        # Add statistical features
        df = self._add_statistical_features(df)
        
        # Clean and handle edge cases
        df = self._clean_features(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        logger.debug("Adding price-based features")
        
        # Price spreads and ratios
        if all(col in df.columns for col in ['High', 'Low', 'Open']):
            df['High_Low_Spread'] = df['High'] - df['Low']
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
            
            # Price changes
            df['Open_Change'] = df['Open'].pct_change()
            df['High_Change'] = df['High'].pct_change()
            df['Low_Change'] = df['Low'].pct_change()
        
        # Rolling price statistics
        for window in self.rolling_windows:
            if 'Open' in df.columns:
                df[f'Open_MA_{window}'] = df['Open'].rolling(window=window).mean()
                df[f'Open_Std_{window}'] = df['Open'].rolling(window=window).std()
                df[f'Open_Min_{window}'] = df['Open'].rolling(window=window).min()
                df[f'Open_Max_{window}'] = df['Open'].rolling(window=window).max()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        logger.debug("Adding volume-based features")
        
        if 'Volume' in df.columns:
            # Volume changes
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
            
            # Rolling volume statistics
            for window in self.rolling_windows:
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
                df[f'Volume_Std_{window}'] = df['Volume'].rolling(window=window).std()
                df[f'Volume_Min_{window}'] = df['Volume'].rolling(window=window).min()
                df[f'Volume_Max_{window}'] = df['Volume'].rolling(window=window).max()
            
            # Volume-price relationship
            if 'Open' in df.columns:
                df['Volume_Price_Ratio'] = df['Volume'] / df['Open']
        
        return df
    
    def _add_marketcap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market capitalization features."""
        logger.debug("Adding market cap features")
        
        if 'Marketcap' in df.columns:
            # Market cap changes
            df['Marketcap_Change'] = df['Marketcap'].pct_change()
            
            # Rolling market cap statistics
            for window in self.rolling_windows:
                df[f'Marketcap_MA_{window}'] = df['Marketcap'].rolling(window=window).mean()
                df[f'Marketcap_Std_{window}'] = df['Marketcap'].rolling(window=window).std()
            
            # Market cap to volume ratio
            if 'Volume' in df.columns:
                df['Marketcap_Volume_Ratio'] = df['Marketcap'] / df['Volume']
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        logger.debug("Adding time-based features")
        
        if 'Date' in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Extract time components
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Quarter'] = df['Date'].dt.quarter
            df['DayOfYear'] = df['Date'].dt.dayofyear
            
            # Cyclical encoding for periodic features
            df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
            
            # Days since start
            df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        logger.debug("Adding technical indicators")
        
        if 'Open' in df.columns:
            # Simple Moving Averages
            for window in [5, 10, 20]:
                df[f'SMA_{window}'] = df['Open'].rolling(window=window).mean()
            
            # Exponential Moving Averages
            for window in [5, 10, 20]:
                df[f'EMA_{window}'] = df['Open'].ewm(span=window).mean()
            
            # Relative Strength Index (RSI)
            df['RSI_14'] = self._calculate_rsi(df['Open'], window=14)
            
            # MACD
            df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = self._calculate_macd(df['Open'])
            
            # Bollinger Bands
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self._calculate_bollinger_bands(df['Open'])
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Open'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        logger.debug("Adding statistical features")
        
        # Price volatility
        if 'Open' in df.columns:
            for window in self.rolling_windows:
                returns = df['Open'].pct_change()
                df[f'Volatility_{window}'] = returns.rolling(window=window).std()
                df[f'Skewness_{window}'] = returns.rolling(window=window).skew()
                df[f'Kurtosis_{window}'] = returns.rolling(window=window).kurt()
        
        # Z-score features
        for col in ['Open', 'Volume', 'Marketcap']:
            if col in df.columns:
                for window in self.rolling_windows:
                    mean_val = df[col].rolling(window=window).mean()
                    std_val = df[col].rolling(window=window).std()
                    df[f'{col}_ZScore_{window}'] = (df[col] - mean_val) / std_val
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean engineered features to handle edge cases."""
        logger.debug("Cleaning engineered features")
        
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Clip extreme values (99th percentile)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        # Fill missing values with median
        for col in numeric_columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower

def create_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Create a summary of engineered features.
    
    Args:
        df: Dataframe with engineered features
        
    Returns:
        Dictionary with feature summary
    """
    feature_groups = {
        'price_features': [col for col in df.columns if any(x in col.lower() for x in ['price', 'high', 'low', 'open', 'close', 'spread', 'ratio'])],
        'volume_features': [col for col in df.columns if 'volume' in col.lower()],
        'marketcap_features': [col for col in df.columns if 'marketcap' in col.lower()],
        'time_features': [col for col in df.columns if any(x in col.lower() for x in ['year', 'month', 'day', 'quarter', 'sin', 'cos'])],
        'technical_indicators': [col for col in df.columns if any(x in col.upper() for x in ['SMA', 'EMA', 'RSI', 'MACD', 'BB'])],
        'statistical_features': [col for col in df.columns if any(x in col.lower() for x in ['volatility', 'skewness', 'kurtosis', 'zscore'])],
        'rolling_features': [col for col in df.columns if any(x in col for x in ['_MA_', '_Std_', '_Min_', '_Max_'])]
    }
    
    summary = {
        'total_features': len(df.columns),
        'feature_groups': {group: len(features) for group, features in feature_groups.items()},
        'feature_names': feature_groups
    }
    
    return summary

def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        Dataframe with feature importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame() 