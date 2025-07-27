"""
Feature engineering for cryptocurrency price prediction.

This module provides comprehensive feature creation including:
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Time-based features (seasonality, cyclical encoding)
- Statistical features (rolling statistics, volatility measures)
- Price and volume derived features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils.config import ROLLING_WINDOWS, TECHNICAL_INDICATORS

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# FEATURE ENGINEERING CONSTANTS
# =============================================================================

# Technical indicator parameters
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2

# Time feature parameters
CYCLICAL_ENCODING = True  # Use cyclical encoding for time features

# =============================================================================
# FEATURE ENGINEER CLASS
# =============================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering transformer for cryptocurrency price data.
    
    This transformer creates a comprehensive set of features for time series prediction:
    - Price-based features (spreads, ratios, changes)
    - Volume-based features (changes, ratios, rolling statistics)
    - Market cap features (changes, ratios)
    - Time-based features (seasonality, cyclical encoding)
    - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
    - Statistical features (rolling statistics, volatility measures)
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
        """
        Fit the feature engineer (no-op for this transformer).
        
        Args:
            X: Input data (not used)
            y: Target values (not used)
            
        Returns:
            self: The fitted transformer
        """
        return self
    
    def transform(self, X):
        """
        Transform the input data by adding engineered features.
        
        Args:
            X: Input dataframe with OHLCV data
            
        Returns:
            Dataframe with additional engineered features
            
        Raises:
            ValueError: If input is not a pandas DataFrame
        """
        logger.info("Starting feature engineering process")
        
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        df = X.copy()
        
        # =====================================================================
        # FEATURE CREATION PIPELINE
        # =====================================================================
        
        # Create basic derived features
        df = self._add_price_features(df)
        df = self._add_volume_features(df)
        df = self._add_marketcap_features(df)
        
        # Create time-based features
        df = self._add_time_features(df)
        
        # Create technical indicators (if enabled)
        if self.add_technical_indicators:
            df = self._add_technical_indicators(df)
        
        # Create statistical features
        df = self._add_statistical_features(df)
        
        # Clean and handle edge cases
        df = self._clean_features(df)
        
        # Store feature names for later use
        self.feature_names_ = [col for col in df.columns if col not in X.columns]
        
        logger.info(f"Feature engineering complete. Added {len(self.feature_names_)} features. Final shape: {df.shape}")
        return df
    
    # =====================================================================
    # PRICE-BASED FEATURES
    # =====================================================================
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features including spreads, ratios, and changes.
        
        Args:
            df: Input dataframe with OHLC data
            
        Returns:
            Dataframe with price features added
        """
        logger.debug("Adding price-based features")
        
        # Price spreads and ratios
        if all(col in df.columns for col in ['High', 'Low', 'Open']):
            # Calculate price spreads
            df['High_Low_Spread'] = df['High'] - df['Low']
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Price_Range'] = (df['High'] - df['Low']) / df['Open']
            
            # Calculate price changes (percentage)
            df['Open_Change'] = df['Open'].pct_change()
            df['High_Change'] = df['High'].pct_change()
            df['Low_Change'] = df['Low'].pct_change()
        
        # Rolling price statistics for different windows
        for window in self.rolling_windows:
            if 'Open' in df.columns:
                # Moving averages and standard deviations
                df[f'Open_MA_{window}'] = df['Open'].rolling(window=window).mean()
                df[f'Open_Std_{window}'] = df['Open'].rolling(window=window).std()
                df[f'Open_Min_{window}'] = df['Open'].rolling(window=window).min()
                df[f'Open_Max_{window}'] = df['Open'].rolling(window=window).max()
                
                # Price relative to moving average
                df[f'Open_MA_Ratio_{window}'] = df['Open'] / df[f'Open_MA_{window}']
        
        return df
    
    # =====================================================================
    # VOLUME-BASED FEATURES
    # =====================================================================
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features including changes and rolling statistics.
        
        Args:
            df: Input dataframe with volume data
            
        Returns:
            Dataframe with volume features added
        """
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
        
        return df
    
    # =====================================================================
    # MARKET CAP FEATURES
    # =====================================================================
    
    def _add_marketcap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market capitalization derived features.
        
        Args:
            df: Input dataframe with market cap data
            
        Returns:
            Dataframe with market cap features added
        """
        logger.debug("Adding market cap features")
        
        if 'Marketcap' in df.columns:
            # Market cap changes
            df['Marketcap_Change'] = df['Marketcap'].pct_change()
            
            # Rolling market cap statistics
            for window in self.rolling_windows:
                df[f'Marketcap_MA_{window}'] = df['Marketcap'].rolling(window=window).mean()
                df[f'Marketcap_Std_{window}'] = df['Marketcap'].rolling(window=window).std()
        
        return df
    
    # =====================================================================
    # TIME-BASED FEATURES
    # =====================================================================
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features including seasonality and cyclical encoding.
        
        Args:
            df: Input dataframe with date column
            
        Returns:
            Dataframe with time features added
        """
        logger.debug("Adding time-based features")
        
        if 'Date' in df.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Extract basic time components
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['Quarter'] = df['Date'].dt.quarter
            
            # Cyclical encoding for periodic features
            if CYCLICAL_ENCODING:
                # Month cyclical encoding (1-12)
                df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
                df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
                
                # Day of week cyclical encoding (0-6)
                df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
                df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
                
                # Day of year cyclical encoding
                day_of_year = df['Date'].dt.dayofyear
                df['DayOfYear_Sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
                df['DayOfYear_Cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        return df
    
    # =====================================================================
    # TECHNICAL INDICATORS
    # =====================================================================
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators including SMA, EMA, RSI, MACD, and Bollinger Bands.
        
        Args:
            df: Input dataframe with price data
            
        Returns:
            Dataframe with technical indicators added
        """
        logger.debug("Adding technical indicators")
        
        if 'Close' not in df.columns:
            logger.warning("Close price not found, skipping technical indicators")
            return df
        
        # Simple Moving Averages (SMA)
        for window in [5, 10, 20]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Exponential Moving Averages (EMA)
        for window in [5, 10, 20]:
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Relative Strength Index (RSI)
        df['RSI_14'] = self._calculate_rsi(df['Close'], RSI_WINDOW)
        
        # Moving Average Convergence Divergence (MACD)
        macd_line, signal_line, macd_histogram = self._calculate_macd(
            df['Close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL
        )
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            df['Close'], BOLLINGER_WINDOW, BOLLINGER_STD
        )
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        df['BB_Width'] = bb_upper - bb_lower
        df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
        
        return df
    
    # =====================================================================
    # STATISTICAL FEATURES
    # =====================================================================
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add statistical features including volatility and momentum measures.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with statistical features added
        """
        logger.debug("Adding statistical features")
        
        if 'Close' in df.columns:
            # Volatility measures
            for window in self.rolling_windows:
                # Rolling volatility (standard deviation of returns)
                returns = df['Close'].pct_change()
                df[f'Volatility_{window}'] = returns.rolling(window=window).std()
                
                # Rolling skewness and kurtosis
                df[f'Skewness_{window}'] = returns.rolling(window=window).skew()
                df[f'Kurtosis_{window}'] = returns.rolling(window=window).kurt()
            
            # Momentum indicators
            for window in [5, 10, 20]:
                df[f'Momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
        
        return df
    
    # =====================================================================
    # FEATURE CLEANING
    # =====================================================================
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean engineered features by handling infinite values, NaNs, and outliers.
        
        Args:
            df: Input dataframe with engineered features
            
        Returns:
            Cleaned dataframe
        """
        logger.debug("Cleaning engineered features")
        
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Clip extreme outliers (beyond 3 standard deviations)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Fill remaining NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # For any remaining NaNs, fill with 0
        df = df.fillna(0)
        
        return df
    
    # =====================================================================
    # TECHNICAL INDICATOR CALCULATIONS
    # =====================================================================
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            window: RSI calculation window
            
        Returns:
            RSI series
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD line, Signal line, MACD histogram)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series
            window: Moving average window
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        return upper_band, middle_band, lower_band

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Create a summary of engineered features.
    
    Args:
        df: Dataframe with engineered features
        
    Returns:
        Dictionary with feature summary information
    """
    summary = {
        'total_features': len(df.columns),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'feature_types': {}
    }
    
    # Categorize features by type
    for col in df.columns:
        if 'price' in col.lower() or any(x in col.lower() for x in ['high', 'low', 'open', 'close']):
            summary['feature_types'][col] = 'price'
        elif 'volume' in col.lower():
            summary['feature_types'][col] = 'volume'
        elif 'marketcap' in col.lower():
            summary['feature_types'][col] = 'marketcap'
        elif any(x in col.lower() for x in ['year', 'month', 'day', 'time']):
            summary['feature_types'][col] = 'time'
        elif any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bb']):
            summary['feature_types'][col] = 'technical'
        else:
            summary['feature_types'][col] = 'statistical'
    
    return summary

def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        Dataframe with feature names and importance scores
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