"""
FastAPI application for cryptocurrency price prediction.

This module provides a REST API for serving cryptocurrency price predictions using
the trained machine learning model. It includes comprehensive endpoints for:
- Single and batch predictions
- Model health checks and information
- Feature importance analysis
- Error handling and validation

Usage:
    python src/app.py
    # or
    uvicorn app:app --reload --host 127.0.0.1 --port 8000
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from utils.config import API_HOST, API_PORT, MODEL_NAME, MODEL_VERSION, LOG_LEVEL, LOG_FORMAT
from training.data import DataPipeline
from training.model import ModelPersistence
from training.features import FeatureEngineer
from training.data import DataPreprocessor

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# =============================================================================
# FASTAPI APPLICATION SETUP
# =============================================================================

# Initialize FastAPI application
app = FastAPI(
    title="Crypto Price Prediction API",
    description="""
    **Crypto Price Prediction API** - A machine learning-powered API for predicting cryptocurrency prices.
    
    ## Features
    * 🎯 **Single Predictions** - Predict price for individual data points
    * 📊 **Batch Predictions** - Process multiple predictions efficiently
    * 🔍 **Model Information** - Get model details and performance metrics
    * 📈 **Feature Importance** - Understand which features drive predictions
    * 🏥 **Health Checks** - Monitor API and model status
    
    ## Usage
    1. **Health Check**: `GET /health` - Verify API is running
    2. **Single Prediction**: `POST /predict` - Predict price for one data point
    3. **Batch Prediction**: `POST /predict_batch` - Predict prices for multiple data points
    4. **Model Info**: `GET /model_info` - Get model details and performance
    5. **Feature Importance**: `GET /feature_importance` - Get feature importance rankings
    
    ## Input Format
    ```json
    {
        "SNo": 1,
        "Name": "Bitcoin",
        "Symbol": "BTC",
        "Date": "2023-12-01",
        "High": 45000.0,
        "Low": 44000.0,
        "Open": 44500.0,
        "Volume": 1000000.0,
        "Marketcap": 850000000000.0
    }
    ```
    """,
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Crypto Price Prediction API",
        "url": "https://github.com/your-repo/crypto-price-prediction",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE VALIDATION
# =============================================================================

class CryptoData(BaseModel):
    """
    Input data model for cryptocurrency price prediction.
    
    This model validates the input data structure and ensures data quality
    before processing by the machine learning model.
    """
    SNo: int = Field(..., description="Serial number", ge=1)
    Name: str = Field(..., description="Cryptocurrency name", min_length=1, max_length=50)
    Symbol: str = Field(..., description="Cryptocurrency symbol", min_length=1, max_length=10)
    Date: str = Field(..., description="Date in YYYY-MM-DD format")
    High: float = Field(..., description="High price", gt=0)
    Low: float = Field(..., description="Low price", gt=0)
    Open: float = Field(..., description="Open price", gt=0)
    Volume: float = Field(..., description="Trading volume", ge=0)
    Marketcap: float = Field(..., description="Market capitalization", ge=0)
    
    @validator('Date')
    def validate_date_format(cls, v):
        """Validate date format is YYYY-MM-DD."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @validator('High', 'Low', 'Open')
    def validate_price_consistency(cls, v, values):
        """Validate price consistency (High >= Low, etc.)."""
        if 'High' in values and 'Low' in values and 'Open' in values:
            high = values.get('High', v)
            low = values.get('Low', v)
            open_price = values.get('Open', v)
            
            if high < low:
                raise ValueError('High price cannot be less than Low price')
            if high < open_price:
                raise ValueError('High price cannot be less than Open price')
            if low > open_price:
                raise ValueError('Low price cannot be greater than Open price')
        
        return v

class PredictionResponse(BaseModel):
    """
    Response model for price predictions.
    
    Provides comprehensive prediction results including confidence scores
    and metadata for transparency and debugging.
    """
    predicted_price: float = Field(..., description="Predicted closing price in USD")
    confidence_score: float = Field(..., description="Prediction confidence score (0-1)", ge=0, le=1)
    model_version: str = Field(..., description="Model version used for prediction")
    prediction_timestamp: str = Field(..., description="ISO timestamp of prediction")
    input_data: Dict = Field(..., description="Input data used for prediction")
    processing_time_ms: float = Field(..., description="Time taken to process prediction in milliseconds")

class HealthResponse(BaseModel):
    """
    Health check response model.
    
    Provides comprehensive health status including model loading state,
    version information, and system status.
    """
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded and ready")
    model_version: str = Field(..., description="Version of the loaded model")
    timestamp: str = Field(..., description="Current server timestamp")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")

class ModelInfoResponse(BaseModel):
    """
    Model information response model.
    
    Provides detailed information about the loaded model including
    performance metrics, training details, and configuration.
    """
    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of machine learning model")
    training_date: str = Field(..., description="Date when model was trained")
    feature_count: int = Field(..., description="Number of features used by the model")
    performance_metrics: Dict = Field(..., description="Model performance metrics")

# =============================================================================
# GLOBAL VARIABLES AND MODEL MANAGEMENT
# =============================================================================

# Global variables for model and pipeline components
model = None
feature_engineer = None
preprocessor = None
model_metadata = None
startup_time = None

def load_model():
    """
    Load the trained model and preprocessing components.
    
    This function loads the trained machine learning model and associated
    preprocessing components (feature engineering and data preprocessing)
    from the model artifacts directory.
    """
    global model, feature_engineer, preprocessor, model_metadata
    
    try:
        logger.info("🔄 Loading trained model and preprocessing components")
        
        # Load model and metadata from artifacts
        from utils.config import MODEL_ARTIFACTS_PATH
        model_path = MODEL_ARTIFACTS_PATH / MODEL_VERSION
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        model, model_metadata = ModelPersistence.load_model(str(model_path))
        
        # Initialize preprocessing components
        feature_engineer = FeatureEngineer(add_technical_indicators=True)
        preprocessor = DataPreprocessor(scale_features=True)
        
        logger.info("✅ Model and preprocessing components loaded successfully")
        logger.info(f"📊 Model type: {model_metadata.get('model_type', 'Unknown')}")
        logger.info(f"🔧 Model parameters: {len(model_metadata.get('model_params', {}))} parameters")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def predict_single(data: CryptoData) -> Dict:
    """
    Make a single price prediction.
    
    Args:
        data: Input cryptocurrency data
        
    Returns:
        Dictionary containing prediction results
        
    Raises:
        Exception: If prediction fails
    """
    try:
        # Convert input data to dataframe
        input_df = pd.DataFrame([data.dict()])
        
        # Apply feature engineering
        engineered_data = feature_engineer.transform(input_df)
        
        # Apply preprocessing
        processed_data = preprocessor.transform(engineered_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Calculate confidence score (simplified - could be enhanced with uncertainty quantification)
        confidence_score = 0.85  # Placeholder - could be based on model uncertainty
        
        return {
            'predicted_price': float(prediction),
            'confidence_score': confidence_score,
            'input_data': data.dict()
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

# =============================================================================
# FASTAPI EVENT HANDLERS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    
    This function is called when the FastAPI application starts up.
    It loads the trained model and initializes the application state.
    """
    global startup_time
    startup_time = datetime.now()
    
    logger.info("🚀 Starting Crypto Price Prediction API")
    
    try:
        load_model()
        logger.info("✅ API startup completed successfully")
    except Exception as e:
        logger.error(f"❌ API startup failed: {e}")
        # In production, you might want to exit here
        # sys.exit(1)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict)
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        Dictionary with API information and available endpoints
    """
    return {
        "message": "Crypto Price Prediction API",
        "version": MODEL_VERSION,
        "description": "Machine learning API for cryptocurrency price prediction",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "model_info": "/model_info",
            "feature_importance": "/feature_importance",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Provides comprehensive health status including model loading state,
    server uptime, and system status.
    
    Returns:
        HealthResponse with detailed health information
    """
    global startup_time
    
    uptime = (datetime.now() - startup_time).total_seconds() if startup_time else 0
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(data: CryptoData):
    """
    Single prediction endpoint.
    
    Makes a price prediction for a single cryptocurrency data point.
    
    Args:
        data: Cryptocurrency data for prediction
        
    Returns:
        PredictionResponse with predicted price and metadata
        
    Raises:
        HTTPException: If prediction fails or model is not loaded
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Make prediction
        prediction_result = predict_single(data)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResponse(
            predicted_price=prediction_result['predicted_price'],
            confidence_score=prediction_result['confidence_score'],
            model_version=MODEL_VERSION,
            prediction_timestamp=datetime.now().isoformat(),
            input_data=prediction_result['input_data'],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(data_list: List[CryptoData]):
    """
    Batch prediction endpoint.
    
    Makes price predictions for multiple cryptocurrency data points efficiently.
    
    Args:
        data_list: List of cryptocurrency data for prediction
        
    Returns:
        List of PredictionResponse objects
        
    Raises:
        HTTPException: If prediction fails or model is not loaded
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(data_list) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large. Maximum 100 items.")
    
    start_time = datetime.now()
    predictions = []
    
    try:
        for i, data in enumerate(data_list):
            item_start_time = datetime.now()
            
            # Make prediction for this item
            prediction_result = predict_single(data)
            
            # Calculate processing time for this item
            processing_time = (datetime.now() - item_start_time).total_seconds() * 1000
            
            predictions.append(PredictionResponse(
                predicted_price=prediction_result['predicted_price'],
                confidence_score=prediction_result['confidence_score'],
                model_version=MODEL_VERSION,
                prediction_timestamp=datetime.now().isoformat(),
                input_data=prediction_result['input_data'],
                processing_time_ms=processing_time
            ))
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Batch prediction completed: {len(predictions)} items in {total_time:.2f}ms")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Model information endpoint.
    
    Provides detailed information about the loaded model including
    performance metrics, training details, and configuration.
    
    Returns:
        ModelInfoResponse with detailed model information
        
    Raises:
        HTTPException: If model is not loaded
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        return ModelInfoResponse(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            model_type=model_metadata.get('model_type', 'Unknown'),
            training_date=model_metadata.get('save_timestamp', 'Unknown'),
            feature_count=len(model_metadata.get('feature_importance', [])),
            performance_metrics=model_metadata.get('evaluation_results', {}).get('summary', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/feature_importance", response_model=List[Dict])
async def get_feature_importance():
    """
    Feature importance endpoint.
    
    Returns the feature importance rankings from the trained model,
    helping users understand which features drive the predictions.
    
    Returns:
        List of dictionaries with feature importance information
        
    Raises:
        HTTPException: If model is not loaded
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        feature_importance = model_metadata.get('feature_importance', [])
        
        if not feature_importance:
            return []
        
        # Return top 20 features
        return feature_importance[:20]
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

# =============================================================================
# ERROR HANDLING
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler.
    
    Catches all unhandled exceptions and returns appropriate HTTP responses
    with error details for debugging.
    
    Args:
        request: The request that caused the exception
        exc: The exception that was raised
        
    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {exc}")
    
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat(),
        "path": str(request.url)
    }

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Application entry point.
    
    Starts the FastAPI server with uvicorn when the script is run directly.
    """
    logger.info(f"🚀 Starting Crypto Price Prediction API server")
    logger.info(f"🌐 Server will be available at: http://{API_HOST}:{API_PORT}")
    logger.info(f"📚 API documentation: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,  # Enable auto-reload for development
        log_level=LOG_LEVEL.lower(),
        access_log=True
    ) 