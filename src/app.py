"""
FastAPI application for cryptocurrency price prediction.
Provides REST API endpoints for model inference.
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
from pydantic import BaseModel, Field
import uvicorn

from utils.config import API_HOST, API_PORT, MODEL_NAME, MODEL_VERSION, LOG_LEVEL, LOG_FORMAT
from training.data import DataPipeline
from training.model import ModelPersistence
from training.features import FeatureEngineer
from training.data import DataPreprocessor

# Set up logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Price Prediction API",
    description="API for predicting cryptocurrency prices using machine learning",
    version=MODEL_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CryptoData(BaseModel):
    """Input data model for cryptocurrency price prediction."""
    SNo: int = Field(..., description="Serial number")
    Name: str = Field(..., description="Cryptocurrency name")
    Symbol: str = Field(..., description="Cryptocurrency symbol")
    Date: str = Field(..., description="Date in YYYY-MM-DD format")
    High: float = Field(..., description="High price")
    Low: float = Field(..., description="Low price")
    Open: float = Field(..., description="Open price")
    Volume: float = Field(..., description="Trading volume")
    Marketcap: float = Field(..., description="Market capitalization")

class PredictionResponse(BaseModel):
    """Response model for price predictions."""
    predicted_price: float = Field(..., description="Predicted closing price")
    confidence_score: float = Field(..., description="Prediction confidence score")
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    input_data: Dict = Field(..., description="Input data used for prediction")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Current timestamp")

# Global variables for model and pipeline
model = None
feature_engineer = None
preprocessor = None
model_metadata = None

def load_model():
    """Load the trained model and preprocessing components."""
    global model, feature_engineer, preprocessor, model_metadata
    
    try:
        logger.info("Loading trained model")
        
        # Load model and metadata
        from utils.config import MODEL_ARTIFACTS_PATH
        model_path = MODEL_ARTIFACTS_PATH / MODEL_VERSION
        model, model_metadata = ModelPersistence.load_model(str(model_path))
        
        # Initialize preprocessing components
        feature_engineer = FeatureEngineer(add_technical_indicators=True)
        preprocessor = DataPreprocessor(scale_features=True)
        
        # Note: In a real deployment, you'd need to fit the preprocessor
        # For now, we'll use a simple approach
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Crypto Price Prediction API")
    try:
        load_model()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"API startup failed: {e}")
        # In production, you might want to exit here
        # sys.exit(1)

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Crypto Price Prediction API",
        "version": MODEL_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=MODEL_VERSION,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(data: CryptoData):
    """
    Predict cryptocurrency closing price.
    
    Args:
        data: Input cryptocurrency data
        
    Returns:
        Predicted price and metadata
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert input data to dataframe
        input_df = pd.DataFrame([data.dict()])
        
        # Convert date string to datetime
        input_df['Date'] = pd.to_datetime(input_df['Date'])
        
        # Apply feature engineering
        if feature_engineer is not None:
            input_df = feature_engineer.transform(input_df)
        
        # Apply preprocessing
        if preprocessor is not None:
            # For simplicity, we'll use basic preprocessing
            # In production, you'd want to use the fitted preprocessor
            numeric_columns = input_df.select_dtypes(include=[np.number]).columns
            input_df = input_df[numeric_columns]
            
            # Handle missing values
            input_df = input_df.fillna(input_df.median())
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Calculate confidence score (simplified)
        # In practice, you might use prediction intervals or model uncertainty
        confidence_score = 0.85  # Placeholder
        
        # Create response
        response = PredictionResponse(
            predicted_price=float(prediction),
            confidence_score=confidence_score,
            model_version=MODEL_VERSION,
            prediction_timestamp=datetime.now().isoformat(),
            input_data=data.dict()
        )
        
        logger.info(f"Prediction made: ${prediction:,.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(data_list: List[CryptoData]):
    """
    Predict cryptocurrency prices for multiple inputs.
    
    Args:
        data_list: List of input cryptocurrency data
        
    Returns:
        List of predicted prices and metadata
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions = []
        
        for data in data_list:
            # Convert input data to dataframe
            input_df = pd.DataFrame([data.dict()])
            
            # Convert date string to datetime
            input_df['Date'] = pd.to_datetime(input_df['Date'])
            
            # Apply feature engineering
            if feature_engineer is not None:
                input_df = feature_engineer.transform(input_df)
            
            # Apply preprocessing
            if preprocessor is not None:
                numeric_columns = input_df.select_dtypes(include=[np.number]).columns
                input_df = input_df[numeric_columns]
                input_df = input_df.fillna(input_df.median())
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Create response
            response = PredictionResponse(
                predicted_price=float(prediction),
                confidence_score=0.85,  # Placeholder
                model_version=MODEL_VERSION,
                prediction_timestamp=datetime.now().isoformat(),
                input_data=data.dict()
            )
            
            predictions.append(response)
        
        logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info", response_model=Dict)
async def get_model_info():
    """Get information about the loaded model."""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        info = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_type": type(model).__name__,
            "model_params": model.get_params() if hasattr(model, 'get_params') else {},
            "feature_count": model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
            "loaded_at": datetime.now().isoformat()
        }
        
        if model_metadata:
            info.update(model_metadata)
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@app.get("/feature_importance", response_model=List[Dict])
async def get_feature_importance():
    """Get feature importance from the model."""
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        if not hasattr(model, 'feature_importances_'):
            raise HTTPException(status_code=400, detail="Model does not support feature importance")
        
        # Get feature names (this would need to be stored/loaded)
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        
        importance_data = [
            {"feature": name, "importance": float(importance)}
            for name, importance in zip(feature_names, model.feature_importances_)
        ]
        
        # Sort by importance
        importance_data.sort(key=lambda x: x["importance"], reverse=True)
        
        return importance_data
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower()
    ) 