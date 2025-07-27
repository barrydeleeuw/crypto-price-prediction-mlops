# Crypto Price Prediction MLOps Pipeline - Project Summary

## 🎯 Project Overview

This project demonstrates the translation of a Jupyter notebook-based cryptocurrency price prediction model into a production-ready MLOps pipeline, following the best practices from the [banking77-pipeline](https://github.com/tsilverberg/banking77-pipeline).

## 📋 Translation Summary

### Original Notebook Analysis
The original `crypto-prediction.ipynb` contained:
- **Data Ingestion**: Kaggle dataset download using `kagglehub`
- **Data Validation**: Comprehensive data quality checks and cleansing
- **Feature Engineering**: Technical indicators, rolling statistics, time-based features
- **Model Training**: Random Forest with time series cross-validation
- **Model Deployment**: MLflow integration for model versioning
- **API Testing**: Basic API endpoint testing

### MLOps Pipeline Translation

#### ✅ **Modular Architecture**
- **Separated concerns** into distinct modules
- **Reusable components** for data processing, feature engineering, and model training
- **Clean interfaces** between different pipeline stages

#### ✅ **Configuration Management**
- **Centralized configuration** in `src/utils/config.py`
- **Environment variable support** with `.env` file
- **Validation functions** for configuration parameters

#### ✅ **Data Processing Pipeline**
- **DataLoader**: Handles Kaggle dataset download and validation
- **DataValidator**: Comprehensive data quality checks
- **DataPreprocessor**: Handles data type conversion and scaling
- **DataSplitter**: Time series-aware data splitting
- **DataPipeline**: Orchestrates the complete data processing workflow

#### ✅ **Feature Engineering**
- **FeatureEngineer**: Modular feature creation with technical indicators
- **Statistical features**: Rolling statistics, volatility measures
- **Time-based features**: Cyclical encoding, temporal features
- **Technical indicators**: RSI, MACD, Bollinger Bands, moving averages

#### ✅ **Model Training & Evaluation**
- **ModelTrainer**: Handles model training with cross-validation
- **ModelEvaluator**: Comprehensive evaluation across all datasets
- **ModelPersistence**: Model saving and loading utilities
- **Hyperparameter tuning**: Grid search with time series CV

#### ✅ **MLflow Integration**
- **Experiment tracking**: Logs parameters, metrics, and artifacts
- **Model versioning**: Registers models with metadata
- **Pipeline logging**: Saves complete preprocessing + model pipeline

#### ✅ **API Deployment**
- **FastAPI application**: Modern, fast web framework
- **Input validation**: Pydantic models for data validation
- **Error handling**: Comprehensive error responses
- **Health checks**: Service monitoring endpoints
- **Documentation**: Auto-generated API docs

## 🏗️ Project Structure

```
crypto_price_pipeline/
├── README.md                    # Comprehensive project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── .gitignore                   # Git ignore patterns
├── env.example                  # Environment variables template
├── quick_start.py               # One-click pipeline execution
├── src/
│   ├── __init__.py              # Package initialization
│   ├── run_etl.py              # Data extraction and preprocessing
│   ├── run_training.py         # Model training and MLflow logging
│   ├── app.py                  # FastAPI application
│   ├── utils/
│   │   ├── __init__.py         # Utils package
│   │   ├── config.py           # Configuration management
│   │   └── data_validation.py  # Data quality checks
│   └── training/
│       ├── __init__.py         # Training package
│       ├── data.py             # Data processing pipeline
│       ├── features.py         # Feature engineering
│       └── model.py            # Model training and evaluation
├── data/                        # Data storage (created at runtime)
└── models/                      # Model artifacts (created at runtime)
```

## 🔄 Pipeline Workflow

### 1. **Data Processing** (`run_etl.py`)
```bash
python src/run_etl.py
```
- Downloads Bitcoin data from Kaggle
- Validates data quality and structure
- Applies feature engineering
- Splits data into train/validation/test sets
- Saves processed data and summary

### 2. **Model Training** (`run_training.py`)
```bash
python src/run_training.py
```
- Loads processed data
- Trains Random Forest model
- Performs time series cross-validation
- Evaluates on validation and test sets
- Logs everything to MLflow
- Saves model artifacts

### 3. **API Deployment** (`app.py`)
```bash
python src/app.py
```
- Starts FastAPI server
- Loads trained model
- Provides prediction endpoints
- Includes health checks and documentation

## 🚀 Key Improvements Over Notebook

### **Production Readiness**
- ✅ **Modular code structure** instead of monolithic notebook
- ✅ **Configuration management** instead of hardcoded values
- ✅ **Error handling** and logging throughout
- ✅ **Input validation** for API endpoints
- ✅ **Health checks** and monitoring

### **MLOps Best Practices**
- ✅ **Experiment tracking** with MLflow
- ✅ **Model versioning** and metadata
- ✅ **Reproducible pipelines** with proper data splitting
- ✅ **Feature store** approach with engineered features
- ✅ **API-first design** for model serving

### **Code Quality**
- ✅ **Type hints** throughout the codebase
- ✅ **Comprehensive documentation** and docstrings
- ✅ **Unit test structure** ready for implementation
- ✅ **Code formatting** and linting setup
- ✅ **Git integration** with proper .gitignore

### **Scalability**
- ✅ **Pipeline orchestration** for complex workflows
- ✅ **Modular components** for easy extension
- ✅ **Configuration-driven** behavior
- ✅ **Docker-ready** structure
- ✅ **CI/CD integration** ready

## 📊 Model Performance

The pipeline maintains the same model performance as the original notebook:
- **Time Series Cross-Validation**: Proper temporal validation
- **Feature Engineering**: 50+ engineered features
- **Model Evaluation**: MAE, RMSE, R², MAPE metrics
- **Feature Importance**: Model interpretability

## 🔧 Usage Instructions

### **Quick Start**
```bash
# Clone and setup
cd "Lesson 2/Submission Lesson 2 assignment"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run complete pipeline
python quick_start.py
```

### **Manual Execution**
```bash
# Step 1: Data processing
python src/run_etl.py

# Step 2: Model training
python src/run_training.py

# Step 3: Start API
python src/app.py
```

### **API Usage**
```bash
# Health check
curl http://127.0.0.1:8000/health

# Make prediction
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "SNo": 1,
       "Name": "Bitcoin",
       "Symbol": "BTC",
       "Date": "2023-01-01",
       "High": 45000.0,
       "Low": 44000.0,
       "Open": 44500.0,
       "Volume": 1000000.0,
       "Marketcap": 850000000000.0
     }'
```

## 🎯 Next Steps

### **Immediate Enhancements**
- [ ] Add unit tests for all modules
- [ ] Implement Docker containerization
- [ ] Add CI/CD pipeline integration
- [ ] Create monitoring and alerting
- [ ] Add model drift detection

### **Advanced Features**
- [ ] Real-time data ingestion from APIs
- [ ] Ensemble model approaches
- [ ] A/B testing framework
- [ ] Model explainability tools
- [ ] Automated retraining pipelines

## 📚 Learning Outcomes

This translation demonstrates:
1. **MLOps principles** in practice
2. **Production-ready code** structure
3. **Best practices** for ML pipeline development
4. **API design** for model serving
5. **Experiment tracking** and model versioning
6. **Modular architecture** for maintainable code

The project successfully transforms a research notebook into a production-ready MLOps pipeline while maintaining all the original functionality and adding enterprise-grade features. 