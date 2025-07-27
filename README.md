# 🚀 Crypto Price Prediction MLOps Pipeline

This project demonstrates a **mini MLOps pipeline** for cryptocurrency price prediction using Bitcoin historical data. It covers:

* ✅ Data ingestion & transformation (Kaggle dataset)
* ✅ Feature engineering (technical indicators, time-based features)
* ✅ Model training (Random Forest with time series validation)
* ✅ Model persistence (MLflow)
* ✅ Mock deployment (FastAPI API)

# 💡 Why This Project?

This project showcases a **production-ready MLOps pipeline** for financial time series prediction, demonstrating:

* ✅ **Time series best practices** - Proper temporal data splitting and validation
* ✅ **Feature engineering** - Technical indicators, rolling statistics, and time-based features
* ✅ **Model versioning** - MLflow integration for experiment tracking and model registry
* ✅ **API deployment** - FastAPI-based inference service
* ✅ **Data validation** - Comprehensive data quality checks and cleansing

It's a practical example of how to build scalable, maintainable ML systems for financial applications.

---

## 📁 Project Structure

```
crypto_price_pipeline/
├── .env                          ← Environment variables
├── data/                         ← Raw and processed data
├── models/                       ← Trained model artifacts
├── src/
│   ├── app.py                    ← FastAPI app for inference
│   ├── run_etl.py               ← Data extraction and preprocessing
│   ├── run_training.py          ← Model training and MLflow logging
│   ├── utils/
│   │   ├── config.py            ← Centralized configuration
│   │   └── data_validation.py   ← Data quality checks
│   └── training/
│       ├── data.py              ← Data processing pipeline
│       ├── features.py          ← Feature engineering
│       └── model.py             ← Model training and evaluation
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## 📦 Setup

### 1. Clone and install dependencies

```bash
cd "Lesson 2/Submission Lesson 2 assignment"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
# .env file at project root
KAGGLE_DATASET=sudalairajkumar/cryptocurrencypricehistory
CRYPTO_SYMBOL=BTC
MODEL_NAME=CryptoPricePredictor
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

---

## 🔄 Run the Pipeline

### ✅ Extract & Preprocess Data

```bash
python3 src/run_etl.py
```

### ✅ Train the Model

```bash
python3 src/run_training.py
```

---

## 🚀 Run the API

### Start server

```bash
cd src/
uvicorn app:app --reload --port 8000
```

### Open in browser

http://127.0.0.1:8000/docs

### Example request

```json
POST /predict
{
  "SNo": 1,
  "Name": "Bitcoin",
  "Symbol": "BTC",
  "Date": "2023-01-01",
  "High": 45000.0,
  "Low": 44000.0,
  "Open": 44500.0,
  "Volume": 1000000.0,
  "Marketcap": 850000000000.0
}
```

### Example response

```json
{
  "predicted_price": 44750.25,
  "confidence_score": 0.85,
  "model_version": "v1.0.0"
}
```

---

## 🧪 Model Performance

The pipeline includes comprehensive evaluation metrics:

* **Time Series Cross-Validation**: Respects temporal order
* **Validation Metrics**: MAE, RMSE, R² score
* **Holdout Testing**: Unseen future data evaluation
* **Feature Importance**: Model interpretability

---

## 📚 Key Features

### Data Processing
- **Temporal validation**: Proper time series data splitting
- **Data quality checks**: Missing values, outliers, data types
- **Feature engineering**: Technical indicators, rolling statistics
- **Data cleansing**: Handling edge cases and invalid values

### Model Training
- **Random Forest**: Robust ensemble method for time series
- **Hyperparameter tuning**: Grid search with time series CV
- **Feature selection**: Automatic feature importance ranking
- **Model persistence**: MLflow integration for versioning

### Deployment
- **FastAPI**: Modern, fast web framework
- **Input validation**: Pydantic models for data validation
- **Error handling**: Comprehensive error responses
- **Documentation**: Auto-generated API docs

---

## 🔧 Development

### Adding New Features

1. Extend `src/training/features.py` with new feature functions
2. Update the preprocessing pipeline in `src/training/data.py`
3. Retrain the model with `python3 src/run_training.py`

### Model Improvements

1. Experiment with different algorithms in `src/training/model.py`
2. Add hyperparameter tuning in the training pipeline
3. Implement ensemble methods for better predictions

---

## 📊 Dataset

The pipeline uses the [Cryptocurrency Price History](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory) dataset from Kaggle, which includes:

- **Bitcoin historical data** from 2013-2023
- **Daily OHLCV data** (Open, High, Low, Close, Volume)
- **Market capitalization** information
- **Clean, structured format** ready for ML

---

## 🚀 Next Steps

* Add **confidence intervals** for predictions
* Implement **real-time data ingestion** from APIs
* Add **model monitoring** and drift detection
* Create **Docker containerization** for deployment
* Add **CI/CD pipeline** integration
* Implement **A/B testing** for model versions

---

## 📚 Credits

* [Kaggle Cryptocurrency Dataset](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory)
* [MLflow](https://mlflow.org/) for experiment tracking
* [FastAPI](https://fastapi.tiangolo.com/) for API development
* [scikit-learn](https://scikit-learn.org/) for ML algorithms
