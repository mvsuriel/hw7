"""
FastAPI for diabetes mellitus prediction using trained models
Compatible with uvicorn --reload
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import joblib
import os
import numpy as np
import json

app = FastAPI(
    title="Diabetes Mellitus Prediction API",
    description="API for predicting diabetes mellitus using trained ML models",
    version="1.0.0"
)

# Get working directory and model paths
wd = os.path.dirname(os.path.abspath(__file__))
model_lr_path = os.path.join(wd, "model_logistic_regression.pkl")
model_rf_path = os.path.join(wd, "model_random_forest.pkl")

# Load the trained models
print("Loading trained models...")
try:
    model_lr = joblib.load(model_lr_path)
    model_rf = joblib.load(model_rf_path)
    print(f"Logistic Regression model loaded from: {model_lr_path}")
    print(f"Random Forest model loaded from: {model_rf_path}")
except FileNotFoundError as e:
    print(f"Error: Model files not found. Please run generate_pickle_file.py first.")
    print(f"Missing file: {e.filename}")
    # Don't exit - let the app start so it can be tested
    model_lr = None
    model_rf = None

# Define expected features
FEATURES = [
    "age", "height", "weight",
    "aids", "cirrhosis", "hepatic_failure",
    "immunosuppression", "leukemia", "lymphoma",
    "solid_tumor_with_metastasis",
]

# Pydantic models for request/response validation
class PatientFeatures(BaseModel):
    age: float = Field(..., description="Patient age in years")
    height: float = Field(..., description="Patient height in cm")
    weight: float = Field(..., description="Patient weight in kg")
    aids: int = Field(..., ge=0, le=1, description="AIDS (0=No, 1=Yes)")
    cirrhosis: int = Field(..., ge=0, le=1, description="Cirrhosis (0=No, 1=Yes)")
    hepatic_failure: int = Field(..., ge=0, le=1, description="Hepatic failure (0=No, 1=Yes)")
    immunosuppression: int = Field(..., ge=0, le=1, description="Immunosuppression (0=No, 1=Yes)")
    leukemia: int = Field(..., ge=0, le=1, description="Leukemia (0=No, 1=Yes)")
    lymphoma: int = Field(..., ge=0, le=1, description="Lymphoma (0=No, 1=Yes)")
    solid_tumor_with_metastasis: int = Field(..., ge=0, le=1, description="Solid tumor with metastasis (0=No, 1=Yes)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "height": 170,
                "weight": 75,
                "aids": 0,
                "cirrhosis": 0,
                "hepatic_failure": 0,
                "immunosuppression": 0,
                "leukemia": 0,
                "lymphoma": 0,
                "solid_tumor_with_metastasis": 0
            }
        }


class PredictionRequest(BaseModel):
    model: str = Field(default="logreg", description="Model type: 'logreg' or 'rf'")
    features: PatientFeatures


class BatchPredictionRequest(BaseModel):
    model: str = Field(default="logreg", description="Model type: 'logreg' or 'rf'")
    samples: List[PatientFeatures]


class FilePredictionRequest(BaseModel):
    model: Optional[str] = Field(default="logreg", description="Model type: 'logreg' or 'rf'")
    model_path: Optional[str] = Field(default=None, description="Path to custom model file")
    data: Union[PatientFeatures, List[PatientFeatures]]


class PredictionResult(BaseModel):
    class_: int = Field(..., alias="class")
    probability: Dict[str, float]
    confidence: float


class PredictionResponse(BaseModel):
    model_used: str
    prediction: PredictionResult
    input_features: Optional[PatientFeatures] = None


@app.get("/")
async def home():
    """Home endpoint with API information"""
    return {
        "message": "Diabetes Mellitus Prediction API",
        "version": "1.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "/": "API information (this page)",
            "/predict": "POST - Make predictions using trained models",
            "/predict/batch": "POST - Batch predictions",
            "/predict/file": "POST - Predictions from JSON data with optional custom model",
            "/health": "GET - Check API health status",
            "/models": "GET - Get information about loaded models"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "logistic_regression": model_lr is not None,
            "random_forest": model_rf is not None
        }
    }


@app.get("/models")
async def models_info():
    """Get information about loaded models"""
    return {
        "models": {
            "logistic_regression": {
                "type": type(model_lr).__name__,
                "path": model_lr_path,
                "loaded": True
            },
            "random_forest": {
                "type": type(model_rf).__name__,
                "path": model_rf_path,
                "loaded": True
            }
        },
        "required_features": FEATURES
    }


def get_model(model_type: str):
    """Helper function to select the appropriate model"""
    model_type = model_type.lower()
    
    if model_type in ['logreg', 'lr', 'logistic_regression']:
        return model_lr, "Logistic Regression"
    elif model_type in ['rf', 'random_forest']:
        return model_rf, "Random Forest"
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Invalid model type: {model_type}",
                "valid_options": ["logreg", "rf"]
            }
        )


def make_prediction(model, features_dict: dict):
    """Helper function to make a prediction"""
    # Create feature array in correct order
    feature_values = [features_dict[f] for f in FEATURES]
    X = np.array([feature_values])
    
    # Make prediction
    prediction_proba = model.predict_proba(X)[0]
    prediction_class = model.predict(X)[0]
    
    return {
        "class": int(prediction_class),
        "probability": {
            "no_diabetes": float(prediction_proba[0]),
            "diabetes": float(prediction_proba[1])
        },
        "confidence": float(max(prediction_proba))
    }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Make predictions using the trained models"""
    try:
        # Select the appropriate model
        model, model_name = get_model(request.model)
        
        # Convert Pydantic model to dict
        features_dict = request.features.model_dump()
        
        # Make prediction
        prediction = make_prediction(model, features_dict)
        
        return {
            "model_used": model_name,
            "prediction": prediction,
            "input_features": features_dict
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction failed",
                "message": str(e)
            }
        )


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions for multiple samples"""
    try:
        # Select the appropriate model
        model, model_name = get_model(request.model)
        
        # Process each sample
        results = []
        for idx, sample in enumerate(request.samples):
            try:
                features_dict = sample.model_dump()
                prediction = make_prediction(model, features_dict)
                
                results.append({
                    "sample_index": idx,
                    "prediction": prediction
                })
            except Exception as e:
                results.append({
                    "sample_index": idx,
                    "error": str(e)
                })
        
        return {
            "model_used": model_name,
            "total_samples": len(request.samples),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Batch prediction failed",
                "message": str(e)
            }
        )


@app.post("/predict/file")
async def predict_from_file(request: FilePredictionRequest):
    """
    Make predictions from JSON data with optional custom model path
    
    Expected JSON structure:
    {
        "model_path": "path/to/model.pkl" (optional, loads custom model),
        "model": "logreg" or "rf" (optional, uses pre-loaded model if model_path not provided),
        "data": {
            "age": 65,
            "height": 170,
            ... (single patient data)
        }
        OR
        "data": [
            {"age": 65, "height": 170, ...},
            {"age": 45, "height": 165, ...}
        ] (multiple patients)
    }
    """
    try:
        # Check if custom model path is provided
        if request.model_path:
            # Load custom model using joblib
            if not os.path.exists(request.model_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {request.model_path}"
                )
            
            try:
                model = joblib.load(request.model_path)
                model_name = f"Custom Model ({os.path.basename(request.model_path)})"
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Failed to load model",
                        "message": str(e)
                    }
                )
        else:
            # Use pre-loaded model
            model, model_name = get_model(request.model)
        
        # Check if single patient or multiple patients
        is_batch = isinstance(request.data, list)
        
        if not is_batch:
            # Single patient prediction
            features_dict = request.data.model_dump()
            prediction = make_prediction(model, features_dict)
            
            return {
                "model_used": model_name,
                "prediction": prediction,
                "input_features": features_dict
            }
        
        else:
            # Batch predictions
            results = []
            for idx, sample in enumerate(request.data):
                try:
                    features_dict = sample.model_dump()
                    prediction = make_prediction(model, features_dict)
                    
                    results.append({
                        "sample_index": idx,
                        "prediction": prediction
                    })
                except Exception as e:
                    results.append({
                        "sample_index": idx,
                        "error": str(e)
                    })
            
            return {
                "model_used": model_name,
                "total_samples": len(request.data),
                "results": results
            }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Prediction from file failed",
                "message": str(e)
            }
        )
