"""
Run the FastAPI server for diabetes mellitus prediction
This file contains the startup logic and model loading
"""
import uvicorn
import os
import joblib

# Global variables for models
model_lr = None
model_rf = None
model_lr_path = None
model_rf_path = None
wd = None

def load_models():
    """Load the trained models from pickle files"""
    global model_lr, model_rf, model_lr_path, model_rf_path, wd
    
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
        return True
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run generate_pickle_file.py first.")
        print(f"Missing file: {e.filename}")
        return False

def get_models():
    """Return the loaded models"""
    return model_lr, model_rf

def get_model_paths():
    """Return the model file paths"""
    return model_lr_path, model_rf_path

if __name__ == '__main__':
    # Load models first
    if not load_models():
        print("Failed to load models. Exiting...")
        exit(1)
    
    print("\n" + "=" * 60)
    print("Starting Diabetes Mellitus Prediction API (FastAPI)")
    print("=" * 60)
    print(f"Working directory: {wd}")
    print(f"Models loaded successfully!")
    print("\nAPI Endpoints:")
    print("  - GET  /                : API information")
    print("  - GET  /health          : Health check")
    print("  - GET  /models          : Model information")
    print("  - POST /predict         : Single prediction")
    print("  - POST /predict/batch   : Batch predictions")
    print("  - POST /predict/file    : Predictions from JSON data")
    print("                            (supports custom model loading)")
    print("\nStarting server with Uvicorn...")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run("api_fastapi:app", host="0.0.0.0", port=8000, reload=True)
