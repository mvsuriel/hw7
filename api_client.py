"""
Robust API Client for Diabetes Mellitus Prediction
Handles various API-related errors and edge cases
"""
import requests
import json
import sys
from requests.exceptions import (
    ConnectionError,
    Timeout,
    HTTPError,
    RequestException
)


def make_prediction(data, model="logreg", api_url="http://localhost:8000", timeout=10):
    """
    Make a prediction request to the API with comprehensive error handling
    
    Parameters:
    -----------
    data : dict
        Dictionary containing patient features
    model : str
        Model to use ('logreg' or 'rf')
    api_url : str
        Base URL of the API
    timeout : int
        Request timeout in seconds
        
    Returns:
    --------
    dict or None
        Prediction result or None if error occurred
    """
    
    # Validate input data has all required features
    required_features = [
        "age", "height", "weight",
        "aids", "cirrhosis", "hepatic_failure",
        "immunosuppression", "leukemia", "lymphoma",
        "solid_tumor_with_metastasis"
    ]
    
    missing_features = [f for f in required_features if f not in data]
    if missing_features:
        print(f"❌ Error: Missing required features: {missing_features}")
        return None
    
    # Prepare request payload
    payload = {
        "model": model,
        "features": data
    }
    
    endpoint = f"{api_url}/predict"
    
    print(f"\n{'='*70}")
    print(f"Making Prediction Request")
    print(f"{'='*70}")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {model}")
    print(f"Patient Data:")
    for key, value in data.items():
        print(f"  - {key}: {value}")
    print(f"{'='*70}\n")
    
    try:
        # Make POST request with timeout
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout
        )
        
        # Raise exception for bad status codes (4xx, 5xx)
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        
        # Display results
        print("✅ Prediction Successful!")
        print(f"\n{'='*70}")
        print("PREDICTION RESULTS")
        print(f"{'='*70}")
        print(f"Model Used: {result.get('model_used', 'Unknown')}")
        
        prediction = result.get('prediction', {})
        print(f"\nPrediction Class: {prediction.get('class', 'N/A')}")
        
        if prediction.get('class') == 0:
            print("  → Patient is predicted to NOT have diabetes mellitus")
        elif prediction.get('class') == 1:
            print("  → Patient is predicted to HAVE diabetes mellitus")
        
        probabilities = prediction.get('probability', {})
        print(f"\nProbabilities:")
        print(f"  - No Diabetes: {probabilities.get('no_diabetes', 0):.4f} ({probabilities.get('no_diabetes', 0)*100:.2f}%)")
        print(f"  - Diabetes:    {probabilities.get('diabetes', 0):.4f} ({probabilities.get('diabetes', 0)*100:.2f}%)")
        
        confidence = prediction.get('confidence', 0)
        print(f"\nConfidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"{'='*70}\n")
        
        return result
        
    except ConnectionError:
        print("❌ Connection Error: Could not connect to the API server.")
        print("   → Make sure the API is running on http://localhost:8000")
        print("   → Start the API with: python api.py")
        return None
        
    except Timeout:
        print(f"❌ Timeout Error: Request took longer than {timeout} seconds.")
        print("   → The API server might be overloaded or not responding.")
        print("   → Try increasing the timeout or check server status.")
        return None
        
    except HTTPError as e:
        print(f"❌ HTTP Error {response.status_code}: {e}")
        
        # Try to parse error message from response
        try:
            error_data = response.json()
            print(f"   → Error: {error_data.get('error', 'Unknown error')}")
            if 'message' in error_data:
                print(f"   → Message: {error_data['message']}")
            if 'missing' in error_data:
                print(f"   → Missing features: {error_data['missing']}")
        except:
            print(f"   → Response: {response.text}")
        
        return None
        
    except json.JSONDecodeError:
        print("❌ JSON Decode Error: Invalid JSON response from server.")
        print(f"   → Response text: {response.text[:200]}")
        return None
        
    except RequestException as e:
        print(f"❌ Request Error: {e}")
        print("   → An unexpected error occurred while making the request.")
        return None
        
    except Exception as e:
        print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
        return None


def check_api_health(api_url="http://localhost:8000", timeout=5):
    """
    Check if the API is running and healthy
    
    Parameters:
    -----------
    api_url : str
        Base URL of the API
    timeout : int
        Request timeout in seconds
        
    Returns:
    --------
    bool
        True if API is healthy, False otherwise
    """
    try:
        response = requests.get(f"{api_url}/health", timeout=timeout)
        response.raise_for_status()
        
        health_data = response.json()
        status = health_data.get('status', 'unknown')
        
        if status == 'healthy':
            print("✅ API is healthy and ready!")
            models = health_data.get('models_loaded', {})
            print(f"   Models loaded: {', '.join([k for k, v in models.items() if v])}")
            return True
        else:
            print(f"⚠️  API status: {status}")
            return False
            
    except ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on", api_url)
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DIABETES MELLITUS PREDICTION API CLIENT")
    print("="*70)
    
    # Step 1: Check API health
    print("\nStep 1: Checking API Health...")
    api_healthy = check_api_health()
    
    if not api_healthy:
        print("\n⚠️  Warning: API is not responding. Attempting prediction anyway...\n")
    
    # Step 2: Define patient data
    patient_data = {
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
    
    # Step 3: Make prediction with Logistic Regression
    print("\n" + "="*70)
    print("EXAMPLE 1: Prediction with Logistic Regression")
    print("="*70)
    result_lr = make_prediction(patient_data, model="logreg")
    
    # Step 4: Make prediction with Random Forest
    print("\n" + "="*70)
    print("EXAMPLE 2: Prediction with Random Forest")
    print("="*70)
    result_rf = make_prediction(patient_data, model="rf")
    
    # Step 5: Test with high-risk patient
    print("\n" + "="*70)
    print("EXAMPLE 3: High-Risk Patient")
    print("="*70)
    high_risk_patient = {
        "age": 72,
        "height": 168,
        "weight": 85,
        "aids": 0,
        "cirrhosis": 1,
        "hepatic_failure": 0,
        "immunosuppression": 1,
        "leukemia": 0,
        "lymphoma": 0,
        "solid_tumor_with_metastasis": 1
    }
    result_high_risk = make_prediction(high_risk_patient, model="logreg")
    
    # Step 6: Test error handling - missing feature
    print("\n" + "="*70)
    print("EXAMPLE 4: Error Handling - Missing Feature")
    print("="*70)
    incomplete_data = {
        "age": 45,
        "height": 175,
        "weight": 80
        # Missing other required features
    }
    result_error = make_prediction(incomplete_data, model="logreg")
    
    # Step 7: Test with invalid endpoint (demonstrate error handling)
    print("\n" + "="*70)
    print("EXAMPLE 5: Error Handling - Wrong API URL")
    print("="*70)
    try:
        response = requests.post(
            "http://localhost:9999/predict",  # Wrong port
            json={"model": "logreg", "features": patient_data},
            timeout=3
        )
    except ConnectionError:
        print("✅ Connection error handled correctly!")
        print("   (Expected behavior when API is not available)")
    except Exception as e:
        print(f"✅ Exception handled: {type(e).__name__}")
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\n📝 Summary of Error Handling:")
    print("  ✓ Connection errors (API not running)")
    print("  ✓ Timeout errors (slow responses)")
    print("  ✓ HTTP errors (4xx, 5xx status codes)")
    print("  ✓ JSON decode errors (invalid responses)")
    print("  ✓ Missing features validation")
    print("  ✓ General exception handling")
    print("\n")
