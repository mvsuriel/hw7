"""
Example script showing how to use the Diabetes Prediction API
"""
import requests
import json

# API base URL
API_URL = "http://localhost:8000"

print("=" * 70)
print("DIABETES MELLITUS PREDICTION API - USAGE EXAMPLES")
print("=" * 70)

# Example 1: Check API health
print("\n1. CHECKING API HEALTH")
print("-" * 70)
response = requests.get(f"{API_URL}/health")
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Example 2: Get API information
print("\n2. GETTING API INFORMATION")
print("-" * 70)
response = requests.get(f"{API_URL}/")
print(f"Status Code: {response.status_code}")
print(f"Available Endpoints: {list(response.json()['endpoints'].keys())}")

# Example 3: Single prediction with dictionary
print("\n3. SINGLE PREDICTION (Dictionary Input)")
print("-" * 70)
patient_data = {
    "model": "logreg",
    "features": {
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

response = requests.post(f"{API_URL}/predict", json=patient_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Example 4: Batch predictions
print("\n4. BATCH PREDICTIONS")
print("-" * 70)
batch_data = {
    "model": "rf",
    "samples": [
        {
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
        },
        {
            "age": 45,
            "height": 165,
            "weight": 68,
            "aids": 0,
            "cirrhosis": 1,
            "hepatic_failure": 0,
            "immunosuppression": 0,
            "leukemia": 0,
            "lymphoma": 0,
            "solid_tumor_with_metastasis": 0
        }
    ]
}

response = requests.post(f"{API_URL}/predict/batch", json=batch_data)
print(f"Status Code: {response.status_code}")
result = response.json()
print(f"Model Used: {result['model_used']}")
print(f"Total Samples: {result['total_samples']}")
print(f"Results: {json.dumps(result['results'], indent=2)}")

# Example 5: Prediction from dictionary using /predict/file endpoint
print("\n5. PREDICTION FROM DICTIONARY (/predict/file endpoint)")
print("-" * 70)
file_data = {
    "model": "logreg",
    "data": {
        "age": 72,
        "height": 175,
        "weight": 85,
        "aids": 0,
        "cirrhosis": 0,
        "hepatic_failure": 0,
        "immunosuppression": 1,
        "leukemia": 0,
        "lymphoma": 0,
        "solid_tumor_with_metastasis": 1
    }
}

response = requests.post(f"{API_URL}/predict/file", json=file_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

# Example 6: Load and send JSON file data
print("\n6. PREDICTION FROM JSON FILE DATA")
print("-" * 70)
try:
    with open('example_patient_data.json', 'r') as f:
        file_data = json.load(f)
        response = requests.post(f"{API_URL}/predict/file", json=file_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
except FileNotFoundError:
    print("File 'example_patient_data.json' not found. Make sure it exists in the current directory.")

# Example 7: Batch prediction from JSON file
print("\n7. BATCH PREDICTION FROM JSON FILE DATA")
print("-" * 70)
try:
    with open('example_batch_patients.json', 'r') as f:
        batch_file_data = json.load(f)
        response = requests.post(f"{API_URL}/predict/file", json=batch_file_data)
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Model Used: {result.get('model_used', 'N/A')}")
        print(f"Total Samples: {result.get('total_samples', 'N/A')}")
        if 'results' in result:
            print(f"Number of Results: {len(result['results'])}")
            # Show first result
            if result['results']:
                print(f"\nFirst Result:")
                print(json.dumps(result['results'][0], indent=2))
        else:
            print(f"Response: {json.dumps(result, indent=2)}")
except FileNotFoundError:
    print("File 'example_batch_patients.json' not found.")

# Example 8: Using custom model path
print("\n8. PREDICTION WITH CUSTOM MODEL PATH")
print("-" * 70)
custom_model_data = {
    "model_path": "model_random_forest.pkl",
    "data": {
        "age": 58,
        "height": 168,
        "weight": 72,
        "aids": 0,
        "cirrhosis": 0,
        "hepatic_failure": 0,
        "immunosuppression": 0,
        "leukemia": 0,
        "lymphoma": 0,
        "solid_tumor_with_metastasis": 0
    }
}

response = requests.post(f"{API_URL}/predict/file", json=custom_model_data)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")

print("\n" + "=" * 70)
print("ALL EXAMPLES COMPLETED")
print("=" * 70)
