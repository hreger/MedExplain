"""
MedExplain - Prediction

This module handles making predictions using trained disease prediction models.
It loads the model and provides functions to make predictions on new patient data.
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler

def load_model(model_path="models/model.joblib"):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load the model
    model = joblib.load(model_path)
    
    return model

def load_scaler(scaler_path="models/scaler.pkl"):
    """
    Load the feature scaler from disk.
    
    Args:
        scaler_path (str): Path to the saved scaler
        
    Returns:
        object: Loaded scaler
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # Load the scaler
    scaler = joblib.load(scaler_path)
    
    return scaler

def load_feature_names(feature_path="models/feature_names.joblib"):
    """
    Load the feature names from disk.
    
    Args:
        feature_path (str): Path to the saved feature names
        
    Returns:
        list: Feature names
    """
    if not os.path.exists(feature_path):
        return None
    
    # Load the feature names
    feature_names = joblib.load(feature_path)
    
    return feature_names

def preprocess_input(patient_data, scaler=None):
    """
    Preprocess patient data for prediction.
    
    Args:
        patient_data (dict or DataFrame): Raw patient data 
        scaler (object): StandardScaler for feature scaling
        
    Returns:
        array: Preprocessed data ready for model input
    """
    # Convert dict to DataFrame if necessary
    if isinstance(patient_data, dict):
        patient_data = pd.DataFrame([patient_data])
    
    # Apply scaler if provided
    if scaler is not None:
        patient_data = scaler.transform(patient_data)
    
    return patient_data

def predict_disease(patient_data, model=None, threshold=0.5):
    """
    Make a disease prediction for a patient.
    
    Args:
        patient_data (dict or DataFrame): Patient features
        model (object): Pre-loaded model (optional)
        threshold (float): Probability threshold for positive class
        
    Returns:
        dict: Prediction result with class and probability
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load scaler
    try:
        scaler = load_scaler()
    except FileNotFoundError:
        scaler = None
    
    # Preprocess data
    processed_data = preprocess_input(patient_data, scaler)
    
    # Make prediction
    try:
        # Get probability for positive class
        probability = model.predict_proba(processed_data)[0, 1]
        
        # Apply threshold to determine predicted class
        prediction = 1 if probability >= threshold else 0
        
        # Map prediction to readable label
        label = "High risk" if prediction == 1 else "Low risk"
        
        return {
            "prediction": prediction,
            "label": label,
            "probability": float(probability)
        }
    except Exception as e:
        return {
            "error": str(e),
            "prediction": None,
            "probability": None
        }

def batch_predict(patient_data_list, model=None, threshold=0.5):
    """
    Make predictions for multiple patients at once.
    
    Args:
        patient_data_list (list): List of patient data (dicts or DataFrame)
        model (object): Pre-loaded model (optional)
        threshold (float): Probability threshold for positive class
        
    Returns:
        list: List of prediction results
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load scaler
    try:
        scaler = load_scaler()
    except FileNotFoundError:
        scaler = None
    
    # Collect all patient data into a DataFrame
    if isinstance(patient_data_list[0], dict):
        df = pd.DataFrame(patient_data_list)
    else:
        df = pd.concat(patient_data_list)
    
    # Preprocess data
    processed_data = preprocess_input(df, scaler)
    
    # Make predictions
    probabilities = model.predict_proba(processed_data)[:, 1]
    predictions = [1 if p >= threshold else 0 for p in probabilities]
    labels = ["High risk" if p == 1 else "Low risk" for p in predictions]
    
    # Compile results
    results = []
    for i, (pred, prob, label) in enumerate(zip(predictions, probabilities, labels)):
        results.append({
            "prediction": pred,
            "label": label,
            "probability": float(prob)
        })
    
    return results

if __name__ == "__main__":
    # Sample code for testing
    sample_patient = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50
    }
    
    print("This is a sample prediction module. Implement actual prediction with:")
    print("result = predict_disease(sample_patient)")

"""
MedExplain - Prediction

This module handles making predictions using trained disease prediction models.
It loads the model and provides functions to make predictions on new patient data.
"""

import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
import pickle
from pathlib import Path

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    # Placeholder for actual implementation
    return None

def preprocess_input(patient_data):
    """
    Preprocess patient data for prediction.
    
    Args:
        patient_data: Raw patient data (dict or dataframe)
        
    Returns:
        Preprocessed data ready for model input
    """
    # Placeholder for actual implementation
    return None

def predict_disease(patient_data, model=None, model_path=None):
    """
    Make a disease prediction for a patient.
    
    Args:
        patient_data: Patient features
        model: Pre-loaded model (optional)
        model_path: Path to model if not pre-loaded
        
    Returns:
        Prediction result and probability
    """
    # Placeholder for actual implementation
    return {"prediction": "Not implemented", "probability": 0.0}

if __name__ == "__main__":
    # Test prediction functionality
    print("Prediction module")

