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
import logging
from pathlib import Path
import lime
import lime.lime_tabular
import shap

class ModelError(Exception):
    """Custom exception for model-related errors"""
    pass

def load_model(model_path="models/model.joblib"):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded model
    """
    try:
        if not Path(model_path).exists():
            raise ModelError(f"Model file not found: {model_path}")
            
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise ModelError(f"Failed to load model: {str(e)}")

def load_scaler(scaler_path="models/scaler.joblib"):
    """
    Load the feature scaler from disk.
    
    Args:
        scaler_path (str): Path to the saved scaler
        
    Returns:
        object: Loaded scaler
    """
    try:
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        logging.error(f"Error loading scaler: {str(e)}")
        raise

def load_feature_names(feature_path="models/feature_names.joblib"):
    """
    Load the feature names from disk.
    
    Args:
        feature_path (str): Path to the saved feature names
        
    Returns:
        list: Feature names
    """
    try:
        if not Path(feature_path).exists():
            return None
        
        feature_names = joblib.load(feature_path)
        return feature_names
    except Exception as e:
        logging.error(f"Error loading feature names: {str(e)}")
        return None

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
    
    # Ensure data is in correct format
    if not isinstance(patient_data, pd.DataFrame):
        raise ValueError("Input must be a dictionary or DataFrame")
    
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
    try:
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
        probability = model.predict_proba(processed_data)[0, 1]
        prediction = 1 if probability >= threshold else 0
        label = "High risk" if prediction == 1 else "Low risk"
        
        return {
            "prediction": prediction,
            "label": label,
            "probability": float(probability)
        }
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {
            "error": str(e),
            "prediction": None,
            "label": None,
            "probability": None
        }

def predict_with_explanation(data):
    """
    Make a prediction with LIME and SHAP explanations.
    
    Args:
        data (dict): Patient features
        
    Returns:
        dict: Prediction result with explanations
    """
    try:
        # Load models and scaler
        model = load_model()
        scaler = load_scaler()
        feature_names = load_feature_names() or [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        
        # Convert input to numpy array
        data_array = np.array(list(data.values())).reshape(1, -1)
        
        # Scale the data
        scaled_data = scaler.transform(data_array)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        # Load training data for LIME explainer
        try:
            X_train = np.load('data/processed/X_train.npy')
            X_train_scaled = scaler.transform(X_train)
        except FileNotFoundError:
            logging.warning("Training data not found, using current data for LIME")
            X_train_scaled = scaled_data
        
        # Generate LIME explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_scaled,
            feature_names=feature_names,
            class_names=['Non-diabetic', 'Diabetic'],
            mode='classification'
        )
        exp = explainer.explain_instance(
            data_array[0], 
            model.predict_proba,
            num_features=len(feature_names)
        )
        
        # Generate SHAP explanation
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(scaled_data)
        
        # Get feature importance
        feature_importance = dict(zip(feature_names, np.abs(shap_values[0])))
        sorted_importance = sorted(
            feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),
            'lime_explanation': exp.as_list(),
            'shap_values': shap_values,
            'feature_importance': sorted_importance,
            'error': None
        }
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {
            'prediction': None,
            'probability': None,
            'lime_explanation': None,
            'shap_values': None,
            'feature_importance': None,
            'error': str(e)
        }

if __name__ == "__main__":
    # Test prediction functionality
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
    
    result = predict_disease(sample_patient)
    print("Sample prediction result:", result)

