"""
MedExplain - Gradio Frontend

This module provides the Gradio-based user interface for the MedExplain application.
It offers an alternative to the Streamlit frontend with similar functionality.
"""

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from src.predict import predict_disease, load_model, load_feature_names, load_scaler
from src.explain import get_lime_explainer, explain_prediction

# Load model and related files
MODEL_PATH = "models/model.joblib"
SCALER_PATH = "models/scaler.joblib"
FEATURE_NAMES_PATH = "models/feature_names.joblib"

# Define feature info for Pima Diabetes dataset
FEATURE_INFO = {
    'Pregnancies': {
        'description': 'Number of pregnancies',
        'min': 0,
        'max': 17,
        'default': 1
    },
    'Glucose': {
        'description': 'Plasma glucose concentration (mg/dL)',
        'min': 0,
        'max': 200,
        'default': 120
    },
    'BloodPressure': {
        'description': 'Diastolic blood pressure (mm Hg)',
        'min': 0,
        'max': 122,
        'default': 70
    },
    'SkinThickness': {
        'description': 'Triceps skin fold thickness (mm)',
        'min': 0,
        'max': 99,
        'default': 20
    },
    'Insulin': {
        'description': '2-Hour serum insulin (mu U/ml)',
        'min': 0,
        'max': 846,
        'default': 80
    },
    'BMI': {
        'description': 'Body mass index (kg/m²)',
        'min': 0,
        'max': 67.1,
        'default': 32
    },
    'DiabetesPedigreeFunction': {
        'description': 'Diabetes pedigree function',
        'min': 0.078,
        'max': 2.42,
        'default': 0.5
    },
    'Age': {
        'description': 'Age (years)',
        'min': 21,
        'max': 81,
        'default': 30
    }
}

def predict_and_explain(*args):
    """Make prediction and generate explanation for input features."""
    try:
        # Convert input to numpy array
        input_data = np.array(args).reshape(1, -1)
        
        # Load necessary components
        model = load_model(MODEL_PATH)
        scaler = load_scaler(SCALER_PATH)
        feature_names = load_feature_names(FEATURE_NAMES_PATH)
        
        # Make prediction
        prediction = predict_disease(input_data, model=model)
        
        # Generate explanation
        explanation = explain_prediction(model, input_data, feature_names)
        
        # Format output
        risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
        probability = prediction[1][0][1] if prediction[0] == 1 else prediction[1][0][0]
        
        # Create explanation text
        explanation_text = f"Prediction: {risk_level} (Probability: {probability:.2%})\n\n"
        explanation_text += "Feature Importance:\n"
        for feature, importance in explanation.items():
            explanation_text += f"- {feature}: {importance:.2f}\n"
        
        return explanation_text
        
    except Exception as e:
        return f"Error: {str(e)}"

def create_interface():
    """Create and configure the Gradio interface."""
    # Create input components
    inputs = []
    for feature, info in FEATURE_INFO.items():
        inputs.append(
            gr.Slider(
                minimum=info['min'],
                maximum=info['max'],
                value=info['default'],
                label=f"{feature} ({info['description']})"
            )
        )
    
    # Create interface
    interface = gr.Interface(
        fn=predict_and_explain,
        inputs=inputs,
        outputs=gr.Textbox(label="Prediction and Explanation"),
        title="MedExplain: AI-Driven Medical Diagnosis Support",
        description="Transparent, interpretable medical predictions using explainable AI",
        examples=[
            [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # High risk example
            [1, 85, 66, 29, 0, 26.6, 0.351, 31]    # Low risk example
        ]
    )
    
    return interface

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found. Please train the model first.")
    else:
        interface = create_interface()
        interface.launch()

