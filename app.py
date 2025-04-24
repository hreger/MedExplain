"""
MedExplain - Streamlit Frontend

This module provides the Streamlit-based user interface for the MedExplain application.
It allows users to input patient data, get disease predictions, and view explanations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import requests
from io import BytesIO
import time

# Import project modules
try:
    from src.predict import predict_disease, load_model, load_feature_names
    from src.explain import get_lime_explainer  # We'll implement this fully later
except ImportError:
    st.error("Error importing project modules. Make sure you're running from the project root directory.")

# Set page configuration
st.set_page_config(
    page_title="MedExplain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define feature info for Pima Diabetes dataset
FEATURE_INFO = {
    'Pregnancies': {
        'description': 'Number of pregnancies',
        'min': 0,
        'max': 17,
        'default': 1,
        'step': 1
    },
    'Glucose': {
        'description': 'Plasma glucose concentration (mg/dL)',
        'min': 0,
        'max': 200,
        'default': 120,
        'step': 1
    },
    'BloodPressure': {
        'description': 'Diastolic blood pressure (mm Hg)',
        'min': 0,
        'max': 122,
        'default': 70,
        'step': 1
    },
    'SkinThickness': {
        'description': 'Triceps skin fold thickness (mm)',
        'min': 0,
        'max': 99,
        'default': 20,
        'step': 1
    },
    'Insulin': {
        'description': '2-Hour serum insulin (mu U/ml)',
        'min': 0,
        'max': 846,
        'default': 79,
        'step': 1
    },
    'BMI': {
        'description': 'Body mass index (weight in kg/(height in m)^2)',
        'min': 0.0,
        'max': 67.1,
        'default': 25.0,
        'step': 0.1
    },
    'DiabetesPedigreeFunction': {
        'description': 'Diabetes pedigree function (genetic influence)',
        'min': 0.078,
        'max': 2.42,
        'default': 0.5,
        'step': 0.01
    },
    'Age': {
        'description': 'Age (years)',
        'min': 21,
        'max': 81,
        'default': 35,
        'step': 1
    }
}

def check_model_exists():
    """Check if a trained model exists."""
    return os.path.exists("models/model.joblib")

def display_header():
    """Display the application header."""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Display logo or icon
        st.image("https://raw.githubusercontent.com/hreger/MedExplain/main/docs/logo.png", 
                 width=150, use_column_width=False)
    
    with col2:
        st.title("🧠 MedExplain")
        st.subheader("AI-Driven Medical Diagnosis Support with Explainable AI")
        st.markdown("""
        MedExplain provides transparent, interpretable results for disease diagnosis
        using patient data, bridging the gap between ML accuracy and medical accountability.
        """)

def data_input_form():
    """Create form for user to input patient data."""
    with st.form("patient_data_form"):
        st.subheader("Patient Data")
        
        # Create a 4x2 grid for data input
        cols = st.columns(2)
        patient_data = {}
        
        for i, (feature, info) in enumerate(FEATURE_INFO.items()):
            col_idx = i % 2
            with cols[col_idx]:
                tooltip = f"{info['description']}\nNormal range: {info['min']} - {info['max']}"
                patient_data[feature] = st.slider(
                    f"{feature} ({info['description']})",
                    min_value=info['min'],
                    max_value=info['max'],
                    value=info['default'],
                    step=info['step'],
                    help=tooltip
                )
        
        # Submit button
        submit = st.form_submit_button("Predict")
        
    return patient_data, submit

def display_prediction_results(prediction_results):
    """Display the prediction results."""
    st.subheader("Prediction Results")
    
    # Create columns for displaying results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display prediction and probability
        if prediction_results["label"] == "High risk":
            st.error(f"Prediction: {prediction_results['label']}")
        else:
            st.success(f"Prediction: {prediction_results['label']}")
        
        # Display probability as percentage
        prob_percentage = prediction_results["probability"] * 100
        st.metric("Probability", f"{prob_percentage:.1f}%")
        
        # Display confidence level
        if prob_percentage > 80 or prob_percentage < 20:
            confidence = "High"
        elif prob_percentage > 60 or prob_percentage < 40:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        st.info(f"Confidence: {confidence}")
    
    with col2:
        # Display probability gauge chart
        fig, ax = plt.subplots(figsize=(4, 0.8))
        
        # Create a simple horizontal bar representing probability
        ax.barh([0], [100], color='lightgray', height=0.6)
        ax.barh([0], [prob_percentage], color='crimson' if prob_percentage > 50 else 'green', height=0.6)
        
        # Add a vertical line at 50%
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
        
        # Set labels
        ax.text(0, 0, "0%", va='center', ha='center', fontsize=9)
        ax.text(50, 0, "50%", va='center', ha='center', fontsize=9)
        ax.text(100, 0, "100%", va='center', ha='center', fontsize=9)
        
        # Remove axes
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        
        st.pyplot(fig)

def display_feature_importance(patient_data):
    """Display feature importance visualization."""
    st.subheader("Feature Importance")
    
    # If we don't have LIME or SHAP working yet, display a placeholder
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Contributing Features")
        
        # Create a simple placeholder visualization showing the top features
        # In a real implementation, this would use LIME or SHAP values
        features = list(patient_data.keys())
        # These are placeholders - in the real app, these would be from LIME/SHAP
        importance = [
            0.35 if f == 'Glucose' else
            0.25 if f == 'BMI' else
            0.15 if f == 'Age' else
            0.08 if f == 'DiabetesPedigreeFunction' else
            0.05 if f == 'BloodPressure' else
            0.03 if f == 'Pregnancies' else
            0.02 if f == 'SkinThickness' else
            0.01
            for f in features
        ]
        
        # Sorting features by importance
        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importance = [importance[i] for i in sorted_indices]
        
        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(sorted_features, sorted_importance, color='crimson')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance (Placeholder)')
        st.pyplot(fig)
        
        st.markdown("""
        > This is a **placeholder visualization**. In the complete implementation, 
        > this would show actual LIME or SHAP values.
        """)
    
    with col2:
        st.markdown("#### Value Analysis")
        
        # Display a table with the patient's values and their significance
        df = pd.DataFrame({
            'Feature': features,
            'Patient Value': [patient_data[f] for f in features],
            'Impact': [
                'High ↑' if f in ['Glucose', 'BMI', 'Age'] else
                'Medium ↑' if f in ['DiabetesPedigreeFunction', 'BloodPressure'] else
                'Low'
                for f in features
            ]
        })
        
        # Sort by importance (same order as visualization)
        df['Importance'] = importance
        df = df.sort_values('Importance', ascending=False).drop('Importance', axis=1)
        
        st.table(df)

def download_pima_dataset():
    """
    Download and save the PIMA Indian Diabetes Dataset if not already present.
    """
    # Create data directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Path for the dataset
    dataset_path = "data/raw/diabetes.csv"
    
    # If dataset already exists, don't download again
    if os.path.exists(dataset_path):
        return True
    
    try:
        # URL for the PIMA dataset (from a reliable source)
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        
        # Download the dataset
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Column names for the dataset
        column_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        ]
        
        # Parse the CSV data
        data = pd.read_csv(BytesIO(response.content), header=None, names=column_names)
        
        # Save the dataset
        data.to_csv(dataset_path, index=False)
        
        # Create a processed version (identical for now)
        data.to_csv("data/processed/diabetes.csv", index=False)
        
        return True
    except Exception as e:
        st.error(f"Error downloading dataset: {str(e)}")
        return False

def main():
    """Main function to run the Streamlit application."""
    # First download the dataset if it doesn't exist
    download_pima_dataset()
    
    # Display the header
    display_header()
    
    # Check if a trained model exists
    model_exists = check_model_exists()
    
    # Display a warning if no model exists
    if not model_exists:
        st.warning("""
        ⚠️ No trained model found! This is a placeholder interface.
        
        In a complete implementation, you would first:
        1. Train a model using `python src/train.py`
        2. Return to this interface to make predictions
        
        Currently showing simulated predictions.
        """)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Information", "Dataset"])
    
    with tab1:
        # Display the patient data input form
        patient_data, submit = data_input_form()
        
        # If submit button is clicked or patient data is changed
        if submit:
            with st.spinner("Making prediction..."):
                time.sleep(1)  # Simulate processing time
                
                # If model exists, use it for prediction
                if model_exists:
                    try:
                        # Load model and make prediction
                        model = load_model()
                        prediction_results = predict_disease(patient_data, model)
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        prediction_results = {
                            "label": "Error",
                            "probability": 0.0
                        }
                else:
                    # Simulate prediction (for placeholder)
                    glucose_impact = (patient_data["Glucose"] - 90) / 110  # Higher glucose increases risk
                    bmi_impact = (patient_data["BMI"] - 18.5) / 25  # Higher BMI increases risk
                    age_impact = (patient_data["Age"] - 20) / 60  # Higher age increases risk
                    family_impact = patient_data["DiabetesPedigreeFunction"] / 2  # Family history impact
                    
                    # Combine impacts for a simulated probability
                    base_prob = 0.3  # Base probability
                    adjusted_prob = base_prob + 0.3 * glucose_impact + 0.2 * bmi_impact + 0.1 * age_impact + 0.1 * family_impact
                    
                    # Clamp to [0, 1]
                    adjusted_prob = max(0.0, min(1.0, adjusted_prob))
                    
                    # Create simulated prediction results
                    prediction_results = {
                        "label": "High risk" if adjusted_prob > 0.5 else "Low risk",
                        "probability": adjusted_prob
                    }
            
            # Display the prediction results
            display_prediction_results(prediction_results)
            
            # Display feature importance visualization
            display_feature_importance(patient_data)
    
    with tab2:
        st.header("Model Information")
        
        # Display model information
        if model_exists:
            # In a full implementation, we would load model metadata
            st.success("✅ Model is trained and ready for predictions")
            
            # Placeholder model metrics
            st.subheader("Model Performance Metrics")
            metrics = {
                "Accuracy": 0.78,
                "Precision": 0.75,
                "Recall": 0.72,
                "F1 Score": 0.73,
                "ROC AUC": 0.82
            }
            
            # Display metrics in a nice format
            metric_cols = st.columns(len(metrics))
            for i, (metric_name, metric_value) in enumerate(metrics.items()):
                with metric_cols[i]:
                    st

"""
MedExplain - Streamlit Frontend

This module provides the Streamlit-based user interface for the MedExplain application.
It allows users to input patient data, get disease predictions, and view explanations.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import predict_disease
from src.explain import explain_prediction

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="MedExplain",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 MedExplain: AI-Driven Medical Diagnosis Support")
    st.subheader("Transparent, Interpretable Medical Predictions")
    
    # Placeholder for actual implementation
    st.write("Coming soon: Interactive medical diagnosis with explanations")
    
if __name__ == "__main__":
    main()

