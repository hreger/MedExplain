"""MedExplain Prediction Component

This module provides the prediction interface and visualization components
for the MedExplain application.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.predict import predict_disease, load_model, predict_with_explanation

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

def display_prediction_page():
    """Display the prediction interface with input form and results."""
    st.header("Disease Prediction")
    st.info("XAI can help medical professionals understand the reasoning behind AI diagnoses and treatment recommendations, leading to more informed decisions.")
    st.markdown("""
        <p style='color:#8892b0;'>Enter patient data below to get a diabetes risk prediction and explanation.</p>
    """, unsafe_allow_html=True)
    
    # Create form for patient data input
    with st.form("prediction_form"):
        patient_data = {}
        col1, col2 = st.columns(2)
        fields = list(FEATURE_INFO.keys())
        for i, field in enumerate(fields):
            info = FEATURE_INFO[field]
            with col1 if i < len(fields)//2 else col2:
                patient_data[field] = st.number_input(
                    f"{field} ({info['description']})",
                    min_value=info['min'],
                    max_value=info['max'],
                    value=info['default'],
                    step=info['step'],
                    help=info['description']
                )
        submitted = st.form_submit_button("Predict", use_container_width=True)
    
    if submitted:
        explanation_results = predict_with_explanation(patient_data)
        if explanation_results.get('error'):
            st.error(f"Error making prediction: {explanation_results['error']}")
        else:
            risk_color = "#FF4B4B" if explanation_results["prediction"] == 1 else "#00CC96"
            emoji = "⚠️" if explanation_results["prediction"] == 1 else "✨"
            label = "High risk" if explanation_results["prediction"] == 1 else "Low risk"
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 20px; padding: 2rem; color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 2rem;'>
                    <div style='font-size:2.5rem; font-weight:700; margin-bottom:1rem; background: linear-gradient(120deg, #ffffff, #64B5F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                        {emoji} {label}
                    </div>
                    <div style='font-size:1.4rem; color:#8892b0;'>
                        Confidence Level: <span style='color:{risk_color}; font-weight:600'>{explanation_results['probability']:.1%}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # --- LIME Visualization ---
            st.markdown("#### LIME Explanation (Feature Contributions)")
            if explanation_results['lime_explanation']:
                import pandas as pd
                lime_df = pd.DataFrame(explanation_results['lime_explanation'], columns=["Feature", "Contribution"])
                lime_df = lime_df.sort_values("Contribution")
                st.bar_chart(lime_df.set_index("Feature"))
            else:
                st.info("No LIME explanation available.")

            # --- SHAP Visualization ---
            st.markdown("#### SHAP Feature Importance")
            if explanation_results['feature_importance']:
                fi = explanation_results['feature_importance']
                fi_df = pd.DataFrame(fi, columns=["Feature", "Importance"])
                fi_df = fi_df.sort_values("Importance")
                st.bar_chart(fi_df.set_index("Feature"))
            else:
                st.info("No SHAP feature importance available.")

            # --- SHAP Force Plot (if possible) ---
            try:
                import shap
                import matplotlib.pyplot as plt
                st.markdown("#### SHAP Force Plot (Prediction Explanation)")
                shap_values = explanation_results['shap_values']
                feature_names = lime_df["Feature"].tolist() if explanation_results['lime_explanation'] else None
                if shap_values and feature_names:
                    shap.initjs()
                    fig, ax = plt.subplots(figsize=(10, 1))
                    shap.force_plot(
                        base_value=0,  # fallback if not available
                        shap_values=shap_values[0][0],
                        features=[patient_data[f] for f in feature_names],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False,
                        ax=ax
                    )
                    st.pyplot(fig)
            except Exception as e:
                st.info(f"SHAP force plot not available: {e}")

            # --- Tabs for metrics and details ---
            tabs = st.tabs(["📊 Metrics", "🧠 Explanation"])
            with tabs[0]:
                st.metric("Prediction", label, delta=None, help="Model's risk assessment")
                st.metric("Probability", f"{explanation_results['probability']:.1%}", help="Model's confidence level")
            with tabs[1]:
                st.markdown("#### LIME Explanation (Top Features)")
                if explanation_results['lime_explanation']:
                    st.dataframe(lime_df, use_container_width=True)
                st.markdown("#### SHAP Feature Importance Table")
                if explanation_results['feature_importance']:
                    st.dataframe(fi_df, use_container_width=True)

def display_prediction_results(prediction_results):
    """Display the prediction results in a visually appealing way."""
    # Check if prediction results contain required fields
    if not prediction_results:
        st.error("No prediction results available")
        return
    
    if prediction_results.get('error'):
        st.error(f"Error in prediction: {prediction_results['error']}")
        return
        
    if not all(key in prediction_results for key in ['prediction', 'label', 'probability']):
        st.error("Invalid prediction results format - missing required fields")
        return
        
    st.markdown("""
        <style>
        .prediction-container {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 20px;
            padding: 2rem;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 2rem;
        }
        .result-label {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(120deg, #ffffff, #64B5F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_color = "#FF4B4B" if prediction_results["label"] == "High risk" else "#00CC96"
        emoji = "⚠️" if prediction_results["label"] == "High risk" else "✨"
        
        st.markdown(f"""
            <div class="prediction-container">
                <div class="result-label">
                    {emoji} {prediction_results["label"]}
                </div>
                <div style="font-size: 1.4rem; color: #8892b0;">
                    Confidence Level: {prediction_results["probability"]:.1%}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create comparison visualization
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1a1a2e')
        
        # Sample historical data
        historical_data = {
            'Low Risk': 0.25,
            'Moderate Risk': 0.45,
            'High Risk': 0.85
        }
        
        # Set up the plot
        categories = list(historical_data.keys())
        historical_values = list(historical_data.values())
        current_value = prediction_results["probability"]
        
        # Plot ranges
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        historical_bars = ax.bar(x - width/2, historical_values, width, 
                               label='Population Average',
                               color='#4facfe',
                               alpha=0.6)
        
        # Add marker for current prediction
        current_category = 'High Risk' if current_value > 0.7 else 'Moderate Risk' if current_value > 0.4 else 'Low Risk'
        category_index = categories.index(current_category)
        
        # Plot current value marker
        ax.scatter(category_index - width/2, current_value, 
                  color='#FF4B4B' if current_value > 0.7 else '#FFA500' if current_value > 0.4 else '#00CC96',
                  s=200, zorder=5, label='Your Risk Level',
                  marker='o', edgecolor='white', linewidth=2)
        
        # Customize the plot
        ax.set_facecolor('#1a1a2e')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Risk Probability', color='white')
        ax.set_xticks(x - width/2)
        ax.set_xticklabels(categories, color='white')
        
        # Add grid
        ax.grid(True, alpha=0.1)
        
        # Customize spines
        for spine in ax.spines.values():
            spine.set_color('#2a2a4e')
        
        # Add legend
        ax.legend(loc='upper left', facecolor='#1a1a2e', labelcolor='white')
        
        # Add percentage labels
        for i, v in enumerate(historical_values):
            ax.text(i - width/2, v + 0.05, f'{v:.1%}',
                   color='white', ha='center', va='bottom')
        
        # Add current value label
        ax.text(category_index - width/2, current_value + 0.05,
               f'Your Risk\n{current_value:.1%}',
               color='white', ha='center', va='bottom')
        
        # Set title
        ax.set_title('Your Risk Level vs Population Average',
                    color='white', pad=20)
        
        # Display the plot
        st.pyplot(fig)
        
        # Add interpretation text
        interpretation = "Low risk" if current_value < 0.4 else "Moderate risk" if current_value < 0.7 else "High risk"
        st.markdown(f"""
            <div style='text-align: center; color: #8892b0; font-size: 1.1rem; margin-top: 1rem;'>
                Risk Assessment: <span style='color: {risk_color};'>{interpretation}</span>
            </div>
        """, unsafe_allow_html=True)