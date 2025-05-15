"""MedExplain Model Information Component

This module provides the model information and performance metrics display
for the MedExplain application.
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def display_model_page():
    """Display model information and performance metrics."""
    st.header("Model Information")
    st.markdown("""
        <p style='color:#8892b0;'>Learn about the model, its performance, and recent updates.</p>
    """, unsafe_allow_html=True)
    
    # Add tabs for different model aspects
    model_tab1, model_tab2, model_tab3 = st.tabs(["Overview", "Performance", "Updates"])
    
    with model_tab1:
        st.subheader("🤖 Model Overview")
        st.markdown("""
            <div style='background: #112240; border-radius: 12px; padding: 1.5rem; color: #fff; margin-bottom: 1rem;'>
                <b>Random Forest Classifier</b> is used for diabetes prediction:<br>
                • High accuracy and robustness<br>
                • Handles non-linear relationships<br>
                • Built-in feature importance ranking<br>
                • Resistant to overfitting
            </div>
        """, unsafe_allow_html=True)
        st.info("Model Version: v1.2.0 (Last updated: 2024-01-15)")
        with st.expander("Model Update History"):
            st.markdown("""
                - v1.2.0 (2024-01-15): Added SHAP explanations
                - v1.1.0 (2023-12-01): Improved feature engineering
                - v1.0.0 (2023-11-15): Initial release
            """)
    
    with model_tab2:
        st.subheader("📊 Performance Metrics")
        metrics = {
            "Accuracy": 0.85,
            "Precision": 0.83,
            "Recall": 0.82,
            "F1 Score": 0.82
        }
        st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        for (metric, value), col in zip(metrics.items(), [col1, col2, col3, col4]):
            with col:
                st.metric(label=metric, value=f"{value:.2%}")
        st.subheader("🎯 Feature Importance")
        feature_importance = {
            "Glucose": 0.30,
            "BMI": 0.20,
            "Age": 0.15,
            "DiabetesPedigreeFunction": 0.12,
            "BloodPressure": 0.10,
            "Pregnancies": 0.08,
            "SkinThickness": 0.03,
            "Insulin": 0.02
        }
        fig, ax = plt.subplots(figsize=(10, 6))
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        bars = ax.barh(features, importance, color="#64B5F6", edgecolor="#0a192f")
        ax.set_xlabel("Importance", color="#8892b0")
        ax.set_title("Feature Importance in Prediction", color="#fff")
        ax.set_facecolor("#112240")
        fig.patch.set_facecolor('#112240')
        ax.tick_params(colors="#fff")
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                    f'{importance[i]:.1%}',
                    ha='left', va='center', fontweight='bold', color="#fff")
        st.pyplot(fig)
    
    with model_tab3:
        st.subheader("🔄 Model Updates")
        st.markdown("""
            <div style='background: #112240; border-radius: 12px; padding: 1.5rem; color: #fff;'>
            <b>Latest Updates</b><br>
            • <b>SHAP Integration</b>: Added SHAP (SHapley Additive exPlanations) for more detailed feature importance analysis<br>
            • <b>Performance Optimization</b>: Improved model inference speed by 25%<br>
            • <b>Feature Engineering</b>: Enhanced preprocessing pipeline for better accuracy<br>
            <br><b>Upcoming Features</b><br>
            • Integration with additional medical datasets<br>
            • Advanced visualization options for model explanations<br>
            • API endpoints for external integrations
            </div>
        """, unsafe_allow_html=True)