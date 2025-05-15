"""MedExplain Dataset Component

This module provides the dataset information and exploration features
for the MedExplain application.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def display_dataset_page():
    """Display dataset information and exploration tools."""
    st.header("Dataset Overview")
    st.markdown("""
        <p style='color:#8892b0;'>Explore the Pima Indians Diabetes Database and its quality.</p>
    """, unsafe_allow_html=True)
    # Add tabs for different dataset aspects
    data_tab1, data_tab2, data_tab3 = st.tabs(["Overview", "Exploration", "Quality"])
    with data_tab1:
        st.markdown("""
            <div style='background: #112240; border-radius: 12px; padding: 1.5rem; color: #fff; margin-bottom: 1rem;'>
                The Pima Indians Diabetes Database is a well-known dataset in machine learning and medical research.<br>
                It contains diagnostic measurements for females of Pima Indian heritage, at least 21 years old.
            </div>
        """, unsafe_allow_html=True)
        try:
            df = pd.read_csv('data/raw/diabetes.csv')
            st.subheader("📊 Dataset Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", df.shape[0])
                st.metric("Features", df.shape[1]-1)
            with col2:
                st.metric("Non-diabetic (0)", f"{(df['Outcome'] == 0).mean():.1%}")
                st.metric("Diabetic (1)", f"{(df['Outcome'] == 1).mean():.1%}")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    with data_tab2:
        st.subheader("🔍 Data Exploration")
        try:
            st.markdown("### Feature Distributions")
            feature = st.selectbox(
                "Select Feature to Visualize",
                options=[col for col in df.columns if col != 'Outcome']
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df, x=feature, hue='Outcome', multiple="stack")
            plt.title(f'Distribution of {feature} by Diabetes Outcome')
            st.pyplot(fig)
            st.markdown("### Feature Correlations")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in data exploration: {str(e)}")
    with data_tab3:
        st.subheader("✨ Data Quality")
        try:
            quality_metrics = pd.DataFrame({
                'Missing Values': df.isnull().sum(),
                'Zero Values': (df == 0).sum(),
                'Unique Values': df.nunique(),
                'Min': df.min(),
                'Max': df.max(),
                'Mean': df.mean(),
                'Std': df.std()
            })
            st.markdown("### Data Quality Metrics")
            st.dataframe(quality_metrics, use_container_width=True)
            st.markdown("""
                <div style='background: #112240; border-radius: 12px; padding: 1.5rem; color: #fff; margin-top: 1rem;'>
                <b>Key Insights</b><br>
                • Zero values in medical measurements might indicate missing data<br>
                • All features are numerical, making them suitable for machine learning<br>
                • The dataset is relatively well-balanced between classes<br>
                <br><b>Data Preprocessing Steps</b><br>
                1. Handle zero values in medical measurements<br>
                2. Scale features to similar ranges<br>
                3. Remove outliers where appropriate<br>
                4. Balance classes if needed
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error in data quality analysis: {str(e)}")