"""MedExplain Hero Component

This module provides the hero/landing page component for the MedExplain application,
showcasing key features and benefits.
"""

import streamlit as st

def display_landing_page():
    """Display the landing page content with feature cards."""
    st.title("Why Choose MedExplain?", anchor=False)
    
    # Create three columns for the feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(145deg, #ffffff, #f5f5f5); padding: 1.5rem; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); height: 100%;'>
                <h3 style='color: #2196F3; margin-bottom: 1rem;'>🎯 Accurate Predictions</h3>
                <p>State-of-the-art machine learning models trained on extensive medical datasets.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(145deg, #ffffff, #f5f5f5); padding: 1.5rem; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); height: 100%;'>
                <h3 style='color: #2196F3; margin-bottom: 1rem;'>🔍 Explainable AI</h3>
                <p>Transparent decision-making process with detailed feature importance analysis.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(145deg, #ffffff, #f5f5f5); padding: 1.5rem; border-radius: 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.1); height: 100%;'>
                <h3 style='color: #2196F3; margin-bottom: 1rem;'>⚡ Real-time Analysis</h3>
                <p>Instant predictions and explanations for immediate medical insights.</p>
            </div>
        """, unsafe_allow_html=True)