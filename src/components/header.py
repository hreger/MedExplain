"""MedExplain Header Component

This module provides the header component for the MedExplain application,
including the logo, title, and navigation elements.
"""

import streamlit as st
import base64
from streamlit_option_menu import option_menu

def get_base64_logo():
    """Get the base64 encoded logo image."""
    with open('assets/medexplain_logo.jpg', 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def display_header():
    """Display the application header with logo and navigation."""
    # Load custom CSS
    with open('src/static/style.css', 'r') as f:
        custom_css = f.read()
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        .header-container {
            background: linear-gradient(135deg, #0a192f 0%, #112240 100%);
            padding: 2rem;
            margin: -4rem -4rem 2rem -4rem;
            text-align: center;
            color: white;
            position: relative;
            overflow: hidden;
            border-radius: 0 0 20px 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .small-logo {
            width: 200px;
            margin-right: 1rem;
            vertical-align: middle;
        }
        .header-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        .title-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .main-title {
            font-size: 2.5rem;
            margin: 0;
            color: white;
            font-weight: 700;
            background: linear-gradient(120deg, #ffffff, #64B5F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            font-size: 1.2rem;
            margin: 0;
            color: #8892b0;
            font-weight: 500;
        }
        .description {
            font-size: 1rem;
            color: #8892b0;
            max-width: 800px;
            margin: 1rem 0;
            line-height: 1.6;
        }
        .nav-container {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
        }
        .nav-button {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            text-decoration: none;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-weight: 500;
            cursor: pointer;
            min-width: 120px;
        }
        .nav-button:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Display the header content (logo, title, subtitle, description)
    st.markdown(f"""
        <div class="header-container">
            <div class="header-content">
                <div class="title-container">
                    <img src="data:image/png;base64,{get_base64_logo()}" class="small-logo" alt="MedExplain Logo">
                    <h1 class="main-title">MedExplain</h1>
                </div>
                <p class="subtitle">AI-Driven Medical Diagnosis Support with Explainable AI</p>
                <p class="description">
                    MedExplain provides transparent, interpretable results for disease diagnosis
                    using patient data, bridging the gap between ML accuracy and medical accountability.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Modern navigation bar using streamlit-option-menu
    selected = option_menu(
        menu_title=None,
        options=["Prediction", "Model Info", "Dataset", "Contact"],
        icons=["activity", "bar-chart", "folder", "telephone"],
        orientation="horizontal",
        default_index=["prediction", "model", "dataset", "contact"].index(st.session_state.get("page", "prediction")),
        styles={
            "container": {"padding": "0!important", "background-color": "#0a192f"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "color": "white", "margin":"0px", "padding":"10px 20px"},
            "nav-link-selected": {"background-color": "#112240", "color": "#64B5F6"},
        }
    )
    st.session_state.page = (
        "prediction" if selected == "Prediction" else
        "model" if selected == "Model Info" else
        "dataset" if selected == "Dataset" else
        "contact"
    )