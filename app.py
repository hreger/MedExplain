"""MedExplain - Streamlit Frontend

This module provides the Streamlit-based user interface for the MedExplain application.
It allows users to input patient data, get disease predictions, and view explanations.
"""

import os
import streamlit as st

# Import project modules
try:
    from src.predict import predict_disease, load_model, load_feature_names
    from src.explain import get_lime_explainer  # We'll implement this fully later
    from src.download_data import download_pima_dataset
    # Import UI components
    from src.components.header import display_header
    from src.components.hero import display_landing_page
    from src.components.prediction import display_prediction_page
    from src.components.model import display_model_page
    from src.components.dataset import display_dataset_page
    from src.components.contact import display_contact_page
except ImportError:
    st.error("Error importing project modules. Make sure you're running from the project root directory.")

# Set page configuration
st.set_page_config(
    page_title="MedExplain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_model_exists():
    """Check if a trained model exists."""
    model_path = "models/model.joblib"
    return os.path.exists(model_path), model_path

def main():
    """Main function to run the Streamlit application."""
    # First check if dataset exists and download if needed
    if not os.path.exists("data/raw/diabetes.csv"):
        download_pima_dataset()
    
    # Display the header (which includes the navigation bar)
    display_header()
    
    # Create container for main content
    main_container = st.container()
    
    with main_container:
        # Remove the duplicate navigation buttons here
        # Only display content based on session state
        if 'page' not in st.session_state:
            st.session_state.page = "home"
        
        # Handle URL hash for navigation
        st.markdown("""
            <script>
                window.addEventListener('load', function() {
                    const hash = window.location.hash.replace('#', '');
                    if (hash) {
                        const event = new CustomEvent('streamlit:setState', {
                            detail: { page: hash }
                        });
                        window.dispatchEvent(event);
                    }
                });
            </script>
        """, unsafe_allow_html=True)
        
        # Display content based on selected page
        if st.session_state.page == "prediction":
            display_prediction_page()
        elif st.session_state.page == "model":
            display_model_page()
        elif st.session_state.page == "dataset":
            display_dataset_page()
        elif st.session_state.page == "contact":
            display_contact_page()
        else:
            # Display landing page content
            display_landing_page()

if __name__ == "__main__":
    main()

