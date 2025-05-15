"""MedExplain Contact Component

This module provides the contact information and feedback form
for the MedExplain application.
"""

import streamlit as st

def display_contact_page():
    """Display contact information and feedback form."""
    st.header("Connect with Us")
    st.markdown("""
        <p style='color:#8892b0;'>Reach out for support, feedback, or documentation.</p>
    """, unsafe_allow_html=True)
    contact_tab1, contact_tab2, contact_tab3 = st.tabs(["Contact", "Feedback", "Documentation"])
    with contact_tab1:
        st.markdown("""
            <div style='background: #112240; border-radius: 12px; padding: 1.5rem; color: #fff; margin-bottom: 1rem;'>
                <h3>📧 Contact Information</h3>
                <b>Email:</b> <a href='mailto:support@medexplain.ai' style='color:#64B5F6;'>support@medexplain.ai</a><br>
                <b>Phone:</b> +1 (555) 123-4567<br>
                <b>Location:</b> Medical AI Research Center
                <hr style='border:1px solid #223; margin:1rem 0;'>
                <h4>🌐 Social Media</h4>
                <a href='https://linkedin.com/medexplain' style='color:#64B5F6;'>LinkedIn</a> |
                <a href='https://twitter.com/medexplain' style='color:#64B5F6;'>Twitter</a> |
                <a href='https://github.com/medexplain' style='color:#64B5F6;'>GitHub</a>
            </div>
        """, unsafe_allow_html=True)
    with contact_tab2:
        st.subheader("📝 Feedback Form")
        st.markdown("<div style='color:#8892b0;'>We value your feedback! Please let us know your thoughts below.</div>", unsafe_allow_html=True)
        feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Feedback"])
        feedback_text = st.text_area("Your Feedback")
        user_email = st.text_input("Your Email (optional)")
        if st.button("Submit Feedback", use_container_width=True):
            st.success("Thank you for your feedback!")
    with contact_tab3:
        st.subheader("📚 Documentation")
        st.markdown("""
            <div style='background: #112240; border-radius: 12px; padding: 1.5rem; color: #fff;'>
            <b>Quick Links</b><br>
            • <a href='https://docs.medexplain.ai/guide' style='color:#64B5F6;'>User Guide</a><br>
            • <a href='https://docs.medexplain.ai/api' style='color:#64B5F6;'>API Documentation</a><br>
            • <a href='https://docs.medexplain.ai/research' style='color:#64B5F6;'>Research Papers</a><br>
            • <a href='https://docs.medexplain.ai/faq' style='color:#64B5F6;'>FAQ</a><br>
            <br><b>Latest Updates</b><br>
            Check our <a href='https://docs.medexplain.ai/changelog' style='color:#64B5F6;'>changelog</a> for recent updates.
            </div>
        """, unsafe_allow_html=True)