#!/usr/bin/env python3
"""
YoutubeNLP: A Streamlit application for NLP analysis of YouTube comments.
"""
import logging
import streamlit as st

from utils.config import setup_logging
from utils.nltk_setup import setup_nltk_resources
from views.main_view import MainView

# Set page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="YoutubeNLP - Comment Analysis",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the YoutubeNLP application."""
    try:
        logger.info("Starting YoutubeNLP application")
        
        # Initialize NLTK resources first
        with st.spinner("Setting up NLP resources..."):
            success = setup_nltk_resources()
            if not success:
                st.warning("Some NLP resources could not be downloaded. Some functionality may be limited.")
                logger.warning("Some NLTK resources failed to download")
        
        # Initialize and render the main view
        view = MainView(configure_page=False)  # Pass flag to avoid page configuration in MainView
        view.render()
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        # Display error in Streamlit
        st.error("An unexpected error occurred. Please check the logs for details.")
        st.exception(e)

if __name__ == "__main__":
    main()