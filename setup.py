#!/usr/bin/env python3
"""
Setup script for YoutubeNLP that handles dependencies, model downloads, and configuration.
"""
import subprocess
import sys
import os
import logging
from pathlib import Path

from utils.nltk_setup import setup_nltk_resources, verify_nltk_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("YoutubeNLP-Setup")

def install_requirements() -> bool:
    """
    Install packages from requirements.txt
    
    Returns:
        bool: True if installation was successful
    """
    try:
        logger.info("Installing requirements from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        return False

def install_spacy_models() -> bool:
    """
    Install spaCy models
    
    Returns:
        bool: True if installation was successful
    """
    try:
        logger.info("Installing spaCy English model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        logger.info("SpaCy model installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing spaCy model: {e}")
        logger.warning("Some entity detection features may be limited.")
        return False

def create_directories():
    """Create necessary directories for the project"""
    dirs = [
        'models/cache',  # Cache for downloaded models and analysis results
        'data/raw',      # Raw downloaded comments
        'data/processed' # Processed analysis results
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            logger.info(f"Creating directory: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)

def check_google_credentials() -> bool:
    """
    Check if Google Cloud credentials are properly configured
    
    Returns:
        bool: True if credentials are properly set up
    """
    creds_var = 'GOOGLE_APPLICATION_CREDENTIALS'
    
    if creds_var not in os.environ:
        logger.warning(f"{creds_var} environment variable not set")
        logger.info("To enable Google Cloud NLP features:")
        logger.info("1. Create a service account key in Google Cloud Console")
        logger.info(f"2. Export {creds_var}=/path/to/credentials.json")
        logger.info("The application will fall back to local NLP processing.")
        return False
    
    creds_path = os.environ[creds_var]
    if not os.path.exists(creds_path):
        logger.error(f"Credentials file not found: {creds_path}")
        return False
    
    try:
        # Attempt to load and validate credentials
        from google.cloud import language_v1
        client = language_v1.LanguageServiceClient()
        logger.info("Google Cloud credentials validated successfully")
        return True
    except Exception as e:
        logger.error(f"Error validating Google Cloud credentials: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("Starting YoutubeNLP setup...")
    success = True
    
    # Step 1: Install package requirements
    if not install_requirements():
        logger.error("Failed to install requirements. Setup cannot continue.")
        sys.exit(1)
    
    # Step 2: Set up NLTK resources
    if not setup_nltk_resources():
        logger.error("Failed to set up critical NLTK resources. Setup cannot continue.")
        sys.exit(1)
    
    # Step 3: Install spaCy models (optional)
    install_spacy_models()
    
    # Step 4: Create necessary directories
    create_directories()
    
    # Step 5: Check credentials (optional)
    has_google = check_google_credentials()
    
    # Final status report
    logger.info("\nSetup Status:")
    logger.info("✓ Package requirements installed")
    logger.info("✓ NLTK resources configured")
    logger.info(f"{'✓' if has_google else '⚠'} Google Cloud NLP: {'Enabled' if has_google else 'Disabled (using fallback)'}")
    
    logger.info("\nSetup complete! You can now run the application:")
    logger.info("  streamlit run app.py")

if __name__ == "__main__":
    main()