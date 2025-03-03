#!/usr/bin/env python3
"""
NLTK Setup Utility: Ensures required NLTK resources are downloaded and available.

This module handles the initialization and downloading of required NLTK resources
for the application to function correctly.
"""
import os
import logging
from typing import List, Optional, Dict, Any
import nltk
from nltk.corpus import stopwords
# LookupError is a built-in exception, not from nltk.data
from pathlib import Path

logger = logging.getLogger(__name__)

# Define all required NLTK resources with their descriptions
REQUIRED_RESOURCES: Dict[str, Dict[str, Any]] = {
    'punkt': {
        'path': 'tokenizers/punkt',
        'description': 'Sentence tokenization',
        'critical': True
    },
    'stopwords': {
        'path': 'corpora/stopwords',
        'description': 'Common stopwords in multiple languages',
        'critical': True
    },
    'averaged_perceptron_tagger': {  # Changed from averaged_perceptron_tagger_eng
        'path': 'taggers/averaged_perceptron_tagger',
        'description': 'POS tagging',
        'critical': True
    },
    'universal_tagset': {
        'path': 'taggers/universal_tagset',
        'description': 'Universal POS tags',
        'critical': True
    },
    'wordnet': {
        'path': 'corpora/wordnet',
        'description': 'WordNet lexical database',
        'critical': False
    },
    'omw-1.4': {
        'path': 'corpora/omw-1.4',
        'description': 'Open Multilingual WordNet',
        'critical': False
    },
    'vader_lexicon': {
        'path': 'sentiment/vader_lexicon',
        'description': 'VADER sentiment lexicon',
        'critical': False
    },
    'maxent_ne_chunker': {
        'path': 'chunkers/maxent_ne_chunker',
        'description': 'Named Entity chunking',
        'critical': False
    },
    'words': {
        'path': 'corpora/words',
        'description': 'Word lists',
        'critical': False
    }
}

def get_nltk_data_dir() -> str:
    """
    Get the NLTK data directory, creating it if it doesn't exist.
    
    Returns:
        str: Path to NLTK data directory
    """
    nltk_data_dir = str(Path.home() / 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    return nltk_data_dir

def download_resource(resource_name: str, data_dir: str) -> bool:
    """
    Download a specific NLTK resource.
    
    Args:
        resource_name: Name of the resource to download
        data_dir: Directory to store the downloaded resource
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        logger.info(f"Downloading NLTK resource: {resource_name}")
        nltk.download(resource_name, download_dir=data_dir, quiet=True)
        return True
    except Exception as e:
        logger.error(f"Failed to download {resource_name}: {str(e)}")
        return False

def setup_nltk_resources(resources: Optional[List[str]] = None) -> bool:
    """
    Download and set up required NLTK resources.
    
    Args:
        resources: Optional list of resource names to download.
                 If None, downloads all REQUIRED_RESOURCES.
    
    Returns:
        bool: True if all critical resources were successfully downloaded/verified
    """
    data_dir = get_nltk_data_dir()
    logger.info(f"Setting up NLTK resources in {data_dir}")
    
    if resources is None:
        resources = list(REQUIRED_RESOURCES.keys())
    
    success = True
    failed_critical = []
    failed_optional = []
    
    for resource_name in resources:
        if resource_name not in REQUIRED_RESOURCES:
            logger.warning(f"Unknown resource: {resource_name}")
            continue
            
        resource_info = REQUIRED_RESOURCES[resource_name]
        resource_path = resource_info['path']
        is_critical = resource_info['critical']
        
        try:
            # Check if resource is already available
            if nltk.data.find(resource_path, paths=[data_dir]):
                logger.info(f"Resource {resource_name} is already available")
                continue
        except LookupError:
            # Resource not found, attempt to download
            if not download_resource(resource_name, data_dir):
                if is_critical:
                    failed_critical.append(resource_name)
                    success = False
                else:
                    failed_optional.append(resource_name)
    
    # Report results
    if failed_critical:
        logger.error(f"Failed to download critical resources: {', '.join(failed_critical)}")
    if failed_optional:
        logger.warning(f"Failed to download optional resources: {', '.join(failed_optional)}")
    
    return success

def verify_nltk_resources() -> bool:
    """
    Verify that all critical NLTK resources are available.
    
    Returns:
        bool: True if all critical resources are available
    """
    critical_resources = [
        name for name, info in REQUIRED_RESOURCES.items() 
        if info['critical']
    ]
    
    for resource_name in critical_resources:
        resource_path = REQUIRED_RESOURCES[resource_name]['path']
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.error(f"Critical resource {resource_name} not found")
            return False
    
    return True

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Download all resources
    setup_nltk_resources()