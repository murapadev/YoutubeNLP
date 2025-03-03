#!/usr/bin/env python3
"""
Configuration: Application-wide settings and configuration utilities.
"""
import os
import logging
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

def download_nltk_resources():
    """Download required NLTK resources."""
    import nltk
    resources = [
        # Tokenizers
        'punkt',
        # 'punkt_tab' - Removed as it doesn't exist in NLTK
        'averaged_perceptron_tagger',
        # Corpora
        'wordnet',
        'omw-1.4',
        'stopwords',
        'universal_tagset',  # For universal POS tags
        'maxent_ne_chunker',  # For named entity recognition
        'words',  # Required for NE chunker
        # Sentiment analysis resources
        'vader_lexicon',
        # Additional resources for multiple languages
        'spanish_grammars',
        'floresta',  # Portuguese treebank
        'conll2002',  # Spanish and Dutch NER
        'europarl_raw/spanish',
    ]
    
    for resource in resources:
        try:
            # Check if resource exists
            if resource.startswith('tokenizers/'):
                nltk.data.find(resource)
            elif '/' in resource:
                nltk.data.find(f'corpora/{resource}')
            else:
                # Try all possible locations
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    try:
                        nltk.data.find(f'corpora/{resource}')
                    except LookupError:
                        nltk.data.find(f'taggers/{resource}')
        except LookupError:
            # Download the missing resource
            logging.info(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource, quiet=False)
                logging.info(f"Successfully downloaded {resource}")
            except Exception as e:
                logging.warning(f"Failed to download NLTK resource {resource}: {str(e)}")

def setup_ml_models():
    """Setup and verify machine learning models."""
    try:
        import tensorflow as tf
        import transformers
        from transformers import pipeline
        
        # Define model paths
        MODEL_PATHS = {
            'sentiment': 'models/sentiment_model',
            'toxicity': 'models/toxicity_model',
            'emotion': 'models/emotion_model',
        }
        
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        # Initialize transformers pipeline for additional analysis
        try:
            # Sentiment analysis with multilingual model
            sentiment_analyzer = pipeline(
                'sentiment-analysis',
                model='nlptown/bert-base-multilingual-uncased-sentiment',
                return_all_scores=True
            )
            
            # Emotion detection
            emotion_analyzer = pipeline(
                'text-classification',
                model='bhadresh-savani/distilbert-base-uncased-emotion',
                return_all_scores=True
            )
            
            logging.info("Successfully loaded transformer models")
            return {
                'sentiment': sentiment_analyzer,
                'emotion': emotion_analyzer
            }
        except Exception as e:
            logging.warning(f"Could not load transformer models: {e}")
            return {}
            
    except ImportError:
        logging.warning("TensorFlow/Transformers not installed. Advanced ML features will be disabled.")
        return {}

# Logging configuration
def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure loggers for external libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)

    # Download NLTK resources
    download_nltk_resources()

# Application settings
APP_CONFIG = {
    "version": "1.0.0",
    "name": "YoutubeNLP",
    "description": "YouTube Comment Analysis with NLP",
    "github_url": "https://github.com/yourusername/YoutubeNLP",
    "issue_url": "https://github.com/yourusername/YoutubeNLP/issues",
    "max_comments_limit": 1000,
    "ml_config": {
        "use_transformers": True,  # Enable/disable transformer models
        "batch_size": 16,  # Batch size for ML inference
        "max_length": 512,  # Max sequence length for transformers
        "cache_dir": "models/cache",  # Cache directory for downloaded models
    }
}

# Language settings
LANGUAGE_SUPPORT = {
    'en': {
        'name': 'English',
        'models': ['sentiment', 'emotion', 'toxicity'],
        'resources': ['punkt', 'averaged_perceptron_tagger', 'wordnet']
    },
    'es': {
        'name': 'Spanish',
        'models': ['sentiment'],
        'resources': ['punkt', 'spanish_grammars']
    },
    'multilingual': {
        'models': ['sentiment'],
        'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl']
    }
}

# Default stopwords to supplement NLTK and WordCloud defaults
ADDITIONAL_STOPWORDS = {
    "youtube", "video", "channel", "subscribe", "like", "comment", 
    "watch", "watching", "watched", "videos", "youtuber", "youtubers",
    "commented", "commenting", "likes", "dislike", "dislikes", "please",
    "thanks", "thank", "plz", "pls", "thx"
}

def has_google_credentials() -> bool:
    """
    Check if Google Cloud credentials are configured.
    
    Returns:
        True if credentials are found, False otherwise
    """
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None

def get_ml_models():
    """Get initialized ML models if available."""
    return setup_ml_models()