#!/usr/bin/env python3
"""
Analyzer Model: Provides NLP analysis capabilities for YouTube comments.
"""
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter

from google.cloud import language_v1
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
from langdetect import detect, LangDetectException

from models.comment_model import CommentData


class NLPAnalyzer:
    """Base class for NLP analysis operations with language detection support."""
    
    def __init__(self):
        """Initialize the NLP analyzer with required resources."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Ensure NLTK resources are available
        self._download_nltk_resources()
        
        # Initialize stop words (will be expanded based on detected languages)
        self.stop_words = set(STOPWORDS)
        self._load_common_stopwords()
        
        # Cache for language-specific models and resources
        self.language_models = {}
        self.language_stopwords = {}
    
    def _download_nltk_resources(self) -> None:
        """
        Download required NLTK resources if not already available.
        """
        resources = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
            ('corpora/words', 'words')
        ]
        
        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
                self.logger.debug(f"NLTK resource {resource_name} already available")
            except LookupError:
                self.logger.info(f"Downloading NLTK resource: {resource_name}")
                try:
                    nltk.download(resource_name, quiet=True)
                    self.logger.info(f"Successfully downloaded {resource_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to download NLTK resource {resource_name}: {str(e)}")
                    self.logger.warning("Some NLP features may be limited")
    
    def _load_common_stopwords(self) -> None:
        """
        Load stopwords for common languages into the main stopwords set.
        """
        common_languages = ['english', 'spanish', 'french', 'german']
        
        try:
            for lang in common_languages:
                try:
                    self.stop_words.update(stopwords.words(lang))
                    self.logger.debug(f"Loaded stopwords for {lang}")
                except Exception as e:
                    self.logger.warning(f"Could not load stopwords for {lang}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error loading common stopwords: {e}")
            # Add a basic set of English stopwords as fallback
            basic_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                              'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                              'such', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                              'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                              'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'}
            self.stop_words.update(basic_stopwords)
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', etc.)
        """
        try:
            # Clean text for more accurate language detection
            # Remove URLs, hashtags, mentions which can confuse detection
            clean_text = re.sub(r'https?://\S+|www\.\S+|@\w+|#\w+', '', text)
            
            # Ensure we have enough text for reliable detection (at least 20 chars)
            if len(clean_text.strip()) < 20:
                # For very short texts, check common language patterns
                if self._likely_english(clean_text):
                    return 'en'
                    
            return detect(clean_text)
        except LangDetectException:
            self.logger.warning(f"Could not detect language for text: {text[:30]}...")
            return 'en'  # Default to English if detection fails
    
    def _likely_english(self, text: str) -> bool:
        """
        Check if a short text is likely to be English based on common word patterns.
        
        Args:
            text: Short text to analyze
            
        Returns:
            True if likely English, False otherwise
        """
        # Common English words/patterns
        english_patterns = [
            r'\bthe\b', r'\ba\b', r'\bis\b', r'\bto\b', r'\band\b',
            r'\bfor\b', r'\bin\b', r'\bthat\b', r'\bit\b'
        ]
        
        text = text.lower()
        matches = 0
        for pattern in english_patterns:
            if re.search(pattern, text):
                matches += 1
        
        # If 2+ common English patterns are found, likely English
        return matches >= 2
    
    def get_stopwords_for_language(self, lang_code: str) -> Set[str]:
        """
        Get stopwords for the specified language.
        
        Args:
            lang_code: ISO 639-1 language code
            
        Returns:
            Set of stopwords for the language
        """
        # Return from cache if already loaded
        if lang_code in self.language_stopwords:
            return self.language_stopwords[lang_code]
        
        lang_map = {
            'en': 'english',
            'es': 'spanish',
            'fr': 'french',
            'de': 'german',
            'it': 'italian',
            'pt': 'portuguese',
            'ru': 'russian',
            'nl': 'dutch',
            'fi': 'finnish',
            'sv': 'swedish',
            'no': 'norwegian',
            'da': 'danish',
            'hu': 'hungarian',
            'tr': 'turkish',
        }
        
        stopword_set = set(STOPWORDS)  # Default WordCloud stopwords
        
        nltk_lang = lang_map.get(lang_code)
        if nltk_lang:
            try:
                stopword_set.update(stopwords.words(nltk_lang))
                self.logger.info(f"Added {nltk_lang} stopwords")
            except Exception as e:
                self.logger.warning(f"Could not load stopwords for {nltk_lang}: {e}")
        
        # Add common social media and YouTube-specific stopwords
        youtube_stopwords = {
            "video", "videos", "youtube", "channel", "subscribe", "like", 
            "comment", "watch", "watching", "subscribe", "channel", "please", 
            "thanks", "thank"
        }
        stopword_set.update(youtube_stopwords)
        
        # Cache for future use
        self.language_stopwords[lang_code] = stopword_set
        return stopword_set
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and extract insights.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze_text method")
    
    def analyze_text_batch(self, texts: List[str], 
                          batch_size: int = 50) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batches for better performance.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of analysis results dictionaries
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.analyze_text(text) for text in batch]
            results.extend(batch_results)
        return results

    def _analyze_emojis(self, text: str) -> Dict[str, Any]:
        """
        Analyze emojis in the text.

        Args:
            text: The text to analyze

        Returns:
            Dict[str, Any]: Dictionary containing found emojis, count, and a placeholder sentiment score.
        """
        try:
            import emoji
            if hasattr(emoji, 'is_emoji'):
                found = [c for c in text if emoji.is_emoji(c)]
            elif hasattr(emoji, 'EMOJI_DATA'):
                found = [c for c in text if c in emoji.EMOJI_DATA]
            else:
                found = []
            # Placeholder for emoji sentiment analysis
            emoji_sentiment = 0.0
            return {"emojis": found, "emoji_count": len(found), "emoji_sentiment": emoji_sentiment}
        except ImportError:
            self.logger.warning("Emoji module not installed; skipping emoji analysis.")
            return {"emojis": [], "emoji_count": 0, "emoji_sentiment": 0.0}
        except Exception as e:
            self.logger.warning(f"Error during emoji analysis: {e}")
            return {"emojis": [], "emoji_count": 0, "emoji_sentiment": 0.0}


class TextBlobAnalyzer(NLPAnalyzer):
    """NLP analyzer using TextBlob with transformer model enhancements."""
    
    def __init__(self):
        """Initialize the TextBlob analyzer."""
        super().__init__()
        self.logger.info("Initialized TextBlob analyzer")
        
        # Initialize transformer models if available
        from utils.config import get_ml_models, APP_CONFIG
        self.ml_models = get_ml_models() if APP_CONFIG["ml_config"]["use_transformers"] else {}
        self.batch_size = APP_CONFIG["ml_config"]["batch_size"]
        
        # Initialize spaCy
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.has_spacy = True
        except Exception as e:
            self.logger.warning(f"Failed to load spaCy model: {e}")
            self.has_spacy = False
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using a combination of TextBlob and transformer models.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if not text or text.isspace():
            return {
                "sentiment": {"score": 0.0, "magnitude": 0.0},
                "keywords": [],
                "language": "en",
                "entities": [],
                "emotions": {},
                "emoji_analysis": self._analyze_emojis(text)
            }
        
        # Detect language first
        lang_code = self.detect_language(text)
        
        # Get language-specific stopwords
        lang_stopwords = self.get_stopwords_for_language(lang_code)
        
        # Collect all analysis results
        results = {}
        
        # 1. Basic sentiment with TextBlob
        blob_sentiment = self._get_textblob_sentiment(text, lang_code)
        
        # 2. Transformer-based sentiment if available
        tf_sentiment = self._get_transformer_sentiment(text) if self.ml_models else None
        
        # Combine sentiments with preference to transformer results
        if tf_sentiment:
            results["sentiment"] = {
                "score": tf_sentiment["score"],
                "magnitude": abs(tf_sentiment["score"]) * tf_sentiment.get("confidence", 1.0)
            }
        else:
            results["sentiment"] = blob_sentiment
        
        # 3. Emotion analysis if available
        if "emotion" in self.ml_models:
            results["emotions"] = self._analyze_emotions(text)
        else:
            results["emotions"] = {}
        
        # 4. Extract keywords using enhanced NLTK approach
        results["keywords"] = self._extract_keywords_nltk(text, lang_stopwords)
        
        # 5. Extract entities using spaCy
        results["entities"] = self._extract_entities_spacy(text, lang_code)
        
        # 6. Add language information
        results["language"] = lang_code
        
        # 7. Add emoji analysis
        results["emoji_analysis"] = self._analyze_emojis(text)
        
        return results
    
    def _get_textblob_sentiment(self, text: str, lang_code: str) -> Dict[str, float]:
        """Get sentiment using TextBlob."""
        # Use language-specific analyzer if available
        if lang_code == 'es':
            from textblob import TextBlob as BaseTextBlob
            from textblob.sentiments import PatternAnalyzer
            blob = BaseTextBlob(text, analyzer=PatternAnalyzer())
        else:
            blob = TextBlob(text)
        
        sentiment = blob.sentiment
        return {
            "score": sentiment.polarity,
            "magnitude": abs(sentiment.polarity) * sentiment.subjectivity
        }
    
    def _get_transformer_sentiment(self, text: str) -> Optional[Dict[str, float]]:
        """Get sentiment using transformer model."""
        try:
            if "sentiment" in self.ml_models:
                results = self.ml_models["sentiment"](text)
                if results:
                    # Convert 5-class output to [-1, 1] range
                    scores = results[0]
                    weighted_score = sum(
                        (i - 2) * score["score"] for i, score in enumerate(scores)
                    ) / 2  # Normalize to [-1, 1]
                    
                    # Get confidence from highest probability
                    confidence = max(score["score"] for score in scores)
                    
                    return {
                        "score": weighted_score,
                        "confidence": confidence
                    }
        except Exception as e:
            self.logger.warning(f"Error in transformer sentiment: {e}")
        return None
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze emotions using the emotion detection model."""
        try:
            results = self.ml_models["emotion"](text)
            if results:
                # Convert list of label/score dicts to a single dict
                emotions = {
                    item["label"]: item["score"] 
                    for item in results[0]
                }
                return emotions
        except Exception as e:
            self.logger.warning(f"Error in emotion analysis: {e}")
        return {}
    
    def _extract_keywords_nltk(self, text: str, 
                             stopwords_set: Optional[Set[str]] = None) -> List[str]:
        """Extract keywords using enhanced NLTK approach with universal tagset."""
        try:
            # Use provided stopwords or default
            stop_words = stopwords_set if stopwords_set else self.stop_words
            
            # First try to tokenize using NLTK's word_tokenize
            try:
                tokens = word_tokenize(text.lower())
            except LookupError as e:
                self.logger.warning(f"NLTK word_tokenize failed: {e}")
                tokens = text.lower().split()

            # First try standard POS tagging
            try:
                # Try loading averaged_perceptron_tagger explicitly if needed
                from nltk.tag import PerceptronTagger
                tagger = PerceptronTagger()
                tagged = tagger.tag(tokens)
            except Exception as e1:
                try:
                    # Fallback to basic POS tagging
                    tagged = pos_tag(tokens)
                except Exception as e2:
                    self.logger.warning(f"POS tagging failed: {e2}")
                    # Return simple keyword extraction without POS tags
                    return [word for word in tokens 
                           if word not in stop_words and len(word) > 2 and word.isalnum()]

            # Process POS tags (renamed variable 'i' to 'idx' for clarity)
            simplified_tags = []
            for word, tag in tagged:
                # Map POS tag to simplified category (NOUN, VERB, ADJ)
                if tag.startswith('NN'):  # Nouns
                    simplified_tags.append((word, 'NOUN'))
                elif tag.startswith('VB'):  # Verbs
                    simplified_tags.append((word, 'VERB'))
                elif tag.startswith('JJ'):  # Adjectives
                    simplified_tags.append((word, 'ADJ'))
                else:
                    simplified_tags.append((word, 'OTHER'))
            
            tagged = simplified_tags

            # Keep content words (nouns, verbs, adjectives)
            keywords = []
            allowed_tags = {'NOUN', 'VERB', 'ADJ'}
            
            for word, tag in tagged:
                if (tag in allowed_tags and 
                    word not in stop_words and 
                    len(word) > 2 and 
                    word.isalnum()):
                    keywords.append(word)
            
            return list(set(keywords))
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            # Last resort fallback
            try:
                # Very simple approach without any NLP tools
                words = text.lower().split()
                return [w for w in words if w not in stop_words and len(w) > 2 and w.isalnum()]
            except:
                return []
    
    def _extract_entities_spacy(self, text: str, lang_code: str) -> List[Dict[str, Any]]:
        """
        Extract named entities using spaCy if available, otherwise return empty list.
        
        Args:
            text: Text to analyze
            lang_code: Language code of the text
            
        Returns:
            List of entity dictionaries with name, type, and salience
        """
        entities = []
        
        if not self.has_spacy:
            return entities
            
        try:
            doc = self.nlp(text)
            
            # Extract entities and calculate basic salience
            total_ents = len(doc.ents)
            if total_ents > 0:
                for ent in doc.ents:
                    # Convert to serializable dictionary
                    entities.append({
                        "name": str(ent.text),  # Ensure string type
                        "type": str(ent.label_),  # Ensure string type
                        "salience": float(1.0/total_ents)  # Ensure float type
                    })
        except Exception as e:
            self.logger.warning(f"Error extracting entities with spaCy: {e}")
            
        return entities

    def analyze_text_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts efficiently using batching for transformer models.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of analysis results
        """
        results = []
        batch_size = self.batch_size
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Get basic analysis for each text
            batch_results = [self.analyze_text(text) for text in batch]
            
            # If transformer models are available, update with batch processing
            if self.ml_models:
                try:
                    # Batch sentiment analysis
                    if "sentiment" in self.ml_models:
                        sentiment_results = self.ml_models["sentiment"](batch)
                        for j, sentiment in enumerate(sentiment_results):
                            scores = sentiment
                            weighted_score = sum(
                                (i - 2) * score["score"] for i, score in enumerate(scores)
                            ) / 2
                            confidence = max(score["score"] for score in scores)
                            batch_results[j]["sentiment"].update({
                                "score": weighted_score,
                                "confidence": confidence
                            })
                    
                    # Batch emotion analysis
                    if "emotion" in self.ml_models:
                        emotion_results = self.ml_models["emotion"](batch)
                        for j, emotions in enumerate(emotion_results):
                            batch_results[j]["emotions"] = {
                                item["label"]: item["score"] 
                                for item in emotions
                            }
                            
                except Exception as e:
                    self.logger.error(f"Error in batch transformer analysis: {e}")
            
            results.extend(batch_results)
        
        return results


class GoogleNLPAnalyzer(NLPAnalyzer):
    """NLP analyzer using Google Cloud Natural Language API."""
    
    def __init__(self):
        """Initialize the Google NLP analyzer."""
        super().__init__()
        try:
            self.language_client = language_v1.LanguageServiceClient()
            self.is_available = True
            self.logger.info("Google Cloud Language API client initialized successfully")
        except Exception as e:
            self.is_available = False
            self.logger.warning(f"Failed to initialize Google Cloud Language API: {e}")
            self.logger.warning("Falling back to TextBlob for NLP analysis")
            # Create a fallback analyzer
            self.fallback_analyzer = TextBlobAnalyzer()
        
        # Cache language models to avoid re-creating them
        self.language_clients = {}
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using Google Cloud Natural Language API with language detection.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentiment, keywords, language, and entities
        """
        if not self.is_available:
            # Fallback when API is not available
            result = self.fallback_analyzer.analyze_text(text)
            result["emoji_analysis"] = self._analyze_emojis(text)
            return result
        
        if not text or text.isspace():
            return {
                "sentiment": {"score": 0.0, "magnitude": 0.0},
                "keywords": [],
                "language": "en",
                "entities": [],
                "emoji_analysis": self._analyze_emojis(text)
            }
        
        self.logger.debug(f"Analyzing text with Google NLP API: {text[:30]}...")
        
        try:
            # Detect language first
            lang_code = self.detect_language(text)
            
            # Get language-specific stopwords
            lang_stopwords = self.get_stopwords_for_language(lang_code)
            
            # Create document with detected language
            document = language_v1.Document(
                content=text,
                type_=language_v1.Document.Type.PLAIN_TEXT,
                language=lang_code
            )
            
            # Analyze sentiment
            sentiment = self.language_client.analyze_sentiment(document=document).document_sentiment
            sentiment_dict = {
                "score": sentiment.score,
                "magnitude": sentiment.magnitude
            }
            
            # Analyze entities
            entity_response = self.language_client.analyze_entities(document=document)
            entities = []
            keywords = []
            
            # Process and filter entities and keywords
            for entity in entity_response.entities:
                entity_name = entity.name.lower()
                
                # Only include entities with appropriate salience
                if entity.salience >= 0.01:
                    entity_type = language_v1.Entity.Type(entity.type_).name
                    
                    # Add as keyword if it's a common noun and not a stopword
                    if (entity_type in ['COMMON', 'OTHER', 'EVENT', 'WORK_OF_ART'] and 
                        entity_name not in lang_stopwords and 
                        len(entity_name) > 2):
                        keywords.append(entity_name)
                    
                    # Add as entity if it's a named entity
                    if entity_type in ['PERSON', 'LOCATION', 'ORGANIZATION', 'EVENT',
                                       'WORK_OF_ART', 'CONSUMER_GOOD']:
                        entities.append({
                            "name": entity.name,
                            "type": entity_type,
                            "salience": entity.salience
                        })
            
            # If no entities were found as keywords, fall back to NLTK extraction
            if not keywords:
                keywords = self._extract_keywords_nltk(text, lang_stopwords)
            
            return {
                "sentiment": sentiment_dict,
                "keywords": keywords,
                "language": lang_code,
                "entities": entities,
                "emoji_analysis": self._analyze_emojis(text)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing text with Google NLP API: {e}")
            self.logger.info("Falling back to TextBlob for this text")
            result = self.fallback_analyzer.analyze_text(text)
            result["emoji_analysis"] = self._analyze_emojis(text)
            return result
    
    def analyze_text_batch(self, texts: List[str], 
                          batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batches using Google Cloud NLP API.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of analysis results dictionaries
        """
        if not self.is_available:
            return self.fallback_analyzer.analyze_text_batch(texts, batch_size)
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.analyze_text(text) for text in batch]
            results.extend(batch_results)
        return results


class AnalyzerFactory:
    """Factory for creating appropriate NLP analyzers."""
    
    @staticmethod
    def create_analyzer(prefer_google: bool = True, 
                       enforce_language_specific: bool = True) -> NLPAnalyzer:
        """
        Create an appropriate NLP analyzer based on preference and availability.
        
        Args:
            prefer_google: Whether to prefer Google NLP API if available
            enforce_language_specific: Whether to enforce language-specific analysis
            
        Returns:
            An NLP analyzer instance
        """
        if prefer_google:
            analyzer = GoogleNLPAnalyzer()
            if analyzer.is_available:
                return analyzer
        
        # Fall back to TextBlob
        return TextBlobAnalyzer()