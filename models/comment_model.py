#!/usr/bin/env python3
"""
Comment Model: Defines data structures for YouTube comments and analysis results.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
import json
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Data class representing a named entity detected in text."""
    name: str
    type: str
    salience: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Entity to a dictionary representation."""
        return {
            'name': self.name,
            'type': self.type,
            'salience': self.salience
        }


@dataclass
class CommentData:
    """Data class representing a YouTube comment with its metadata and analysis.
    
    Attributes:
        text (str): Comment text.
        author (str): Comment author.
        likes (int): Number of likes.
        time (str): Timestamp as string.
        sentiment_score (float): Sentiment score.
        sentiment_magnitude (float): Sentiment magnitude.
        sentiment_confidence (float): Confidence of transformer sentiment.
        keywords (List[str]): Extracted keywords.
        text_length (int): Length of the comment text.
        language (str): Detected language (ISO 639-1).
        entities (List[Entity]): Named entities.
        emotions (Dict[str, float]): Emotion analysis scores.
        reply_to (Optional[str]): ID of parent comment.
        reply_count (int): Number of replies.
        video_id (Optional[str]): Associated video ID.
        emoji_count (int): Number of detected emojis.
        emojis (List[str]): List of detected emojis.
        emoji_sentiment (float): Emoji-based sentiment score.
    """
    text: str
    author: str
    likes: int = 0
    time: str = ""
    sentiment_score: float = 0.0
    sentiment_magnitude: float = 0.0
    sentiment_confidence: float = 0.0  # Added for transformer model confidence
    keywords: List[str] = field(default_factory=list)
    text_length: int = 0
    language: str = "en"  # Default to English
    entities: List[Entity] = field(default_factory=list)
    emotions: Dict[str, float] = field(default_factory=dict)  # New field for emotion analysis
    reply_to: Optional[str] = None
    reply_count: int = 0
    video_id: Optional[str] = None
    emoji_count: int = 0
    emojis: List[str] = field(default_factory=list)
    emoji_sentiment: float = 0.0

    def __post_init__(self):
        """Initialize calculated fields after instance creation."""
        # Calculate text length
        self.text_length = len(self.text) if self.text else 0
        
        # Improved emoji extraction with support for multiple library versions.
        try:
            import emoji
            if hasattr(emoji, 'is_emoji'):
                # For newer emoji versions providing is_emoji()
                self.emojis = [c for c in self.text if emoji.is_emoji(c)]
            elif hasattr(emoji, 'EMOJI_DATA'):
                self.emojis = [c for c in self.text if c in emoji.EMOJI_DATA]
            else:
                self.emojis = []
            self.emoji_count = len(self.emojis)
        except ImportError:
            logger.warning("Emoji module not installed; skipping emoji extraction.")
            self.emojis = []
            self.emoji_count = 0
        except Exception as e:
            logger.warning(f"Error during emoji extraction: {e}")
            self.emojis = []
            self.emoji_count = 0
        
        # Default empty collections if None
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = []
        if self.emotions is None:
            self.emotions = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert CommentData to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the comment.
        """
        basic_dict = {
            'text': self.text,
            'author': self.author,
            'likes': self.likes,
            'time': self.time,
            'sentiment_score': self.sentiment_score,
            'sentiment_magnitude': self.sentiment_magnitude,
            'sentiment_confidence': self.sentiment_confidence,
            'keywords': self.keywords,
            'text_length': self.text_length,
            'language': self.language,
            'entities': [entity.to_dict() for entity in self.entities],
            'emotions': self.emotions,
            'reply_to': self.reply_to,
            'reply_count': self.reply_count,
            'video_id': self.video_id,
            'emoji_count': self.emoji_count,
            'emojis': self.emojis,
            'emoji_sentiment': self.emoji_sentiment
        }
        return basic_dict

    def get_sentiment_category(self) -> str:
        """Get the sentiment category based on score."""
        if self.sentiment_score <= -0.6:
            return "Very Negative"
        elif self.sentiment_score <= -0.2:
            return "Negative"
        elif self.sentiment_score < 0.2:
            return "Neutral"
        elif self.sentiment_score < 0.6:
            return "Positive"
        else:
            return "Very Positive"
    
    def get_primary_emotion(self) -> Optional[Tuple[str, float]]:
        """Get the primary emotion and its score if emotions are available.
        
        Returns:
            Optional[Tuple[str, float]]: Tuple of (emotion, score) or None.
        """
        if not self.emotions:
            return None
        return max(self.emotions.items(), key=lambda x: x[1])
    
    def parse_relative_time(self) -> Optional[datetime]:
        """
        Parse relative time strings in multiple languages to datetime objects.
        Uses dateparser for universal relative date parsing with multi-language support.
        
        Returns:
            Optional[datetime]: Parsed datetime object, or None if parsing fails
        """
        if not self.time:
            return None
        
        try:
            import dateparser
            
            # Configure dateparser settings
            settings = {
                'RELATIVE_BASE': datetime.now(),  # Use current time as reference
                'TIMEZONE': 'UTC',  # Normalize to UTC
                'TO_TIMEZONE': 'UTC',
                'RETURN_AS_TIMEZONE_AWARE': False,  # Return timezone-naive datetime
                'PREFER_DATES_FROM': 'past'  # YouTube comments are always in the past
            }
            
            # Parse the relative date string
            result = dateparser.parse(
                self.time,
                settings=settings,
                languages=['en', 'es', 'fr', 'de', 'it', 'pt', 'nl']  # Add more languages as needed
            )
            
            return result
            
        except ImportError:
            logger.warning("dateparser module not installed. Date parsing disabled.")
            return None
        except Exception as e:
            # Log the error but don't raise it to avoid breaking the analysis
            logger.warning(f"Failed to parse date '{self.time}': {str(e)}")
            return None

    def get_datetime(self) -> Optional[datetime]:
        """Convert the comment's time string to a datetime object.
        
        Returns:
            Optional[datetime]: Parsed datetime or None if parsing fails.
        """
        return self.parse_relative_time()


class CommentsCollection:
    """Collection of comments with data processing capabilities."""
    
    def __init__(self, comments: List[CommentData] = None):
        """Initialize CommentsCollection with a list of CommentData objects.
        
        Args:
            comments: List of CommentData objects
        """
        self.comments = comments or []
        self._dataframe = None
        self._metadata = {
            "language_counts": {},
            "sentiment_stats": {},
            "entity_counts": {}
        }
        self._is_analyzed = False
    
    def add_comment(self, comment: CommentData) -> None:
        """Add a comment to the collection.
        
        Args:
            comment: CommentData object
        """
        self.comments.append(comment)
        # Reset cached dataframe and analysis
        self._dataframe = None
        self._is_analyzed = False
    
    def add_comments(self, comments: List[CommentData]) -> None:
        """Add multiple comments to the collection.
        
        Args:
            comments: List of CommentData objects
        """
        self.comments.extend(comments)
        # Reset cached dataframe and analysis
        self._dataframe = None
        self._is_analyzed = False

    def to_dataframe(self) -> pd.DataFrame:
        """Convert comments to a pandas DataFrame.
        
        Returns:
            DataFrame with comment data
        """
        if self._dataframe is None or len(self._dataframe) != len(self.comments):
            self._dataframe = pd.DataFrame([c.to_dict() for c in self.comments])
        return self._dataframe

    def to_csv(self) -> str:
        """Convert comments to CSV format.
        
        Returns:
            CSV string representation of comments
        """
        return self.to_dataframe().to_csv(index=False)
    
    def filter_by_language(self, language_code: str) -> 'CommentsCollection':
        """
        Filter comments by language.
        
        Args:
            language_code: ISO 639-1 language code to filter by
            
        Returns:
            A new CommentsCollection containing only comments in the specified language
        """
        filtered = [comment for comment in self.comments if comment.language == language_code]
        return CommentsCollection(filtered)
    
    def filter_by_sentiment(self, 
                          min_score: float = -1.0, 
                          max_score: float = 1.0) -> 'CommentsCollection':
        """
        Filter comments by sentiment score range.
        
        Args:
            min_score: Minimum sentiment score (default: -1.0)
            max_score: Maximum sentiment score (default: 1.0)
            
        Returns:
            A new CommentsCollection containing only comments in the specified sentiment range
        """
        filtered = [
            comment for comment in self.comments 
            if min_score <= comment.sentiment_score <= max_score
        ]
        return CommentsCollection(filtered)
    
    def analyze_metadata(self) -> Dict[str, Any]:
        """
        Analyze the collection of comments to extract metadata and statistics.
        
        Returns:
            Dictionary containing metadata and statistics about the comments
        """
        if self._is_analyzed:
            return self._metadata
            
        # Initialize counters
        language_counts = {}
        sentiment_counts = {"Very Negative": 0, "Negative": 0, "Neutral": 0, "Positive": 0, "Very Positive": 0}
        entity_counts = {}
        sentiment_scores = []
        sentiment_by_language = {}
        
        for comment in self.comments:
            # Count languages
            lang = comment.language
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            # Count sentiment categories
            sentiment_cat = comment.get_sentiment_category()
            sentiment_counts[sentiment_cat] = sentiment_counts.get(sentiment_cat, 0) + 1
            
            # Track sentiment scores for stats
            sentiment_scores.append(comment.sentiment_score)
            
            # Track sentiment by language
            if lang not in sentiment_by_language:
                sentiment_by_language[lang] = []
            sentiment_by_language[lang].append(comment.sentiment_score)
            
            # Count entities
            for entity in comment.entities:
                entity_key = f"{entity.type}:{entity.name}"
                entity_counts[entity_key] = entity_counts.get(entity_key, 0) + 1
        
        # Calculate sentiment statistics (safely)
        sentiment_stats = {
            "counts": sentiment_counts,
            "mean": sum(sentiment_scores) / max(len(sentiment_scores), 1),
            "min": min(sentiment_scores) if sentiment_scores else 0,
            "max": max(sentiment_scores) if sentiment_scores else 0,
        }
        
        # Calculate sentiment by language
        sentiment_by_lang_stats = {}
        for lang, scores in sentiment_by_language.items():
            sentiment_by_lang_stats[lang] = {
                "mean": sum(scores) / max(len(scores), 1),  # Avoid division by zero
                "count": len(scores)
            }
        
        # Save metadata
        self._metadata = {
            "language_counts": language_counts,
            "sentiment_stats": sentiment_stats,
            "sentiment_by_language": sentiment_by_lang_stats,
            "entity_counts": entity_counts,
            "total_comments": len(self.comments),
            "total_likes": sum(c.likes for c in self.comments)
        }
        
        self._is_analyzed = True
        return self._metadata

    def __getitem__(self, index):
        """Allow indexing into the collection."""
        return self.comments[index]
    
    def __len__(self) -> int:
        """Get the number of comments in the collection."""
        return len(self.comments)

    def __iter__(self):
        """Iterate over comments in the collection."""
        return iter(self.comments)

    @property
    def is_empty(self) -> bool:
        """Check if the collection is empty."""
        return len(self.comments) == 0