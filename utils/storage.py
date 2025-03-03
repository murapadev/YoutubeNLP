#!/usr/bin/env python3
"""
Storage Module: Provides persistent storage capabilities for the application.
"""
import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path

class AnalysisStorage:
    """
    Handles persistent storage of analysis results and application state.
    
    This class provides capabilities to:
    1. Cache analysis results by video ID
    2. Store and retrieve user preferences
    3. Track analysis history
    4. Export/import analysis data
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the storage system.
        
        Args:
            db_path: Path to the SQLite database file (default: ~/.youtubeNLP/cache.db)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set default path if not provided
        if db_path is None:
            home_dir = Path.home()
            app_dir = home_dir / ".youtubeNLP"
            app_dir.mkdir(exist_ok=True)
            db_path = str(app_dir / "cache.db")
            
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Set up the database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_cache (
                video_id TEXT PRIMARY KEY,
                timestamp TEXT,
                comment_count INTEGER,
                metadata TEXT,
                results_json TEXT
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            ''')
            
            # Modified history table without the unique constraint
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                timestamp TEXT,
                url TEXT,
                comment_count INTEGER,
                title TEXT
            )
            ''')
            
            # Add indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_timestamp ON analysis_history(timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_video_id ON analysis_history(video_id)')
            
            conn.commit()
            conn.close()
            self.logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_analysis(self, 
                     video_id: str, 
                     comment_count: int,
                     metadata: Dict[str, Any],
                     results: Union[Dict[str, Any], str],
                     url: str = None,
                     title: str = None) -> bool:
        """
        Save analysis results to storage.
        
        Args:
            video_id: YouTube video ID
            comment_count: Number of comments analyzed
            metadata: Analysis metadata (language stats, etc.)
            results: Analysis results (as dict or JSON string)
            url: Optional video URL
            title: Optional video title
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert dictionaries to JSON
            def make_json_safe(obj):
                if isinstance(obj, (list, dict)):
                    return json.dumps(obj)
                return obj
                
            metadata_json = json.dumps(metadata)
            results_json = json.dumps(results) if isinstance(results, dict) else results
            
            # Current timestamp
            timestamp = datetime.now().isoformat()
            
            # Save to cache (replacing old entry if it exists)
            cursor.execute(
                "INSERT OR REPLACE INTO analysis_cache VALUES (?, ?, ?, ?, ?)",
                (video_id, timestamp, comment_count, metadata_json, results_json)
            )
            
            # Add to history (always insert new entry)
            if url:
                cursor.execute(
                    "INSERT INTO analysis_history (video_id, timestamp, url, comment_count, title) VALUES (?, ?, ?, ?, ?)",
                    (video_id, timestamp, url, comment_count, title)
                )
            
            conn.commit()
            conn.close()
            self.logger.info(f"Saved analysis for video {video_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {e}")
            self.logger.debug("Error details", exc_info=True)
            return False
    
    def get_analysis(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis results for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with analysis results or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Get results as dictionaries
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM analysis_cache WHERE video_id = ?",
                (video_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                conn.close()
                return None
                
            # Parse JSON fields
            result = dict(row)
            result["metadata"] = json.loads(result["metadata"])
            result["results"] = json.loads(result["results_json"])
            del result["results_json"]
            
            conn.close()
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve analysis: {e}")
            return None
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get list of previously analyzed videos.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of history items, most recent first
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM analysis_history ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            
            history = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve history: {e}")
            return []
    
    def set_preference(self, key: str, value: Any) -> bool:
        """
        Save a user preference.
        
        Args:
            key: Preference name
            value: Preference value (will be JSON serialized)
            
        Returns:
            True if saved successfully
        """
        try:
            # Convert value to JSON if it's not a string
            if not isinstance(value, str):
                value = json.dumps(value)
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT OR REPLACE INTO user_preferences VALUES (?, ?)",
                (key, value)
            )
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save preference: {e}")
            return False
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a user preference.
        
        Args:
            key: Preference name
            default: Default value if preference not found
            
        Returns:
            The preference value or default if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT value FROM user_preferences WHERE key = ?",
                (key,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return default
                
            value = row[0]
            
            # Try to parse as JSON if it looks like JSON
            if value.startswith('{') or value.startswith('[') or value in ['true', 'false', 'null'] or value.isdigit():
                try:
                    return json.loads(value)
                except:
                    pass
            
            return value
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve preference: {e}")
            return default
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> bool:
        """
        Clear the analysis cache.
        
        Args:
            older_than_days: Only clear entries older than this many days (None for all)
            
        Returns:
            True if cleared successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if older_than_days is not None:
                # Calculate cutoff date
                cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                
                cursor.execute(
                    "DELETE FROM analysis_cache WHERE timestamp < ?",
                    (cutoff_date,)
                )
            else:
                cursor.execute("DELETE FROM analysis_cache")
                
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleared {affected} entries from cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return False
            
    def clear_history(self, older_than_days: Optional[int] = None) -> bool:
        """
        Clear the analysis history.
        
        Args:
            older_than_days: Only clear entries older than this many days (None for all)
            
        Returns:
            True if cleared successfully
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if older_than_days is not None:
                # Calculate cutoff date
                cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
                
                cursor.execute(
                    "DELETE FROM analysis_history WHERE timestamp < ?",
                    (cutoff_date,)
                )
            else:
                cursor.execute("DELETE FROM analysis_history")
                
            affected = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleared {affected} entries from history")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear history: {e}")
            return False


# Create a singleton instance
_storage_instance = None

def get_storage() -> AnalysisStorage:
    """
    Get or create the singleton storage instance.
    
    Returns:
        The global AnalysisStorage instance
    """
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = AnalysisStorage()
    return _storage_instance