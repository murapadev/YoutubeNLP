#!/usr/bin/env python3
"""
Analysis Controller: Coordinates between models and views to handle the analysis workflow.
"""
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
from youtube_comment_downloader import SORT_BY_POPULAR, SORT_BY_RECENT
import streamlit as st

from models.comment_model import CommentsCollection, CommentData, Entity
from models.analyzer_model import AnalyzerFactory
from models.downloader_model import YouTubeDownloader
from utils.config import has_google_credentials
from utils.storage import get_storage


class AnalysisController:
    """Controller for coordinating YouTube comment analysis workflow."""
    
    def __init__(self):
        """Initialize the analysis controller with required components."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.downloader = YouTubeDownloader()
        self.analyzer = AnalyzerFactory.create_analyzer(
            prefer_google=has_google_credentials(),
            enforce_language_specific=True
        )
        
        # Initialize storage system
        self.storage = get_storage()
        
        # Current state
        self.comments = None
        self.dataframe = None
        self.metadata = None
        self.video_id = None
        self.video_url = None
        self.video_title = None
        self.analysis_complete = False
        self.using_cached_data = False
        
        # Processing settings
        self.batch_size = 10  # Number of comments to analyze in a single batch
        self.use_batched_analysis = True
        
        # Load preferences from storage
        self._load_preferences()
    
    def _load_preferences(self):
        """Load user preferences from storage."""
        try:
            # Load batch size preference
            batch_size = self.storage.get_preference('batch_size')
            if batch_size is not None:
                self.batch_size = batch_size
                
            # Load batched analysis preference
            use_batched = self.storage.get_preference('use_batched_analysis')
            if use_batched is not None:
                self.use_batched_analysis = use_batched
        except Exception as e:
            self.logger.warning(f"Failed to load preferences: {e}")
    
    def fetch_comments(self, 
                      url: str, 
                      max_comments: Optional[int] = None, 
                      sort_by: str = SORT_BY_POPULAR,
                      use_cache: bool = True) -> bool:
        """
        Fetch comments from the given YouTube URL with progress tracking.
        
        Args:
            url: YouTube video URL
            max_comments: Maximum number of comments to fetch (None for all available)
            sort_by: Comment sorting method (SORT_BY_POPULAR or SORT_BY_RECENT)
            use_cache: Whether to check for cached results first
            
        Returns:
            True if comments were fetched successfully, False otherwise
        """
        try:
            # Get video metadata first
            from utils.video_utils import extract_video_id, get_video_metadata
            video_id = extract_video_id(url)
            if video_id:
                # Try to get metadata from video
                metadata = None
                try:
                    metadata = get_video_metadata(url)
                    if metadata:
                        # Store metadata in session state for visualization
                        st.session_state.video_metadata = metadata.__dict__
                except Exception as e:
                    self.logger.warning(f"Could not fetch video metadata: {e}")
                    
            if not self.downloader.validate_url(url):
                self.logger.error(f"Invalid YouTube URL: {url}")
                st.error("Please enter a valid YouTube URL.")
                return False
            
            # Extract and save video ID and URL
            self.video_id = video_id
            self.video_url = url
            
            if not self.video_id:
                st.error("Could not extract video ID from URL.")
                return False
            
            # Check if we have cached results for this video
            if use_cache:
                cached_data = self.storage.get_analysis(self.video_id)
                if cached_data:
                    # Display info about found cache
                    cache_time = cached_data["timestamp"]
                    comment_count = cached_data["comment_count"]
                    
                    # Format the timestamp for display
                    from datetime import datetime
                    try:
                        dt = datetime.fromisoformat(cache_time)
                        formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                    except:
                        formatted_time = cache_time
                    
                    # Ask user if they want to use cached data or fetch again
                    st.info(f"Found cached analysis for this video from {formatted_time} with {comment_count} comments.")
                    use_cached = st.radio(
                        "Would you like to use the cached results or fetch new comments?",
                        ["Use cached results", "Fetch new comments"],
                        index=0
                    ) == "Use cached results"
                    
                    if use_cached:
                        self._load_from_cache(cached_data)
                        return True
                
            # Progress tracking variables for UI
            progress = st.progress(0.0)
            progress_text = st.empty()
            counter = {"comments": 0}
            
            def update_progress(count):
                counter["comments"] = count
                if max_comments:
                    progress.progress(min(count / max_comments, 1.0))
                progress_text.text(f"Fetched {count} comments...")
            
            with st.spinner("Fetching comments..."):
                # Try to get video title for better UX
                try:
                    import requests
                    from bs4 import BeautifulSoup
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        title_tag = soup.find('title')
                        if title_tag and title_tag.string:
                            self.video_title = title_tag.string.replace(" - YouTube", "").strip()
                except Exception as e:
                    self.logger.warning(f"Could not fetch video title: {e}")
                
                self.comments = self.downloader.get_comments(
                    url, 
                    max_comments, 
                    sort_by,
                    progress_callback=update_progress
                )
                
            if not self.comments or len(self.comments) == 0:
                st.error("No comments found for this video.")
                return False
                
            # Add video_id to each comment
            for comment in self.comments:
                comment.video_id = self.video_id
                
            final_count = len(self.comments)
            progress.progress(1.0)
            progress_text.text(f"Successfully fetched {final_count} comments.")
            time.sleep(0.5)  # Brief pause for UI
            progress.empty()
            progress_text.empty()
                
            st.success(f"Successfully fetched {final_count} comments.")
            return True
            
        except Exception as e:
            self.logger.error(f"Error fetching comments: {str(e)}")
            st.error(f"Error fetching comments: {str(e)}")
            return False
    
    def _load_from_cache(self, cached_data: Dict[str, Any]) -> None:
        """
        Load analysis data from cache.
        
        Args:
            cached_data: Cached analysis data from storage
        """
        try:
            self.logger.info(f"Loading cached data for video {self.video_id}")
            
            # Extract data from cache
            results = cached_data["results"]
            self.metadata = cached_data["metadata"]
            
            # Convert cached results back to a DataFrame
            self.dataframe = pd.DataFrame.from_dict(results)
            
            # Reconstruct comments collection if needed
            comments_list = []
            for _, row in self.dataframe.iterrows():
                comment = CommentData(
                    text=row.get('text', ''),
                    author=row.get('author', ''),
                    likes=row.get('likes', 0),
                    time=row.get('time', ''),
                )
                
                # Add analysis results
                comment.sentiment_score = row.get('sentiment_score', 0.0)
                comment.sentiment_magnitude = row.get('sentiment_magnitude', 0.0)
                comment.language = row.get('language', 'en')
                comment.keywords = row.get('keywords', [])
                
                # Add entities if available
                if 'entities' in row and row['entities']:
                    entities_data = row['entities'] if isinstance(row['entities'], list) else json.loads(row['entities'])
                    comment.entities = [Entity(**entity) for entity in entities_data]
                
                comments_list.append(comment)
            
            self.comments = CommentsCollection(comments_list)
            
            # Mark as complete and using cached data
            self.analysis_complete = True
            self.using_cached_data = True
            
            st.success(f"Loaded cached analysis with {len(self.comments)} comments.")
            
        except Exception as e:
            self.logger.error(f"Error loading cached data: {e}")
            st.warning(f"Error loading cached data: {e}. Will perform new analysis.")
            self.using_cached_data = False
    
    def analyze_comments(self) -> bool:
        """
        Analyze the fetched comments using NLP techniques.
        
        Returns:
            True if analysis was successful, False otherwise
        """
        # If we're using cached data, no need to analyze again
        if self.using_cached_data and self.analysis_complete:
            return True
        
        if not self.comments:
            self.logger.error("No comments to analyze")
            return False
            
        try:
            analyzed_comments = CommentsCollection()
            language_counter = Counter()
            entity_counter = Counter()
            
            with st.spinner('Analyzing comments...'):
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                total_comments = len(self.comments)
                
                # Determine whether to use batched processing based on comment count
                if self.use_batched_analysis and total_comments > self.batch_size:
                    # Batch processing for better performance
                    self._analyze_comments_batched(
                        analyzed_comments, 
                        language_counter,
                        entity_counter,
                        progress_bar, 
                        progress_text
                    )
                else:
                    # Individual processing - better for smaller sets or when more details needed
                    self._analyze_comments_individual(
                        analyzed_comments, 
                        language_counter,
                        entity_counter,
                        progress_bar, 
                        progress_text
                    )
            
            # Replace the comments collection with the analyzed one
            self.comments = analyzed_comments
            # Generate statistics and metadata
            self.metadata = self.comments.analyze_metadata()
            # Create dataframe for visualizations
            self.dataframe = self.comments.to_dataframe()
            
            # Save to storage
            self._save_to_cache()
            
            # Clear progress indicators
            progress_bar.empty()
            progress_text.empty()
            
            # Mark analysis as complete
            self.analysis_complete = True
            self.using_cached_data = False
            
            # Display language statistics
            self._display_language_stats(language_counter)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error analyzing comments: {str(e)}")
            st.error(f"Error analyzing comments: {str(e)}")
            return False
    
    def _save_to_cache(self) -> None:
        """Save analysis results to cache."""
        if not self.video_id or self.dataframe is None:  # Fixed condition check
            return
            
        try:
            # Convert DataFrame to JSON-serializable format
            results = [
                {k: v for k, v in record.items() if pd.notna(v)}  # Filter out NaN values
                for record in self.dataframe.to_dict('records')
            ]
            
            # Save to storage
            self.storage.save_analysis(
                video_id=self.video_id,
                comment_count=len(self.comments) if self.comments else 0,
                metadata=self.metadata or {},  # Ensure metadata is never None
                results=results,
                url=self.video_url,
                title=self.video_title
            )
            self.logger.info(f"Saved analysis results to cache for video {self.video_id}")
        except Exception as e:
            self.logger.error(f"Failed to save to cache: {e}")
            self.logger.debug("Error details", exc_info=True)
    
    def _analyze_comments_individual(self, 
                                   analyzed_comments: CommentsCollection,
                                   language_counter: Counter,
                                   entity_counter: Counter,
                                   progress_bar,
                                   progress_text) -> None:
        """
        Process comments individually, showing detailed progress.
        
        Args:
            analyzed_comments: Collection to store processed comments
            language_counter: Counter to track language statistics
            entity_counter: Counter to track entity statistics
            progress_bar: Streamlit progress bar element
            progress_text: Streamlit text element for progress updates
        """
        total_comments = len(self.comments)
        
        for i, comment in enumerate(self.comments):
            # Update progress
            if i % 5 == 0 or i == total_comments - 1:
                progress = min(i / total_comments, 1.0)
                progress_bar.progress(progress)
                progress_text.text(f"Analyzing comment {i+1} of {total_comments}...")
            
            # Skip empty comments
            if not comment.text or comment.text.isspace():
                continue
                
            # Analyze the comment text
            analysis_results = self.analyzer.analyze_text(comment.text)
            
            # Update the comment with analysis results
            comment.sentiment_score = analysis_results["sentiment"]["score"]
            comment.sentiment_magnitude = analysis_results["sentiment"]["magnitude"]
            comment.keywords = analysis_results["keywords"]
            comment.language = analysis_results.get("language", "en")
            
            # Add detected entities if available
            if "entities" in analysis_results:
                for entity_data in analysis_results["entities"]:
                    entity = Entity(
                        name=entity_data["name"],
                        type=entity_data["type"],
                        salience=entity_data.get("salience", 0.0)
                    )
                    comment.entities.append(entity)
                    entity_counter[f"{entity.type}:{entity.name}"] += 1
            
            # Track language statistics
            language_counter[comment.language] += 1
            
            # Add processed comment
            analyzed_comments.add_comment(comment)
    
    def _analyze_comments_batched(self, 
                                analyzed_comments: CommentsCollection,
                                language_counter: Counter,
                                entity_counter: Counter,
                                progress_bar,
                                progress_text) -> None:
        """
        Process comments in batches for better performance.
        
        Args:
            analyzed_comments: Collection to store processed comments
            language_counter: Counter to track language statistics
            entity_counter: Counter to track entity statistics
            progress_bar: Streamlit progress bar element 
            progress_text: Streamlit text element for progress updates
        """
        total_comments = len(self.comments)
        batch_size = self.batch_size
        
        # Process in batches
        for i in range(0, total_comments, batch_size):
            # Get current batch
            batch_end = min(i + batch_size, total_comments)
            current_batch = self.comments[i:batch_end]
            
            # Update progress
            progress = min(i / total_comments, 1.0)
            progress_bar.progress(progress)
            progress_text.text(f"Analyzing batch {i//batch_size + 1} of {(total_comments + batch_size - 1)//batch_size}...")
            
            # Extract texts for batch processing
            batch_texts = [comment.text for comment in current_batch if comment.text and not comment.text.isspace()]
            valid_indices = [j for j, comment in enumerate(current_batch) if comment.text and not comment.text.isspace()]
            
            if not batch_texts:
                continue
                
            # Analyze batch
            batch_results = self.analyzer.analyze_text_batch(batch_texts)
            
            # Process results and update comments
            for idx, result in zip(valid_indices, batch_results):
                comment = current_batch[idx]
                
                # Update with analysis results
                comment.sentiment_score = result["sentiment"]["score"]
                comment.sentiment_magnitude = result["sentiment"]["magnitude"]
                comment.keywords = result["keywords"]
                comment.language = result.get("language", "en")
                
                # Add detected entities if available
                if "entities" in result:
                    for entity_data in result["entities"]:
                        entity = Entity(
                            name=entity_data["name"],
                            type=entity_data["type"],
                            salience=entity_data.get("salience", 0.0)
                        )
                        comment.entities.append(entity)
                        entity_counter[f"{entity.type}:{entity.name}"] += 1
                
                # Track language statistics  
                language_counter[comment.language] += 1
                
                # Add processed comment
                analyzed_comments.add_comment(comment)
        
        # Final progress update
        progress_bar.progress(1.0)
        progress_text.text(f"Analyzed {total_comments} comments.")
    
    def _display_language_stats(self, language_counter: Counter) -> None:
        """
        Display statistics about detected languages.
        
        Args:
            language_counter: Counter with language codes and their counts
        """
        if not language_counter:
            return
            
        # Language name mapping for common ISO codes
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'pl': 'Polish',
            'tr': 'Turkish',
            'sv': 'Swedish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'da': 'Danish',
            'cs': 'Czech',
            'el': 'Greek',
            'he': 'Hebrew',
            'id': 'Indonesian',
            'ro': 'Romanian',
            'hu': 'Hungarian',
        }
        
        st.subheader("Comment Languages")
        
        # Convert to more readable format
        lang_data = [(language_names.get(code, code), count) for code, count in language_counter.most_common()]
        
        # Display as a small table
        lang_df = pd.DataFrame(lang_data, columns=["Language", "Count"])
        total_comments = lang_df["Count"].sum()
        lang_df["Percentage"] = (lang_df["Count"] / total_comments * 100).round(1).astype(str) + '%'
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(lang_df, use_container_width=True)
            
        with col2:
            # Only show chart if there's more than one language
            if len(lang_data) > 1:
                import plotly.express as px
                fig = px.pie(
                    lang_df, 
                    values='Count', 
                    names='Language',
                    title='Comment Language Distribution'
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Get the dataframe containing analyzed comments.
        
        Returns:
            DataFrame with analysis results or None if not available
        """
        if self.dataframe is None and self.comments:
            self.dataframe = self.comments.to_dataframe()
        return self.dataframe
    
    def get_csv(self) -> Optional[str]:
        """
        Get CSV representation of the analyzed comments.
        
        Returns:
            CSV string or None if not available
        """
        if not self.comments:
            return None
        return self.comments.to_csv()
    
    def get_language_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get statistics about detected languages.
        
        Returns:
            Dictionary with language statistics
        """
        if not self.metadata:
            return {}
        return self.metadata.get("language_counts", {})
    
    def get_entity_stats(self) -> Dict[str, int]:
        """
        Get statistics about detected named entities.
        
        Returns:
            Dictionary mapping entity types to counts
        """
        if not self.metadata:
            return {}
            
        # Group entity counts by type
        entity_counts = defaultdict(int)
        for entity_key, count in self.metadata.get("entity_counts", {}).items():
            entity_type = entity_key.split(":", 1)[0]
            entity_counts[entity_type] += count
            
        return dict(entity_counts)
    
    def set_preference(self, key: str, value: Any) -> bool:
        """
        Save a user preference both in memory and to persistent storage.
        
        Args:
            key: Preference name
            value: Preference value
            
        Returns:
            True if saved successfully
        """
        try:
            # Update in-memory setting
            if key == 'batch_size':
                self.batch_size = int(value)
            elif key == 'use_batched_analysis':
                self.use_batched_analysis = bool(value)
            
            # Save to storage
            return self.storage.set_preference(key, value)
        except Exception as e:
            self.logger.error(f"Failed to save preference {key}: {e}")
            return False
        
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of previously analyzed videos.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of history entries
        """
        return self.storage.get_analysis_history(limit)
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> bool:
        """
        Clear the analysis cache.
        
        Args:
            older_than_days: Only clear entries older than this many days (None for all)
            
        Returns:
            True if cleared successfully
        """
        return self.storage.clear_cache(older_than_days)
    
    def clear_history(self, older_than_days: Optional[int] = None) -> bool:
        """
        Clear the analysis history.
        
        Args:
            older_than_days: Only clear entries older than this many days (None for all)
            
        Returns:
            True if cleared successfully
        """
        return self.storage.clear_history(older_than_days)
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.comments = None
        self.dataframe = None
        self.metadata = None
        self.video_id = None
        self.video_url = None
        self.video_title = None
        self.analysis_complete = False
        self.using_cached_data = False