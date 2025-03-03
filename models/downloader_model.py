#!/usr/bin/env python3
"""
Downloader Model: Handles fetching comments from YouTube videos.
"""
import logging
import re
import time
from typing import List, Dict, Any, Optional, Iterator, Callable
from itertools import islice

import streamlit as st
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT

from models.comment_model import CommentData, CommentsCollection


class YouTubeDownloader:
    """
    Handles downloading and processing YouTube comments with progress tracking
    and adaptive rate limiting to avoid API throttling.
    """
    
    def __init__(self):
        """Initialize the YouTube downloader with appropriate configuration."""
        self.downloader = YoutubeCommentDownloader()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Rate limiting settings to avoid throttling
        self.batch_size = 100
        self.batch_pause = 1.0  # seconds between batches
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if the given URL is a valid YouTube URL.
        
        Args:
            url: YouTube URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        pattern = r'^(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$'
        return bool(re.match(pattern, url))
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract the video ID from a YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None if not found
        """
        # Look for v= pattern
        v_pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        match = re.search(v_pattern, url)
        if match:
            return match.group(1)
        return None
    
    def get_comments(self, 
                    url: str, 
                    max_comments: Optional[int] = None, 
                    sort_by: str = SORT_BY_POPULAR,
                    progress_callback: Optional[Callable[[int], None]] = None) -> CommentsCollection:
        """
        Fetch comments from a YouTube video with progress tracking.
        
        Args:
            url: YouTube video URL
            max_comments: Maximum number of comments to fetch (None for all available)
            sort_by: Comment sorting method (SORT_BY_POPULAR or SORT_BY_RECENT)
            progress_callback: Optional callback function to report progress
            
        Returns:
            CommentsCollection object containing the fetched comments
        
        Raises:
            ValueError: If URL is invalid
            RuntimeError: If comments cannot be fetched
        """
        if not self.validate_url(url):
            self.logger.error(f"Invalid YouTube URL: {url}")
            raise ValueError(f"Invalid YouTube URL: {url}")
        
        video_id = self.extract_video_id(url)
        if not video_id:
            self.logger.error(f"Could not extract video ID from URL: {url}")
            raise ValueError(f"Could not extract video ID from URL: {url}")
            
        try:
            self.logger.info(f"Fetching comments from video ID: {video_id}")
            raw_comments = self.downloader.get_comments_from_url(url, sort_by=sort_by)
            
            comment_collection = self._process_comments(
                raw_comments, 
                max_comments, 
                progress_callback
            )
            
            self.logger.info(f"Fetched {len(comment_collection)} comments from {url}")
            return comment_collection
            
        except Exception as e:
            self.logger.error(f"Failed to fetch comments: {str(e)}")
            raise RuntimeError(f"Failed to fetch comments: {str(e)}")
    
    def _process_comments(self, 
                         raw_comments: Iterator[Dict[str, Any]], 
                         max_comments: Optional[int] = None,
                         progress_callback: Optional[Callable[[int], None]] = None) -> CommentsCollection:
        """
        Process raw comment data from the downloader with progress tracking.
        
        Args:
            raw_comments: Iterator of raw comment dictionaries
            max_comments: Maximum number of comments to process (None for all)
            progress_callback: Optional callback function to report progress
            
        Returns:
            CommentsCollection containing processed comments
        """
        comments_collection = CommentsCollection()
        comments_processed = 0
        last_progress_update = 0
        batch_counter = 0
        
        # Set up initial progress reporting
        if progress_callback:
            progress_callback(0)
        
        # Handle explicit limit or unlimited
        has_limit = max_comments is not None
        
        try:
            for comment_dict in raw_comments:
                # Stop if we hit the limit
                if has_limit and comments_processed >= max_comments:
                    break
                
                # Skip empty comments
                if not comment_dict.get('text'):
                    continue
                
                # Process timestamp
                time_str = comment_dict.get('time', '')
                
                # Process likes - ensure we have a valid integer
                try:
                    # Different versions of the downloader may return likes as string or int
                    likes_value = comment_dict.get('likes', 0)
                    # If it's a string format like "1.5K", convert to approximate number
                    if isinstance(likes_value, str):
                        likes_value = likes_value.strip()
                        if likes_value.endswith('K'):
                            likes = int(float(likes_value[:-1]) * 1000)
                        elif likes_value.endswith('M'):
                            likes = int(float(likes_value[:-1]) * 1000000)
                        elif likes_value:
                            likes = int(likes_value)
                        else:
                            likes = 0
                    else:
                        likes = int(likes_value) if likes_value else 0
                except (ValueError, TypeError):
                    self.logger.warning(f"Could not parse likes value: {comment_dict.get('likes')}")
                    likes = 0
                
                # YouTube timestamps are relative like "1 year ago", "2 months ago", etc.
                # Convert them to ISO format datetime strings if possible
                try:
                    from dateutil.parser import parse
                    from dateutil.relativedelta import relativedelta
                    from datetime import datetime
                    
                    if time_str:
                        # Current time as reference
                        now = datetime.now()
                        
                        # Parse relative time string
                        parts = time_str.lower().split()
                        if len(parts) >= 2:
                            quantity = int(parts[0])
                            unit = parts[1]
                            
                            # Map YouTube's time units to relativedelta arguments
                            unit_map = {
                                'second': 'seconds',
                                'seconds': 'seconds',
                                'minute': 'minutes',
                                'minutes': 'minutes',
                                'hour': 'hours',
                                'hours': 'hours',
                                'day': 'days',
                                'days': 'days',
                                'week': 'weeks',
                                'weeks': 'weeks',
                                'month': 'months',
                                'months': 'months',
                                'year': 'years',
                                'years': 'years'
                            }
                            
                            if unit.rstrip('s') in unit_map:
                                unit_key = unit_map[unit.rstrip('s')]
                                delta_args = {unit_key: quantity}
                                timestamp = now - relativedelta(**delta_args)
                                time_str = timestamp.isoformat()
                except Exception as e:
                    self.logger.warning(f"Could not parse timestamp '{time_str}': {str(e)}")
                
                # Create comment object
                comment = CommentData(
                    text=comment_dict.get('text', ''),
                    author=comment_dict.get('author', 'Anonymous'),
                    likes=likes,  # Use the parsed likes value
                    time=time_str,  # Use the processed timestamp
                    reply_to=comment_dict.get('reply_to'),
                    reply_count=comment_dict.get('reply_count', 0),
                    video_id=comment_dict.get('video_id')
                )
                
                # Add comment to collection
                comments_collection.add_comment(comment)
                comments_processed += 1
                batch_counter += 1
                
                # Update progress periodically (not every iteration for efficiency)
                if progress_callback and comments_processed - last_progress_update >= 10:
                    progress_callback(comments_processed)
                    last_progress_update = comments_processed
                
                # Throttle requests to avoid API limits
                if batch_counter >= self.batch_size:
                    time.sleep(self.batch_pause)
                    batch_counter = 0
                
        except Exception as e:
            self.logger.error(f"Error while processing comments: {str(e)}")
            # Continue with comments successfully fetched so far
        
        # Final progress update
        if progress_callback:
            progress_callback(comments_processed)
        
        return comments_collection