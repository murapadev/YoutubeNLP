#!/usr/bin/env python3
"""
Video Utilities: Functions for working with YouTube video metadata.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from urllib.parse import urlparse, parse_qs
import json

import requests

logger = logging.getLogger(__name__)

class VideoMetadata:
    """Represents metadata for a YouTube video."""
    
    def __init__(
        self, 
        video_id: str, 
        title: Optional[str] = None,
        channel: Optional[str] = None,
        view_count: Optional[int] = None,
        like_count: Optional[int] = None,
        publish_date: Optional[str] = None
    ):
        """
        Initialize video metadata.
        
        Args:
            video_id: YouTube video ID
            title: Video title
            channel: Channel name
            view_count: Number of views
            like_count: Number of likes
            publish_date: Publication date
        """
        self.video_id = video_id
        self.title = title
        self.channel = channel
        self.view_count = view_count
        self.like_count = like_count
        self.publish_date = publish_date
        
    def get_embed_url(self) -> str:
        """Get the embed URL for this video."""
        return f"https://www.youtube.com/embed/{self.video_id}"
    
    def get_watch_url(self) -> str:
        """Get the watch URL for this video."""
        return f"https://www.youtube.com/watch?v={self.video_id}"
    
    def __str__(self) -> str:
        """Return a string representation of the video metadata."""
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.channel:
            parts.append(f"Channel: {self.channel}")
        if self.view_count is not None:
            parts.append(f"Views: {self.view_count:,}")
        if self.like_count is not None:
            parts.append(f"Likes: {self.like_count:,}")
        if self.publish_date:
            parts.append(f"Published: {self.publish_date}")
        
        return f"Video: {self.video_id}\n" + "\n".join(parts)


def extract_video_id(url: str) -> Optional[str]:
    """
    Extract the YouTube video ID from a URL.
    
    Args:
        url: YouTube video URL
        
    Returns:
        Video ID if found, None otherwise
    """
    try:
        parsed_url = urlparse(url)
        
        # Handle youtube.com URLs
        if parsed_url.netloc in ('youtube.com', 'www.youtube.com'):
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                if 'v' in query_params:
                    return query_params['v'][0]
            elif parsed_url.path.startswith('/embed/'):
                return parsed_url.path.split('/')[2]
            elif parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]
                
        # Handle youtu.be URLs
        elif parsed_url.netloc == 'youtu.be':
            return parsed_url.path.lstrip('/')
            
        return None
    except Exception as e:
        logger.error(f"Error extracting video ID: {e}")
        return None


def get_video_metadata(url: str) -> Optional[VideoMetadata]:
    """
    Get metadata for a YouTube video.
    
    Args:
        url: YouTube video URL
        
    Returns:
        VideoMetadata object if successful, None otherwise
    """
    try:
        video_id = extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            return None
            
        # Initialize with video ID only
        metadata = VideoMetadata(video_id)
        
        # Try to get more metadata using YouTube Data API if available
        api_key = os.environ.get('YOUTUBE_API_KEY')
        if api_key:
            api_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('items') and len(data['items']) > 0:
                    item = data['items'][0]
                    snippet = item.get('snippet', {})
                    statistics = item.get('statistics', {})
                    
                    metadata.title = snippet.get('title')
                    metadata.channel = snippet.get('channelTitle')
                    metadata.publish_date = snippet.get('publishedAt')
                    
                    # Convert string values to integers
                    try:
                        if 'viewCount' in statistics:
                            metadata.view_count = int(statistics['viewCount'])
                        if 'likeCount' in statistics:
                            metadata.like_count = int(statistics['likeCount'])
                    except (ValueError, TypeError):
                        pass
                    
                    return metadata
        
        # Fallback method - try to get the title using requests and BeautifulSoup
        if not metadata.title:
            try:
                from bs4 import BeautifulSoup
                response = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title_tag = soup.find('title')
                    if title_tag and title_tag.string:
                        metadata.title = title_tag.string.replace(" - YouTube", "").strip()
            except ImportError:
                logger.warning("BeautifulSoup not available for title extraction")
            except Exception as e:
                logger.warning(f"Failed to extract title from HTML: {e}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get video metadata: {e}")
        return None