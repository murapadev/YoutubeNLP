#!/usr/bin/env python3
"""
YoutubeNLP:

Usage:

youtubeNLP "video_url"


It returns many plots and a csv file with the data.

- The first plot is the sentiment analysis of the comments of the video.
- The second plot a wordcloud of the comments.
- The csv will content the user name, the comment and the sentiment of the comment, and the keywords of the comment.

The script will get all the comments made in the video and will analyze them with the help of the Google Natural Language API.

"""
import sys
import os
import csv
import re
import logging
from typing import Dict, List, Iterator, Optional, Any, Tuple

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
from itertools import islice
from google.cloud import language_v1
import requests
from urllib.parse import urlparse, parse_qs
import json

from utils.video_utils import VideoMetadata, extract_video_id, get_video_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("YoutubeNLP")

def get_comments(url: str, comments_len: int = 100) -> Iterator:
    """
    Get comments from a YouTube video.
    
    Args:
        url: YouTube video URL
        comments_len: Maximum number of comments to retrieve
        
    Returns:
        Iterator of comments
    """
    try:
        downloader = YoutubeCommentDownloader()
        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
        # Get the specified number of comments from the video
        return islice(comments, comments_len)
    except Exception as e:
        logger.error(f"Failed to get comments: {e}")
        return iter([])  # Return empty iterator

def get_sentiment(comment: Any, client: Optional[language_v1.LanguageServiceClient] = None) -> float:
    """
    Get the sentiment score of a comment.
    
    Args:
        comment: Comment object
        client: Language service client (created if None)
        
    Returns:
        Sentiment score (-1.0 to 1.0)
    """
    try:
        if client is None:
            client = language_v1.LanguageServiceClient()
        # The text to analyze
        text = comment.text
        document = language_v1.Document(
            content=text,
            type=language_v1.Document.Type.PLAIN_TEXT)
        # Detects the sentiment of the text
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        return sentiment.score
    except Exception as e:
        logger.warning(f"Failed to analyze sentiment: {e}")
        return 0.0  # Neutral sentiment as fallback

def get_keywords(comment: Any, client: Optional[language_v1.LanguageServiceClient] = None) -> List[str]:
    """
    Extract keywords from a comment.
    
    Args:
        comment: Comment object
        client: Language service client (created if None)
        
    Returns:
        List of keywords
    """
    try:
        if client is None:
            client = language_v1.LanguageServiceClient()
            
        # The text to analyze
        text = comment.text
        document = language_v1.Document(
            content=text,
            type=language_v1.Document.Type.PLAIN_TEXT)
        # Detects syntax and extracts tokens
        keywords = client.analyze_syntax(document=document).tokens
        return [keyword.lemma for keyword in keywords if keyword.part_of_speech.tag == 1]
    except Exception as e:
        logger.warning(f"Failed to extract keywords: {e}")
        return []

def analyze_comments(url: str, comments_len: int = 100) -> Tuple[Dict[str, Any], str]:
    """
    Analyze comments from a YouTube video and generate visualizations.
    
    Args:
        url: YouTube video URL
        comments_len: Maximum number of comments to analyze
        
    Returns:
        Tuple containing analysis results and CSV filename
    """
    # Get video metadata
    video_metadata = get_video_metadata(url)
    if not video_metadata:
        logger.error("Failed to get video metadata")
    else:
        logger.info(f"Analyzing video: {video_metadata.title or video_metadata.video_id}")
    
    # Define output file based on video ID if available
    output_filename = f"comments_{video_metadata.video_id}.csv" if video_metadata else "comments.csv"
    
    # Get comments
    comments = get_comments(url, comments_len)
    if not comments:
        logger.error("No comments found or error occurred")
        return {"error": "No comments found"}, output_filename
    
    # Initialize Language API client (reuse for all comments)
    try:
        nlp_client = language_v1.LanguageServiceClient()
    except Exception as e:
        logger.error(f"Failed to initialize Google Language API: {e}")
        logger.warning("Continuing without sentiment analysis")
        nlp_client = None
    
    # Create the csv file
    try:
        with open(output_filename, "w", newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # Include video metadata in header
            if video_metadata and video_metadata.title:
                csv_writer.writerow(["Video Title", video_metadata.title])
                if video_metadata.channel:
                    csv_writer.writerow(["Channel", video_metadata.channel])
                csv_writer.writerow([])  # Empty row for separation
            
            # Write column headers
            csv_writer.writerow(["user", "comment", "sentiment", "keywords"])
            
            # Process all comments
            all_comments = ""
            sentiments = []
            
            comment_count = 0
            
            for comment in comments:
                comment_count += 1
                if comment_count % 10 == 0:
                    logger.info(f"Processed {comment_count} comments")
                
                # Get the sentiment of the comment
                sentiment = get_sentiment(comment, nlp_client)
                # Get the keywords of the comment
                keywords = get_keywords(comment, nlp_client)
                # Write the data in the csv file
                csv_writer.writerow([comment.author, comment.text, sentiment, ",".join(keywords)])
                # Add the comment to the wordcloud
                all_comments += comment.text + " "
                # Add the sentiment to the sentiments list
                sentiments.append(sentiment)
        
        logger.info(f"Analysis complete. Processed {comment_count} comments.")
        
        # Generate visualizations
        results = {
            "comment_count": comment_count,
            "sentiments": sentiments,
            "wordcloud_text": all_comments,
            "video_metadata": video_metadata.__dict__ if video_metadata else {}
        }
        
        return results, output_filename
        
    except Exception as e:
        logger.error(f"Error analyzing comments: {e}")
        return {"error": str(e)}, output_filename

def display_results(results: Dict[str, Any], output_filename: str) -> None:
    """
    Display analysis results including video information and visualizations.
    
    Args:
        results: Analysis results from analyze_comments
        output_filename: Path to the CSV file containing comment data
    """
    if "error" in results:
        logger.error(f"Analysis error: {results['error']}")
        return
        
    # Display video information
    video_metadata = results.get("video_metadata", {})
    if video_metadata:
        print("\n" + "="*40)
        print("VIDEO INFORMATION:")
        print("="*40)
        if "title" in video_metadata and video_metadata["title"]:
            print(f"Title: {video_metadata['title']}")
        if "channel" in video_metadata and video_metadata["channel"]:
            print(f"Channel: {video_metadata['channel']}")
        if "view_count" in video_metadata and video_metadata["view_count"]:
            print(f"Views: {video_metadata['view_count']:,}")
        if "like_count" in video_metadata and video_metadata["like_count"]:
            print(f"Likes: {video_metadata['like_count']:,}")
        
        # Display embed information
        if "video_id" in video_metadata:
            print("\nEmbed URL:")
            print(f"https://www.youtube.com/embed/{video_metadata['video_id']}")
            
        print("="*40 + "\n")
    
    # Display comment information
    print(f"Analyzed {results['comment_count']} comments")
    print(f"Results saved to: {output_filename}")
    
    # Show the sentiment analysis
    if "sentiments" in results and results["sentiments"]:
        plt.figure(figsize=(10, 6))
        plt.hist(results["sentiments"], bins=10, edgecolor='black')
        plt.title('Comment Sentiment Distribution')
        plt.xlabel('Sentiment Score (-1 to +1)')
        plt.ylabel('Number of Comments')
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()
        plt.show()
        
    # Generate and show the wordcloud
    if "wordcloud_text" in results and results["wordcloud_text"]:
        # Generate the wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=200,
            contour_width=3,
            contour_color='steelblue'
        ).generate(results["wordcloud_text"])
        
        # Show the wordcloud
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('Word Cloud of Comment Text')
        plt.tight_layout()
        plt.show()

def main() -> None:
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <youtube_url> [comment_count]")
        return
    
    # Get the URL from command line
    url = sys.argv[1]
    
    # Get optional comment count parameter
    comments_len = 100
    if len(sys.argv) > 2:
        try:
            comments_len = int(sys.argv[2])
        except ValueError:
            logger.warning(f"Invalid comment count: {sys.argv[2]}. Using default (100).")
    
    # Analyze comments
    results, csv_filename = analyze_comments(url, comments_len)
    
    # Display results
    display_results(results, csv_filename)

if __name__ == "__main__":
    main()


