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
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from youtube_comment_downloader import *
from itertools import islice
from google.cloud import language_v1



def getComments(comments_len=100):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(sys.argv[1], sort_by=SORT_BY_POPULAR)

    # Get the first 100 comments of the video

    return islice(comments, comments_len)

"""
getWordcloud:

Given a list of comments, it will return a wordcloud of the comments, and a csv file with the data.
"""

def getWordcloud(comments):
    # Create the csv file
    csv_file = open("comments.csv", "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["user", "comment", "sentiment", "keywords"])

    # Create the wordcloud
    all_comments = ""

    # Create the sentiment analysis
    sentiments = []
    for comment in comments:
        # Get the sentiment of the comment
        sentiment = getSentiment(comment)

        # Get the keywords of the comment
        keywords = getKeywords(comment)

        # Write the data in the csv file
        csv_writer.writerow([comment.author, comment.text, sentiment, keywords])

        # Add the comment to the wordcloud
        all_comments += comment.text + " "

        # Add the sentiment to the sentiments list
        sentiments.append(sentiment)

    # Write the data in the csv file
    csv_file.close()

    # Generate the wordcloud
    wordcloud = WordCloud().generate(all_comments)

    # Show the wordcloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Show the sentiment analysis
    plt.hist(sentiments, bins=5)
    plt.show()

"""
getSentiment:

Given a comment, it will return the sentiment of the comment.
"""

def getSentiment(comment, client=language_v1.LanguageServiceClient()):

    # The text to analyze
    text = comment.text
    document = language_v1.Document(
        content=text,
        type=language_v1.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment

    return sentiment.score

"""
getKeywords:

Given a comment, it will return the keywords of the comment.
"""

def getKeywords(comment, client = language_v1.LanguageServiceClient()):
        # The text to analyze
    text = comment.text
    document = language_v1.Document(
        content=text,
        type=language_v1.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    keywords = client.analyze_syntax(document=document).tokens
    return [keyword.lemma for keyword in keywords if keyword.part_of_speech.tag == 1]

if __name__ == "__main__":
    # Get the comments
    comments = getComments()
    # Get the wordcloud
    getWordcloud(comments)


