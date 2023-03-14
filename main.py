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



""" for comment in comments:
    print(comment)
 """

def getComments(comments_len=100):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(sys.argv[1], sort_by=SORT_BY_POPULAR)

    # Get the first 100 comments of the video

    return islice(comments, comments_len)
