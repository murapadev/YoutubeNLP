#!/usr/bin/env python3
"""
Visualization View: Components for displaying analysis visualizations.
"""
import logging
from typing import Dict, List, Any, Optional
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import json


class VisualizationView:
    """Handles the visualization components of the application."""
    
    def __init__(self):
        """Initialize the visualization view."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def display_video_info(self, video_metadata: Dict[str, Any]) -> None:
        """
        Display video information and embed player.
        
        Args:
            video_metadata: Dictionary containing video metadata
        """
        if not video_metadata:
            st.warning("No video metadata available")
            return

        # Create two columns: video info and embed
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Video Information")
            # Display video title
            if 'title' in video_metadata and video_metadata['title']:
                st.markdown(f"#### {video_metadata['title']}")
            
            # Display statistics
            if 'channel' in video_metadata and video_metadata['channel']:
                st.write(f"**Channel:** {video_metadata['channel']}")
            
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                if 'view_count' in video_metadata and video_metadata['view_count']:
                    st.metric("Views", f"{video_metadata['view_count']:,}")
            with metrics_col2:
                if 'like_count' in video_metadata and video_metadata['like_count']:
                    st.metric("Likes", f"{video_metadata['like_count']:,}")
            
            if 'publish_date' in video_metadata and video_metadata['publish_date']:
                st.write(f"**Published:** {video_metadata['publish_date']}")

        with col2:
            st.subheader("Video Preview")
            video_id = video_metadata.get('video_id')
            if video_id:
                embed_url = f"https://www.youtube.com/embed/{video_id}"
                st.markdown(
                    f'<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">'
                    f'<iframe style="position:absolute;top:0;left:0;width:100%;height:100%;" '
                    f'src="{embed_url}" frameborder="0" '
                    f'allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
                    f'allowfullscreen></iframe></div>',
                    unsafe_allow_html=True
                )

        st.divider()
    
    def display_dataframe(self, df: pd.DataFrame) -> None:
        """
        Display the dataframe of analyzed comments.
        
        Args:
            df: DataFrame containing comment data
        """
        if df.empty:
            st.warning("No data to display.")
            return
            
        st.subheader("Comment Data")
        st.dataframe(
            df[['author', 'text', 'sentiment_score', 'sentiment_magnitude', 'likes']],
            use_container_width=True
        )
        
    def display_download_button(self, csv_data: str) -> None:
        """
        Display a button to download the analyzed data as CSV.
        
        Args:
            csv_data: CSV string representation of the data
        """
        st.download_button(
            label="Download data as CSV",
            data=csv_data,
            file_name="youtube_comments.csv",
            mime="text/csv"
        )
    
    def display_sentiment_distribution(self, df: pd.DataFrame) -> None:
        """
        Display sentiment distribution visualization.
        
        Args:
            df: DataFrame containing comment data
        """
        if df.empty:
            return
            
        st.subheader("Sentiment Distribution")
        
        # Create sentiment categories
        df['sentiment_category'] = pd.cut(
            df['sentiment_score'], 
            bins=[-1.1, -0.6, -0.2, 0.2, 0.6, 1.1],
            labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        )
        
        # Plot sentiment distribution
        fig = px.histogram(
            df, 
            x='sentiment_category',
            color='sentiment_category',
            labels={'sentiment_category': 'Sentiment'},
            title='Comment Sentiment Distribution',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Improve layout
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Comments",
            legend_title="Sentiment",
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        sentiment_counts = df['sentiment_category'].value_counts().to_dict()
        total = sum(sentiment_counts.values())
        
        st.markdown("### Sentiment Summary")
        col1, col2, col3 = st.columns(3)
        
        positive_percent = sum(sentiment_counts.get(cat, 0) for cat in ['Positive', 'Very Positive']) / total * 100
        neutral_percent = sentiment_counts.get('Neutral', 0) / total * 100
        negative_percent = sum(sentiment_counts.get(cat, 0) for cat in ['Negative', 'Very Negative']) / total * 100
        
        col1.metric("Positive Comments", f"{positive_percent:.1f}%")
        col2.metric("Neutral Comments", f"{neutral_percent:.1f}%")
        col3.metric("Negative Comments", f"{negative_percent:.1f}%")
    
    def display_comment_length_vs_sentiment(self, df: pd.DataFrame) -> None:
        """
        Display comment length vs sentiment visualization.
        
        Args:
            df: DataFrame containing comment data
        """
        if df.empty:
            return
            
        st.subheader("Comment Length vs. Sentiment")
        
        fig = px.scatter(
            df, 
            x='text_length', 
            y='sentiment_score',
            color='sentiment_magnitude',
            hover_data=['author', 'text'],
            labels={
                'text_length': 'Comment Length (characters)',
                'sentiment_score': 'Sentiment Score',
                'sentiment_magnitude': 'Sentiment Intensity'
            },
            title='Comment Length vs. Sentiment',
            color_continuous_scale='Viridis'
        )
        
        # Add trend line
        fig.update_layout(
            xaxis_title="Comment Length (characters)",
            yaxis_title="Sentiment Score",
            font=dict(size=12)
        )
        
        # Add regression line
        if len(df) > 1:  # Need at least 2 points for regression
            x = df['text_length']
            y = df['sentiment_score']
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=sorted(x),
                    y=p(sorted(x)),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', dash='dash')
                )
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add correlation statistics
        if len(df) > 1:
            correlation = df['text_length'].corr(df['sentiment_score'])
            st.info(f"Correlation between comment length and sentiment: {correlation:.3f}")
            
            if abs(correlation) < 0.2:
                st.write("There's little to no correlation between comment length and sentiment.")
            elif abs(correlation) < 0.4:
                st.write("There's a weak correlation between comment length and sentiment.")
            elif abs(correlation) < 0.6:
                st.write("There's a moderate correlation between comment length and sentiment.")
            else:
                st.write("There's a strong correlation between comment length and sentiment.")
    
    def display_wordcloud(self, df: pd.DataFrame, stop_words: set) -> None:
        """
        Display word cloud visualization.
        
        Args:
            df: DataFrame containing comment data
            stop_words: Set of stop words to exclude
        """
        if df.empty:
            return
            
        st.subheader("Comment Word Cloud")
        
        # Filter options for word cloud with unique key
        sentiment_filter = st.radio(
            "Filter by sentiment:",
            ('All', 'Positive', 'Neutral', 'Negative'),
            horizontal=True,
            key="wordcloud_sentiment_filter"  # Added unique key
        )
        
        # Filter comments based on selection
        if sentiment_filter == 'Positive':
            filtered_df = df[df['sentiment_score'] > 0.2]
        elif sentiment_filter == 'Negative':
            filtered_df = df[df['sentiment_score'] < -0.2]
        elif sentiment_filter == 'Neutral':
            filtered_df = df[(df['sentiment_score'] >= -0.2) & (df['sentiment_score'] <= 0.2)]
        else:
            filtered_df = df
        
        if filtered_df.empty:
            st.warning(f"No {sentiment_filter.lower()} comments found.")
            return
            
        # Generate word cloud
        all_text = ' '.join(filtered_df['text'].tolist())
        
        # Color function based on sentiment
        if sentiment_filter == 'Positive':
            colormap = 'Greens'
        elif sentiment_filter == 'Negative':
            colormap = 'Reds'
        elif sentiment_filter == 'Neutral':
            colormap = 'Blues'
        else:
            colormap = 'viridis'
        
        wordcloud = WordCloud(
            stopwords=stop_words,
            max_words=100,
            max_font_size=40,
            scale=3,
            random_state=42,
            background_color='white',
            colormap=colormap,
            width=800,
            height=400
        ).generate(all_text)
        
        # Display word cloud
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Add explanation
        with st.expander("📖 About the Word Cloud"):
            st.write(
                "The word cloud shows the most frequent words in the comments, "
                "with more frequent words appearing larger. Stop words like 'the', 'a', etc. "
                "are excluded. The color scheme reflects the selected sentiment filter."
            )
    
    def display_keyword_analysis(self, df: pd.DataFrame) -> None:
        """
        Display keyword analysis visualization.
        
        Args:
            df: DataFrame containing comment data
        """
        if df.empty:
            return
            
        st.subheader("Top Keywords")
        
        # Extract all keywords
        all_keywords = []
        for keywords in df['keywords']:
            if keywords:
                # Handle both string and list formats for keywords
                if isinstance(keywords, str):
                    # Handle comma-separated string format
                    all_keywords.extend([k.strip() for k in keywords.split(',')])
                elif isinstance(keywords, list):
                    # Handle list format directly
                    all_keywords.extend([k.strip() if isinstance(k, str) else str(k) for k in keywords])
        
        # Count keywords
        keyword_counts = Counter(all_keywords)
        top_keywords = pd.DataFrame(keyword_counts.most_common(20), columns=['Keyword', 'Count'])
        
        if top_keywords.empty:
            st.warning("No keywords extracted from comments.")
            return
            
        fig = px.bar(
            top_keywords,
            x='Count',
            y='Keyword',
            orientation='h',
            title='Top 20 Keywords in Comments',
            color='Count',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_title="Occurrence Count",
            yaxis_title="Keyword",
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add keyword frequency table with download option
        with st.expander("View Complete Keyword Data"):
            st.dataframe(top_keywords)
            
            # Allow downloading the keyword data
            csv = top_keywords.to_csv(index=False)
            st.download_button(
                label="Download keyword data as CSV",
                data=csv,
                file_name="youtube_comment_keywords.csv",
                mime="text/csv"
            )
    
    def display_sentiment_over_time(self, df: pd.DataFrame) -> None:
        """
        Display sentiment over time visualization (if time data is available).
        
        Args:
            df: DataFrame containing comment data
        """
        if df.empty or 'time' not in df.columns or not df['time'].notna().any():
            return
            
        try:
            # Convert time strings to datetime objects
            df['time_parsed'] = pd.to_datetime(df['time'], errors='coerce')
            
            if not df['time_parsed'].notna().any():
                return
                
            st.subheader("Sentiment Over Time")
            
            # Create time-based visualization
            df_time = df.dropna(subset=['time_parsed']).sort_values('time_parsed')
            
            # Option for smoothing
            window_size = st.slider(
                "Smoothing window (number of comments)",
                min_value=1,
                max_value=max(20, len(df_time) // 5),
                value=min(10, max(1, len(df_time) // 10))
            )
            
            # Calculate rolling average for sentiment
            df_time['rolling_sentiment'] = df_time['sentiment_score'].rolling(
                window=window_size, 
                min_periods=1
            ).mean()
            
            # Create time series chart
            fig = px.line(
                df_time,
                x='time_parsed',
                y=['sentiment_score', 'rolling_sentiment'],
                title='Comment Sentiment Over Time',
                labels={
                    'time_parsed': 'Time',
                    'value': 'Sentiment Score',
                    'variable': 'Metric'
                },
                color_discrete_map={
                    'sentiment_score': 'lightgrey',
                    'rolling_sentiment': 'darkblue'
                }
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Sentiment Score",
                legend_title="Metric",
                font=dict(size=12),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add trend analysis
            with st.expander("Trend Analysis"):
                # Identify the overall trend
                first_half = df_time.iloc[:len(df_time)//2]['sentiment_score'].mean()
                second_half = df_time.iloc[len(df_time)//2:]['sentiment_score'].mean()
                
                trend = second_half - first_half
                
                if abs(trend) < 0.05:
                    st.info("The sentiment has remained relatively stable over time.")
                elif trend > 0:
                    st.success(f"The sentiment has improved over time (change of +{trend:.2f}).")
                else:
                    st.error(f"The sentiment has declined over time (change of {trend:.2f}).")
                    
        except Exception as e:
            self.logger.error(f"Error displaying sentiment over time: {str(e)}")
    
    def display_sentiment_by_language(self, df: pd.DataFrame) -> None:
        """
        Display sentiment analysis broken down by language.
        
        Args:
            df: DataFrame containing comment data with language information
        """
        if df.empty or 'language' not in df.columns:
            return
            
        st.subheader("Sentiment by Language")
        
        # Get the top languages (up to 5)
        top_languages = df['language'].value_counts().head(5).index.tolist()
        
        # Filter for only comments in the top languages
        df_top_langs = df[df['language'].isin(top_languages)]
        
        if df_top_langs.empty:
            st.info("Not enough data to analyze sentiment by language.")
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
        }
        
        # Replace language codes with friendly names
        df_top_langs['language_name'] = df_top_langs['language'].map(
            lambda x: language_names.get(x, x.upper())
        )
        
        # Calculate average sentiment by language
        sentiment_by_lang = df_top_langs.groupby('language_name')['sentiment_score'].agg(
            ['mean', 'count', 'std']
        ).reset_index()
        sentiment_by_lang['mean'] = sentiment_by_lang['mean'].round(3)
        sentiment_by_lang['std'] = sentiment_by_lang['std'].round(3)
        sentiment_by_lang = sentiment_by_lang.rename(columns={
            'language_name': 'Language',
            'mean': 'Avg. Sentiment',
            'count': 'Comments',
            'std': 'Std. Dev.'
        })
        
        # Create horizontal bar chart for average sentiment by language
        fig = px.bar(
            sentiment_by_lang,
            y='Language',
            x='Avg. Sentiment',
            error_x='Std. Dev.',
            text='Avg. Sentiment',
            color='Avg. Sentiment',
            color_continuous_scale='RdBu',
            title='Average Sentiment Score by Language',
            hover_data=['Comments'],
            orientation='h'
        )
        
        fig.update_layout(
            xaxis_title="Average Sentiment (-1 = Negative, +1 = Positive)",
            yaxis_title="Language",
            coloraxis_colorbar=dict(
                title="Sentiment",
            ),
            height=400
        )
        
        # Center sentiment scale around 0
        max_abs = max(abs(fig.data[0].x.min()), abs(fig.data[0].x.max()), 1)
        fig.update_xaxes(range=[-max_abs, max_abs])
        
        # Add zero reference line
        fig.add_shape(
            type='line',
            x0=0, x1=0,
            y0=-0.5, y1=len(sentiment_by_lang)-0.5,
            line=dict(color='black', width=1, dash='dash')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add detailed table with sentiment breakdown
        with st.expander("View detailed sentiment by language"):
            st.dataframe(sentiment_by_lang, use_container_width=True)
            
            # Add explanation
            st.write("""
            **What this means:** 
            - A positive score indicates overall positive sentiment in comments for that language
            - A negative score indicates overall negative sentiment
            - The standard deviation shows how much variation exists in the sentiment
            - Languages with few comments may not provide statistically significant results
            """)


    def display_entity_analysis(self, df: pd.DataFrame) -> None:
        """
        Display named entity analysis visualization.
        
        Args:
            df: DataFrame containing comment data with entity information
        """
        if df.empty or 'entities' not in df.columns:
            return
            
        st.subheader("Named Entity Analysis")
        
        try:
            # Extract entities from JSON strings or lists
            entity_data = []
            
            for entities in df['entities'].dropna():
                if not entities:
                    continue
                
                # Handle both string and list inputs
                if isinstance(entities, str):
                    try:
                        entities = json.loads(entities)
                    except json.JSONDecodeError:
                        continue
                
                # Now entities should be a list
                if isinstance(entities, list):
                    for entity in entities:
                        if isinstance(entity, dict):
                            entity_data.append({
                                'name': str(entity.get('name', '')),
                                'type': str(entity.get('type', 'UNKNOWN')),
                                'salience': float(entity.get('salience', 0.0))
                            })
                    
            if not entity_data:
                st.info("No entities were detected in the comments.")
                return
                
            entity_df = pd.DataFrame(entity_data)
            
            # Count entity types
            entity_type_counts = entity_df['type'].value_counts().reset_index()
            entity_type_counts.columns = ['Entity Type', 'Count']
            
            # Display entity type distribution
            st.subheader("Entity Type Distribution")
            fig = px.pie(
                entity_type_counts,
                values='Count',
                names='Entity Type',
                title='Types of Entities Mentioned in Comments'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Find top entities by type
            st.subheader("Top Entities by Category")
            
            # Get top entity types
            top_entity_types = entity_type_counts['Entity Type'].head(5).tolist()
            
            # Create tabs for each top entity type
            tabs = st.tabs([f"{entity_type}" for entity_type in top_entity_types])
            
            # Fill each tab with top entities of that type
            for i, entity_type in enumerate(top_entity_types):
                with tabs[i]:
                    # Filter for this type
                    type_entities = entity_df[entity_df['type'] == entity_type]
                    
                    # Get top entities by count and average salience
                    # Fix: Use size() for counting instead of count() on the same column
                    top_entities = (type_entities.groupby('name')
                                   .agg(
                                        Mentions=('name', 'size'),  # Count occurrences without name collision
                                        Avg_Salience=('salience', 'mean')
                                    )
                                   .reset_index()
                                   .rename(columns={'name': 'Entity'})  # Rename for clarity
                                   .sort_values('Mentions', ascending=False)
                                   .head(15))
                    
                    # Bar chart of top entities
                    if not top_entities.empty:
                        fig = px.bar(
                            top_entities,
                            x='Mentions',
                            y='Entity',
                            orientation='h',
                            title=f'Top {entity_type} Entities',
                            color='Avg_Salience',
                            color_continuous_scale='Viridis'
                        )
                        fig.update_layout(
                            xaxis_title="Number of Mentions",
                            yaxis_title=f"{entity_type}",
                            yaxis=dict(autorange="reversed")
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No {entity_type} entities found in comments.")
                        
        except Exception as e:
            st.error(f"Error analyzing entities: {str(e)}")
            self.logger.error(f"Entity analysis error: {str(e)}", exc_info=True)


    def display_advanced_sentiment_analysis(self, df: pd.DataFrame) -> None:
        """
        Display advanced sentiment analysis including cross-analysis with other factors.
        
        Args:
            df: DataFrame containing comment data
        """
        if df.empty:
            return
        
        st.subheader("Advanced Sentiment Analysis")
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution by likes/popularity
            if 'likes' in df.columns and df['likes'].sum() > 0:
                # Group comments by sentiment category and sum likes
                df['sentiment_category'] = pd.cut(
                    df['sentiment_score'], 
                    bins=[-1.1, -0.6, -0.2, 0.2, 0.6, 1.1],
                    labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
                )
                
                likes_by_sentiment = df.groupby('sentiment_category')['likes'].agg(['sum', 'count']).reset_index()
                likes_by_sentiment['average_likes'] = likes_by_sentiment['sum'] / likes_by_sentiment['count']
                
                fig = px.bar(
                    likes_by_sentiment, 
                    x='sentiment_category', 
                    y='average_likes',
                    color='sentiment_category',
                    labels={'sentiment_category': 'Sentiment', 'average_likes': 'Average Likes'},
                    title='Average Likes by Sentiment Category'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add insight
                most_liked = likes_by_sentiment.loc[likes_by_sentiment['average_likes'].idxmax()]
                st.markdown(f"**Insight:** {most_liked['sentiment_category']} comments receive the most likes on average " + 
                           f"({most_liked['average_likes']:.1f} likes per comment).")
        
        with col2:
            # Sentiment distribution by comment length
            df['length_category'] = pd.cut(
                df['text_length'],
                bins=[0, 50, 100, 200, 500, float('inf')],
                labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
            )
            
            sentiment_by_length = df.groupby('length_category')['sentiment_score'].mean().reset_index()
            
            fig = px.bar(
                sentiment_by_length,
                x='length_category',
                y='sentiment_score',
                color='sentiment_score',
                color_continuous_scale='RdBu',
                labels={'length_category': 'Comment Length', 'sentiment_score': 'Average Sentiment'},
                title='Sentiment by Comment Length'
            )
            
            # Center the color scale
            fig.update_coloraxes(cmin=-1, cmax=1, colorbar_title='Sentiment')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insight
            correlation = df['text_length'].corr(df['sentiment_score'])
            st.markdown(f"**Insight:** The correlation between comment length and sentiment is {correlation:.3f}.")
            
        # Sentiment magnitude analysis (intensity regardless of positive/negative)
        st.subheader("Sentiment Intensity Analysis")
        
        # Create columns for sentiment intensity analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram of sentiment magnitude
            fig = px.histogram(
                df,
                x='sentiment_magnitude',
                nbins=20,
                color_discrete_sequence=['darkblue'],
                title='Distribution of Sentiment Intensity',
                labels={'sentiment_magnitude': 'Sentiment Intensity'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of sentiment magnitude
            with st.expander("About Sentiment Intensity"):
                st.write("""
                **Sentiment Intensity** (magnitude) measures how strongly the sentiment is expressed, 
                regardless of whether it's positive or negative. Higher values indicate stronger emotional content.
                
                - Low intensity (0-0.3): Neutral or mildly expressed opinions
                - Medium intensity (0.3-0.6): Clearly expressed opinions
                - High intensity (0.6+): Strongly expressed emotional content
                """)
        
        with col2:
            # Scatter plot of sentiment polarity vs magnitude
            fig = px.scatter(
                df,
                x='sentiment_score',
                y='sentiment_magnitude',
                color='sentiment_magnitude',
                color_continuous_scale='Viridis',
                hover_data=['text'],
                labels={
                    'sentiment_score': 'Sentiment Score (-1 to +1)',
                    'sentiment_magnitude': 'Sentiment Intensity'
                },
                title='Sentiment Polarity vs. Intensity'
            )
            
            # Add quadrant lines
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                         annotation_text="High Intensity", annotation_position="top right")
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            
            # Add annotations for quadrants
            fig.add_annotation(x=-0.5, y=0.75, text="Strong Negative", showarrow=False, font_size=10)
            fig.add_annotation(x=0.5, y=0.75, text="Strong Positive", showarrow=False, font_size=10)
            fig.add_annotation(x=-0.5, y=0.25, text="Mild Negative", showarrow=False, font_size=10)
            fig.add_annotation(x=0.5, y=0.25, text="Mild Positive", showarrow=False, font_size=10)
            
            st.plotly_chart(fig, use_container_width=True)


    def display_all_visualizations(self, df: pd.DataFrame, stop_words: set) -> None:
        """
        Display all visualizations in appropriate layout.
        
        Args:
            df: DataFrame containing comment data
            stop_words: Set of stop words to exclude from text analysis
        """
        if df is None or df.empty:
            st.warning("No data to visualize.")
            return
            
        # Store the complete dataframe in session state for filtering
        if 'complete_analysis_df' not in st.session_state or st.session_state.complete_analysis_df is None:
            st.session_state.complete_analysis_df = df.copy()
            st.session_state.complete_stop_words = stop_words

        # NOTE: Video info is now displayed by MainView, so we removed it from here
        # to avoid showing it twice
        
        # Create tabs for different types of visualizations
        tabs = st.tabs([
            "📊 Sentiment Analysis", 
            "🧠 Advanced Insights",
            "☁️ Word Cloud & Keywords", 
            "🌐 Language Analysis",
            "👤 Named Entities",
            "📈 Temporal Analysis",
            "🔍 Data Explorer"
        ])
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                self.display_sentiment_distribution(df)
            with col2:
                self.display_comment_length_vs_sentiment(df)
        
        with tabs[1]:
            self.display_advanced_sentiment_analysis(df)
        
        with tabs[2]:
            self.display_wordcloud(df, stop_words)
            self.display_keyword_analysis(df)
        
        with tabs[3]:
            if 'language' in df.columns:
                self.display_sentiment_by_language(df)
                
                # Add language distribution statistics
                st.subheader("Language Distribution")
                lang_dist = df['language'].value_counts()
                total_comments = len(df)
                
                # Create language distribution chart
                fig = px.pie(
                    values=lang_dist.values,
                    names=lang_dist.index,
                    title='Comment Language Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add language statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Languages", len(lang_dist))
                with col2:
                    primary_lang = lang_dist.index[0]
                    primary_pct = (lang_dist[primary_lang] / total_comments) * 100
                    st.metric("Primary Language", f"{primary_lang} ({primary_pct:.1f}%)")
                
                # Show language details in an expander
                with st.expander("View Language Details"):
                    st.dataframe(pd.DataFrame({
                        'Language': lang_dist.index,
                        'Comments': lang_dist.values,
                        'Percentage': (lang_dist.values / total_comments * 100).round(1)
                    }))
            else:
                st.info("Language information is not available in the analyzed data.")
        
        with tabs[4]:
            self.display_entity_analysis(df)
        
        with tabs[5]:
            self.display_temporal_analysis(df)
            
        with tabs[6]:
            self.display_dataframe(df)
            if len(df) > 0:
                csv = df.to_csv(index=False)
                self.display_download_button(csv)
    
    def get_stopwords_for_language(self, lang_code: str) -> set:
        """
        Get a set of stopwords for the given language.
        
        Args:
            lang_code: ISO 639-1 language code (e.g., 'en', 'es')
            
        Returns:
            Set of stopwords for the specified language
        """
        # Language name mapping for NLTK languages
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
        
        # Start with default WordCloud stopwords
        stopword_set = set(STOPWORDS)
        
        # Add NLTK stopwords if available for this language
        nltk_lang = lang_map.get(lang_code)
        if nltk_lang:
            try:
                from nltk.corpus import stopwords
                stopword_set.update(stopwords.words(nltk_lang))
                self.logger.info(f"Added {nltk_lang} stopwords")
            except Exception as e:
                self.logger.warning(f"Could not load stopwords for {nltk_lang}: {e}")
        
        # Add common social media and YouTube-specific stopwords
        youtube_stopwords = {
            "video", "videos", "youtube", "channel", "like", "subscribe", 
            "comment", "watch", "watching"
        }
        stopword_set.update(youtube_stopwords)
        
        return stopword_set

    def _format_datetime(self, date_str: str) -> str:
        """Format datetime string for display.
        
        Args:
            date_str: ISO format datetime string
            
        Returns:
            Formatted datetime string for display
        """
        try:
            from dateutil import parser
            dt = parser.parse(date_str)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return date_str

    def display_temporal_analysis(self, df: pd.DataFrame) -> None:
        """
        Display temporal analysis of comments.
        
        Args:
            df: DataFrame containing comment data
        """
        if df.empty:
            return
            
        st.subheader("Temporal Analysis")
        
        try:
            # Convert time to datetime using the comment model's parsing
            if 'time' in df.columns:
                # Create a fresh copy to avoid chained indexing
                df_temp = df.copy()
                
                # Parse dates using dateparser from CommentData
                from models.comment_model import CommentData
                
                def parse_date(time_str):
                    if pd.isna(time_str):
                        return None
                    comment = CommentData(text="", author="", time=str(time_str))
                    return comment.parse_relative_time()
                
                # Safe assignment using loc
                df_temp.loc[:, 'time_parsed'] = df_temp['time'].apply(parse_date)
                df_temp = df_temp.dropna(subset=['time_parsed'])
                
                if len(df_temp) > 0:
                    # Sort by time - create new DataFrame instead of modifying in place
                    df_temp = df_temp.sort_values('time_parsed')
                    
                    # Add date column safely
                    df_temp.loc[:, 'date'] = df_temp['time_parsed'].dt.date
                    
                    # Create time-based visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Comments over time
                        st.subheader("Comment Activity Over Time")
                        daily_counts = df_temp.groupby('date').size().reset_index(name='count')
                        
                        fig = px.line(
                            daily_counts,
                            x='date',
                            y='count',
                            title='Comments per Day',
                            labels={'date': 'Date', 'count': 'Number of Comments'}
                        )
                        
                        # Improve x-axis date formatting
                        fig.update_xaxes(tickformat='%Y-%m-%d')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Sentiment over time with smoothing
                        st.subheader("Sentiment Trends")
                        window_size = st.slider(
                            "Smoothing window (number of comments)",
                            min_value=1,
                            max_value=max(20, len(df_temp) // 5),
                            value=min(10, max(1, len(df_temp) // 10))
                        )
                        
                        # Calculate rolling average safely
                        df_temp.loc[:, 'rolling_sentiment'] = df_temp['sentiment_score'].rolling(
                            window=window_size,
                            min_periods=1
                        ).mean()
                        
                        fig = px.line(
                            df_temp,
                            x='time_parsed',
                            y=['sentiment_score', 'rolling_sentiment'],
                            title='Sentiment Evolution',
                            labels={
                                'time_parsed': 'Time',
                                'value': 'Sentiment Score',
                                'variable': 'Metric'
                            },
                            color_discrete_map={
                                'sentiment_score': 'lightgrey',
                                'rolling_sentiment': 'darkblue'
                            }
                        )
                        
                        fig.update_layout(
                            xaxis_title="Time",
                            yaxis_title="Sentiment Score",
                            legend_title="Metric",
                            showlegend=True,
                            xaxis=dict(tickformat='%Y-%m-%d\n%H:%M')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Time-of-Day Analysis
                    st.subheader("Time-of-Day Analysis")
                    
                    # Add hour column safely
                    df_temp.loc[:, 'hour'] = df_temp['time_parsed'].dt.hour
                    
                    # Aggregate hourly stats
                    hourly_stats = (df_temp.groupby('hour')
                                  .agg({
                                      'sentiment_score': ['mean', 'count']
                                  })
                                  .reset_index())
                    
                    # Flatten multi-level columns
                    hourly_stats.columns = ['hour', 'avg_sentiment', 'comment_count']
                    
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # Comments by hour
                        fig = px.bar(
                            hourly_stats,
                            x='hour',
                            y='comment_count',
                            title='Comments by Hour of Day (UTC)',
                            labels={
                                'hour': 'Hour (24h)',
                                'comment_count': 'Number of Comments'
                            }
                        )
                        
                        # Improve hour formatting
                        fig.update_xaxes(
                            tickmode='array',
                            ticktext=[f'{h:02d}:00' for h in range(24)],
                            tickvals=list(range(24))
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col4:
                        # Average sentiment by hour
                        fig = px.line(
                            hourly_stats,
                            x='hour',
                            y='avg_sentiment',
                            title='Average Sentiment by Hour of Day (UTC)',
                            labels={
                                'hour': 'Hour (24h)',
                                'avg_sentiment': 'Average Sentiment'
                            }
                        )
                        
                        # Improve hour formatting and add zero reference line
                        fig.update_xaxes(
                            tickmode='array',
                            ticktext=[f'{h:02d}:00' for h in range(24)],
                            tickvals=list(range(24))
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate trends
                    with st.expander("Trend Analysis"):
                        first_date = df_temp['time_parsed'].min()
                        last_date = df_temp['time_parsed'].max()
                        total_days = (last_date - first_date).days
                        
                        st.write(f"Analysis period: {total_days} days")
                        st.write(f"From: {first_date.strftime('%Y-%m-%d %H:%M')} to {last_date.strftime('%Y-%m-%d %H:%M')}")
                        
                        # Calculate daily change in sentiment
                        daily_sentiment = df_temp.groupby('date')['sentiment_score'].mean()
                        sentiment_trend = daily_sentiment.iloc[-1] - daily_sentiment.iloc[0]
                        
                        if abs(sentiment_trend) < 0.1:
                            st.info("📊 Sentiment remained relatively stable over the period")
                        elif sentiment_trend > 0:
                            st.success(f"📈 Overall sentiment improved by {sentiment_trend:.2f} points")
                        else:
                            st.error(f"📉 Overall sentiment declined by {abs(sentiment_trend):.2f} points")
                        
                        # Most active periods
                        peak_hour = hourly_stats.loc[hourly_stats['comment_count'].idxmax()]
                        st.write(f"Most active hour: {int(peak_hour['hour']):02d}:00 UTC ({int(peak_hour['comment_count'])} comments)")
                        
                        # Calculate activity distribution
                        active_hours = hourly_stats[hourly_stats['comment_count'] > 0].shape[0]
                        st.write(f"Activity spread: Comments across {active_hours} different hours of the day")
                        
                        # Calculate and show peak periods
                        peak_threshold = hourly_stats['comment_count'].mean() + hourly_stats['comment_count'].std()
                        peak_hours = hourly_stats[hourly_stats['comment_count'] > peak_threshold]
                        if not peak_hours.empty:
                            peak_times = [f"{h:02d}:00" for h in peak_hours['hour']]
                            st.write(f"Peak activity times (UTC): {', '.join(peak_times)}")
                else:
                    st.warning("No valid timestamps found in the comments data.")
            else:
                st.warning("Timestamp information is not available in the comments data.")
                
        except Exception as e:
            st.error("Error analyzing temporal data")
            self.logger.error(f"Error in temporal analysis: {str(e)}", exc_info=True)