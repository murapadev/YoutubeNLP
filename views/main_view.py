#!/usr/bin/env python3
"""
Main View: Primary UI components and layout for the application.
"""
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple

import streamlit as st
from youtube_comment_downloader import SORT_BY_POPULAR, SORT_BY_RECENT

from views.visualization_view import VisualizationView
from controllers.analysis_controller import AnalysisController
from models.analyzer_model import AnalyzerFactory


class MainView:
    """Main view class handling the primary UI components and layout."""
    
    def __init__(self, configure_page: bool = True):
        """
        Initialize the main view with required components.
        
        Args:
            configure_page: Whether to configure the Streamlit page. Set to False
                           if page config is handled elsewhere (recommended).
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.visualization = VisualizationView()
        self.controller = AnalysisController()
        
        # Configure Streamlit page if requested
        # Note: This should typically be False, with page config in app.py
        if configure_page:
            self._configure_page()
        else:
            self._add_custom_styles()
            
        # Initialize session state variables
        self._initialize_session_state()
    
    def _add_custom_styles(self) -> None:
        """Add custom CSS styles to the Streamlit application."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .subheader {
            font-size: 1.5rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .info-box {
            background-color: #e1f5fe;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _configure_page(self) -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="YoutubeNLP - Comment Analysis",
            page_icon="🎥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        self._add_custom_styles()
    
    def _initialize_session_state(self) -> None:
        """
        Initialize session state variables for tracking app state.
        
        This method ensures that all required session state variables exist
        before they are accessed by the application code.
        """
        # Define the list of required session state variables with their default values
        required_state = {
            "initialized": True,
            "analysis_complete": False,
            "analysis_in_progress": False,
            "analysis_params_hash": "",
            "analysis_triggered": False,
            "last_url": "",
            "last_max_comments": 0,
            "last_sort_by": "",
            "selected_language": "All Languages",
        }
        
        # Initialize each variable if it doesn't exist in session state
        for key, default_value in required_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
        # Initialize data storage variables
        if "complete_analysis_df" not in st.session_state:
            st.session_state["complete_analysis_df"] = None
        
        if "video_metadata" not in st.session_state:
            st.session_state["video_metadata"] = None
    
    def render_header(self) -> None:
        """Render the application header with title and description."""
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown('<div class="main-header">📊 YouTube Comment Analyzer</div>', 
                      unsafe_allow_html=True)
            st.markdown(
                '<div class="subheader">Analyze YouTube comments using Natural Language Processing</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=100)
    
    def render_input_form(self) -> Tuple[str, int, str]:
        """
        Render the input form for entering YouTube URL and analysis parameters.
        
        Returns:
            Tuple containing (url, max_comments, sort_by)
        """
        with st.form(key="youtube_form"):
            url = st.text_input(
                "YouTube Video URL", 
                help="Enter the full URL of a YouTube video",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_comments = st.number_input(
                    "Maximum Comments to Analyze",
                    min_value=10,
                    max_value=5000,
                    value=100,
                    step=10,
                    help="Higher values may increase processing time"
                )
            
            with col2:
                sort_options = {
                    SORT_BY_POPULAR: "Most Popular",
                    SORT_BY_RECENT: "Most Recent"
                }
                sort_by = st.selectbox(
                    "Sort Comments By",
                    options=list(sort_options.keys()),
                    format_func=lambda x: sort_options[x],
                    index=0
                )
            
            submit_button = st.form_submit_button(
                label="Analyze Comments", 
                use_container_width=True,
                type="primary"
            )
            
            if submit_button:
                if not url or not url.strip().startswith("http"):
                    st.error("Please enter a valid YouTube URL")
                    return "", max_comments, sort_by
                    
                st.session_state["analysis_triggered"] = True
                
            return url, max_comments, sort_by
    
    def _get_params_hash(self, url: str, max_comments: Optional[int], sort_by: str) -> str:
        """
        Generate a hash of the analysis parameters to detect changes.
        
        Args:
            url: YouTube video URL
            max_comments: Maximum number of comments to analyze
            sort_by: Comment sorting method
            
        Returns:
            String hash of parameters
        """
        params_str = f"{url}_{max_comments}_{sort_by}"
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def render_sidebar(self) -> None:
        """Render the sidebar with options and history."""
        st.sidebar.title("Options")
        
        # Advanced options in expander
        with st.sidebar.expander("Advanced Options", expanded=False):
            # Batch processing options
            st.checkbox(
                "Use batched analysis",
                value=True,
                key="use_batched_analysis",
                help="Process comments in batches for better performance"
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=5,
                max_value=100,
                value=10,
                step=5,
                help="Number of comments to process in each batch"
            )
            
            # Save preferences
            if st.button("Save Preferences"):
                self.controller.set_preference("batch_size", batch_size)
                self.controller.set_preference(
                    "use_batched_analysis", 
                    st.session_state.get("use_batched_analysis", True)
                )
                st.success("Preferences saved!")
        
        # Add separator
        st.sidebar.divider()
        
        # History section
        self._render_history_section()
        
        # About section
        st.sidebar.divider()
        st.sidebar.info(
            """
            **About YoutubeNLP**
            
            This tool analyzes YouTube comments using Natural Language Processing techniques.
            
            Features include:
            - Sentiment analysis
            - Keyword extraction
            - Named entity recognition
            - Language detection
            
            [GitHub Repository](https://github.com/murapa/YoutubeNLP)
            """
        )
    
    def _render_history_section(self) -> None:
        """Render analysis history section in sidebar."""
        st.sidebar.subheader("Analysis History")
        
        # Get history from controller
        history = self.controller.get_analysis_history(limit=5)
        
        if not history:
            st.sidebar.info("No history available")
            return
            
        # Display history items
        for item in history:
            col1, col2 = st.sidebar.columns([3, 1])
            
            title = item.get("title") or "Untitled Video"
            if len(title) > 30:
                title = title[:27] + "..."
                
            comment_count = item.get("comment_count", 0)
            timestamp = item.get("timestamp", "").split("T")[0]
            video_id = item.get("video_id", "")
            
            with col1:
                st.write(f"**{title}**")
                st.write(f"{comment_count} comments • {timestamp}")
            
            with col2:
                if st.button("Load", key=f"load_{video_id}"):
                    # Load cached analysis
                    url = item.get("url", f"https://youtube.com/watch?v={video_id}")
                    
                    # Update session state to trigger loading from cache
                    st.session_state["last_url"] = url
                    st.session_state["analysis_triggered"] = True
                    st.session_state["use_cached_analysis"] = True
                    
                    # Force UI refresh
                    st.experimental_rerun()
    
    def render(self) -> None:
        """Render the main view with all components."""
        # Render the header
        self.render_header()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render the input form
        url, max_comments, sort_by = self.render_input_form()
        
        # Check if analysis has been triggered
        if st.session_state.get("analysis_triggered", False):
            # Use provided URL or last used URL
            analysis_url = url if url else st.session_state.get("last_url", "")
            
            if not analysis_url:
                st.error("Please enter a YouTube URL")
                st.session_state["analysis_triggered"] = False
                return
                
            # Generate hash of current parameters
            params_hash = self._get_params_hash(
                analysis_url, 
                max_comments, 
                sort_by
            )
            
            # Check if we need to re-run the analysis
            need_reanalysis = (
                params_hash != st.session_state.get("analysis_params_hash", "") or
                not st.session_state.get("analysis_complete", False)
            )
            
            if need_reanalysis:
                # Process the YouTube video
                success = self._process_analysis(
                    analysis_url, 
                    max_comments, 
                    sort_by
                )
                
                if success:
                    # Update session state
                    st.session_state["analysis_params_hash"] = params_hash
                    st.session_state["analysis_complete"] = True
                    st.session_state["last_url"] = analysis_url
                    st.session_state["last_max_comments"] = max_comments
                    st.session_state["last_sort_by"] = sort_by
                else:
                    # Analysis failed
                    st.session_state["analysis_complete"] = False
            
            # Reset trigger flag
            st.session_state["analysis_triggered"] = False
            
            # Show results if analysis is complete
            if st.session_state.get("analysis_complete", False):
                # Display the analysis results
                if st.session_state.get("video_metadata"):
                    self.visualization.display_video_info(
                        st.session_state["video_metadata"]
                    )
                
                # Get the dataframe from controller
                df = self.controller.get_dataframe()
                
                if df is not None and not df.empty:
                    # Store in session state
                    st.session_state["complete_analysis_df"] = df
                    
                    # Set filter options in sidebar
                    if "language" in df.columns:
                        languages = ["All Languages"] + sorted(df["language"].unique().tolist())
                        selected_lang = st.sidebar.selectbox(
                            "Filter by Language",
                            languages,
                            index=0
                        )
                        st.session_state["selected_language"] = selected_lang
                    
                    # Apply filters
                    filtered_df = df
                    if st.session_state["selected_language"] != "All Languages":
                        filtered_df = df[df["language"] == st.session_state["selected_language"]]
                    
                    # Display the visualizations
                    # Get stopwords based on detected languages
                    try:
                        from wordcloud import STOPWORDS
                        stop_words = set(STOPWORDS)
                        # Get language-specific stopwords 
                        for lang in df["language"].unique():
                            stop_words.update(self.controller.analyzer.get_stopwords_for_language(lang))
                    except Exception as e:
                        self.logger.warning(f"Error getting stopwords: {e}")
                        stop_words = set()
                    
                    # Display all visualizations
                    self.visualization.display_all_visualizations(filtered_df, stop_words)
    
    def _process_analysis(self, url: str, max_comments: int, sort_by: str) -> bool:
        """
        Process the analysis of YouTube comments.
        
        Args:
            url: YouTube video URL
            max_comments: Maximum number of comments to analyze
            sort_by: Comment sorting method
            
        Returns:
            True if analysis was successful
        """
        try:
            # Reset controller state
            self.controller.reset()
            
            # Step 1: Fetch comments
            success = self.controller.fetch_comments(
                url=url,
                max_comments=max_comments,
                sort_by=sort_by,
                use_cache=True
            )
            
            if not success:
                return False
                
            # Step 2: Analyze comments
            success = self.controller.analyze_comments()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing analysis: {e}", exc_info=True)
            st.error(f"An error occurred during analysis: {str(e)}")
            return False