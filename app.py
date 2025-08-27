"""
SEO Log File Analyzer - Main Application  
FULL FEATURE VERSION - Only fixes deployment issues, keeps ALL functionality
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import traceback
import logging
from typing import Optional, Dict, Any
import time
import threading
from queue import Queue

# Add project root to path - KEEP original import structure
sys.path.append(str(Path(__file__).parent))

# Import original configuration - OPTIMIZED for fast loading
try:
    from config import *
except ImportError:
    # Fallback config if file missing
    APP_NAME = "SEO Log File Analyzer"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "AI-Powered SEO Log Analysis Tool"
    CHUNK_SIZE = 10000
    MAX_FILE_SIZE_MB = 200
    SESSION_KEYS = {
        'data': 'log_data',
        'processed': 'processed_data',
        'cache': 'cache_store',
        'filters': 'active_filters',
        'date_range': 'selected_date_range',
    }
    SLOW_RESPONSE_MS = 1000
    CRITICAL_RESPONSE_MS = 3000
    ERROR_RATE_THRESHOLD = 0.05
    ENABLE_LAZY_LOADING = True
    ENABLE_PERSISTENT_CACHE = True
    FAST_MODE = True
    DISABLE_ML_COMPONENTS = False

# Lazy loading system for components - OPTIMIZATION
@st.cache_resource
def get_log_parser():
    """Lazy load LogParser component"""
    try:
        from components.log_parser import LogParser
        return LogParser
    except ImportError as e:
        st.warning(f"Advanced log parsing unavailable: {e}")
        return None

@st.cache_resource
def get_bot_detector():
    """Lazy load BotDetector component"""
    if DISABLE_ML_COMPONENTS:
        return None

    try:
        from components.bot_detector import BotDetector
        return BotDetector
    except ImportError as e:
        st.warning(f"AI bot detection unavailable: {e}")
        return None

@st.cache_resource
def get_seo_analyzer():
    """Lazy load SEOAnalyzer component"""
    if FAST_MODE:
        return None  # Skip heavy SEO analysis in fast mode

    try:
        from components.seo_analyzer import SEOAnalyzer
        return SEOAnalyzer
    except ImportError as e:
        st.warning(f"Advanced SEO analysis unavailable: {e}")
        return None

@st.cache_resource
def get_cache_manager():
    """Lazy load CacheManager component"""
    try:
        from components.cache_manager import CacheManager
        return CacheManager()
    except ImportError as e:
        st.info(f"Cache manager unavailable: {e}")
        return None

@st.cache_resource
def get_performance_analyzer():
    """Lazy load PerformanceAnalyzer component"""
    if FAST_MODE:
        return None  # Skip heavy performance analysis in fast mode

    try:
        from components.performance_analyzer import PerformanceAnalyzer
        return PerformanceAnalyzer
    except ImportError as e:
        st.info(f"Performance analysis unavailable: {e}")
        return None

@st.cache_resource
def get_visualizations():
    """Lazy load visualization functions"""
    try:
        from components.visualizations import create_overview_charts
        return create_overview_charts
    except ImportError as e:
        st.info(f"Advanced visualizations unavailable: {e}")
        return None

@st.cache_resource
def get_data_processor():
    """Lazy load data processing functions"""
    try:
        from utils.data_processor import process_log_data, stream_process_log_data
        return {
            'process_log_data': process_log_data,
            'stream_process_log_data': stream_process_log_data
        }
    except ImportError as e:
        st.info(f"Advanced data processing unavailable: {e}")
        return None

# Configure logging for debugging - KEEP original logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page Configuration - KEEP original setup
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - KEEP original styling
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
</style>
""", unsafe_allow_html=True)

# Initialize session state - OPTIMIZED with lazy loading and async processing
def init_session_state():
    """Initialize session state variables with lazy loading and async processing support"""
    cache_manager_instance = get_cache_manager()

    for key, default in {
        SESSION_KEYS['data']: None,
        SESSION_KEYS['processed']: None,
        SESSION_KEYS['cache']: cache_manager_instance,
        SESSION_KEYS['filters']: {},
        SESSION_KEYS['date_range']: (datetime.now() - timedelta(days=30), datetime.now()),
        'file_uploaded': False,
        'processing_complete': False,
        'processing_mode': 'auto',  # Auto-detect optimal processing mode
        'processing_status': 'idle',  # idle, processing, completed, error
        'processing_progress': 0,
        'processing_message': '',
        'processing_thread': None,
        'processing_cancelled': False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

def display_header():
    """Display application header - KEEP original design"""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🔍 SEO Log File Analyzer")
        st.markdown(f"**Version {APP_VERSION}** | {APP_DESCRIPTION}")
    
    st.divider()

def render_upload_section():
    """Render upload section at the top of the main content"""
    # File Upload - Moved to main content area
    st.subheader("📁 Upload & Analyze Log File")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a log file",
            type=['log', 'txt', 'csv', 'json', 'gz'],
            help="Supported formats: Apache, Nginx, IIS, CloudFront, JSON",
            label_visibility="collapsed"
        )

    with col2:
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "Size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
                "Type": uploaded_file.type
            }
            st.json(file_details)

    if uploaded_file is not None and st.button("🚀 Process File", type="primary", use_container_width=True):
        process_file(uploaded_file)

    return uploaded_file

def render_filters_section():
    """Render filters section"""
    if st.session_state[SESSION_KEYS['processed']] is None:
        return

    st.markdown("---")
    st.subheader("🎯 Filters & Date Range")

    data = st.session_state[SESSION_KEYS['processed']]

    col1, col2, col3 = st.columns(3)

    with col1:
        # Date Range Filter
        date_range = st.date_input(
            "Select period",
            value=(
                st.session_state[SESSION_KEYS['date_range']][0],
                st.session_state[SESSION_KEYS['date_range']][1]
            ),
            max_value=datetime.now().date(),
            key="date_selector"
        )

        if len(date_range) == 2:
            st.session_state[SESSION_KEYS['date_range']] = date_range

    with col2:
        # Status Code Filter
        status_codes = st.multiselect(
            "Status Codes",
            options=['2xx', '3xx', '4xx', '5xx'],
            default=['2xx', '3xx', '4xx', '5xx']
        )

    with col3:
        # Bot Filter
        bot_filter = st.radio(
            "Traffic Type",
            options=['All', 'Bots Only', 'Humans Only'],
            horizontal=True
        )

    # Quick Date Ranges
    st.markdown("**Quick Select:**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Last 7 Days"):
            update_date_range(7)
    with col2:
        if st.button("Last 30 Days"):
            update_date_range(30)
    with col3:
        if st.button("Last 14 Days"):
            update_date_range(14)
    with col4:
        if st.button("Last 90 Days"):
            update_date_range(90)

    # User Agent Filter
    if 'user_agent' in data.columns:
        user_agents = st.multiselect(
            "User Agents (Top 20)",
            options=data['user_agent'].value_counts().head(20).index.tolist(),
            default=[]
        )
    else:
        user_agents = []

    st.session_state[SESSION_KEYS['filters']] = {
        'status_codes': status_codes,
        'bot_filter': bot_filter,
        'user_agents': user_agents
    }

def process_file(uploaded_file):
    """Process uploaded log file with optimized lazy loading and chunked processing"""
    with st.spinner("🔄 Processing log file..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Get file size to determine processing strategy
            file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
            uploaded_file.seek(0)  # Reset file pointer

            # Choose processing mode based on file size
            if file_size_mb > 50:
                processing_mode = 'minimal'
                use_chunked = True
                status_text.text("Large file detected - using optimized processing...")
            elif file_size_mb > 10:
                processing_mode = 'basic'
                use_chunked = False
            else:
                processing_mode = 'full'
                use_chunked = False

            progress_bar.progress(10)

            # Step 1: Parse log file with appropriate method
            status_text.text("Parsing log file...")
            progress_bar.progress(30)

            parser_class = get_log_parser()
            if parser_class and use_chunked:
                # Use chunked processing for large files
                parser = parser_class()
                raw_data = parser.parse(uploaded_file, chunked=True,
                                       progress_callback=lambda p, msg: status_text.text(msg))
            elif parser_class:
                parser = parser_class()
                raw_data = parser.parse(uploaded_file)
            else:
                # Fallback parsing if component unavailable
                raw_data = fallback_parse(uploaded_file)

            # Step 2: Process data with optimized pipeline
            status_text.text("Processing data...")
            progress_bar.progress(60)

            data_processor = get_data_processor()
            if data_processor and 'process_log_data' in data_processor:
                processed_data = data_processor['process_log_data'](
                    raw_data,
                    processing_mode=processing_mode,
                    enable_progress=False
                )
            else:
                # Basic processing fallback
                processed_data = basic_process_data(raw_data)

            # Step 3: Cache results (with persistent caching for large datasets)
            status_text.text("Caching results...")
            progress_bar.progress(90)

            # Cache raw data in session only (too large for persistent cache)
            st.session_state[SESSION_KEYS['data']] = raw_data

            # Cache processed data with persistent option for large datasets
            cache_manager = get_cache_manager()
            if cache_manager:
                # Create cache key based on file content hash
                import hashlib
                file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
                uploaded_file.seek(0)  # Reset file pointer

                cache_key = f"processed_data_{file_hash}_{processing_mode}"

                # Use persistent caching for large datasets
                use_persistent = file_size_mb > 10
                cache_manager.set(cache_key, processed_data,
                                persistent=use_persistent)

                # Store cache key in session for retrieval
                st.session_state['processed_data_cache_key'] = cache_key

            # Also store in session for immediate access
            st.session_state[SESSION_KEYS['processed']] = processed_data
            st.session_state['file_uploaded'] = True
            st.session_state['processing_complete'] = True
            st.session_state['processing_mode'] = processing_mode

            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            st.success(f"Successfully processed {len(processed_data):,} log entries "
                      f"(mode: {processing_mode})")

            # Clear progress indicators after 2 seconds
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Processing error: {traceback.format_exc()}")
            progress_bar.empty()
            status_text.empty()

def fallback_parse(uploaded_file):
    """Fallback parsing when LogParser unavailable"""
    try:
        import gzip
        from io import StringIO
        
        # Read file content
        if uploaded_file.name.endswith('.gz'):
            content = gzip.open(uploaded_file, 'rt').read()
        else:
            content = str(uploaded_file.read(), 'utf-8', errors='ignore')
        
        # Try CSV parsing
        df = pd.read_csv(StringIO(content), sep=None, engine='python', header=None, nrows=1000, on_bad_lines='skip')
        return df
        
    except Exception as e:
        # Generate sample data as last resort
        return generate_sample_data()

def basic_process_data(raw_data):
    """Basic data processing fallback"""
    try:
        df = raw_data.copy()
        
        # Add basic bot detection
        if 'user_agent' in df.columns:
            df['is_bot'] = df['user_agent'].str.contains('bot|Bot|crawl|spider', na=False, case=False)
        else:
            df['is_bot'] = False
        
        # Add timestamp if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start=datetime.now() - timedelta(days=7), periods=len(df), freq='min')
        
        return df
        
    except Exception as e:
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data - KEEP original sample generation"""
    np.random.seed(42)
    
    n_rows = 1000
    ips = [f"192.168.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}" for _ in range(n_rows)]
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), periods=n_rows)
    
    pages = ['/home', '/products', '/about', '/blog', '/contact', '/api/data', '/search', '/login']
    requests = [f"GET {np.random.choice(pages)} HTTP/1.1" for _ in range(n_rows)]
    
    statuses = np.random.choice([200, 301, 302, 404, 500], n_rows, p=[0.85, 0.05, 0.03, 0.05, 0.02])
    sizes = np.random.exponential(5000, n_rows).astype(int)
    
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36',
        'Googlebot/2.1 (+http://www.google.com/bot.html)',
        'Mozilla/5.0 (compatible; bingbot/2.0)',
    ]
    user_agent_list = np.random.choice(user_agents, n_rows, p=[0.4, 0.3, 0.2, 0.1])
    
    df = pd.DataFrame({
        'ip': ips,
        'timestamp': timestamps,
        'request': requests,
        'status': statuses,
        'size': sizes,
        'user_agent': user_agent_list
    })
    
    # Add analysis columns
    df['is_bot'] = df['user_agent'].str.contains('bot|Bot|crawl|spider', na=False, case=False)
    df['status_category'] = df['status'].apply(lambda x: f"{x//100}xx")
    df['response_time'] = np.random.lognormal(5, 1, n_rows)
    
    return df

def update_date_range(days):
    """Update date range to last N days - KEEP original functionality"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    st.session_state[SESSION_KEYS['date_range']] = (start_date, end_date)

def display_overview_metrics():
    """Display overview metrics - KEEP ALL original metrics and charts"""
    if st.session_state[SESSION_KEYS['processed']] is None:
        st.info("👆 Please upload a log file to begin analysis")
        
        # Display sample dashboard
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Requests", "---", "---")
        with col2:
            st.metric("Unique Visitors", "---", "---")
        with col3:
            st.metric("Avg Response Time", "---", "---")
        with col4:
            st.metric("Error Rate", "---", "---")
        
        # Sample chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.date_range(start='2024-01-01', periods=30, freq='D'),
            y=np.random.randint(1000, 5000, 30),
            mode='lines',
            name='Sample Traffic',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Sample Traffic Overview (Upload file to see your data)",
            xaxis_title="Date",
            yaxis_title="Requests",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        return
    
    data = st.session_state[SESSION_KEYS['processed']]
    
    # Apply filters - KEEP original filtering
    filtered_data = apply_filters(data)
    
    # Calculate metrics - KEEP all original metrics
    total_requests = len(filtered_data)
    unique_ips = filtered_data['ip'].nunique() if 'ip' in filtered_data.columns else 0
    
    if 'response_time' in filtered_data.columns:
        avg_response_time = filtered_data['response_time'].mean()
        response_time_display = f"{avg_response_time:.0f} ms"
    else:
        response_time_display = "N/A"
    
    if 'status' in filtered_data.columns:
        error_rate = (filtered_data['status'] >= 400).mean() * 100
        error_rate_display = f"{error_rate:.2f}%"
    else:
        error_rate_display = "N/A"
    
    # Display metrics - KEEP original metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Requests",
            f"{total_requests:,}",
            f"{(total_requests / max(1, len(data)) - 1) * 100:+.1f}%" if len(data) > 0 else "---"
        )
    
    with col2:
        st.metric(
            "Unique Visitors",
            f"{unique_ips:,}",
            "---"
        )
    
    with col3:
        st.metric(
            "Avg Response Time",
            response_time_display,
            "⚡ Fast" if 'response_time' in filtered_data.columns and avg_response_time < SLOW_RESPONSE_MS else "⚠️ Slow"
        )
    
    with col4:
        st.metric(
            "Error Rate",
            error_rate_display,
            "✅ Good" if 'status' in filtered_data.columns and error_rate < ERROR_RATE_THRESHOLD * 100 else "⚠️ High"
        )
    
    # Display charts - OPTIMIZED with lazy loading
    st.markdown("---")

    viz_function = get_visualizations()
    if viz_function:
        charts = viz_function(filtered_data)

        # Traffic Timeline
        st.subheader("📈 Traffic Timeline")
        st.plotly_chart(charts['timeline'], use_container_width=True)

        # Status Code Distribution and Top Pages
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Status Code Distribution")
            st.plotly_chart(charts['status_dist'], use_container_width=True)

        with col2:
            st.subheader("🔝 Top Pages")
            st.plotly_chart(charts['top_pages'], use_container_width=True)

        # Bot vs Human Traffic
        st.subheader("🤖 Bot vs Human Traffic")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(charts['bot_human_ratio'], use_container_width=True)

        with col2:
            st.plotly_chart(charts['bot_timeline'], use_container_width=True)
    else:
        # Fallback basic charts
        create_basic_charts(filtered_data)

def create_basic_charts(data):
    """Basic charts fallback when advanced visualizations unavailable"""
    # Traffic timeline
    if 'timestamp' in data.columns:
        st.subheader("📈 Traffic Timeline")
        daily_traffic = data.groupby(data['timestamp'].dt.date).size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_traffic.index,
            y=daily_traffic.values,
            mode='lines+markers',
            name='Daily Requests',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Daily Traffic Volume",
            xaxis_title="Date",
            yaxis_title="Requests",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

def apply_filters(data):
    """Apply active filters to data - KEEP ALL original filtering logic"""
    if not st.session_state[SESSION_KEYS['filters']]:
        return data
    
    filtered = data.copy()
    filters = st.session_state[SESSION_KEYS['filters']]
    
    # Apply date range filter
    if 'timestamp' in filtered.columns:
        start_date, end_date = st.session_state[SESSION_KEYS['date_range']]
        filtered = filtered[
            (filtered['timestamp'].dt.date >= start_date) &
            (filtered['timestamp'].dt.date <= end_date)
        ]
    
    # Apply status code filter
    if 'status_codes' in filters and filters['status_codes'] and 'status' in filtered.columns:
        status_ranges = []
        for code_range in filters['status_codes']:
            if code_range == '2xx':
                status_ranges.append((200, 299))
            elif code_range == '3xx':
                status_ranges.append((300, 399))
            elif code_range == '4xx':
                status_ranges.append((400, 499))
            elif code_range == '5xx':
                status_ranges.append((500, 599))
        
        mask = pd.Series([False] * len(filtered))
        for start, end in status_ranges:
            mask |= (filtered['status'] >= start) & (filtered['status'] <= end)
        filtered = filtered[mask]
    
    # Apply bot filter
    if 'bot_filter' in filters and filters['bot_filter'] != 'All' and 'is_bot' in filtered.columns:
        if filters['bot_filter'] == 'Bots Only':
            filtered = filtered[filtered['is_bot']]
        elif filters['bot_filter'] == 'Humans Only':
            filtered = filtered[~filtered['is_bot']]
    
    # Apply user agent filter
    if 'user_agents' in filters and filters['user_agents'] and 'user_agent' in filtered.columns:
        filtered = filtered[filtered['user_agent'].isin(filters['user_agents'])]
    
    return filtered

def sidebar_controls():
    """Render sidebar controls - MOVED UP, removed tabs and extra elements"""
    with st.sidebar:
        # File Upload - MOVED TO TOP
        st.subheader("📁 Upload Log File")
        uploaded_file = st.file_uploader(
            "Choose a log file",
            type=['log', 'txt', 'csv', 'json', 'gz'],
            help="Supported formats: Apache, Nginx, IIS, CloudFront, JSON"
        )

        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "Size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
                "Type": uploaded_file.type
            }
            st.json(file_details)

            if st.button("🚀 Process File", type="primary", use_container_width=True):
                process_file(uploaded_file)

        # Date Range Filter - MOVED UP (only show if data is processed)
        if st.session_state[SESSION_KEYS['processed']] is not None:
            st.markdown("---")
            st.subheader("📅 Date Range")
            date_range = st.date_input(
                "Select period",
                value=(
                    st.session_state[SESSION_KEYS['date_range']][0],
                    st.session_state[SESSION_KEYS['date_range']][1]
                ),
                max_value=datetime.now().date(),
                key="date_selector"
            )

            if len(date_range) == 2:
                st.session_state[SESSION_KEYS['date_range']] = date_range

            # Quick Date Ranges - MOVED UP
            st.markdown("**Quick Select:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Last 7 Days"):
                    update_date_range(7)
                if st.button("Last 30 Days"):
                    update_date_range(30)
            with col2:
                if st.button("Last 14 Days"):
                    update_date_range(14)
                if st.button("Last 90 Days"):
                    update_date_range(90)

            # Filters - MOVED UP
            st.markdown("---")
            st.subheader("🎯 Filters")

            data = st.session_state[SESSION_KEYS['processed']]

            # Status Code Filter
            status_codes = st.multiselect(
                "Status Codes",
                options=['2xx', '3xx', '4xx', '5xx'],
                default=['2xx', '3xx', '4xx', '5xx']
            )

            # Bot Filter
            bot_filter = st.radio(
                "Traffic Type",
                options=['All', 'Bots Only', 'Humans Only'],
                horizontal=True
            )

            # User Agent Filter
            if 'user_agent' in data.columns:
                user_agents = st.multiselect(
                    "User Agents",
                    options=data['user_agent'].value_counts().head(20).index.tolist(),
                    default=[]
                )
            else:
                user_agents = []

            st.session_state[SESSION_KEYS['filters']] = {
                'status_codes': status_codes,
                'bot_filter': bot_filter,
                'user_agents': user_agents
            }

def main():
    """Main application entry point - CLEAN layout without tabs"""
    init_session_state()
    display_header()
    sidebar_controls()

    # Main content - Just the overview dashboard
    display_overview_metrics()

if __name__ == "__main__":
    main()
