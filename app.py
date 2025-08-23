"""
SEO Log File Analyzer - Main Application
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import *
from components.log_parser import LogParser
from components.cache_manager import CacheManager
from components.visualizations import create_overview_charts
from utils.data_processor import process_log_data

# Page Configuration
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .uploadedFile {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    for key, default in {
        SESSION_KEYS['data']: None,
        SESSION_KEYS['processed']: None,
        SESSION_KEYS['cache']: CacheManager(),
        SESSION_KEYS['filters']: {},
        SESSION_KEYS['date_range']: (datetime.now() - timedelta(days=30), datetime.now()),
        'file_uploaded': False,
        'processing_complete': False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

def display_header():
    """Display application header"""
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🔍 SEO Log File Analyzer")
        st.markdown(f"**Version {APP_VERSION}** | {APP_DESCRIPTION}")
    
    st.divider()

def sidebar_controls():
    """Render sidebar controls"""
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=SEO+Log+Analyzer", use_column_width=True)
        st.markdown("---")
        
        # File Upload
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
        
        st.markdown("---")
        
        # Date Range Filter
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
        
        # Quick Date Ranges
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
        
        st.markdown("---")
        
        # Filters
        if st.session_state[SESSION_KEYS['processed']] is not None:
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
            
            st.session_state[SESSION_KEYS['filters']] = {
                'status_codes': status_codes,
                'bot_filter': bot_filter,
                'user_agents': user_agents if 'user_agents' in locals() else []
            }
        
        st.markdown("---")
        
        # Help Section
        with st.expander("ℹ️ Help & Documentation"):
            st.markdown("""
            ### Quick Start Guide
            1. Upload your server log file
            2. Click 'Process File' to analyze
            3. Use filters to refine analysis
            4. Navigate tabs for detailed insights
            
            ### Supported Formats
            - Apache (Common/Combined)
            - Nginx
            - IIS (W3C Extended)
            - CloudFront
            - JSON structured logs
            
            ### Need Help?
            - [Documentation](https://github.com/yourusername/seo-log-analyzer)
            - [Report Issues](https://github.com/yourusername/seo-log-analyzer/issues)
            """)

def process_file(uploaded_file):
    """Process uploaded log file"""
    with st.spinner("🔄 Processing log file..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Parse log file
            status_text.text("Parsing log file...")
            progress_bar.progress(20)
            
            parser = LogParser()
            raw_data = parser.parse(uploaded_file)
            
            # Step 2: Process data
            status_text.text("Processing data...")
            progress_bar.progress(50)
            
            processed_data = process_log_data(raw_data)
            
            # Step 3: Cache results
            status_text.text("Caching results...")
            progress_bar.progress(80)
            
            st.session_state[SESSION_KEYS['data']] = raw_data
            st.session_state[SESSION_KEYS['processed']] = processed_data
            st.session_state['file_uploaded'] = True
            st.session_state['processing_complete'] = True
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            st.success(f"Successfully processed {len(processed_data):,} log entries")
            
            # Clear progress indicators after 2 seconds
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            progress_bar.empty()
            status_text.empty()

def update_date_range(days):
    """Update date range to last N days"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    st.session_state[SESSION_KEYS['date_range']] = (start_date, end_date)

def display_overview_metrics():
    """Display overview metrics"""
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
    
    # Apply filters
    filtered_data = apply_filters(data)
    
    # Calculate metrics
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
    
    # Display metrics
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
    
    # Display charts
    st.markdown("---")
    charts = create_overview_charts(filtered_data)
    
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

def apply_filters(data):
    """Apply active filters to data"""
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

def main():
    """Main application entry point"""
    init_session_state()
    display_header()
    sidebar_controls()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📈 Quick Stats",
        "⚡ Performance",
        "🎯 SEO Insights",
        "ℹ️ About"
    ])
    
    with tab1:
        display_overview_metrics()
    
    with tab2:
        if st.session_state[SESSION_KEYS['processed']] is not None:
            data = apply_filters(st.session_state[SESSION_KEYS['processed']])
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Request Methods")
                if 'method' in data.columns:
                    method_dist = data['method'].value_counts()
                    fig = px.pie(
                        values=method_dist.values,
                        names=method_dist.index,
                        title="HTTP Methods Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Method data not available")
            
            with col2:
                st.subheader("🌍 Top Referrers")
                if 'referrer' in data.columns:
                    referrers = data['referrer'].value_counts().head(10)
                    fig = px.bar(
                        x=referrers.values,
                        y=referrers.index,
                        orientation='h',
                        title="Top 10 Referrers"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Referrer data not available")
        else:
            st.info("Upload a log file to view statistics")
    
    with tab3:
        if st.session_state[SESSION_KEYS['processed']] is not None:
            st.subheader("⚡ Performance Metrics")
            data = apply_filters(st.session_state[SESSION_KEYS['processed']])
            
            if 'response_time' in data.columns:
                # Response time percentiles
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("P50 (Median)", f"{data['response_time'].quantile(0.5):.0f} ms")
                with col2:
                    st.metric("P75", f"{data['response_time'].quantile(0.75):.0f} ms")
                with col3:
                    st.metric("P95", f"{data['response_time'].quantile(0.95):.0f} ms")
                with col4:
                    st.metric("P99", f"{data['response_time'].quantile(0.99):.0f} ms")
                
                # Response time distribution
                fig = px.histogram(
                    data,
                    x='response_time',
                    nbins=50,
                    title="Response Time Distribution",
                    labels={'response_time': 'Response Time (ms)', 'count': 'Number of Requests'}
                )
                fig.add_vline(x=SLOW_RESPONSE_MS, line_dash="dash", line_color="orange",
                            annotation_text="Slow Threshold")
                fig.add_vline(x=CRITICAL_RESPONSE_MS, line_dash="dash", line_color="red",
                            annotation_text="Critical Threshold")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Response time data not available in log file")
        else:
            st.info("Upload a log file to view performance metrics")
    
    with tab4:
        st.subheader("🎯 SEO Insights Preview")
        st.info("Navigate to the 'SEO Insights' page in the sidebar for detailed analysis")
        
        if st.session_state[SESSION_KEYS['processed']] is not None:
            data = apply_filters(st.session_state[SESSION_KEYS['processed']])
            
            # Quick SEO metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                bot_traffic = data['is_bot'].sum() if 'is_bot' in data.columns else 0
                st.metric("Bot Traffic", f"{bot_traffic:,}", f"{bot_traffic/len(data)*100:.1f}%")
            
            with col2:
                if 'status' in data.columns:
                    not_found = (data['status'] == 404).sum()
                    st.metric("404 Errors", f"{not_found:,}", "Check Error page for details")
                else:
                    st.metric("404 Errors", "N/A", "---")
            
            with col3:
                if 'url' in data.columns:
                    unique_urls = data['url'].nunique()
                    st.metric("Unique URLs", f"{unique_urls:,}", "---")
                else:
                    st.metric("Unique URLs", "N/A", "---")
    
    with tab5:
        st.subheader("ℹ️ About SEO Log File Analyzer")
        st.markdown(f"""
        ### Version {APP_VERSION}
        
        **SEO Log File Analyzer** is a comprehensive tool for analyzing server log files to gain SEO insights.
        
        #### Features:
        - 🔍 Multi-format log file support
        - 🤖 AI-powered bot detection
        - ⚡ Performance analysis
        - 📊 Real-time visualizations
        - 📈 SEO metrics and insights
        - 📥 Export capabilities
        
        #### Supported Log Formats:
        - Apache (Common/Combined)
        - Nginx
        - IIS (W3C Extended)
        - CloudFront
        - JSON structured logs
        
        #### Technology Stack:
        - Streamlit for UI
        - Pandas/Polars for data processing
        - Plotly for visualizations
        - Scikit-learn for ML models
        
        ---
        
        **Need Help?**
        - 📖 [Documentation](https://github.com/yourusername/seo-log-analyzer)
        - 🐛 [Report Issues](https://github.com/yourusername/seo-log-analyzer/issues)
        - 💡 [Feature Requests](https://github.com/yourusername/seo-log-analyzer/discussions)
        """)

if __name__ == "__main__":
    main()
