# pages/1_📊_Overview.py

"""
Overview page for SEO Log Analyzer
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from components.visualizations import create_overview_charts
from utils.data_processor import apply_filters

# Page config
st.set_page_config(
    page_title="Overview - SEO Log Analyzer",
    page_icon="📊",
    layout="wide"
)

def display_metrics():
    """Display key metrics"""
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        st.warning("No data loaded. Please upload a log file from the main page.")
        return
    
    data = st.session_state[SESSION_KEYS['processed']]
    
    # Apply date range filter
    if 'timestamp' in data.columns:
        date_range = st.session_state.get(SESSION_KEYS['date_range'], (data['timestamp'].min(), data['timestamp'].max()))
        mask = (data['timestamp'].dt.date >= date_range[0]) & (data['timestamp'].dt.date <= date_range[1])
        filtered_data = data[mask]
    else:
        filtered_data = data
    
    # Calculate metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_requests = len(filtered_data)
        prev_requests = len(data) - total_requests if len(data) > total_requests else 0
        delta = ((total_requests - prev_requests) / prev_requests * 100) if prev_requests > 0 else 0
        st.metric(
            "Total Requests",
            f"{total_requests:,}",
            f"{delta:+.1f}%" if delta != 0 else None
        )
    
    with col2:
        unique_visitors = filtered_data['ip'].nunique() if 'ip' in filtered_data.columns else 0
        st.metric(
            "Unique Visitors",
            f"{unique_visitors:,}",
            help="Unique IP addresses"
        )
    
    with col3:
        if 'response_time' in filtered_data.columns:
            avg_response = filtered_data['response_time'].mean()
            p95_response = filtered_data['response_time'].quantile(0.95)
            st.metric(
                "Avg Response Time",
                f"{avg_response:.0f}ms",
                f"P95: {p95_response:.0f}ms"
            )
        else:
            st.metric("Avg Response Time", "N/A")
    
    with col4:
        if 'status' in filtered_data.columns:
            error_rate = (filtered_data['status'] >= 400).mean() * 100
            st.metric(
                "Error Rate",
                f"{error_rate:.2f}%",
                "⚠️ High" if error_rate > 5 else "✅ Normal"
            )
        else:
            st.metric("Error Rate", "N/A")

def display_traffic_analysis():
    """Display traffic analysis charts"""
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        return
    
    data = st.session_state[SESSION_KEYS['processed']]
    
    # Apply date range filter
    if 'timestamp' in data.columns:
        date_range = st.session_state.get(SESSION_KEYS['date_range'], (data['timestamp'].min(), data['timestamp'].max()))
        mask = (data['timestamp'].dt.date >= date_range[0]) & (data['timestamp'].dt.date <= date_range[1])
        filtered_data = data[mask]
    else:
        filtered_data = data
    
    # Generate charts
    charts = create_overview_charts(filtered_data)
    
    # Traffic Timeline
    st.subheader("📈 Traffic Timeline")
    st.plotly_chart(charts['timeline'], use_container_width=True)
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Status Code Distribution")
        st.plotly_chart(charts['status_dist'], use_container_width=True)
        
        # Additional status code details
        if 'status' in filtered_data.columns:
            with st.expander("Status Code Details"):
                status_summary = filtered_data['status'].value_counts().head(10)
                st.dataframe(
                    pd.DataFrame({
                        'Status Code': status_summary.index,
                        'Count': status_summary.values,
                        'Percentage': (status_summary.values / len(filtered_data) * 100).round(2)
                    })
                )
    
    with col2:
        st.subheader("🔝 Top Pages")
        st.plotly_chart(charts['top_pages'], use_container_width=True)
        
        # Additional page details
        if 'url' in filtered_data.columns:
            with st.expander("Page Details"):
                page_stats = filtered_data.groupby('url').agg({
                    'ip': 'count',
                    'response_time': 'mean' if 'response_time' in filtered_data.columns else lambda x: 0
                }).round(2)
                page_stats.columns = ['Requests', 'Avg Response Time']
                st.dataframe(page_stats.head(10))

def display_bot_analysis():
    """Display bot analysis section"""
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        return
    
    data = st.session_state[SESSION_KEYS['processed']]
    
    # Apply date range filter
    if 'timestamp' in data.columns:
        date_range = st.session_state.get(SESSION_KEYS['date_range'], (data['timestamp'].min(), data['timestamp'].max()))
        mask = (data['timestamp'].dt.date >= date_range[0]) & (data['timestamp'].dt.date <= date_range[1])
        filtered_data = data[mask]
    else:
        filtered_data = data
    
    st.subheader("🤖 Bot Traffic Analysis")
    
    # Generate bot charts
    charts = create_overview_charts(filtered_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(charts['bot_human_ratio'], use_container_width=True)
        
        # Bot statistics
        if 'is_bot' in filtered_data.columns:
            bot_count = filtered_data['is_bot'].sum()
            human_count = len(filtered_data) - bot_count
            
            st.info(f"""
            **Bot Traffic:** {bot_count:,} requests ({bot_count/len(filtered_data)*100:.1f}%)  
            **Human Traffic:** {human_count:,} requests ({human_count/len(filtered_data)*100:.1f}%)
            """)
    
    with col2:
        st.plotly_chart(charts['bot_timeline'], use_container_width=True)
        
        # Bot type breakdown
        if 'bot_type' in filtered_data.columns:
            with st.expander("Bot Type Breakdown"):
                bot_types = filtered_data[filtered_data['is_bot']]['bot_type'].value_counts()
                st.dataframe(
                    pd.DataFrame({
                        'Bot Type': bot_types.index,
                        'Count': bot_types.values,
                        'Percentage': (bot_types.values / bot_types.sum() * 100).round(2)
                    })
                )

def display_geographic_analysis():
    """Display geographic analysis"""
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        return
    
    data = st.session_state[SESSION_KEYS['processed']]
    
    if 'country' not in data.columns:
        st.info("Geographic data not available")
        return
    
    st.subheader("🌍 Geographic Distribution")
    
    # Country distribution
    country_counts = data['country'].value_counts().head(20)
    
    fig = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        labels={'x': 'Requests', 'y': 'Country'},
        title="Top Countries by Request Volume"
    )
    st.plotly_chart(fig, use_container_width=True)

def display_session_analysis():
    """Display session analysis"""
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        return
    
    data = st.session_state[SESSION_KEYS['processed']]
    
    if 'session_id' not in data.columns:
        st.info("Session data not available")
        return
    
    st.subheader("👥 Session Analysis")
    
    # Session metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sessions = data.groupby(['ip', 'session_id']).ngroups
        st.metric("Total Sessions", f"{total_sessions:,}")
    
    with col2:
        if 'session_duration' in data.columns:
            avg_duration = data.groupby(['ip', 'session_id'])['session_duration'].first().mean()
            st.metric("Avg Session Duration", f"{avg_duration/60:.1f} min")
        else:
            st.metric("Avg Session Duration", "N/A")
    
    with col3:
        if 'page_views' in data.columns:
            avg_pages = data.groupby(['ip', 'session_id'])['page_views'].first().mean()
            st.metric("Avg Pages/Session", f"{avg_pages:.1f}")
        else:
            st.metric("Avg Pages/Session", "N/A")
    
    with col4:
        # Bounce rate (sessions with only 1 page view)
        if 'page_views' in data.columns:
            session_pages = data.groupby(['ip', 'session_id'])['page_views'].first()
            bounce_rate = (session_pages == 1).mean() * 100
            st.metric("Bounce Rate", f"{bounce_rate:.1f}%")
        else:
            st.metric("Bounce Rate", "N/A")

def main():
    """Main overview page"""
    st.title("📊 Overview Dashboard")
    st.markdown("---")
    
    # Check if data is loaded
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        st.warning("⚠️ No data loaded. Please upload a log file from the main page.")
        
        # Show sample dashboard
        if st.button("Load Sample Data"):
            from components.log_parser import LogParser
            parser = LogParser()
            sample_data = parser.get_sample_data(5000)
            st.session_state[SESSION_KEYS['processed']] = sample_data
            st.rerun()
        
        return
    
    # Display sections
    display_metrics()
    st.markdown("---")
    
    # Tab layout for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Traffic Analysis",
        "🤖 Bot Analysis", 
        "🌍 Geographic",
        "👥 Sessions"
    ])
    
    with tab1:
        display_traffic_analysis()
    
    with tab2:
        display_bot_analysis()
    
    with tab3:
        display_geographic_analysis()
    
    with tab4:
        display_session_analysis()
    
    # Export options
    st.markdown("---")
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to Excel"):
            from utils.export_handler import ExportHandler
            handler = ExportHandler()
            excel_data = handler.export_excel(st.session_state[SESSION_KEYS['processed']])
            st.download_button(
                "Download Excel",
                excel_data,
                "seo_log_analysis.xlsx",
                "application/vnd.ms-excel"
            )
    
    with col2:
        if st.button("📄 Export to CSV"):
            from utils.export_handler import ExportHandler
            handler = ExportHandler()
            csv_data = handler.export_csv(st.session_state[SESSION_KEYS['processed']])
            st.download_button(
                "Download CSV",
                csv_data,
                "seo_log_analysis.csv",
                "text/csv"
            )
    
    with col3:
        if st.button("📑 Generate PDF Report"):
            st.info("Navigate to the Reports page for comprehensive PDF generation")

if __name__ == "__main__":
    main()
