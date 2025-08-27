# pages/4_❌_Errors.py

"""
Error Analysis page for SEO Log Analyzer
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

# Page config
st.set_page_config(
    page_title="Error Analysis - SEO Log Analyzer",
    page_icon="❌",
    layout="wide"
)

def display_error_metrics():
    """Display error metrics overview"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'status' not in data.columns:
        st.warning("No error data available")
        return
    
    # Calculate error metrics
    total_requests = len(data)
    error_4xx = ((data['status'] >= 400) & (data['status'] < 500)).sum()
    error_5xx = (data['status'] >= 500).sum()
    error_404 = (data['status'] == 404).sum()
    
    total_errors = error_4xx + error_5xx
    error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Errors",
            f"{total_errors:,}",
            f"{error_rate:.2f}% of requests"
        )
    
    with col2:
        st.metric(
            "404 Errors",
            f"{error_404:,}",
            f"{error_404/total_requests*100:.2f}%" if total_requests > 0 else "0%"
        )
    
    with col3:
        st.metric(
            "4xx Errors",
            f"{error_4xx:,}",
            "Client errors"
        )
    
    with col4:
        st.metric(
            "5xx Errors",
            f"{error_5xx:,}",
            "⚠️ Server errors" if error_5xx > 0 else "✅ No server errors"
        )

def display_404_analysis():
    """Display 404 error analysis"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'status' not in data.columns:
        return
    
    st.subheader("🔍 404 Error Analysis")
    
    # Filter 404 errors
    errors_404 = data[data['status'] == 404]
    
    if errors_404.empty:
        st.success("✅ No 404 errors detected!")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top 404 pages
        if 'url' in errors_404.columns:
            top_404_pages = errors_404['url'].value_counts().head(20)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_404_pages.values,
                    y=top_404_pages.index,
                    orientation='h',
                    marker_color='#f59e0b',
                    text=top_404_pages.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Top 404 Error Pages",
                xaxis_title="Number of 404 Errors",
                yaxis_title="",
                height=500,
                margin=dict(l=300)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 404 statistics
        st.markdown("### 404 Error Statistics")
        
        st.metric("Total 404s", f"{len(errors_404):,}")
        
        if 'url' in errors_404.columns:
            st.metric("Unique 404 URLs", f"{errors_404['url'].nunique():,}")
        
        if 'referrer' in errors_404.columns:
            # Top referrers to 404 pages
            st.markdown("**Top Referrers to 404s:**")
            top_referrers = errors_404['referrer'].value_counts().head(5)
            
            for referrer, count in top_referrers.items():
                if referrer and referrer != '-':
                    st.text(f"• {referrer[:50]}... ({count})")
        
        # Export 404 list
        if st.button("📥 Export 404 List"):
            csv = errors_404[['url', 'timestamp']].to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "404_errors.csv",
                "text/csv"
            )

def display_5xx_analysis():
    """Display 5xx server error analysis"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'status' not in data.columns:
        return
    
    st.subheader("🚨 5xx Server Error Analysis")
    
    # Filter 5xx errors
    errors_5xx = data[data['status'] >= 500]
    
    if errors_5xx.empty:
        st.success("✅ No server errors detected!")
        return
    
    # Error code breakdown
    error_breakdown = errors_5xx['status'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Error code distribution
        fig = go.Figure(data=[
            go.Pie(
                labels=[f"{code} ({self._get_error_name(code)})" for code in error_breakdown.index],
                values=error_breakdown.values,
                hole=0.3,
                marker_colors=px.colors.sequential.Reds
            )
        ])
        
        fig.update_layout(
            title="5xx Error Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top pages with 5xx errors
        if 'url' in errors_5xx.columns:
            st.markdown("### Pages with Server Errors")
            
            top_5xx_pages = errors_5xx['url'].value_counts().head(10)
            
            for url, count in top_5xx_pages.items():
                st.error(f"**{url[:50]}...** - {count} errors")
    
    # Timeline of 5xx errors
    if 'timestamp' in errors_5xx.columns:
        st.markdown("### Server Error Timeline")
        
        errors_5xx['hour'] = pd.to_datetime(errors_5xx['timestamp']).dt.floor('H')
        hourly_5xx = errors_5xx.groupby('hour').size()
        
        fig = go.Figure(data=[
            go.Scatter(
                x=hourly_5xx.index,
                y=hourly_5xx.values,
                mode='lines+markers',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=8)
            )
        ])
        
        fig.update_layout(
            title="5xx Errors Over Time",
            xaxis_title="Time",
            yaxis_title="Number of Errors",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_error_trends():
    """Display error trends over time"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'status' not in data.columns or 'timestamp' not in data.columns:
        return
    
    st.subheader("📈 Error Trends")
    
    # Create hourly error data
    data['hour'] = pd.to_datetime(data['timestamp']).dt.floor('H')
    
    # Calculate error rates by hour
    hourly_stats = data.groupby('hour').agg({
        'status': [
            'count',
            lambda x: ((x >= 400) & (x < 500)).sum(),
            lambda x: (x >= 500).sum(),
            lambda x: (x == 404).sum()
        ]
    })
    
    hourly_stats.columns = ['total', '4xx', '5xx', '404']
    hourly_stats['error_rate'] = (hourly_stats['4xx'] + hourly_stats['5xx']) / hourly_stats['total'] * 100
    
    # Create trend chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly_stats.index,
        y=hourly_stats['4xx'],
        name='4xx Errors',
        mode='lines',
        stackgroup='one',
        line=dict(color='#f59e0b')
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly_stats.index,
        y=hourly_stats['5xx'],
        name='5xx Errors',
        mode='lines',
        stackgroup='one',
        line=dict(color='#ef4444')
    ))
    
    fig.update_layout(
        title="Error Trends Over Time",
        xaxis_title="Time",
        yaxis_title="Number of Errors",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Error rate trend
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rate = go.Figure(data=[
            go.Scatter(
                x=hourly_stats.index,
                y=hourly_stats['error_rate'],
                mode='lines+markers',
                line=dict(color='#ef4444', width=2),
                marker=dict(size=6)
            )
        ])
        
        # Add threshold line
        fig_rate.add_hline(
            y=ERROR_RATE_THRESHOLD * 100,
            line_dash="dash",
            line_color="orange",
            annotation_text="Warning Threshold"
        )
        
        fig_rate.update_layout(
            title="Error Rate Trend",
            xaxis_title="Time",
            yaxis_title="Error Rate (%)",
            height=350
        )
        
        st.plotly_chart(fig_rate, use_container_width=True)
    
    with col2:
        # Error spikes detection
        st.markdown("### Error Spikes Detected")
        
        # Find spikes (error rate > 2x average)
        avg_error_rate = hourly_stats['error_rate'].mean()
        spikes = hourly_stats[hourly_stats['error_rate'] > avg_error_rate * 2]
        
        if not spikes.empty:
            for idx, row in spikes.iterrows():
                st.warning(f"**{idx.strftime('%Y-%m-%d %H:%M')}** - Error rate: {row['error_rate']:.1f}%")
        else:
            st.success("No significant error spikes detected")

def display_redirect_analysis():
    """Display redirect analysis"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'status' not in data.columns:
        return
    
    st.subheader("↪️ Redirect Analysis")
    
    # Filter redirects
    redirects = data[data['status'].isin([301, 302, 303, 307, 308])]
    
    if redirects.empty:
        st.info("No redirects detected")
        return
    
    # Redirect type breakdown
    redirect_types = redirects['status'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Redirect distribution
        fig = go.Figure(data=[
            go.Bar(
                x=redirect_types.index.astype(str),
                y=redirect_types.values,
                marker_color=['#3b82f6' if code == 301 else '#06b6d4' for code in redirect_types.index],
                text=redirect_types.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Redirect Type Distribution",
            xaxis_title="HTTP Status Code",
            yaxis_title="Count",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Redirect Statistics")
        
        st.metric("Total Redirects", f"{len(redirects):,}")
        st.metric("Permanent (301)", f"{(redirects['status'] == 301).sum():,}")
        st.metric("Temporary (302)", f"{(redirects['status'] == 302).sum():,}")
        
        # Redirect chains warning
        if 'url' in redirects.columns:
            # Check for potential redirect chains
            redirect_urls = redirects['url'].value_counts()
            chains = redirect_urls[redirect_urls > 1]
            
            if not chains.empty:
                st.warning(f"⚠️ {len(chains)} URLs have multiple redirects (potential chains)")

def display_error_impact():
    """Display error impact analysis"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'status' not in data.columns:
        return
    
    st.subheader("💥 Error Impact Analysis")
    
    # Separate errors by bot vs human traffic
    if 'is_bot' in data.columns:
        bot_errors = data[(data['is_bot'] == True) & (data['status'] >= 400)]
        human_errors = data[(data['is_bot'] == False) & (data['status'] >= 400)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Impact on Bots")
            
            if not bot_errors.empty:
                bot_error_rate = len(bot_errors) / data['is_bot'].sum() * 100
                st.metric("Bot Error Rate", f"{bot_error_rate:.2f}%")
                
                # Top errors affecting bots
                if 'url' in bot_errors.columns:
                    st.markdown("**Top Pages Affecting Bots:**")
                    top_bot_errors = bot_errors['url'].value_counts().head(5)
                    for url, count in top_bot_errors.items():
                        st.text(f"• {url[:50]}... ({count})")
            else:
                st.success("No errors affecting bots")
        
        with col2:
            st.markdown("### Impact on Users")
            
            if not human_errors.empty:
                human_error_rate = len(human_errors) / (~data['is_bot']).sum() * 100
                st.metric("User Error Rate", f"{human_error_rate:.2f}%")
                
                # Top errors affecting users
                if 'url' in human_errors.columns:
                    st.markdown("**Top Pages Affecting Users:**")
                    top_human_errors = human_errors['url'].value_counts().head(5)
                    for url, count in top_human_errors.items():
                        st.text(f"• {url[:50]}... ({count})")
            else:
                st.success("No errors affecting users")
    
    # Error impact on SEO
    st.markdown("### SEO Impact")
    
    errors = data[data['status'] >= 400]
    
    if 'bot_type' in errors.columns:
        # Errors by search engine bot
        search_bots = ['google', 'bing', 'yandex', 'baidu']
        search_bot_errors = errors[errors['bot_type'].isin(search_bots)]
        
        if not search_bot_errors.empty:
            bot_error_counts = search_bot_errors['bot_type'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=bot_error_counts.index,
                    y=bot_error_counts.values,
                    marker_color='#ef4444',
                    text=bot_error_counts.values,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Errors Encountered by Search Engine Bots",
                xaxis_title="Search Engine",
                yaxis_title="Number of Errors",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.warning("⚠️ Search engine crawlers are encountering errors. This may impact indexing and rankings.")

def _get_error_name(self, code):
    """Get human-readable error name"""
    error_names = {
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
        505: "HTTP Version Not Supported"
    }
    return error_names.get(code, "Server Error")

def main():
    """Main error analysis page"""
    st.title("❌ Error Analysis")
    st.markdown("Comprehensive error detection and analysis")
    st.markdown("---")
    
    # Check if data is loaded
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        st.warning("⚠️ No data loaded. Please upload a log file from the main page.")
        return
    
    # Display metrics
    display_error_metrics()
    st.markdown("---")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 404 Errors",
        "🚨 5xx Errors",
        "📈 Error Trends",
        "↪️ Redirects",
        "💥 Impact Analysis"
    ])
    
    with tab1:
        display_404_analysis()
    
    with tab2:
        display_5xx_analysis()
    
    with tab3:
        display_error_trends()
    
    with tab4:
        display_redirect_analysis()
    
    with tab5:
        display_error_impact()

if __name__ == "__main__":
    # Add the method to the module level for the error name lookup
    import types
    import sys
    module = sys.modules[__name__]
    module._get_error_name = _get_error_name
    
    main()
