# pages/2_🤖_Bot_Analysis.py

"""
Bot Analysis page for SEO Log Analyzer
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
from components.bot_detector import BotDetector
from models.ml_models import BotClassifier
from components.visualizations import create_empty_chart

# Page config
st.set_page_config(
    page_title="Bot Analysis - SEO Log Analyzer",
    page_icon="🤖",
    layout="wide"
)

def display_bot_metrics():
    """Display bot detection metrics"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'is_bot' not in data.columns:
        st.warning("Bot detection not available. Please process data first.")
        return
    
    # Calculate metrics
    total_requests = len(data)
    bot_requests = data['is_bot'].sum()
    human_requests = total_requests - bot_requests
    bot_percentage = (bot_requests / total_requests * 100) if total_requests > 0 else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Bot Requests",
            f"{bot_requests:,}",
            f"{bot_percentage:.1f}% of total"
        )
    
    with col2:
        st.metric(
            "Human Requests",
            f"{human_requests:,}",
            f"{100-bot_percentage:.1f}% of total"
        )
    
    with col3:
        if 'bot_verified' in data.columns:
            verified_bots = data[data['is_bot']]['bot_verified'].sum()
            verification_rate = (verified_bots / bot_requests * 100) if bot_requests > 0 else 0
            st.metric(
                "Verified Bots",
                f"{verified_bots:,}",
                f"{verification_rate:.1f}% verified"
            )
        else:
            st.metric("Verified Bots", "N/A")
    
    with col4:
        if 'bot_confidence' in data.columns:
            avg_confidence = data[data['is_bot']]['bot_confidence'].mean()
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.2%}",
                "High" if avg_confidence > 0.8 else "Medium"
            )
        else:
            st.metric("Avg Confidence", "N/A")

def display_bot_types():
    """Display bot type breakdown"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'bot_type' not in data.columns:
        return
    
    st.subheader("🔍 Bot Type Distribution")
    
    # Filter for bots only
    bot_data = data[data['is_bot'] == True]
    
    if bot_data.empty:
        st.info("No bots detected in the data")
        return
    
    # Bot type distribution
    bot_types = bot_data['bot_type'].value_counts()
    
    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=bot_types.index,
            values=bot_types.values,
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )
    ])
    
    fig.update_layout(
        title="Bot Types Distribution",
        height=400
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bot type table
        bot_type_df = pd.DataFrame({
            'Bot Type': bot_types.index,
            'Count': bot_types.values,
            'Percentage': (bot_types.values / bot_types.sum() * 100).round(2)
        })
        st.dataframe(bot_type_df, hide_index=True)

def display_bot_activity():
    """Display bot activity patterns"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'is_bot' not in data.columns:
        return
    
    st.subheader("📊 Bot Activity Patterns")
    
    bot_data = data[data['is_bot'] == True]
    
    if bot_data.empty or 'timestamp' not in bot_data.columns:
        st.info("No bot activity data available")
        return
    
    # Hourly bot activity
    bot_data['hour'] = pd.to_datetime(bot_data['timestamp']).dt.hour
    hourly_activity = bot_data.groupby('hour').size()
    
    # Daily bot activity
    bot_data['date'] = pd.to_datetime(bot_data['timestamp']).dt.date
    daily_activity = bot_data.groupby('date').size()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly pattern
        fig_hourly = go.Figure(data=[
            go.Bar(
                x=hourly_activity.index,
                y=hourly_activity.values,
                marker_color='#8b5cf6',
                text=hourly_activity.values,
                textposition='auto'
            )
        ])
        
        fig_hourly.update_layout(
            title="Bot Activity by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Number of Requests",
            height=350
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Daily trend
        fig_daily = go.Figure(data=[
            go.Scatter(
                x=daily_activity.index,
                y=daily_activity.values,
                mode='lines+markers',
                line=dict(color='#8b5cf6', width=2),
                marker=dict(size=8)
            )
        ])
        
        fig_daily.update_layout(
            title="Daily Bot Activity Trend",
            xaxis_title="Date",
            yaxis_title="Number of Requests",
            height=350
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)

def display_crawler_analysis():
    """Display search engine crawler analysis"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'bot_type' not in data.columns:
        return
    
    st.subheader("🔎 Search Engine Crawler Analysis")
    
    # Filter for search engine bots
    search_engines = ['google', 'bing', 'yandex', 'baidu']
    search_bot_data = data[data['bot_type'].isin(search_engines)]
    
    if search_bot_data.empty:
        st.info("No search engine crawlers detected")
        return
    
    # Crawler comparison
    crawler_stats = search_bot_data.groupby('bot_type').agg({
        'url': 'count',
        'status': lambda x: (x < 400).mean() * 100 if 'status' in data.columns else 100,
        'response_time': 'mean' if 'response_time' in data.columns else lambda x: 0
    }).round(2)
    
    crawler_stats.columns = ['Requests', 'Success Rate (%)', 'Avg Response Time (ms)']
    
    # Display table
    st.dataframe(
        crawler_stats.style.highlight_max(axis=0, color='lightgreen'),
        use_container_width=True
    )
    
    # Crawled pages analysis
    if 'url' in search_bot_data.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Most crawled pages
            st.markdown("**Most Crawled Pages by Search Engines**")
            top_crawled = search_bot_data['url'].value_counts().head(10)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_crawled.values,
                    y=top_crawled.index,
                    orientation='h',
                    marker_color='#3b82f6'
                )
            ])
            
            fig.update_layout(
                xaxis_title="Crawl Count",
                height=400,
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Crawler preferences
            st.markdown("**Page Preferences by Crawler**")
            
            crawler_preferences = {}
            for crawler in search_engines:
                crawler_data = search_bot_data[search_bot_data['bot_type'] == crawler]
                if not crawler_data.empty:
                    top_page = crawler_data['url'].value_counts().head(1)
                    if not top_page.empty:
                        crawler_preferences[crawler] = {
                            'page': top_page.index[0],
                            'count': top_page.values[0]
                        }
            
            if crawler_preferences:
                pref_df = pd.DataFrame(crawler_preferences).T
                st.dataframe(pref_df, use_container_width=True)

def display_bot_verification():
    """Display bot verification analysis"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'bot_verified' not in data.columns:
        st.info("Bot verification data not available")
        return
    
    st.subheader("✅ Bot Verification Status")
    
    bot_data = data[data['is_bot'] == True]
    
    # Verification stats
    verified = bot_data['bot_verified'].sum()
    unverified = len(bot_data) - verified
    
    # Create donut chart
    fig = go.Figure(data=[
        go.Pie(
            labels=['Verified', 'Unverified'],
            values=[verified, unverified],
            hole=0.4,
            marker_colors=['#10b981', '#ef4444'],
            textinfo='label+percent+value'
        )
    ])
    
    fig.update_layout(
        title="Bot Verification Status",
        height=400
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Verification Details")
        st.info(f"""
        **Verified Bots:** {verified:,}  
        **Unverified Bots:** {unverified:,}  
        **Verification Rate:** {(verified/len(bot_data)*100):.1f}%
        
        ⚠️ Unverified bots may be:
        - Malicious crawlers
        - Scrapers
        - Monitoring tools
        - Misconfigured bots
        """)
        
        if st.button("🔍 Analyze Unverified Bots"):
            unverified_data = bot_data[bot_data['bot_verified'] == False]
            if not unverified_data.empty and 'ip' in unverified_data.columns:
                st.markdown("**Top Unverified Bot IPs:**")
                top_ips = unverified_data['ip'].value_counts().head(10)
                st.dataframe(
                    pd.DataFrame({
                        'IP Address': top_ips.index,
                        'Requests': top_ips.values
                    }),
                    hide_index=True
                )

def display_ml_bot_detection():
    """Display ML-based bot detection section"""
    st.subheader("🤖 Machine Learning Bot Detection")
    
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None:
        st.warning("No data available for ML analysis")
        return
    
    # Initialize ML classifier
    if 'bot_classifier' not in st.session_state:
        st.session_state['bot_classifier'] = BotClassifier()
    
    classifier = st.session_state['bot_classifier']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎯 Train Bot Classifier", type="primary"):
            with st.spinner("Training bot classifier..."):
                classifier.train(data)
                st.success("Bot classifier trained successfully!")
        
        if classifier.is_trained:
            # Display feature importance
            st.markdown("**Feature Importance:**")
            importance_df = pd.DataFrame({
                'Feature': classifier.feature_importance.keys(),
                'Importance': classifier.feature_importance.values()
            }).sort_values('Importance', ascending=False)
            
            fig = go.Figure(data=[
                go.Bar(
                    x=importance_df['Importance'],
                    y=importance_df['Feature'],
                    orientation='h',
                    marker_color='#8b5cf6'
                )
            ])
            
            fig.update_layout(
                title="Bot Detection Feature Importance",
                xaxis_title="Importance Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if classifier.is_trained:
            st.markdown("### Model Status")
            st.success("✅ Model Trained")
            
            # Get predictions
            predictions = classifier.predict(data)
            ml_bot_count = predictions.sum()
            
            st.metric(
                "ML-Detected Bots",
                f"{ml_bot_count:,}",
                f"{ml_bot_count/len(data)*100:.1f}% of traffic"
            )
            
            # Confidence distribution
            if st.button("📊 Show Confidence Distribution"):
                probabilities = classifier.predict_proba(data)
                
                fig = px.histogram(
                    probabilities,
                    nbins=20,
                    title="Bot Probability Distribution",
                    labels={'value': 'Bot Probability', 'count': 'Number of IPs'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train the model to see ML-based detection results")

def main():
    """Main bot analysis page"""
    st.title("🤖 Bot Analysis")
    st.markdown("Comprehensive bot detection and analysis")
    st.markdown("---")
    
    # Check if data is loaded
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        st.warning("⚠️ No data loaded. Please upload a log file from the main page.")
        return
    
    # Display sections
    display_bot_metrics()
    st.markdown("---")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Bot Types",
        "📈 Activity Patterns",
        "🔎 Search Crawlers",
        "✅ Verification",
        "🤖 ML Detection"
    ])
    
    with tab1:
        display_bot_types()
    
    with tab2:
        display_bot_activity()
    
    with tab3:
        display_crawler_analysis()
    
    with tab4:
        display_bot_verification()
    
    with tab5:
        display_ml_bot_detection()

if __name__ == "__main__":
    main()
