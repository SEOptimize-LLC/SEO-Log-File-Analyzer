"""
Advanced SEO Insights page for SEO Log Analyzer
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path
from collections import Counter
import networkx as nx

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from components.seo_analyzer import SEOAnalyzer
from components.visualizations import create_empty_chart
from models.predictive_analytics import PredictiveAnalytics

# Page config
st.set_page_config(
    page_title="SEO Insights - SEO Log Analyzer",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Advanced SEO Insights")
    st.markdown("Deep dive into SEO patterns, crawl budget optimization, and predictive analytics")
    
    # Check for data
    if 'parsed_data' not in st.session_state or st.session_state.parsed_data is None:
        st.warning("⚠️ No data available. Please upload and process a log file first.")
        if st.button("Go to Main Dashboard"):
            st.switch_page("app.py")
        return
    
    data = st.session_state.parsed_data
    
    # Initialize SEO Analyzer
    seo_analyzer = SEOAnalyzer()
    seo_metrics = seo_analyzer.analyze(data)
    
    # Display tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Crawl Budget", 
        "📊 Content Discovery", 
        "🔗 Internal Linking",
        "📱 Mobile vs Desktop",
        "🌍 International SEO",
        "🔮 Predictive Analytics"
    ])
    
    with tab1:
        display_crawl_budget_analysis(data, seo_metrics)
    
    with tab2:
        display_content_discovery(data, seo_metrics)
    
    with tab3:
        display_internal_linking(data, seo_metrics)
    
    with tab4:
        display_mobile_desktop_analysis(data, seo_metrics)
    
    with tab5:
        display_international_seo(data, seo_metrics)
    
    with tab6:
        display_predictive_analytics(data, seo_metrics)

def display_crawl_budget_analysis(data, metrics):
    """Display crawl budget optimization insights"""
    st.header("🎯 Crawl Budget Optimization")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Crawl Budget Used",
            f"{metrics.get('total_crawls', 0):,}",
            help="Total number of bot requests"
        )
    
    with col2:
        waste_pct = metrics.get('crawl_waste_percentage', 0)
        st.metric(
            "Crawl Waste",
            f"{waste_pct:.1f}%",
            delta=f"{-waste_pct:.1f}%" if waste_pct > 10 else "Optimal",
            delta_color="inverse",
            help="Percentage of crawls on low-value pages"
        )
    
    with col3:
        st.metric(
            "Unique Pages Crawled",
            f"{metrics.get('unique_pages_crawled', 0):,}",
            help="Number of unique URLs crawled"
        )
    
    with col4:
        avg_depth = metrics.get('avg_crawl_depth', 0)
        st.metric(
            "Average Crawl Depth",
            f"{avg_depth:.1f}",
            delta="Good" if avg_depth < 4 else "Deep",
            help="Average directory depth of crawled pages"
        )
    
    # Crawl distribution chart
    st.subheader("📊 Crawl Distribution by Page Type")
    
    if 'url' in data.columns and 'is_bot' in data.columns:
        bot_data = data[data['is_bot'] == True]
        
        # Categorize pages
        page_categories = categorize_pages(bot_data['url'])
        category_counts = pd.Series(page_categories).value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                marker_color=['#10b981' if cat in ['Product', 'Category', 'Content'] 
                             else '#ef4444' for cat in category_counts.index],
                text=category_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Crawl Budget Distribution by Page Type",
            xaxis_title="Page Category",
            yaxis_title="Number of Crawls",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Crawl waste analysis
    st.subheader("🗑️ Crawl Waste Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Parameters and duplicates
        waste_sources = metrics.get('crawl_waste_sources', {})
        if waste_sources:
            waste_df = pd.DataFrame(
                list(waste_sources.items()),
                columns=['Waste Type', 'Count']
            )
            waste_df['Percentage'] = (waste_df['Count'] / waste_df['Count'].sum() * 100).round(1)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=waste_df['Waste Type'],
                    values=waste_df['Count'],
                    hole=0.3,
                    marker_colors=['#ef4444', '#f97316', '#eab308', '#84cc16']
                )
            ])
            
            fig.update_layout(
                title="Crawl Waste Sources",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Recommendations
        st.markdown("### 💡 Optimization Recommendations")
        
        recommendations = generate_crawl_budget_recommendations(metrics)
        for i, rec in enumerate(recommendations, 1):
            if rec['priority'] == 'high':
                st.error(f"{i}. 🔴 {rec['text']}")
            elif rec['priority'] == 'medium':
                st.warning(f"{i}. 🟡 {rec['text']}")
            else:
                st.info(f"{i}. 🟢 {rec['text']}")

def display_content_discovery(data, metrics):
    """Display content discovery insights"""
    st.header("📊 Content Discovery Analysis")
    
    # Orphan pages analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        orphan_count = metrics.get('orphan_pages', 0)
        st.metric(
            "Orphan Pages",
            orphan_count,
            delta="Found" if orphan_count > 0 else "None",
            delta_color="inverse",
            help="Pages receiving traffic but not crawled"
        )
    
    with col2:
        discovery_rate = metrics.get('discovery_rate', 0)
        st.metric(
            "Discovery Rate",
            f"{discovery_rate:.1f}%",
            help="Percentage of pages discovered by bots"
        )
    
    with col3:
        new_pages = metrics.get('new_pages_discovered', 0)
        st.metric(
            "New Pages Found",
            new_pages,
            delta=f"+{new_pages}" if new_pages > 0 else "0",
            help="Recently discovered pages"
        )
    
    # Content freshness analysis
    st.subheader("🔄 Content Freshness Impact")
    
    if 'last_modified' in data.columns and 'is_bot' in data.columns:
        bot_data = data[data['is_bot'] == True]
        
        # Group by content age
        bot_data['content_age'] = pd.to_datetime('now') - pd.to_datetime(bot_data['last_modified'])
        bot_data['age_category'] = pd.cut(
            bot_data['content_age'].dt.days,
            bins=[0, 7, 30, 90, 365, float('inf')],
            labels=['< 1 week', '1-4 weeks', '1-3 months', '3-12 months', '> 1 year']
        )
        
        age_crawls = bot_data.groupby('age_category').size()
        
        fig = go.Figure(data=[
            go.Bar(
                x=age_crawls.index,
                y=age_crawls.values,
                marker_color=['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444'],
                text=age_crawls.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Crawl Frequency by Content Age",
            xaxis_title="Content Age",
            yaxis_title="Number of Crawls",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top discovered content
    st.subheader("🌟 Top Discovered Content")
    
    if 'url' in data.columns and 'is_bot' in data.columns:
        bot_data = data[data['is_bot'] == True]
        top_pages = bot_data['url'].value_counts().head(20)
        
        top_pages_df = pd.DataFrame({
            'URL': top_pages.index,
            'Bot Crawls': top_pages.values,
            'Crawl Share': (top_pages.values / top_pages.sum() * 100).round(2)
        })
        
        st.dataframe(
            top_pages_df,
            use_container_width=True,
            hide_index=True
        )

def display_internal_linking(data, metrics):
    """Display internal linking analysis"""
    st.header("🔗 Internal Linking Analysis")
    
    # Link metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Average Internal Links",
            metrics.get('avg_internal_links', 0),
            help="Average internal links per page"
        )
    
    with col2:
        st.metric(
            "Link Depth",
            metrics.get('avg_link_depth', 0),
            help="Average click depth from homepage"
        )
    
    with col3:
        st.metric(
            "Orphan Rate",
            f"{metrics.get('orphan_rate', 0):.1f}%",
            delta_color="inverse",
            help="Percentage of pages with no internal links"
        )
    
    with col4:
        st.metric(
            "Link Velocity",
            f"{metrics.get('link_velocity', 0):.1f}/day",
            help="New internal links discovered per day"
        )
    
    # Link flow visualization
    st.subheader("🕸️ Link Flow Visualization")
    
    if 'referer' in data.columns and 'url' in data.columns:
        # Create link graph
        link_data = data[data['referer'].notna()].head(1000)
        
        # Build network graph
        G = nx.DiGraph()
        
        for _, row in link_data.iterrows():
            G.add_edge(row['referer'], row['url'])
        
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(G, max_iter=100)
            top_pages = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:20]
            
            # Create visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=[p[1] for p in top_pages],
                    y=[p[0][-30:] for p in top_pages],  # Truncate URLs for display
                    orientation='h',
                    marker_color='#8b5cf6'
                )
            ])
            
            fig.update_layout(
                title="Top Pages by Internal PageRank",
                xaxis_title="PageRank Score",
                yaxis_title="Page URL",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Insufficient data for PageRank calculation")

def display_mobile_desktop_analysis(data, metrics):
    """Display mobile vs desktop crawler analysis"""
    st.header("📱 Mobile vs Desktop Analysis")
    
    # Detect mobile vs desktop bots
    if 'user_agent' in data.columns and 'is_bot' in data.columns:
        bot_data = data[data['is_bot'] == True]
        
        # Classify mobile vs desktop
        bot_data['is_mobile'] = bot_data['user_agent'].str.contains(
            'Mobile|Android', case=False, na=False
        )
        
        mobile_count = bot_data['is_mobile'].sum()
        desktop_count = len(bot_data) - mobile_count
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mobile Bot Crawls",
                f"{mobile_count:,}",
                help="Googlebot smartphone and other mobile bots"
            )
        
        with col2:
            st.metric(
                "Desktop Bot Crawls", 
                f"{desktop_count:,}",
                help="Traditional desktop crawler requests"
            )
        
        with col3:
            mobile_ratio = (mobile_count / len(bot_data) * 100) if len(bot_data) > 0 else 0
            st.metric(
                "Mobile-First Index",
                f"{mobile_ratio:.1f}%",
                delta="Mobile-First" if mobile_ratio > 60 else "Desktop-Heavy",
                help="Percentage of mobile bot traffic"
            )
        
        # Comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Mobile', 'Desktop'],
                    values=[mobile_count, desktop_count],
                    hole=0.3,
                    marker_colors=['#3b82f6', '#8b5cf6']
                )
            ])
            
            fig.update_layout(
                title="Mobile vs Desktop Bot Distribution",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Timeline comparison
            if 'timestamp' in bot_data.columns:
                bot_data['date'] = pd.to_datetime(bot_data['timestamp']).dt.date
                
                daily_mobile = bot_data[bot_data['is_mobile']].groupby('date').size()
                daily_desktop = bot_data[~bot_data['is_mobile']].groupby('date').size()
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=daily_mobile.index,
                    y=daily_mobile.values,
                    mode='lines',
                    name='Mobile',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=daily_desktop.index,
                    y=daily_desktop.values,
                    mode='lines',
                    name='Desktop',
                    line=dict(color='#8b5cf6', width=2)
                ))
                
                fig.update_layout(
                    title="Mobile vs Desktop Crawl Trend",
                    xaxis_title="Date",
                    yaxis_title="Crawls",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)

def display_international_seo(data, metrics):
    """Display international SEO insights"""
    st.header("🌍 International SEO Analysis")
    
    # Check for international indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Unique Countries",
            metrics.get('unique_countries', 'N/A'),
            help="Countries detected from IP addresses"
        )
    
    with col2:
        st.metric(
            "Language Variants",
            metrics.get('language_variants', 'N/A'),
            help="Detected language versions"
        )
    
    with col3:
        st.metric(
            "Hreflang Pages",
            metrics.get('hreflang_pages', 'N/A'),
            help="Pages with hreflang tags"
        )
    
    # Geographic distribution
    st.subheader("🗺️ Geographic Crawler Distribution")
    
    # Placeholder for geographic data
    st.info("Geographic analysis requires IP geolocation data. Enable IP geolocation in settings.")
    
    # Language detection from URLs
    if 'url' in data.columns:
        # Detect language patterns in URLs
        language_patterns = {
            'en': r'/en/|/en-',
            'es': r'/es/|/es-',
            'fr': r'/fr/|/fr-',
            'de': r'/de/|/de-',
            'it': r'/it/|/it-',
            'pt': r'/pt/|/pt-',
            'ja': r'/ja/|/ja-',
            'zh': r'/zh/|/zh-'
        }
        
        language_counts = {}
        for lang, pattern in language_patterns.items():
            count = data['url'].str.contains(pattern, case=False, na=False).sum()
            if count > 0:
                language_counts[lang.upper()] = count
        
        if language_counts:
            lang_df = pd.DataFrame(
                list(language_counts.items()),
                columns=['Language', 'Pages']
            )
            
            fig = go.Figure(data=[
                go.Bar(
                    x=lang_df['Language'],
                    y=lang_df['Pages'],
                    marker_color='#10b981',
                    text=lang_df['Pages'],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Content by Language Code",
                xaxis_title="Language",
                yaxis_title="Number of Pages",
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)

def display_predictive_analytics(data, metrics):
    """Display predictive analytics and forecasting"""
    st.header("🔮 Predictive Analytics")
    
    # Initialize predictive model
    predictor = PredictiveAnalytics()
    
    st.subheader("📈 Crawl Rate Prediction")
    
    if 'timestamp' in data.columns and 'is_bot' in data.columns:
        bot_data = data[data['is_bot'] == True]
        
        # Prepare time series data
        bot_data['date'] = pd.to_datetime(bot_data['timestamp']).dt.date
        daily_crawls = bot_data.groupby('date').size().reset_index(name='crawls')
        
        if len(daily_crawls) > 7:  # Need at least a week of data
            # Train model and predict
            with st.spinner("Training predictive model..."):
                forecast = predictor.predict_crawl_rate(daily_crawls)
            
            if forecast is not None:
                # Visualization
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=daily_crawls['date'],
                    y=daily_crawls['crawls'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='#3b82f6', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast['date'],
                    y=forecast['predicted'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#10b981', width=2, dash='dash')
                ))
                
                # Confidence interval
                if 'lower_bound' in forecast.columns:
                    fig.add_trace(go.Scatter(
                        x=forecast['date'],
                        y=forecast['upper_bound'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast['date'],
                        y=forecast['lower_bound'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Confidence Interval'
                    ))
                
                fig.update_layout(
                    title="Crawl Rate Forecast (Next 30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Daily Crawls",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_forecast = forecast['predicted'].mean()
                    avg_historical = daily_crawls['crawls'].mean()
                    change = ((avg_forecast - avg_historical) / avg_historical * 100)
                    
                    st.metric(
                        "Predicted Avg Daily Crawls",
                        f"{avg_forecast:.0f}",
                        delta=f"{change:+.1f}%",
                        help="Expected average for next 30 days"
                    )
                
                with col2:
                    trend = "Increasing" if change > 5 else "Decreasing" if change < -5 else "Stable"
                    st.metric(
                        "Crawl Trend",
                        trend,
                        help="Overall crawl trajectory"
                    )
                
                with col3:
                    confidence = predictor.get_confidence_score()
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.1f}%",
                        help="Prediction confidence level"
                    )
        else:
            st.info("Need at least 7 days of data for predictions")
    
    # SEO Impact Predictions
    st.subheader("🎯 SEO Impact Predictions")
    
    predictions = predictor.predict_seo_impact(metrics)
    
    if predictions:
        for prediction in predictions:
            if prediction['risk_level'] == 'high':
                st.error(f"⚠️ **{prediction['issue']}**: {prediction['impact']}")
            elif prediction['risk_level'] == 'medium':
                st.warning(f"📊 **{prediction['issue']}**: {prediction['impact']}")
            else:
                st.info(f"💡 **{prediction['issue']}**: {prediction['impact']}")

def categorize_pages(urls):
    """Categorize URLs into page types"""
    categories = []
    
    for url in urls:
        url_lower = str(url).lower()
        
        if any(term in url_lower for term in ['/product', '/item', '/p/']):
            categories.append('Product')
        elif any(term in url_lower for term in ['/category', '/c/', '/shop']):
            categories.append('Category')
        elif any(term in url_lower for term in ['/blog', '/article', '/post']):
            categories.append('Content')
        elif any(term in url_lower for term in ['.jpg', '.png', '.gif', '.webp']):
            categories.append('Images')
        elif any(term in url_lower for term in ['.js', '.css']):
            categories.append('Assets')
        elif any(term in url_lower for term in ['?', '&', 'session', 'utm_']):
            categories.append('Parameters')
        else:
            categories.append('Other')
    
    return categories

def generate_crawl_budget_recommendations(metrics):
    """Generate crawl budget optimization recommendations"""
    recommendations = []
    
    waste_pct = metrics.get('crawl_waste_percentage', 0)
    if waste_pct > 20:
        recommendations.append({
            'priority': 'high',
            'text': f'High crawl waste ({waste_pct:.1f}%). Block low-value pages in robots.txt'
        })
    
    if metrics.get('parameter_crawls', 0) > 1000:
        recommendations.append({
            'priority': 'high',
            'text': 'Excessive parameter crawling detected. Use canonical tags or parameter handling in GSC'
        })
    
    if metrics.get('orphan_pages', 0) > 0:
        recommendations.append({
            'priority': 'medium',
            'text': f"{metrics.get('orphan_pages', 0)} orphan pages found. Add internal links to improve discovery"
        })
    
    if metrics.get('avg_crawl_depth', 0) > 4:
        recommendations.append({
            'priority': 'medium',
            'text': 'Deep site structure detected. Flatten architecture for better crawl efficiency'
        })
    
    if metrics.get('404_rate', 0) > 5:
        recommendations.append({
            'priority': 'high',
            'text': 'High 404 error rate. Fix broken links to conserve crawl budget'
        })
    
    if not recommendations:
        recommendations.append({
            'priority': 'low',
            'text': 'Crawl budget is well optimized. Continue monitoring'
        })
    
    return recommendations

if __name__ == "__main__":
    main()
