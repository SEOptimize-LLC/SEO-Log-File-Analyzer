# pages/3_⚡_Performance.py

"""
Performance Analysis page for SEO Log Analyzer
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from components.performance_analyzer import PerformanceAnalyzer
from components.visualizations import create_performance_charts
from models.predictive_analytics import PerformanceForecaster

# Page config
st.set_page_config(
    page_title="Performance Analysis - SEO Log Analyzer",
    page_icon="⚡",
    layout="wide"
)

def display_performance_metrics():
    """Display key performance metrics"""
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None:
        st.warning("No data available for performance analysis")
        return
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    performance_data = analyzer.analyze(data)
    
    # Store in session state for other functions
    st.session_state['performance_analysis'] = performance_data
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'response_times' in performance_data and 'percentiles' in performance_data['response_times']:
            p50 = performance_data['response_times']['percentiles']['p50']
            st.metric(
                "Median Response Time",
                f"{p50:.0f}ms",
                "✅ Fast" if p50 < 1000 else "⚠️ Slow"
            )
        else:
            st.metric("Median Response Time", "N/A")
    
    with col2:
        if 'response_times' in performance_data and 'percentiles' in performance_data['response_times']:
            p95 = performance_data['response_times']['percentiles']['p95']
            st.metric(
                "P95 Response Time",
                f"{p95:.0f}ms",
                "✅ Good" if p95 < SLOW_RESPONSE_MS else "⚠️ High"
            )
        else:
            st.metric("P95 Response Time", "N/A")
    
    with col3:
        if 'ttfb' in performance_data and 'mean' in performance_data['ttfb']:
            ttfb_mean = performance_data['ttfb']['mean']
            st.metric(
                "Avg TTFB",
                f"{ttfb_mean:.0f}ms",
                "✅ Good" if ttfb_mean < TTFB_WARNING_MS else "⚠️ High"
            )
        else:
            st.metric("Avg TTFB", "N/A")
    
    with col4:
        if 'performance_score' in performance_data:
            score = performance_data['performance_score']['score']
            grade = performance_data['performance_score']['grade']
            st.metric(
                "Performance Score",
                f"{score}/100",
                f"Grade: {grade}"
            )
        else:
            st.metric("Performance Score", "N/A")

def display_response_time_analysis():
    """Display response time analysis"""
    performance_data = st.session_state.get('performance_analysis', {})
    
    if not performance_data or 'response_times' not in performance_data:
        st.info("Response time data not available")
        return
    
    st.subheader("📊 Response Time Analysis")
    
    response_data = performance_data['response_times']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time distribution
        if 'categories' in response_data:
            categories = response_data['categories']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Fast\n(<1s)', 'Moderate\n(1-3s)', 'Slow\n(3-5s)', 'Critical\n(>5s)'],
                    y=[categories.get('fast', 0), categories.get('moderate', 0),
                       categories.get('slow', 0), categories.get('critical', 0)],
                    marker_color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444'],
                    text=[categories.get('fast', 0), categories.get('moderate', 0),
                          categories.get('slow', 0), categories.get('critical', 0)],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Response Time Distribution",
                xaxis_title="Category",
                yaxis_title="Number of Requests",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Percentiles
        if 'percentiles' in response_data:
            percentiles = response_data['percentiles']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['P50', 'P75', 'P90', 'P95', 'P99'],
                    y=[percentiles.get('p50', 0), percentiles.get('p75', 0),
                       percentiles.get('p90', 0), percentiles.get('p95', 0),
                       percentiles.get('p99', 0)],
                    marker_color='#3b82f6',
                    text=[f"{v:.0f}ms" for v in [percentiles.get('p50', 0),
                                                  percentiles.get('p75', 0),
                                                  percentiles.get('p90', 0),
                                                  percentiles.get('p95', 0),
                                                  percentiles.get('p99', 0)]],
                    textposition='auto'
                )
            ])
            
            # Add threshold lines
            fig.add_hline(y=SLOW_RESPONSE_MS, line_dash="dash", line_color="orange",
                         annotation_text="Slow Threshold")
            fig.add_hline(y=CRITICAL_RESPONSE_MS, line_dash="dash", line_color="red",
                         annotation_text="Critical Threshold")
            
            fig.update_layout(
                title="Response Time Percentiles",
                xaxis_title="Percentile",
                yaxis_title="Response Time (ms)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    if 'statistics' in response_data:
        stats = response_data['statistics']
        
        st.markdown("### 📈 Response Time Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{stats.get('mean', 0):.0f}ms")
        with col2:
            st.metric("Median", f"{stats.get('median', 0):.0f}ms")
        with col3:
            st.metric("Std Dev", f"{stats.get('std', 0):.0f}ms")
        with col4:
            st.metric("Max", f"{stats.get('max', 0):.0f}ms")

def display_slow_endpoints():
    """Display slow endpoints analysis"""
    performance_data = st.session_state.get('performance_analysis', {})
    
    if 'slow_endpoints' not in performance_data or not performance_data['slow_endpoints']:
        st.info("No slow endpoints detected")
        return
    
    st.subheader("🐌 Slowest Endpoints")
    
    slow_endpoints = performance_data['slow_endpoints']
    
    # Create DataFrame
    endpoints_df = pd.DataFrame(slow_endpoints)
    
    # Display table
    st.dataframe(
        endpoints_df[['url', 'avg_response_time', 'median_response_time', 'total_requests', 'impact_score']].round(2),
        use_container_width=True,
        hide_index=True
    )
    
    # Visualization
    top_10 = endpoints_df.head(10)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_10['avg_response_time'],
            y=top_10['url'],
            orientation='h',
            marker_color='#ef4444',
            text=[f"{t:.0f}ms" for t in top_10['avg_response_time']],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top 10 Slowest Endpoints",
        xaxis_title="Average Response Time (ms)",
        yaxis_title="",
        height=400,
        margin=dict(l=300)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_ttfb_analysis():
    """Display TTFB analysis"""
    performance_data = st.session_state.get('performance_analysis', {})
    
    if 'ttfb' not in performance_data:
        st.info("TTFB data not available")
        return
    
    st.subheader("⏱️ Time To First Byte (TTFB) Analysis")
    
    ttfb_data = performance_data['ttfb']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # TTFB distribution
        if 'categories' in ttfb_data:
            categories = ttfb_data['categories']
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Good (<600ms)', 'Warning (600-1200ms)', 'Critical (>1200ms)'],
                    values=[categories.get('good', 0), categories.get('warning', 0), 
                           categories.get('critical', 0)],
                    marker_colors=['#10b981', '#f59e0b', '#ef4444'],
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="TTFB Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # TTFB percentiles
        if 'percentiles' in ttfb_data:
            percentiles = ttfb_data['percentiles']
            
            st.markdown("### TTFB Percentiles")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("P50", f"{percentiles.get('p50', 0):.0f}ms")
                st.metric("P75", f"{percentiles.get('p75', 0):.0f}ms")
            
            with metrics_col2:
                st.metric("P90", f"{percentiles.get('p90', 0):.0f}ms")
                st.metric("P95", f"{percentiles.get('p95', 0):.0f}ms")
            
            # Recommendation
            if 'recommendation' in ttfb_data:
                st.info(ttfb_data['recommendation'])

def display_performance_trends():
    """Display performance trends over time"""
    performance_data = st.session_state.get('performance_analysis', {})
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None or 'trends' not in performance_data:
        st.info("Trend data not available")
        return
    
    st.subheader("📈 Performance Trends")
    
    trends = performance_data['trends']
    
    if 'daily_stats' in trends and trends['daily_stats']:
        daily_df = pd.DataFrame(trends['daily_stats'])
        
        # Create trend chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=daily_df['date'],
            y=daily_df['mean'],
            mode='lines+markers',
            name='Mean Response Time',
            line=dict(color='#3b82f6', width=2)
        ))
        
        if 'p95' in daily_df.columns:
            fig.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['p95'],
                mode='lines+markers',
                name='P95 Response Time',
                line=dict(color='#f59e0b', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Daily Performance Trend",
            xaxis_title="Date",
            yaxis_title="Response Time (ms)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        if 'trend' in trends and trends['trend'].get('direction'):
            trend_info = trends['trend']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Trend Direction",
                    trend_info['direction'].capitalize(),
                    "📈" if trend_info['direction'] == 'degrading' else "📉"
                )
            
            with col2:
                if 'slope' in trend_info:
                    st.metric(
                        "Change Rate",
                        f"{abs(trend_info['slope']):.2f}ms/day"
                    )
            
            with col3:
                if 'r_squared' in trend_info:
                    st.metric(
                        "Confidence (R²)",
                        f"{trend_info['r_squared']:.3f}"
                    )
        
        # Anomalies
        if 'anomaly_dates' in trends and trends['anomaly_dates']:
            st.warning(f"⚠️ Performance anomalies detected on: {', '.join(map(str, trends['anomaly_dates']))}")

def display_performance_forecast():
    """Display performance predictions"""
    st.subheader("🔮 Performance Forecast")
    
    data = st.session_state.get(SESSION_KEYS['processed'])
    
    if data is None:
        st.warning("No data available for forecasting")
        return
    
    # Initialize forecaster
    if 'performance_forecaster' not in st.session_state:
        st.session_state['performance_forecaster'] = PerformanceForecaster()
    
    forecaster = st.session_state['performance_forecaster']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        metric = st.selectbox(
            "Select Metric to Forecast",
            ['response_time', 'error_rate', 'traffic_volume']
        )
        
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner(f"Training forecast model for {metric}..."):
                forecaster.train(data, metric)
                
                # Generate forecast
                forecast = forecaster.predict(metric, periods_hours=168)  # 1 week
                
                if not forecast.empty:
                    # Plot forecast
                    fig = go.Figure()
                    
                    # Historical data
                    historical = forecaster.prepare_data(data, metric)
                    if not historical.empty:
                        fig.add_trace(go.Scatter(
                            x=historical['ds'],
                            y=historical['y'],
                            mode='lines',
                            name='Historical',
                            line=dict(color='#3b82f6')
                        ))
                    
                    # Forecast
                    future = forecast[forecast['ds'] > pd.Timestamp.now()]
                    fig.add_trace(go.Scatter(
                        x=future['ds'],
                        y=future['yhat'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#10b981', dash='dash')
                    ))
                    
                    # Confidence interval
                    fig.add_trace(go.Scatter(
                        x=future['ds'],
                        y=future['yhat_upper'],
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future['ds'],
                        y=future['yhat_lower'],
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f"{metric.replace('_', ' ').title()} Forecast",
                        xaxis_title="Time",
                        yaxis_title=metric.replace('_', ' ').title(),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success(f"Forecast generated for {metric}")
    
    with col2:
        st.markdown("### Forecast Settings")
        
        horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=7
        )
        
        st.info(f"""
        **Current Settings:**
        - Metric: {metric if 'metric' in locals() else 'response_time'}
        - Horizon: {horizon} days
        - Update Frequency: Hourly
        """)

def display_resource_analysis():
    """Display resource size analysis"""
    performance_data = st.session_state.get('performance_analysis', {})
    
    if 'resource_sizes' not in performance_data:
        st.info("Resource size data not available")
        return
    
    st.subheader("📦 Resource Size Analysis")
    
    resource_data = performance_data['resource_sizes']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Size distribution
        if 'categories' in resource_data:
            categories = resource_data['categories']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Small\n(<100KB)', 'Medium\n(100KB-1MB)', 
                       'Large\n(1MB-5MB)', 'Very Large\n(>5MB)'],
                    y=[categories.get('small', 0), categories.get('medium', 0),
                       categories.get('large', 0), categories.get('very_large', 0)],
                    marker_color=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
                )
            ])
            
            fig.update_layout(
                title="Resource Size Distribution",
                xaxis_title="Size Category",
                yaxis_title="Number of Resources",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metrics
        st.markdown("### Resource Metrics")
        
        if 'mean_size_mb' in resource_data:
            st.metric("Average Size", f"{resource_data['mean_size_mb']:.2f} MB")
        
        if 'total_bandwidth_gb' in resource_data:
            st.metric("Total Bandwidth", f"{resource_data['total_bandwidth_gb']:.2f} GB")
        
        # Largest resources
        if 'largest_resources' in resource_data and resource_data['largest_resources']:
            st.markdown("**Largest Resources:**")
            largest_df = pd.DataFrame(resource_data['largest_resources'])
            st.dataframe(
                largest_df[['url', 'size_mb']].head(5),
                hide_index=True
            )

def main():
    """Main performance page"""
    st.title("⚡ Performance Analysis")
    st.markdown("Comprehensive performance metrics and analysis")
    st.markdown("---")
    
    # Check if data is loaded
    if SESSION_KEYS['processed'] not in st.session_state or st.session_state[SESSION_KEYS['processed']] is None:
        st.warning("⚠️ No data loaded. Please upload a log file from the main page.")
        return
    
    # Display metrics
    display_performance_metrics()
    st.markdown("---")
    
    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Response Times",
        "🐌 Slow Endpoints",
        "⏱️ TTFB",
        "📈 Trends",
        "🔮 Forecast",
        "📦 Resources"
    ])
    
    with tab1:
        display_response_time_analysis()
    
    with tab2:
        display_slow_endpoints()
    
    with tab3:
        display_ttfb_analysis()
    
    with tab4:
        display_performance_trends()
    
    with tab5:
        display_performance_forecast()
    
    with tab6:
        display_resource_analysis()

if __name__ == "__main__":
    main()
