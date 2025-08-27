# components/visualizations.py

"""
Visualization components for SEO Log Analyzer
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import COLOR_SCHEME, CHART_HEIGHT, CHART_WIDTH

def create_overview_charts(df: pd.DataFrame) -> Dict:
    """Create overview dashboard charts"""
    charts = {}
    
    # Traffic timeline
    charts['timeline'] = create_traffic_timeline(df)
    
    # Status code distribution
    charts['status_dist'] = create_status_distribution(df)
    
    # Top pages chart
    charts['top_pages'] = create_top_pages_chart(df)
    
    # Bot vs Human ratio
    charts['bot_human_ratio'] = create_bot_human_pie(df)
    
    # Bot timeline
    charts['bot_timeline'] = create_bot_timeline(df)
    
    return charts

def create_traffic_timeline(df: pd.DataFrame) -> go.Figure:
    """Create traffic timeline chart"""
    if 'timestamp' not in df.columns:
        return create_empty_chart("No timestamp data available")
    
    # Aggregate by hour
    df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
    
    # Group by hour and status category
    if 'status_category' in df.columns:
        timeline_data = df.groupby(['hour', 'status_category']).size().reset_index(name='count')
        
        fig = go.Figure()
        
        for status in ['2xx', '3xx', '4xx', '5xx']:
            status_data = timeline_data[timeline_data['status_category'] == status]
            if not status_data.empty:
                fig.add_trace(go.Scatter(
                    x=status_data['hour'],
                    y=status_data['count'],
                    mode='lines',
                    name=status,
                    line=dict(color=COLOR_SCHEME.get(status, '#666'), width=2),
                    stackgroup='one'
                ))
    else:
        hourly_counts = df.groupby('hour').size().reset_index(name='count')
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_counts['hour'],
            y=hourly_counts['count'],
            mode='lines+markers',
            name='Requests',
            line=dict(color='#1f77b4', width=2)
        ))
    
    fig.update_layout(
        title="Traffic Timeline",
        xaxis_title="Time",
        yaxis_title="Requests",
        hovermode='x unified',
        height=CHART_HEIGHT,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_status_distribution(df: pd.DataFrame) -> go.Figure:
    """Create status code distribution chart"""
    if 'status_category' not in df.columns:
        return create_empty_chart("No status code data available")
    
    status_counts = df['status_category'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=status_counts.index,
            y=status_counts.values,
            marker_color=[COLOR_SCHEME.get(s, '#666') for s in status_counts.index],
            text=status_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Status Code Distribution",
        xaxis_title="Status Code",
        yaxis_title="Count",
        height=CHART_HEIGHT,
        showlegend=False
    )
    
    return fig

def create_top_pages_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Create top pages horizontal bar chart"""
    if 'url' not in df.columns:
        return create_empty_chart("No URL data available")
    
    top_pages = df['url'].value_counts().head(top_n)
    
    fig = go.Figure(data=[
        go.Bar(
            x=top_pages.values,
            y=top_pages.index,
            orientation='h',
            marker_color='#3b82f6',
            text=top_pages.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f"Top {top_n} Pages",
        xaxis_title="Requests",
        yaxis_title="",
        height=CHART_HEIGHT,
        margin=dict(l=200)  # More space for URLs
    )
    
    return fig

def create_bot_human_pie(df: pd.DataFrame) -> go.Figure:
    """Create bot vs human traffic pie chart"""
    if 'is_bot' not in df.columns:
        return create_empty_chart("No bot detection data available")
    
    bot_counts = df['is_bot'].value_counts()
    labels = ['Bot Traffic', 'Human Traffic']
    values = [bot_counts.get(True, 0), bot_counts.get(False, 0)]
    
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=[COLOR_SCHEME['bot'], COLOR_SCHEME['human']],
            textinfo='label+percent',
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Bot vs Human Traffic",
        height=CHART_HEIGHT,
        showlegend=True
    )
    
    return fig

def create_bot_timeline(df: pd.DataFrame) -> go.Figure:
    """Create bot traffic timeline"""
    if 'timestamp' not in df.columns or 'is_bot' not in df.columns:
        return create_empty_chart("No bot timeline data available")
    
    df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
    
    bot_timeline = df.groupby(['hour', 'is_bot']).size().reset_index(name='count')
    
    fig = go.Figure()
    
    # Bot traffic
    bot_data = bot_timeline[bot_timeline['is_bot'] == True]
    if not bot_data.empty:
        fig.add_trace(go.Scatter(
            x=bot_data['hour'],
            y=bot_data['count'],
            mode='lines',
            name='Bot Traffic',
            line=dict(color=COLOR_SCHEME['bot'], width=2)
        ))
    
    # Human traffic
    human_data = bot_timeline[bot_timeline['is_bot'] == False]
    if not human_data.empty:
        fig.add_trace(go.Scatter(
            x=human_data['hour'],
            y=human_data['count'],
            mode='lines',
            name='Human Traffic',
            line=dict(color=COLOR_SCHEME['human'], width=2)
        ))
    
    fig.update_layout(
        title="Bot vs Human Traffic Over Time",
        xaxis_title="Time",
        yaxis_title="Requests",
        hovermode='x unified',
        height=CHART_HEIGHT,
        showlegend=True
    )
    
    return fig

def create_performance_charts(performance_data: Dict) -> Dict:
    """Create performance analysis charts"""
    charts = {}
    
    # Response time distribution
    if 'response_times' in performance_data:
        charts['response_dist'] = create_response_time_distribution(performance_data['response_times'])
        charts['response_percentiles'] = create_percentile_chart(performance_data['response_times'])
    
    # TTFB chart
    if 'ttfb' in performance_data:
        charts['ttfb_dist'] = create_ttfb_chart(performance_data['ttfb'])
    
    # Slow endpoints
    if 'slow_endpoints' in performance_data:
        charts['slow_endpoints'] = create_slow_endpoints_chart(performance_data['slow_endpoints'])
    
    # Performance trends
    if 'trends' in performance_data:
        charts['trends'] = create_performance_trend_chart(performance_data['trends'])
    
    return charts

def create_response_time_distribution(response_data: Dict) -> go.Figure:
    """Create response time distribution histogram"""
    if 'categories' not in response_data:
        return create_empty_chart("No response time data available")
    
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
        height=CHART_HEIGHT,
        showlegend=False
    )
    
    return fig

def create_percentile_chart(response_data: Dict) -> go.Figure:
    """Create response time percentile chart"""
    if 'percentiles' not in response_data:
        return create_empty_chart("No percentile data available")
    
    percentiles = response_data['percentiles']
    
    fig = go.Figure(data=[
        go.Bar(
            x=['P50', 'P75', 'P90', 'P95', 'P99'],
            y=[percentiles.get('p50', 0), percentiles.get('p75', 0),
               percentiles.get('p90', 0), percentiles.get('p95', 0),
               percentiles.get('p99', 0)],
            marker_color='#3b82f6',
            text=[f"{v:.0f}ms" for v in [percentiles.get('p50', 0), percentiles.get('p75', 0),
                                          percentiles.get('p90', 0), percentiles.get('p95', 0),
                                          percentiles.get('p99', 0)]],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Response Time Percentiles",
        xaxis_title="Percentile",
        yaxis_title="Response Time (ms)",
        height=CHART_HEIGHT,
        showlegend=False
    )
    
    return fig

def create_ttfb_chart(ttfb_data: Dict) -> go.Figure:
    """Create TTFB distribution chart"""
    if 'categories' not in ttfb_data:
        return create_empty_chart("No TTFB data available")
    
    categories = ttfb_data['categories']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Good (<600ms)', 'Warning (600-1200ms)', 'Critical (>1200ms)'],
            values=[categories.get('good', 0), categories.get('warning', 0), categories.get('critical', 0)],
            marker_colors=['#10b981', '#f59e0b', '#ef4444'],
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="Time To First Byte Distribution",
        height=CHART_HEIGHT
    )
    
    return fig

def create_slow_endpoints_chart(slow_endpoints: List[Dict]) -> go.Figure:
    """Create slow endpoints chart"""
    if not slow_endpoints:
        return create_empty_chart("No slow endpoints detected")
    
    # Take top 10 slowest
    top_slow = slow_endpoints[:10]
    
    urls = [ep['url'] for ep in top_slow]
    response_times = [ep['avg_response_time'] for ep in top_slow]
    
    fig = go.Figure(data=[
        go.Bar(
            x=response_times,
            y=urls,
            orientation='h',
            marker_color='#ef4444',
            text=[f"{rt:.0f}ms" for rt in response_times],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top 10 Slowest Endpoints",
        xaxis_title="Average Response Time (ms)",
        yaxis_title="",
        height=CHART_HEIGHT,
        margin=dict(l=300)  # More space for URLs
    )
    
    return fig

def create_performance_trend_chart(trend_data: Dict) -> go.Figure:
    """Create performance trend chart"""
    if 'daily_stats' not in trend_data or not trend_data['daily_stats']:
        return create_empty_chart("No trend data available")
    
    daily_stats = pd.DataFrame(trend_data['daily_stats'])
    
    fig = go.Figure()
    
    # Mean response time
    fig.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['mean'],
        mode='lines+markers',
        name='Mean',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # P95 response time
    if 'p95' in daily_stats.columns:
        fig.add_trace(go.Scatter(
            x=daily_stats['date'],
            y=daily_stats['p95'],
            mode='lines+markers',
            name='P95',
            line=dict(color='#f59e0b', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title="Performance Trend",
        xaxis_title="Date",
        yaxis_title="Response Time (ms)",
        hovermode='x unified',
        height=CHART_HEIGHT,
        showlegend=True
    )
    
    return fig

def create_seo_charts(seo_data: Dict) -> Dict:
    """Create SEO analysis charts"""
    charts = {}
    
    # Crawl budget efficiency
    if 'crawl_budget' in seo_data:
        charts['crawl_efficiency'] = create_crawl_efficiency_chart(seo_data['crawl_budget'])
    
    # Mobile vs Desktop
    if 'mobile_desktop' in seo_data:
        charts['mobile_desktop'] = create_mobile_desktop_chart(seo_data['mobile_desktop'])
    
    # Orphan pages
    if 'orphan_pages' in seo_data:
        charts['orphan_pages'] = create_orphan_pages_chart(seo_data['orphan_pages'])
    
    # Crawl frequency
    if 'crawl_frequency' in seo_data:
        charts['crawl_freq'] = create_crawl_frequency_chart(seo_data['crawl_frequency'])
    
    return charts

def create_crawl_efficiency_chart(crawl_data: Dict) -> go.Figure:
    """Create crawl budget efficiency chart"""
    if 'wasted_crawls' not in crawl_data:
        return create_empty_chart("No crawl budget data available")
    
    wasted = crawl_data['wasted_crawls']
    efficient = crawl_data.get('total_crawls', 0) - crawl_data.get('total_wasted', 0)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Efficient Crawls', '404 Pages', '5xx Errors', 'Redirects'],
            values=[efficient, wasted.get('404_pages', 0), 
                   wasted.get('5xx_errors', 0), wasted.get('redirects', 0)],
            marker_colors=['#10b981', '#f59e0b', '#ef4444', '#3b82f6'],
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="Crawl Budget Efficiency",
        height=CHART_HEIGHT
    )
    
    return fig

def create_mobile_desktop_chart(mobile_data: Dict) -> go.Figure:
    """Create mobile vs desktop crawler chart"""
    mobile = mobile_data.get('mobile_crawls', 0)
    desktop = mobile_data.get('desktop_crawls', 0)
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Mobile', 'Desktop'],
            y=[mobile, desktop],
            marker_color=['#8b5cf6', '#06b6d4'],
            text=[mobile, desktop],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Mobile vs Desktop Crawler Activity",
        xaxis_title="Crawler Type",
        yaxis_title="Number of Crawls",
        height=CHART_HEIGHT,
        showlegend=False
    )
    
    return fig

def create_orphan_pages_chart(orphan_data: Dict) -> go.Figure:
    """Create orphan pages visualization"""
    if 'orphan_pages' not in orphan_data or not orphan_data['orphan_pages']:
        return create_empty_chart("No orphan pages detected")
    
    # Take top 10 orphan pages by visits
    top_orphans = orphan_data['orphan_pages'][:10]
    
    urls = [page['url'] for page in top_orphans]
    visits = [page['visits'] for page in top_orphans]
    
    fig = go.Figure(data=[
        go.Bar(
            x=visits,
            y=urls,
            orientation='h',
            marker_color='#f59e0b',
            text=visits,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top Orphan Pages (No Internal Links)",
        xaxis_title="Number of Visits",
        yaxis_title="",
        height=CHART_HEIGHT,
        margin=dict(l=300)  # More space for URLs
    )
    
    return fig

def create_crawl_frequency_chart(freq_data: Dict) -> go.Figure:
    """Create crawl frequency chart"""
    if 'most_crawled' not in freq_data or not freq_data['most_crawled']:
        return create_empty_chart("No crawl frequency data available")
    
    # Get top 10 most crawled pages
    most_crawled = dict(list(freq_data['most_crawled'].items())[:10])
    
    urls = list(most_crawled.keys())
    crawl_counts = [data['crawl_count'] for data in most_crawled.values()]
    
    fig = go.Figure(data=[
        go.Bar(
            x=crawl_counts,
            y=urls,
            orientation='h',
            marker_color='#3b82f6',
            text=crawl_counts,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Most Frequently Crawled Pages",
        xaxis_title="Number of Crawls",
        yaxis_title="",
        height=CHART_HEIGHT,
        margin=dict(l=300)  # More space for URLs
    )
    
    return fig

def create_error_charts(error_data: Dict) -> Dict:
    """Create error analysis charts"""
    charts = {}
    
    if '404_errors' in error_data:
        charts['404_chart'] = create_404_chart(error_data)
    
    if '5xx_errors' in error_data:
        charts['5xx_chart'] = create_5xx_chart(error_data)
    
    if 'error_timeline' in error_data:
        charts['error_timeline'] = create_error_timeline(error_data)
    
    return charts

def create_404_chart(error_data: Dict) -> go.Figure:
    """Create 404 error chart"""
    if 'top_404_pages' not in error_data or not error_data['top_404_pages']:
        return create_empty_chart("No 404 errors detected")
    
    top_404s = dict(list(error_data['top_404_pages'].items())[:10])
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_404s.values()),
            y=list(top_404s.keys()),
            orientation='h',
            marker_color='#f59e0b',
            text=list(top_404s.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top 404 Error Pages",
        xaxis_title="Number of Errors",
        yaxis_title="",
        height=CHART_HEIGHT,
        margin=dict(l=300)
    )
    
    return fig

def create_5xx_chart(error_data: Dict) -> go.Figure:
    """Create 5xx error chart"""
    if 'top_5xx_pages' not in error_data or not error_data['top_5xx_pages']:
        return create_empty_chart("No 5xx errors detected")
    
    top_5xx = dict(list(error_data['top_5xx_pages'].items())[:10])
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(top_5xx.values()),
            y=list(top_5xx.keys()),
            orientation='h',
            marker_color='#ef4444',
            text=list(top_5xx.values()),
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Top 5xx Server Error Pages",
        xaxis_title="Number of Errors",
        yaxis_title="",
        height=CHART_HEIGHT,
        margin=dict(l=300)
    )
    
    return fig

def create_error_timeline(error_data: Dict) -> go.Figure:
    """Create error timeline chart"""
    # This would need actual timeline data from the error analysis
    # For now, return a placeholder
    return create_empty_chart("Error timeline data not available")

def create_empty_chart(message: str) -> go.Figure:
    """Create an empty chart with a message"""
    fig = go.Figure()
    
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=20, color="#999")
    )
    
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_heatmap(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str) -> go.Figure:
    """Create a heatmap visualization"""
    pivot_table = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc='sum')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        colorbar=dict(title=z_col)
    ))
    
    fig.update_layout(
        title=title,
        height=CHART_HEIGHT,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig

def create_sankey_diagram(source: List, target: List, value: List, title: str) -> go.Figure:
    """Create a Sankey diagram for flow visualization"""
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(set(source + target))
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(
        title=title,
        height=CHART_HEIGHT
    )
    
    return fig
