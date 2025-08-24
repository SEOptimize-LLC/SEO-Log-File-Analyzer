# utils/data_processor.py

"""
Data processing utilities
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
from datetime import datetime, timedelta
import streamlit as st
from config import (
    CHUNK_SIZE,
    ENABLE_SAMPLING,
    SAMPLE_RATE,
    MIN_ROWS_FOR_SAMPLING
)
from components.bot_detector import BotDetector

def process_log_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw log data with all enrichments
    
    Args:
        df: Raw log DataFrame
        
    Returns:
        Processed DataFrame with additional columns
    """
    with st.spinner("Processing log data..."):
        # Apply sampling if needed
        if ENABLE_SAMPLING and len(df) > MIN_ROWS_FOR_SAMPLING:
            df = apply_sampling(df, SAMPLE_RATE)
            st.info(f"Sampling applied: {len(df):,} rows")
        
        # Ensure timestamp column
        df = ensure_timestamp(df)
        
        # Add calculated fields
        df = add_calculated_fields(df)
        
        # Detect bots
        bot_detector = BotDetector()
        df = bot_detector.detect_bots(df)
        
        # Add session information
        df = identify_sessions(df)
        
        # Add geographic information (if IP available)
        if 'ip' in df.columns:
            df = add_geo_info(df)
        
        # Clean and standardize URLs
        if 'url' in df.columns:
            df = clean_urls(df)
        
        # Add time-based features
        df = add_time_features(df)
        
        return df

def apply_sampling(df: pd.DataFrame, sample_rate: float) -> pd.DataFrame:
    """
    Apply sampling to large datasets
    
    Args:
        df: Input DataFrame
        sample_rate: Sampling rate (0-1)
        
    Returns:
        Sampled DataFrame
    """
    if sample_rate >= 1:
        return df
    
    return df.sample(frac=sample_rate, random_state=42)

def chunk_dataframe(df: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> Generator:
    """
    Split DataFrame into chunks for processing
    
    Args:
        df: Input DataFrame
        chunk_size: Size of each chunk
        
    Yields:
        DataFrame chunks
    """
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        yield df.iloc[start:end]

def ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a proper timestamp column"""
    if 'timestamp' not in df.columns:
        # Try to find or create timestamp
        if 'date' in df.columns and 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        elif 'datetime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['datetime'])
        else:
            # Generate timestamps for demo
            df['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(days=30),
                periods=len(df),
                freq='T'  # Minute frequency
            )
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add calculated fields to DataFrame"""
    # Status category
    if 'status' in df.columns:
        df['status_category'] = df['status'].apply(lambda x: f"{x//100}xx" if pd.notna(x) else 'unknown')
        df['is_error'] = df['status'] >= 400
        df['is_redirect'] = df['status'].between(300, 399)
        df['is_success'] = df['status'].between(200, 299)
    
    # Size in MB
    if 'size' in df.columns:
        df['size_mb'] = df['size'] / (1024 * 1024)
        df['size_category'] = pd.cut(
            df['size_mb'],
            bins=[0, 0.1, 1, 5, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
    
    # Response time categories
    if 'response_time' in df.columns:
        df['response_category'] = pd.cut(
            df['response_time'],
            bins=[0, 1000, 3000, 5000, float('inf')],
            labels=['Fast', 'Moderate', 'Slow', 'Critical']
        )
    
    # URL depth (number of path segments)
    if 'url' in df.columns:
        df['url_depth'] = df['url'].str.count('/')
        df['has_query'] = df['url'].str.contains('\\?', na=False)
        df['file_extension'] = df['url'].str.extract(r'\.([a-zA-Z0-9]+)(?:\?|$)', expand=False)
    
    return df

def identify_sessions(df: pd.DataFrame, timeout_minutes: int = 30) -> pd.DataFrame:
    """
    Identify user sessions based on IP and time gaps
    
    Args:
        df: DataFrame with IP and timestamp columns
        timeout_minutes: Minutes of inactivity to define new session
        
    Returns:
        DataFrame with session information
    """
    if 'ip' not in df.columns or 'timestamp' not in df.columns:
        return df
    
    # Sort by IP and timestamp
    df = df.sort_values(['ip', 'timestamp'])
    
    # Calculate time differences within each IP
    df['time_diff'] = df.groupby('ip')['timestamp'].diff()
    
    # Mark new sessions
    df['new_session'] = (
        (df['time_diff'] > pd.Timedelta(minutes=timeout_minutes)) | 
        df['time_diff'].isna()
    )
    
    # Assign session IDs
    df['session_id'] = df.groupby('ip')['new_session'].cumsum()
    
    # Calculate session metrics
    session_stats = df.groupby(['ip', 'session_id']).agg({
        'timestamp': ['min', 'max', 'count']
    })
    session_stats.columns = ['session_start', 'session_end', 'page_views']
    session_stats['session_duration'] = (
        session_stats['session_end'] - session_stats['session_start']
    ).dt.total_seconds()
    
    # Merge session stats back
    df = df.merge(
        session_stats.reset_index()[['ip', 'session_id', 'session_duration', 'page_views']],
        on=['ip', 'session_id'],
        how='left'
    )
    
    # Clean up
    df = df.drop(['time_diff', 'new_session'], axis=1)
    
    return df

def add_geo_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geographic information based on IP
    
    Args:
        df: DataFrame with IP column
        
    Returns:
        DataFrame with geo information
    """
    # For demo purposes, add synthetic geo data
    # In production, you would use a GeoIP database
    
    if 'ip' in df.columns:
        # Simple IP-based country assignment (demo only)
        df['country'] = df['ip'].apply(lambda x: assign_country_demo(x))
        df['is_domestic'] = df['country'] == 'US'
    
    return df

def assign_country_demo(ip: str) -> str:
    """Demo function to assign country based on IP"""
    # This is a simplified demo - in production use GeoIP2 or similar
    ip_parts = ip.split('.')
    if ip_parts[0] == '192' or ip_parts[0] == '10':
        return 'US'
    elif ip_parts[0] == '172':
        return 'UK'
    elif ip_parts[0] == '203':
        return 'IN'
    else:
        countries = ['US', 'UK', 'DE', 'FR', 'JP', 'CN', 'BR', 'AU']
        return np.random.choice(countries)

def clean_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize URLs
    
    Args:
        df: DataFrame with URL column
        
    Returns:
        DataFrame with cleaned URLs
    """
    if 'url' not in df.columns:
        return df
    
    # Remove query parameters for grouping
    df['url_path'] = df['url'].str.split('?').str[0]
    
    # Extract query string
    df['query_string'] = df['url'].str.split('?').str[1]
    
    # Normalize trailing slashes
    df['url_normalized'] = df['url_path'].str.rstrip('/')
    df.loc[df['url_normalized'] == '', 'url_normalized'] = '/'
    
    # Extract domain if full URL
    if df['url'].str.contains('http', na=False).any():
        df['domain'] = df['url'].str.extract(r'https?://([^/]+)', expand=False)
    
    # Categorize URLs
    df['url_category'] = categorize_url(df['url_path'])
    
    return df

def categorize_url(url_series: pd.Series) -> pd.Series:
    """Categorize URLs based on patterns"""
    categories = []
    
    for url in url_series:
        if pd.isna(url):
            categories.append('unknown')
        elif url == '/' or url == '/index' or url == '/home':
            categories.append('homepage')
        elif '/product' in url or '/item' in url:
            categories.append('product')
        elif '/category' in url or '/collection' in url:
            categories.append('category')
        elif '/blog' in url or '/article' in url or '/post' in url:
            categories.append('content')
        elif '/api' in url:
            categories.append('api')
        elif '/admin' in url or '/wp-admin' in url:
            categories.append('admin')
        elif '/search' in url:
            categories.append('search')
        elif '/cart' in url or '/checkout' in url:
            categories.append('commerce')
        elif '.css' in url or '.js' in url or '.jpg' in url or '.png' in url:
            categories.append('static')
        else:
            categories.append('other')
    
    return pd.Series(categories)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        DataFrame with time features
    """
    if 'timestamp' not in df.columns:
        return df
    
    # Extract time components
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.isocalendar().week
    df['month'] = df['timestamp'].dt.month
    
    # Business hours flag
    df['is_business_hours'] = (
        (df['hour'] >= 9) & (df['hour'] <= 17) & 
        (df['day_of_week'] < 5)  # Monday to Friday
    )
    
    # Peak hours flag
    df['is_peak_hour'] = df['hour'].isin([10, 11, 14, 15, 16])
    
    # Weekend flag
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    return df

def aggregate_metrics(df: pd.DataFrame, group_by: List[str], metrics: Dict) -> pd.DataFrame:
    """
    Aggregate metrics by specified columns
    
    Args:
        df: Input DataFrame
        group_by: Columns to group by
        metrics: Dictionary of column: aggregation function
        
    Returns:
        Aggregated DataFrame
    """
    return df.groupby(group_by).agg(metrics).reset_index()

def calculate_percentiles(series: pd.Series, percentiles: List[float] = [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]) -> Dict:
    """
    Calculate percentiles for a series
    
    Args:
        series: Pandas Series
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary of percentile values
    """
    result = {}
    for p in percentiles:
        result[f'p{int(p*100)}'] = series.quantile(p)
    return result

def detect_anomalies(df: pd.DataFrame, column: str, threshold: float = 3) -> pd.DataFrame:
    """
    Detect anomalies using z-score method
    
    Args:
        df: Input DataFrame
        column: Column to check for anomalies
        threshold: Z-score threshold
        
    Returns:
        DataFrame with anomaly flag
    """
    if column not in df.columns:
        return df
    
    mean = df[column].mean()
    std = df[column].std()
    
    df[f'{column}_zscore'] = (df[column] - mean) / std
    df[f'{column}_anomaly'] = abs(df[f'{column}_zscore']) > threshold
    
    return df

def calculate_rolling_stats(df: pd.DataFrame, column: str, window: str = '1H') -> pd.DataFrame:
    """
    Calculate rolling statistics
    
    Args:
        df: Input DataFrame with timestamp
        column: Column to calculate stats for
        window: Rolling window size
        
    Returns:
        DataFrame with rolling stats
    """
    if 'timestamp' not in df.columns or column not in df.columns:
        return df
    
    df = df.sort_values('timestamp')
    df = df.set_index('timestamp')
    
    df[f'{column}_rolling_mean'] = df[column].rolling(window).mean()
    df[f'{column}_rolling_std'] = df[column].rolling(window).std()
    df[f'{column}_rolling_max'] = df[column].rolling(window).max()
    df[f'{column}_rolling_min'] = df[column].rolling(window).min()
    
    df = df.reset_index()
    
    return df
