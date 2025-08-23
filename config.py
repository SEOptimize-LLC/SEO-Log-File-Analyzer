"""
Configuration settings for SEO Log File Analyzer
"""
import os
from datetime import timedelta

# App Settings
APP_NAME = "SEO Log File Analyzer"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "AI-Powered SEO Log Analysis Tool"

# Processing Settings
CHUNK_SIZE = 10000  # Rows to process at once
MAX_FILE_SIZE_MB = 1000  # Maximum file size in MB
SAMPLE_RATE = 0.1  # Sampling rate for large files (10%)
ENABLE_SAMPLING = True
MIN_ROWS_FOR_SAMPLING = 100000

# Cache Settings
ENABLE_CACHE = True
CACHE_TTL = 3600  # Cache duration in seconds
MAX_CACHE_SIZE_MB = 512

# Analysis Settings
TIME_ZONE = 'UTC'
DEFAULT_DATE_RANGE = 30  # Days
SESSION_TIMEOUT = timedelta(minutes=30)
MIN_BOT_CONFIDENCE = 0.7  # Minimum confidence for bot detection

# Performance Thresholds
SLOW_RESPONSE_MS = 1000  # Milliseconds
CRITICAL_RESPONSE_MS = 3000
ERROR_RATE_THRESHOLD = 0.05  # 5%
TTFB_WARNING_MS = 600
TTFB_CRITICAL_MS = 1200

# SEO Settings
CRAWL_BUDGET_EFFICIENCY_TARGET = 0.8  # 80%
ORPHAN_PAGE_THRESHOLD = 30  # Days without crawl
MOBILE_FIRST_PRIORITY = True
MIN_CRAWL_FREQUENCY = 7  # Days

# Bot Detection
KNOWN_BOTS = {
    'google': ['Googlebot', 'Googlebot-Image', 'Googlebot-News', 'Googlebot-Video'],
    'bing': ['bingbot', 'msnbot', 'BingPreview'],
    'yandex': ['YandexBot', 'YandexImages', 'YandexVideo'],
    'baidu': ['Baiduspider', 'Baiduspider-image'],
    'facebook': ['facebookexternalhit', 'facebookcatalog'],
    'twitter': ['Twitterbot'],
    'linkedin': ['LinkedInBot'],
    'pinterest': ['Pinterestbot'],
    'ai_bots': ['GPTBot', 'ChatGPT-User', 'Claude-Web', 'anthropic-ai', 'CCBot'],
    'seo_tools': ['AhrefsBot', 'SemrushBot', 'MJ12bot', 'DotBot', 'DataForSeoBot'],
}

# Verified Bot IPs (Google)
GOOGLE_BOT_IPS = [
    '66.249.64.0/19',
    '66.249.64.0/27',
    '66.249.79.0/27',
    '66.249.79.0/24',
]

# Log Format Patterns
LOG_PATTERNS = {
    'apache_common': r'^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+|-)$',
    'apache_combined': r'^(\S+) \S+ \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+|-) "([^"]*)" "([^"]*)"$',
    'nginx': r'^(\S+) - \S+ \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+) (\S+)" (\d{3}) (\d+) "([^"]*)" "([^"]*)"',
    'iis': r'^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) (\S+) (\S+) (\S+) (\d+) (\S+) (\S+) (\S+) (\d{3})',
    'cloudfront': r'^(\d{4}-\d{2}-\d{2})\t(\d{2}:\d{2}:\d{2})\t(\S+)\t(\d+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\d{3})',
}

# Export Settings
EXPORT_FORMATS = ['CSV', 'Excel', 'JSON', 'PDF']
MAX_EXPORT_ROWS = 1000000
PDF_REPORT_TEMPLATE = 'executive_summary'

# Visualization Settings
CHART_HEIGHT = 400
CHART_WIDTH = 800
COLOR_SCHEME = {
    '2xx': '#10b981',  # Green
    '3xx': '#3b82f6',  # Blue
    '4xx': '#f59e0b',  # Orange
    '5xx': '#ef4444',  # Red
    'bot': '#8b5cf6',   # Purple
    'human': '#06b6d4',  # Cyan
}

# ML Model Settings
MODEL_UPDATE_FREQUENCY = 7  # Days
MIN_TRAINING_SAMPLES = 1000
ANOMALY_DETECTION_SENSITIVITY = 0.95
FORECAST_HORIZON_DAYS = 30

# Alert Thresholds
ALERTS = {
    'error_spike': {'threshold': 0.1, 'window': '5min'},
    'bot_spike': {'threshold': 2.0, 'window': '1hour'},
    'performance_degradation': {'threshold': 1.5, 'window': '15min'},
    'crawl_drop': {'threshold': 0.5, 'window': '1day'},
}

# Session State Keys
SESSION_KEYS = {
    'data': 'log_data',
    'processed': 'processed_data',
    'cache': 'cache_store',
    'filters': 'active_filters',
    'date_range': 'selected_date_range',
    'bot_model': 'bot_classifier',
    'predictions': 'forecast_data',
}
