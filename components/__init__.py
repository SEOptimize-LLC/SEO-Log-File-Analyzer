# components/__init__.py
"""
SEO Log File Analyzer Components

This package contains core analysis components for the SEO Log File Analyzer.
"""

# Import main components for easy access
try:
    from .log_parser import LogParser
    from .bot_detector import BotDetector
    from .seo_analyzer import SEOAnalyzer
    from .cache_manager import CacheManager
    from .performance_analyzer import PerformanceAnalyzer
    from .visualizations import create_overview_charts
    
    __all__ = [
        'LogParser',
        'BotDetector', 
        'SEOAnalyzer',
        'CacheManager',
        'PerformanceAnalyzer',
        'create_overview_charts'
    ]
except ImportError as e:
    # Handle missing modules gracefully during development
    print(f"Warning: Some components could not be imported: {e}")
    __all__ = []

__version__ = "1.0.0"
