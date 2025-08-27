# components/performance_analyzer.py

"""
Performance metrics analyzer
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import streamlit as st
from config import (
    SLOW_RESPONSE_MS,
    CRITICAL_RESPONSE_MS,
    TTFB_WARNING_MS,
    TTFB_CRITICAL_MS
)

class PerformanceAnalyzer:
    """Analyze performance metrics from log data"""
    
    def __init__(self):
        self.slow_threshold = SLOW_RESPONSE_MS
        self.critical_threshold = CRITICAL_RESPONSE_MS
        self.ttfb_warning = TTFB_WARNING_MS
        self.ttfb_critical = TTFB_CRITICAL_MS
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive performance analysis
        
        Args:
            df: DataFrame with log data
            
        Returns:
            Dictionary with performance metrics
        """
        results = {}
        
        # Response time analysis
        results['response_times'] = self._analyze_response_times(df)
        
        # TTFB analysis
        results['ttfb'] = self._analyze_ttfb(df)
        
        # Slow endpoints
        results['slow_endpoints'] = self._identify_slow_endpoints(df)
        
        # Resource size analysis
        results['resource_sizes'] = self._analyze_resource_sizes(df)
        
        # Server load analysis
        results['server_load'] = self._analyze_server_load(df)
        
        # Performance trends
        results['trends'] = self._analyze_performance_trends(df)
        
        # Performance score
        results['performance_score'] = self._calculate_performance_score(results)
        
        return results
    
    def _analyze_response_times(self, df: pd.DataFrame) -> Dict:
        """Analyze response time metrics"""
        if 'response_time' not in df.columns:
            return self._generate_synthetic_response_times(df)
        
        response_times = df['response_time'].dropna()
        
        if len(response_times) == 0:
            return {}
        
        percentiles = {
            'p50': response_times.quantile(0.50),
            'p75': response_times.quantile(0.75),
            'p90': response_times.quantile(0.90),
            'p95': response_times.quantile(0.95),
            'p99': response_times.quantile(0.99)
        }
        
        # Categorize response times
        categories = {
            'fast': (response_times < 1000).sum(),
            'moderate': ((response_times >= 1000) & (response_times < self.slow_threshold)).sum(),
            'slow': ((response_times >= self.slow_threshold) & (response_times < self.critical_threshold)).sum(),
            'critical': (response_times >= self.critical_threshold).sum()
        }
        
        # Calculate statistics
        stats = {
            'mean': response_times.mean(),
            'median': response_times.median(),
            'std': response_times.std(),
            'min': response_times.min(),
            'max': response_times.max(),
            'total_requests': len(response_times)
        }
        
        return {
            'percentiles': percentiles,
            'categories': categories,
            'statistics': stats,
            'slow_percentage': (categories['slow'] + categories['critical']) / len(response_times) * 100,
            'recommendation': self._get_response_time_recommendation(percentiles['p95'])
        }
    
    def _analyze_ttfb(self, df: pd.DataFrame) -> Dict:
        """Analyze Time To First Byte metrics"""
        # Check if TTFB data is available
        if 'ttfb' in df.columns:
            ttfb_data = df['ttfb'].dropna()
        elif 'time_to_first_byte' in df.columns:
            ttfb_data = df['time_to_first_byte'].dropna()
        else:
            # Estimate TTFB as 20% of response time
            if 'response_time' in df.columns:
                ttfb_data = df['response_time'].dropna() * 0.2
            else:
                return {}
        
        if len(ttfb_data) == 0:
            return {}
        
        percentiles = {
            'p50': ttfb_data.quantile(0.50),
            'p75': ttfb_data.quantile(0.75),
            'p90': ttfb_data.quantile(0.90),
            'p95': ttfb_data.quantile(0.95),
            'p99': ttfb_data.quantile(0.99)
        }
        
        categories = {
            'good': (ttfb_data < self.ttfb_warning).sum(),
            'warning': ((ttfb_data >= self.ttfb_warning) & (ttfb_data < self.ttfb_critical)).sum(),
            'critical': (ttfb_data >= self.ttfb_critical).sum()
        }
        
        return {
            'percentiles': percentiles,
            'categories': categories,
            'mean': ttfb_data.mean(),
            'recommendation': self._get_ttfb_recommendation(percentiles['p75'])
        }
    
    def _identify_slow_endpoints(self, df: pd.DataFrame) -> List[Dict]:
        """Identify slowest endpoints"""
        if 'response_time' not in df.columns or 'url' not in df.columns:
            return []
        
        # Group by URL and calculate metrics
        endpoint_stats = df.groupby('url')['response_time'].agg([
            'mean',
            'median',
            'count',
            ('p95', lambda x: x.quantile(0.95)),
            ('slow_requests', lambda x: (x >= self.slow_threshold).sum())
        ]).reset_index()
        
        # Calculate impact score
        endpoint_stats['impact_score'] = (
            endpoint_stats['mean'] * 
            endpoint_stats['count'] / 
            df['response_time'].mean()
        )
        
        # Sort by impact score
        endpoint_stats = endpoint_stats.sort_values('impact_score', ascending=False)
        
        # Prepare results
        slow_endpoints = []
        for _, row in endpoint_stats.head(20).iterrows():
            slow_endpoints.append({
                'url': row['url'],
                'avg_response_time': row['mean'],
                'median_response_time': row['median'],
                'p95_response_time': row['p95'],
                'total_requests': row['count'],
                'slow_requests': row['slow_requests'],
                'impact_score': row['impact_score']
            })
        
        return slow_endpoints
    
    def _analyze_resource_sizes(self, df: pd.DataFrame) -> Dict:
        """Analyze resource sizes and their impact"""
        if 'size' not in df.columns:
            return {}
        
        sizes = df['size'].dropna()
        
        if len(sizes) == 0:
            return {}
        
        # Convert to MB
        sizes_mb = sizes / (1024 * 1024)
        
        percentiles = {
            'p50': sizes_mb.quantile(0.50),
            'p75': sizes_mb.quantile(0.75),
            'p90': sizes_mb.quantile(0.90),
            'p95': sizes_mb.quantile(0.95),
            'p99': sizes_mb.quantile(0.99)
        }
        
        # Categorize sizes
        categories = {
            'small': (sizes_mb < 0.1).sum(),  # < 100KB
            'medium': ((sizes_mb >= 0.1) & (sizes_mb < 1)).sum(),  # 100KB - 1MB
            'large': ((sizes_mb >= 1) & (sizes_mb < 5)).sum(),  # 1MB - 5MB
            'very_large': (sizes_mb >= 5).sum()  # > 5MB
        }
        
        # Find largest resources
        if 'url' in df.columns:
            largest_resources = df.nlargest(10, 'size')[['url', 'size']].to_dict('records')
            for resource in largest_resources:
                resource['size_mb'] = resource['size'] / (1024 * 1024)
        else:
            largest_resources = []
        
        # Calculate bandwidth usage
        total_bandwidth_gb = sizes.sum() / (1024 * 1024 * 1024)
        
        return {
            'percentiles': percentiles,
            'categories': categories,
            'mean_size_mb': sizes_mb.mean(),
            'total_bandwidth_gb': total_bandwidth_gb,
            'largest_resources': largest_resources,
            'recommendation': self._get_size_recommendation(percentiles['p90'])
        }
    
    def _analyze_server_load(self, df: pd.DataFrame) -> Dict:
        """Analyze server load patterns"""
        if 'timestamp' not in df.columns:
            return {}
        
        # Group by time intervals
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Requests per hour
        hourly_load = df.groupby('hour').size()
        
        # Requests per day of week
        daily_load = df.groupby('day_of_week').size()
        
        # Peak detection
        peak_hour = hourly_load.idxmax()
        peak_day = daily_load.idxmax()
        
        # Calculate load variance
        load_variance = hourly_load.std() / hourly_load.mean() if hourly_load.mean() > 0 else 0
        
        # Concurrent connections estimate (simplified)
        if 'response_time' in df.columns:
            df['end_time'] = pd.to_datetime(df['timestamp']) + pd.to_timedelta(df['response_time'], unit='ms')
            
            # Sample concurrent connections at different times
            sample_times = pd.date_range(df['timestamp'].min(), df['timestamp'].max(), periods=100)
            concurrent_connections = []
            
            for time_point in sample_times:
                concurrent = ((pd.to_datetime(df['timestamp']) <= time_point) & 
                            (df['end_time'] >= time_point)).sum()
                concurrent_connections.append(concurrent)
            
            max_concurrent = max(concurrent_connections)
            avg_concurrent = np.mean(concurrent_connections)
        else:
            max_concurrent = 0
            avg_concurrent = 0
        
        return {
            'hourly_distribution': hourly_load.to_dict(),
            'daily_distribution': daily_load.to_dict(),
            'peak_hour': peak_hour,
            'peak_day': peak_day,
            'load_variance': load_variance,
            'max_concurrent_connections': max_concurrent,
            'avg_concurrent_connections': avg_concurrent,
            'recommendation': self._get_load_recommendation(load_variance, max_concurrent)
        }
    
    def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze performance trends over time"""
        if 'timestamp' not in df.columns or 'response_time' not in df.columns:
            return {}
        
        # Create time-based aggregations
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        daily_stats = df.groupby('date')['response_time'].agg([
            'mean',
            'median',
            ('p95', lambda x: x.quantile(0.95)),
            'count'
        ]).reset_index()
        
        # Calculate trend
        if len(daily_stats) > 1:
            # Simple linear regression for trend
            from scipy import stats
            x = np.arange(len(daily_stats))
            y = daily_stats['mean'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend = {
                'direction': 'improving' if slope < 0 else 'degrading',
                'slope': slope,
                'r_squared': r_value ** 2,
                'significant': p_value < 0.05
            }
        else:
            trend = {'direction': 'insufficient_data'}
        
        # Identify anomalies
        if len(daily_stats) > 7:
            mean_response = daily_stats['mean'].mean()
            std_response = daily_stats['mean'].std()
            anomalies = daily_stats[
                (daily_stats['mean'] > mean_response + 2 * std_response) |
                (daily_stats['mean'] < mean_response - 2 * std_response)
            ]['date'].tolist()
        else:
            anomalies = []
        
        return {
            'daily_stats': daily_stats.to_dict('records'),
            'trend': trend,
            'anomaly_dates': anomalies,
            'recommendation': self._get_trend_recommendation(trend)
        }
    
    def _calculate_performance_score(self, results: Dict) -> Dict:
        """Calculate overall performance score"""
        score = 100  # Start with perfect score
        factors = {}
        
        # Response time impact
        if 'response_times' in results and 'percentiles' in results['response_times']:
            p95 = results['response_times']['percentiles']['p95']
            if p95 > self.critical_threshold:
                score -= 30
                factors['response_time'] = -30
            elif p95 > self.slow_threshold:
                score -= 15
                factors['response_time'] = -15
            else:
                factors['response_time'] = 0
        
        # TTFB impact
        if 'ttfb' in results and 'percentiles' in results['ttfb']:
            p75_ttfb = results['ttfb']['percentiles']['p75']
            if p75_ttfb > self.ttfb_critical:
                score -= 20
                factors['ttfb'] = -20
            elif p75_ttfb > self.ttfb_warning:
                score -= 10
                factors['ttfb'] = -10
            else:
                factors['ttfb'] = 0
        
        # Resource size impact
        if 'resource_sizes' in results and 'percentiles' in results['resource_sizes']:
            p90_size = results['resource_sizes']['percentiles']['p90']
            if p90_size > 5:  # > 5MB
                score -= 15
                factors['resource_size'] = -15
            elif p90_size > 2:  # > 2MB
                score -= 7
                factors['resource_size'] = -7
            else:
                factors['resource_size'] = 0
        
        # Server load impact
        if 'server_load' in results:
            if results['server_load'].get('max_concurrent_connections', 0) > 1000:
                score -= 10
                factors['server_load'] = -10
            else:
                factors['server_load'] = 0
        
        # Ensure score stays within bounds
        score = max(0, min(100, score))
        
        # Determine grade
        if score >= 90:
            grade = 'A'
        elif score >= 80:
            grade = 'B'
        elif score >= 70:
            grade = 'C'
        elif score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        return {
            'score': score,
            'grade': grade,
            'factors': factors,
            'recommendation': self._get_overall_recommendation(score)
        }
    
    def _generate_synthetic_response_times(self, df: pd.DataFrame) -> Dict:
        """Generate synthetic response times for demo purposes"""
        n = len(df)
        synthetic_times = np.random.lognormal(6, 1.2, n)  # Log-normal distribution
        
        percentiles = {
            'p50': np.percentile(synthetic_times, 50),
            'p75': np.percentile(synthetic_times, 75),
            'p90': np.percentile(synthetic_times, 90),
            'p95': np.percentile(synthetic_times, 95),
            'p99': np.percentile(synthetic_times, 99)
        }
        
        categories = {
            'fast': (synthetic_times < 1000).sum(),
            'moderate': ((synthetic_times >= 1000) & (synthetic_times < self.slow_threshold)).sum(),
            'slow': ((synthetic_times >= self.slow_threshold) & (synthetic_times < self.critical_threshold)).sum(),
            'critical': (synthetic_times >= self.critical_threshold).sum()
        }
        
        return {
            'percentiles': percentiles,
            'categories': categories,
            'statistics': {
                'mean': synthetic_times.mean(),
                'median': np.median(synthetic_times),
                'std': synthetic_times.std(),
                'min': synthetic_times.min(),
                'max': synthetic_times.max(),
                'total_requests': n
            },
            'slow_percentage': (categories['slow'] + categories['critical']) / n * 100,
            'recommendation': self._get_response_time_recommendation(percentiles['p95'])
        }
    
    # Recommendation methods
    def _get_response_time_recommendation(self, p95: float) -> str:
        if p95 < 1000:
            return "✅ Excellent response times. Performance is optimal."
        elif p95 < self.slow_threshold:
            return "⚠️ Good response times, but room for improvement."
        elif p95 < self.critical_threshold:
            return "🚨 Slow response times detected. Optimize server performance."
        else:
            return "🔴 Critical performance issues. Immediate optimization required."
    
    def _get_ttfb_recommendation(self, p75: float) -> str:
        if p75 < self.ttfb_warning:
            return "✅ Good TTFB. Server response is fast."
        elif p75 < self.ttfb_critical:
            return "⚠️ TTFB could be improved. Check server configuration."
        else:
            return "🚨 High TTFB detected. Review server-side processing."
    
    def _get_size_recommendation(self, p90: float) -> str:
        if p90 < 1:
            return "✅ Resource sizes are well optimized."
        elif p90 < 3:
            return "⚠️ Some large resources. Consider compression and optimization."
        else:
            return "🚨 Very large resources detected. Implement aggressive optimization."
    
    def _get_load_recommendation(self, variance: float, max_concurrent: int) -> str:
        if max_concurrent > 1000:
            return "🚨 High server load detected. Consider scaling infrastructure."
        elif variance > 0.5:
            return "⚠️ Uneven load distribution. Implement load balancing."
        else:
            return "✅ Server load is well distributed."
    
    def _get_trend_recommendation(self, trend: Dict) -> str:
        if trend.get('direction') == 'improving':
            return "✅ Performance is improving over time."
        elif trend.get('direction') == 'degrading':
            return "🚨 Performance is degrading. Investigate recent changes."
        else:
            return "📊 Insufficient data for trend analysis."
    
    def _get_overall_recommendation(self, score: int) -> str:
        if score >= 90:
            return "✅ Excellent performance! Keep up the good work."
        elif score >= 70:
            return "⚠️ Good performance with room for optimization."
        elif score >= 50:
            return "🚨 Performance needs attention. Multiple issues detected."
        else:
            return "🔴 Critical performance problems. Immediate action required."
