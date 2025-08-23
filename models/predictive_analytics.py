# models/predictive_analytics.py

"""
Predictive analytics models for SEO Log Analyzer
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import streamlit as st
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CrawlPredictor:
    """Predict future crawl patterns"""
    
    def __init__(self):
        self.models = {}
        self.forecast_horizon = 30  # days
        self.is_trained = False
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for crawl prediction
        
        Args:
            df: Log DataFrame with bot traffic
            
        Returns:
            Prepared DataFrame for Prophet
        """
        if 'timestamp' not in df.columns or 'is_bot' not in df.columns:
            return pd.DataFrame()
        
        # Filter for bot traffic
        bot_df = df[df['is_bot'] == True]
        
        # Aggregate by day
        daily_crawls = bot_df.groupby(bot_df['timestamp'].dt.date).size().reset_index()
        daily_crawls.columns = ['ds', 'y']
        
        return daily_crawls
    
    def train(self, df: pd.DataFrame, bot_type: str = 'all'):
        """
        Train crawl prediction model
        
        Args:
            df: Log DataFrame
            bot_type: Type of bot to predict ('all', 'google', 'bing', etc.)
        """
        # Prepare data
        if bot_type != 'all' and 'bot_type' in df.columns:
            df = df[df['bot_type'] == bot_type]
        
        prophet_data = self.prepare_data(df)
        
        if len(prophet_data) < 7:
            st.warning("Insufficient data for crawl prediction (need at least 7 days)")
            return
        
        # Initialize and train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonalities if enough data
        if len(prophet_data) > 30:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
        
        model.fit(prophet_data)
        self.models[bot_type] = model
        self.is_trained = True
        
        st.success(f"Crawl predictor trained for {bot_type} bots")
    
    def predict(self, periods: int = None, bot_type: str = 'all') -> pd.DataFrame:
        """
        Predict future crawl patterns
        
        Args:
            periods: Number of days to predict
            bot_type: Type of bot
            
        Returns:
            DataFrame with predictions
        """
        if bot_type not in self.models:
            st.warning(f"No model trained for {bot_type} bots")
            return pd.DataFrame()
        
        periods = periods or self.forecast_horizon
        model = self.models[bot_type]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Make predictions
        forecast = model.predict(future)
        
        # Add metadata
        forecast['bot_type'] = bot_type
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'bot_type']]
    
    def get_crawl_insights(self, forecast: pd.DataFrame) -> Dict:
        """
        Extract insights from crawl predictions
        
        Args:
            forecast: Forecast DataFrame
            
        Returns:
            Insights dictionary
        """
        if forecast.empty:
            return {}
        
        # Future predictions only
        today = pd.Timestamp.now().date()
        future_forecast = forecast[forecast['ds'].dt.date > today]
        
        if future_forecast.empty:
            return {}
        
        insights = {
            'avg_daily_crawls': future_forecast['yhat'].mean(),
            'peak_crawl_day': future_forecast.loc[future_forecast['yhat'].idxmax(), 'ds'],
            'min_crawl_day': future_forecast.loc[future_forecast['yhat'].idxmin(), 'ds'],
            'trend': 'increasing' if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else 'decreasing',
            'weekly_pattern': self._detect_weekly_pattern(future_forecast)
        }
        
        return insights
    
    def _detect_weekly_pattern(self, forecast: pd.DataFrame) -> Dict:
        """Detect weekly crawl patterns"""
        if 'ds' not in forecast.columns:
            return {}
        
        forecast['day_of_week'] = pd.to_datetime(forecast['ds']).dt.dayofweek
        weekly_avg = forecast.groupby('day_of_week')['yhat'].mean()
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pattern = {days[i]: round(avg, 2) for i, avg in weekly_avg.items()}
        
        return pattern


class PerformanceForecaster:
    """Forecast performance metrics"""
    
    def __init__(self):
        self.models = {}
        self.metrics = ['response_time', 'error_rate', 'traffic_volume']
        self.forecast_horizon = 7  # days
    
    def prepare_data(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Prepare data for performance forecasting
        
        Args:
            df: Log DataFrame
            metric: Metric to forecast
            
        Returns:
            Prepared DataFrame
        """
        if 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        # Aggregate by hour
        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        
        if metric == 'response_time' and 'response_time' in df.columns:
            hourly_data = df.groupby('hour')['response_time'].mean().reset_index()
            hourly_data.columns = ['ds', 'y']
        elif metric == 'error_rate' and 'status' in df.columns:
            hourly_data = df.groupby('hour')['status'].apply(
                lambda x: (x >= 400).mean() * 100
            ).reset_index()
            hourly_data.columns = ['ds', 'y']
        elif metric == 'traffic_volume':
            hourly_data = df.groupby('hour').size().reset_index(name='y')
            hourly_data.columns = ['ds', 'y']
        else:
            return pd.DataFrame()
        
        return hourly_data
    
    def train(self, df: pd.DataFrame, metric: str = 'response_time'):
        """
        Train performance forecasting model
        
        Args:
            df: Log DataFrame
            metric: Metric to forecast
        """
        prophet_data = self.prepare_data(df, metric)
        
        if len(prophet_data) < 48:  # At least 48 hours of data
            st.warning(f"Insufficient data for {metric} forecasting")
            return
        
        # Initialize Prophet with appropriate parameters
        if metric == 'response_time':
            model = Prophet(
                changepoint_prior_scale=0.1,
                seasonality_prior_scale=10
            )
        elif metric == 'error_rate':
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=5
            )
        else:  # traffic_volume
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True
            )
        
        model.fit(prophet_data)
        self.models[metric] = model
        
        st.success(f"Performance forecaster trained for {metric}")
    
    def predict(self, metric: str = 'response_time', periods_hours: int = 168) -> pd.DataFrame:
        """
        Predict future performance metrics
        
        Args:
            metric: Metric to predict
            periods_hours: Number of hours to predict (default 1 week)
            
        Returns:
            Forecast DataFrame
        """
        if metric not in self.models:
            st.warning(f"No model trained for {metric}")
            return pd.DataFrame()
        
        model = self.models[metric]
        
        # Create future dataframe (hourly)
        future = model.make_future_dataframe(periods=periods_hours, freq='H')
        
        # Make predictions
        forecast = model.predict(future)
        
        # Add metric name
        forecast['metric'] = metric
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'metric']]
    
    def detect_anomalies(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Detect anomalies in performance metrics using forecast
        
        Args:
            df: Log DataFrame
            metric: Metric to check
            
        Returns:
            DataFrame with anomaly flags
        """
        if metric not in self.models:
            return df
        
        # Get historical forecast
        model = self.models[metric]
        historical_data = self.prepare_data(df, metric)
        
        if historical_data.empty:
            return df
        
        # Predict on historical data
        forecast = model.predict(historical_data[['ds']])
        
        # Calculate residuals
        residuals = historical_data['y'].values - forecast['yhat'].values
        threshold = 2 * np.std(residuals)
        
        # Flag anomalies
        historical_data['is_anomaly'] = np.abs(residuals) > threshold
        historical_data['anomaly_score'] = residuals / threshold
        
        # Merge back
        df = df.merge(
            historical_data[['ds', 'is_anomaly', 'anomaly_score']],
            left_on=pd.to_datetime(df['timestamp']).dt.floor('H'),
            right_on='ds',
            how='left'
        )
        
        return df
    
    def get_performance_trends(self, forecasts: Dict[str, pd.DataFrame]) -> Dict:
        """
        Analyze performance trends from forecasts
        
        Args:
            forecasts: Dictionary of metric: forecast DataFrame
            
        Returns:
            Trends analysis
        """
        trends = {}
        
        for metric, forecast in forecasts.items():
            if forecast.empty:
                continue
            
            # Analyze future predictions
            future = forecast[forecast['ds'] > pd.Timestamp.now()]
            
            if not future.empty:
                # Calculate trend
                start_value = future['yhat'].iloc[0]
                end_value = future['yhat'].iloc[-1]
                change_pct = ((end_value - start_value) / start_value) * 100
                
                trends[metric] = {
                    'direction': 'improving' if change_pct < 0 else 'degrading',
                    'change_percentage': abs(change_pct),
                    'peak_time': future.loc[future['yhat'].idxmax(), 'ds'],
                    'lowest_time': future.loc[future['yhat'].idxmin(), 'ds'],
                    'avg_predicted': future['yhat'].mean(),
                    'confidence_interval': (future['yhat_lower'].mean(), future['yhat_upper'].mean())
                }
        
        return trends
    
    def generate_alerts(self, forecast: pd.DataFrame, metric: str, thresholds: Dict) -> List[Dict]:
        """
        Generate alerts based on forecasted values
        
        Args:
            forecast: Forecast DataFrame
            metric: Metric name
            thresholds: Alert thresholds
            
        Returns:
            List of alerts
        """
        alerts = []
        
        # Check future predictions against thresholds
        future = forecast[forecast['ds'] > pd.Timestamp.now()]
        
        if metric in thresholds and not future.empty:
            threshold = thresholds[metric]
            
            # Check for threshold violations
            if metric == 'response_time':
                violations = future[future['yhat'] > threshold]
                if not violations.empty:
                    alerts.append({
                        'type': 'warning',
                        'metric': metric,
                        'message': f"Response time expected to exceed {threshold}ms",
                        'times': violations['ds'].tolist(),
                        'severity': 'high' if violations['yhat'].max() > threshold * 1.5 else 'medium'
                    })
            
            elif metric == 'error_rate':
                violations = future[future['yhat'] > threshold]
                if not violations.empty:
                    alerts.append({
                        'type': 'error',
                        'metric': metric,
                        'message': f"Error rate expected to exceed {threshold}%",
                        'times': violations['ds'].tolist(),
                        'severity': 'critical' if violations['yhat'].max() > threshold * 2 else 'high'
                    })
        
        return alerts
