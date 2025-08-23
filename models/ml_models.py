# models/ml_models.py

"""
Machine Learning models for SEO Log Analyzer
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
from typing import Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class BotClassifier:
    """Machine learning model for bot classification"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'request_rate',
            'unique_urls_ratio',
            'error_rate',
            'session_duration',
            'page_depth_avg',
            'night_activity_ratio',
            'weekend_activity_ratio',
            'user_agent_length'
        ]
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for bot classification
        
        Args:
            df: Raw log DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame()
        
        # Group by IP to calculate features
        for ip in df['ip'].unique():
            ip_data = df[df['ip'] == ip]
            
            # Calculate features
            feature_row = {
                'ip': ip,
                'request_rate': len(ip_data) / ((ip_data['timestamp'].max() - ip_data['timestamp'].min()).total_seconds() + 1),
                'unique_urls_ratio': ip_data['url'].nunique() / len(ip_data) if 'url' in ip_data.columns else 0,
                'error_rate': (ip_data['status'] >= 400).mean() if 'status' in ip_data.columns else 0,
                'session_duration': (ip_data['timestamp'].max() - ip_data['timestamp'].min()).total_seconds(),
                'page_depth_avg': ip_data['url'].str.count('/').mean() if 'url' in ip_data.columns else 0,
                'night_activity_ratio': ip_data['hour'].isin(range(0, 6)).mean() if 'hour' in ip_data.columns else 0,
                'weekend_activity_ratio': ip_data['is_weekend'].mean() if 'is_weekend' in ip_data.columns else 0,
                'user_agent_length': ip_data['user_agent'].str.len().mean() if 'user_agent' in ip_data.columns else 0
            }
            
            features = pd.concat([features, pd.DataFrame([feature_row])], ignore_index=True)
        
        return features
    
    def train(self, df: pd.DataFrame, labels: Optional[pd.Series] = None):
        """
        Train the bot classifier
        
        Args:
            df: DataFrame with log data
            labels: Optional labels for supervised learning
        """
        # Prepare features
        features_df = self.prepare_features(df)
        
        # Use provided labels or generate synthetic ones for demo
        if labels is None:
            # Generate synthetic labels based on heuristics
            labels = self._generate_synthetic_labels(features_df)
        
        # Prepare training data
        X = features_df[self.feature_columns].fillna(0)
        y = labels
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = (y_pred == y_test).mean()
        
        self.is_trained = True
        st.success(f"Bot classifier trained with {accuracy:.2%} accuracy")
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Predict bot classification
        
        Args:
            df: DataFrame with log data
            
        Returns:
            Series with bot predictions
        """
        if not self.is_trained:
            st.warning("Model not trained. Using heuristic-based detection.")
            return self._heuristic_detection(df)
        
        # Prepare features
        features_df = self.prepare_features(df)
        X = features_df[self.feature_columns].fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Map predictions back to original DataFrame
        ip_to_prediction = dict(zip(features_df['ip'], predictions))
        return df['ip'].map(ip_to_prediction).fillna(False)
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction probabilities
        
        Args:
            df: DataFrame with log data
            
        Returns:
            DataFrame with bot probabilities
        """
        if not self.is_trained:
            return pd.DataFrame({'bot_probability': [0.5] * len(df)})
        
        # Prepare features
        features_df = self.prepare_features(df)
        X = features_df[self.feature_columns].fillna(0)
        
        # Scale and predict probabilities
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Map probabilities back
        ip_to_prob = dict(zip(features_df['ip'], probabilities[:, 1]))
        return df['ip'].map(ip_to_prob).fillna(0.5)
    
    def _generate_synthetic_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate synthetic labels for training"""
        # Simple heuristic for synthetic labels
        is_bot = (
            (features_df['request_rate'] > 1) |
            (features_df['unique_urls_ratio'] < 0.1) |
            (features_df['error_rate'] > 0.3) |
            (features_df['night_activity_ratio'] > 0.5)
        )
        return is_bot.astype(int)
    
    def _heuristic_detection(self, df: pd.DataFrame) -> pd.Series:
        """Fallback heuristic-based bot detection"""
        features_df = self.prepare_features(df)
        is_bot = self._generate_synthetic_labels(features_df)
        ip_to_bot = dict(zip(features_df['ip'], is_bot))
        return df['ip'].map(ip_to_bot).fillna(False)
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'feature_importance': self.feature_importance
            }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            saved = joblib.load(filepath)
            self.model = saved['model']
            self.scaler = saved['scaler']
            self.feature_columns = saved['feature_columns']
            self.feature_importance = saved['feature_importance']
            self.is_trained = True
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")


class AnomalyDetector:
    """Anomaly detection for log data"""
    
    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for anomaly detection
        
        Args:
            df: Log DataFrame
            
        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame()
        
        # Time-based aggregations
        if 'timestamp' in df.columns:
            hourly_stats = df.groupby(df['timestamp'].dt.floor('H')).agg({
                'ip': 'count',
                'status': lambda x: (x >= 400).mean() if 'status' in df.columns else 0,
                'response_time': 'mean' if 'response_time' in df.columns else lambda x: 0,
                'size': 'mean' if 'size' in df.columns else lambda x: 0
            }).reset_index()
            
            hourly_stats.columns = ['hour', 'request_count', 'error_rate', 'avg_response_time', 'avg_size']
            features = hourly_stats
        
        return features
    
    def train(self, df: pd.DataFrame):
        """
        Train anomaly detector
        
        Args:
            df: Log DataFrame
        """
        features = self.prepare_features(df)
        
        if features.empty:
            st.warning("Insufficient data for anomaly detection training")
            return
        
        # Select numeric columns only
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        st.success("Anomaly detector trained successfully")
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in log data
        
        Args:
            df: Log DataFrame
            
        Returns:
            DataFrame with anomaly flags
        """
        if not self.is_trained:
            st.warning("Anomaly detector not trained")
            return df
        
        features = self.prepare_features(df)
        
        if features.empty:
            return df
        
        # Select numeric columns
        numeric_features = features.select_dtypes(include=[np.number]).fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(numeric_features)
        predictions = self.model.predict(X_scaled)
        
        # Add anomaly flag
        features['is_anomaly'] = predictions == -1
        features['anomaly_score'] = self.model.score_samples(X_scaled)
        
        # Merge back to original DataFrame
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.floor('H')
            df = df.merge(
                features[['hour', 'is_anomaly', 'anomaly_score']],
                on='hour',
                how='left'
            )
        
        return df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of detected anomalies
        
        Args:
            df: DataFrame with anomaly detection results
            
        Returns:
            Summary dictionary
        """
        if 'is_anomaly' not in df.columns:
            return {}
        
        anomalies = df[df['is_anomaly'] == True]
        
        summary = {
            'total_anomalies': len(anomalies),
            'anomaly_percentage': len(anomalies) / len(df) * 100 if len(df) > 0 else 0,
            'anomaly_hours': anomalies['hour'].unique().tolist() if 'hour' in anomalies.columns else [],
            'avg_anomaly_score': anomalies['anomaly_score'].mean() if 'anomaly_score' in anomalies.columns else 0
        }
        
        # Identify anomaly patterns
        if len(anomalies) > 0:
            if 'status' in anomalies.columns:
                summary['anomaly_status_codes'] = anomalies['status'].value_counts().head(5).to_dict()
            
            if 'url' in anomalies.columns:
                summary['anomaly_urls'] = anomalies['url'].value_counts().head(5).to_dict()
        
        return summary
