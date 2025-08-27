# components/bot_detector.py

"""
ML-powered bot detection system
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import dns.resolver
import dns.reversename
from typing import Dict, List, Tuple, Optional
import streamlit as st
from user_agents import parse
import httpagentparser
from config import KNOWN_BOTS, GOOGLE_BOT_IPS, MIN_BOT_CONFIDENCE
import concurrent.futures
import threading
from functools import lru_cache
import time

class BotDetector:
    """Advanced bot detection using ML and verification techniques"""
    
    def __init__(self):
        self.known_bots = KNOWN_BOTS
        self.google_ips = GOOGLE_BOT_IPS
        self.model = None
        self.scaler = StandardScaler()
        self.bot_patterns = self._compile_bot_patterns()
        self.dns_cache = {}  # Cache for DNS verification results
        self.dns_timeout = 2.0  # DNS timeout in seconds
        
    def detect_bots(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect bots in log data
        
        Args:
            df: DataFrame with log data
            
        Returns:
            DataFrame with bot detection columns added
        """
        # Method 1: User agent pattern matching
        df['bot_ua_match'] = df['user_agent'].apply(self._check_user_agent) if 'user_agent' in df.columns else False
        
        # Method 2: Behavioral analysis
        if all(col in df.columns for col in ['ip', 'timestamp']):
            df = self._analyze_behavior(df)
        
        # Method 3: DNS verification for claimed bots (optimized batch processing)
        if 'ip' in df.columns and 'user_agent' in df.columns:
            df['bot_verified'] = self._batch_verify_bots(df)
        
        # Method 4: ML-based detection
        if self.model is not None:
            df = self._ml_detection(df)
        
        # Combine all methods
        df['is_bot'] = df[['bot_ua_match', 'bot_behavior', 'bot_verified']].any(axis=1) if 'bot_behavior' in df.columns else df['bot_ua_match']
        
        # Classify bot types
        df['bot_type'] = df.apply(self._classify_bot_type, axis=1) if 'user_agent' in df.columns else 'unknown'
        
        # Calculate confidence score
        df['bot_confidence'] = self._calculate_confidence(df)
        
        return df
    
    def _compile_bot_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for bot detection"""
        patterns = {}
        
        # Create patterns for each bot category
        for category, bots in self.known_bots.items():
            pattern = '|'.join([re.escape(bot) for bot in bots])
            patterns[category] = re.compile(pattern, re.IGNORECASE)
        
        # General bot pattern
        patterns['general'] = re.compile(
            r'bot|crawl|spider|scraper|scan|fetch|check|monitor|test|ping',
            re.IGNORECASE
        )
        
        return patterns
    
    def _check_user_agent(self, user_agent: str) -> bool:
        """Check if user agent matches known bot patterns"""
        if pd.isna(user_agent):
            return False
        
        # Check against compiled patterns
        for pattern in self.bot_patterns.values():
            if pattern.search(user_agent):
                return True
        
        # Parse user agent
        try:
            ua = parse(user_agent)
            if ua.is_bot:
                return True
        except:
            pass
        
        return False
    
    def _analyze_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze request behavior to identify bots"""
        # Calculate request patterns per IP
        ip_stats = df.groupby('ip').agg({
            'timestamp': ['count', lambda x: (x.max() - x.min()).total_seconds()],
            'url': 'nunique',
            'status': lambda x: (x >= 400).mean()
        }).reset_index()
        
        ip_stats.columns = ['ip', 'request_count', 'session_duration', 'unique_urls', 'error_rate']
        
        # Calculate request rate
        ip_stats['request_rate'] = ip_stats['request_count'] / (ip_stats['session_duration'] + 1)
        
        # Bot behavior indicators
        ip_stats['bot_behavior'] = (
            (ip_stats['request_rate'] > 1) |  # More than 1 request per second
            (ip_stats['request_count'] > 1000) |  # High volume
            (ip_stats['unique_urls'] / ip_stats['request_count'] < 0.1) |  # Repetitive requests
            (ip_stats['error_rate'] > 0.5)  # High error rate
        )
        
        # Merge back to original DataFrame
        df = df.merge(ip_stats[['ip', 'bot_behavior']], on='ip', how='left')
        
        return df

    def _batch_verify_bots(self, df: pd.DataFrame) -> pd.Series:
        """
        Batch verify bots using optimized DNS lookups

        Args:
            df: DataFrame with ip and user_agent columns

        Returns:
            Series of boolean verification results
        """
        # Filter to only suspected bots to reduce DNS lookups
        suspected_bots = df[df['bot_ua_match'] == True].copy()

        if suspected_bots.empty:
            return pd.Series(False, index=df.index)

        # Group by IP and user agent type for batching
        google_ips = []
        bing_ips = []
        other_bots = []

        for idx, row in suspected_bots.iterrows():
            ip = row.get('ip', '')
            ua = row.get('user_agent', '').lower()

            if 'googlebot' in ua:
                google_ips.append(ip)
            elif 'bingbot' in ua:
                bing_ips.append(ip)
            else:
                other_bots.append((ip, ua))

        # Batch verify Google bots
        google_results = self._batch_verify_googlebots(google_ips) if google_ips else {}

        # Batch verify Bing bots
        bing_results = self._batch_verify_bingbots(bing_ips) if bing_ips else {}

        # For other bots, trust UA (no DNS verification needed)
        other_results = {ip: True for ip, ua in other_bots}

        # Combine results
        verification_results = {**google_results, **bing_results, **other_results}

        # Create result series
        results = []
        for idx, row in df.iterrows():
            ip = row.get('ip', '')
            if ip in verification_results:
                results.append(verification_results[ip])
            else:
                results.append(False)

        return pd.Series(results, index=df.index)

    @lru_cache(maxsize=1000)
    def _cached_dns_lookup(self, ip: str, bot_type: str) -> bool:
        """
        Cached DNS lookup with timeout

        Args:
            ip: IP address to verify
            bot_type: 'google' or 'bing'

        Returns:
            True if verified as legitimate bot
        """
        if ip in self.dns_cache:
            return self.dns_cache[ip]

        try:
            # Set DNS resolver timeout
            resolver = dns.resolver.Resolver()
            resolver.timeout = self.dns_timeout
            resolver.lifetime = self.dns_timeout

            rev_name = dns.reversename.from_address(ip)
            ptr_records = resolver.resolve(rev_name, 'PTR')

            for ptr in ptr_records:
                hostname = str(ptr).lower()

                if bot_type == 'google':
                    if hostname.endswith('.googlebot.com.') or hostname.endswith('.google.com.'):
                        forward_records = resolver.resolve(hostname.rstrip('.'), 'A')
                        for forward in forward_records:
                            if str(forward) == ip:
                                self.dns_cache[ip] = True
                                return True
                elif bot_type == 'bing':
                    if hostname.endswith('.search.msn.com.'):
                        forward_records = resolver.resolve(hostname.rstrip('.'), 'A')
                        for forward in forward_records:
                            if str(forward) == ip:
                                self.dns_cache[ip] = True
                                return True

        except (dns.resolver.Timeout, dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            # DNS timeout or no record - cache as False
            self.dns_cache[ip] = False
            return False
        except Exception:
            # Other DNS errors - cache as False
            self.dns_cache[ip] = False
            return False

        self.dns_cache[ip] = False
        return False

    def _batch_verify_googlebots(self, ips: List[str]) -> Dict[str, bool]:
        """Batch verify Googlebot IPs"""
        results = {}
        for ip in ips:
            results[ip] = self._cached_dns_lookup(ip, 'google')
        return results

    def _batch_verify_bingbots(self, ips: List[str]) -> Dict[str, bool]:
        """Batch verify Bingbot IPs"""
        results = {}
        for ip in ips:
            results[ip] = self._cached_dns_lookup(ip, 'bing')
        return results

    def _verify_bot(self, row: pd.Series) -> bool:
        """Verify if claimed bot is legitimate using DNS"""
        if not row.get('bot_ua_match'):
            return False
        
        user_agent = row.get('user_agent', '')
        ip = row.get('ip', '')
        
        # Check if it claims to be Googlebot
        if 'googlebot' in user_agent.lower():
            return self._verify_googlebot(ip)
        
        # Check if it claims to be Bingbot
        if 'bingbot' in user_agent.lower():
            return self._verify_bingbot(ip)
        
        # For other bots, trust the UA for now
        return True
    
    def _verify_googlebot(self, ip: str) -> bool:
        """Verify Googlebot using reverse DNS lookup"""
        try:
            # Reverse DNS lookup
            rev_name = dns.reversename.from_address(ip)
            ptr_records = dns.resolver.resolve(rev_name, 'PTR')
            
            for ptr in ptr_records:
                hostname = str(ptr).lower()
                # Check if hostname ends with googlebot.com or google.com
                if hostname.endswith('.googlebot.com.') or hostname.endswith('.google.com.'):
                    # Forward DNS lookup to verify
                    forward_records = dns.resolver.resolve(hostname.rstrip('.'), 'A')
                    for forward in forward_records:
                        if str(forward) == ip:
                            return True
        except:
            pass
        
        return False
    
    def _verify_bingbot(self, ip: str) -> bool:
        """Verify Bingbot using reverse DNS lookup"""
        try:
            rev_name = dns.reversename.from_address(ip)
            ptr_records = dns.resolver.resolve(rev_name, 'PTR')
            
            for ptr in ptr_records:
                hostname = str(ptr).lower()
                # Check if hostname ends with search.msn.com
                if hostname.endswith('.search.msn.com.'):
                    # Forward DNS lookup to verify
                    forward_records = dns.resolver.resolve(hostname.rstrip('.'), 'A')
                    for forward in forward_records:
                        if str(forward) == ip:
                            return True
        except:
            pass
        
        return False
    
    def _ml_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use machine learning model for bot detection"""
        features = self._extract_ml_features(df)
        
        if features is not None and len(features) > 0:
            try:
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Predict
                predictions = self.model.predict(features_scaled)
                df['bot_ml'] = predictions == -1  # Isolation Forest returns -1 for anomalies
            except:
                df['bot_ml'] = False
        else:
            df['bot_ml'] = False
        
        return df
    
    def _extract_ml_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features for ML model"""
        required_cols = ['request_rate', 'unique_urls', 'error_rate']
        
        if not all(col in df.columns for col in required_cols):
            return None
        
        features = df[required_cols].fillna(0).values
        return features
    
    def _classify_bot_type(self, row: pd.Series) -> str:
        """Classify the type of bot"""
        user_agent = row.get('user_agent', '').lower()
        
        # Check each bot category
        for category, pattern in self.bot_patterns.items():
            if category != 'general' and pattern.search(user_agent):
                return category
        
        # Check if it's a bot but unknown type
        if row.get('is_bot'):
            return 'other'
        
        return 'human'
    
    def _calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate bot detection confidence score"""
        confidence = pd.Series(0.0, index=df.index)
        
        # Add confidence from different detection methods
        if 'bot_ua_match' in df.columns:
            confidence += df['bot_ua_match'] * 0.4
        
        if 'bot_behavior' in df.columns:
            confidence += df['bot_behavior'] * 0.3
        
        if 'bot_verified' in df.columns:
            confidence += df['bot_verified'] * 0.3
        
        if 'bot_ml' in df.columns:
            confidence += df['bot_ml'] * 0.2
        
        return confidence.clip(0, 1)
    
    def train_ml_model(self, df: pd.DataFrame):
        """Train ML model for bot detection"""
        # Prepare training data
        if 'is_bot' not in df.columns:
            st.warning("No labeled bot data available for training")
            return
        
        features = self._prepare_training_features(df)
        labels = df['is_bot'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest for anomaly detection
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X_train_scaled)
        
        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        accuracy = ((predictions == -1) == y_test).mean()
        
        st.success(f"Bot detection model trained with {accuracy:.2%} accuracy")
    
    def _prepare_training_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for training"""
        # Calculate behavioral features
        features = []
        
        for ip in df['ip'].unique():
            ip_data = df[df['ip'] == ip]
            
            # Calculate features
            request_count = len(ip_data)
            time_span = (ip_data['timestamp'].max() - ip_data['timestamp'].min()).total_seconds()
            request_rate = request_count / (time_span + 1)
            unique_urls = ip_data['url'].nunique() if 'url' in ip_data.columns else 0
            error_rate = (ip_data['status'] >= 400).mean() if 'status' in ip_data.columns else 0
            
            features.append([request_rate, unique_urls, error_rate])
        
        return np.array(features)
    
    def get_bot_summary(self, df: pd.DataFrame) -> Dict:
        """Generate bot detection summary"""
        if 'is_bot' not in df.columns:
            return {}
        
        total_requests = len(df)
        bot_requests = df['is_bot'].sum()
        
        summary = {
            'total_requests': total_requests,
            'bot_requests': bot_requests,
            'human_requests': total_requests - bot_requests,
            'bot_percentage': (bot_requests / total_requests * 100) if total_requests > 0 else 0
        }
        
        # Bot type breakdown
        if 'bot_type' in df.columns:
            bot_types = df[df['is_bot']]['bot_type'].value_counts().to_dict()
            summary['bot_types'] = bot_types
        
        # Top bot IPs
        if 'ip' in df.columns:
            top_bot_ips = df[df['is_bot']]['ip'].value_counts().head(10).to_dict()
            summary['top_bot_ips'] = top_bot_ips
        
        # Verified vs unverified bots
        if 'bot_verified' in df.columns:
            verified_bots = df[df['is_bot']]['bot_verified'].sum()
            summary['verified_bots'] = verified_bots
            summary['unverified_bots'] = bot_requests - verified_bots
        
        return summary
