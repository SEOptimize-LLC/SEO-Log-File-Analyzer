# components/cache_manager.py

"""
Cache management for SEO Log Analyzer
"""
import streamlit as st
import pandas as pd
import hashlib
import pickle
import json
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
from config import CACHE_TTL, ENABLE_CACHE, MAX_CACHE_SIZE_MB

class CacheManager:
    """Manage caching for processed data"""
    
    def __init__(self):
        self.enabled = ENABLE_CACHE
        self.ttl = CACHE_TTL
        self.max_size_mb = MAX_CACHE_SIZE_MB
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache in session state"""
        if 'cache_store' not in st.session_state:
            st.session_state.cache_store = {}
        if 'cache_metadata' not in st.session_state:
            st.session_state.cache_metadata = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        if not self.enabled:
            return None
        
        if key in st.session_state.cache_store:
            # Check if expired
            metadata = st.session_state.cache_metadata.get(key, {})
            if self._is_expired(metadata):
                self.delete(key)
                return None
            
            # Update access time
            st.session_state.cache_metadata[key]['last_accessed'] = datetime.now()
            return st.session_state.cache_store[key]
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (overrides default)
        """
        if not self.enabled:
            return
        
        # Check cache size
        if self._get_cache_size() > self.max_size_mb:
            self._evict_lru()
        
        st.session_state.cache_store[key] = value
        st.session_state.cache_metadata[key] = {
            'created': datetime.now(),
            'last_accessed': datetime.now(),
            'ttl': ttl or self.ttl
        }
    
    def delete(self, key: str):
        """Delete cached value"""
        if key in st.session_state.cache_store:
            del st.session_state.cache_store[key]
        if key in st.session_state.cache_metadata:
            del st.session_state.cache_metadata[key]
    
    def clear(self):
        """Clear all cache"""
        st.session_state.cache_store = {}
        st.session_state.cache_metadata = {}
    
    def _is_expired(self, metadata: Dict) -> bool:
        """Check if cache entry is expired"""
        if not metadata:
            return True
        
        created = metadata.get('created')
        ttl = metadata.get('ttl', self.ttl)
        
        if created and ttl:
            return datetime.now() > created + timedelta(seconds=ttl)
        
        return False
    
    def _get_cache_size(self) -> float:
        """Get approximate cache size in MB"""
        try:
            # Serialize cache to get size
            cache_bytes = pickle.dumps(st.session_state.cache_store)
            return len(cache_bytes) / (1024 * 1024)
        except:
            return 0
    
    def _evict_lru(self):
        """Evict least recently used cache entries"""
        if not st.session_state.cache_metadata:
            return
        
        # Sort by last accessed time
        sorted_keys = sorted(
            st.session_state.cache_metadata.keys(),
            key=lambda k: st.session_state.cache_metadata[k].get('last_accessed', datetime.min)
        )
        
        # Remove oldest 20% of cache
        num_to_remove = max(1, len(sorted_keys) // 5)
        for key in sorted_keys[:num_to_remove]:
            self.delete(key)
    
    @staticmethod
    def create_key(*args) -> str:
        """
        Create cache key from arguments
        
        Args:
            *args: Arguments to create key from
            
        Returns:
            Cache key string
        """
        key_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()

# Streamlit cache decorators for heavy computations
@st.cache_data(ttl=CACHE_TTL)
def cache_dataframe_operation(df: pd.DataFrame, operation: str, **kwargs) -> pd.DataFrame:
    """
    Cache DataFrame operations
    
    Args:
        df: Input DataFrame
        operation: Operation name
        **kwargs: Operation parameters
        
    Returns:
        Processed DataFrame
    """
    if operation == 'groupby':
        return df.groupby(**kwargs)
    elif operation == 'aggregate':
        return df.agg(**kwargs)
    elif operation == 'filter':
        return df[kwargs.get('condition', True)]
    else:
        return df

@st.cache_data(ttl=CACHE_TTL)
def cache_analysis_result(data_hash: str, analysis_type: str) -> Dict:
    """
    Cache analysis results
    
    Args:
        data_hash: Hash of input data
        analysis_type: Type of analysis
        
    Returns:
        Analysis results
    """
    # This is a placeholder - actual analysis would be done here
    return {}

@st.cache_resource
def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance"""
    return CacheManager()

class FileCache:
    """File-based cache for large datasets"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        self._init_cache_dir()
    
    def _init_cache_dir(self):
        """Initialize cache directory"""
        import os
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_file_path(self, key: str) -> str:
        """Get file path for cache key"""
        import os
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def exists(self, key: str) -> bool:
        """Check if cache file exists"""
        import os
        return os.path.exists(self.get_file_path(key))
    
    def save(self, key: str, data: Any):
        """Save data to cache file"""
        with open(self.get_file_path(key), 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, key: str) -> Optional[Any]:
        """Load data from cache file"""
        if not self.exists(key):
            return None
        
        try:
            with open(self.get_file_path(key), 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def delete(self, key: str):
        """Delete cache file"""
        import os
        if self.exists(key):
            os.remove(self.get_file_path(key))
    
    def clear(self):
        """Clear all cache files"""
        import os
        import glob
        
        pattern = os.path.join(self.cache_dir, "*.pkl")
        for file in glob.glob(pattern):
            os.remove(file)
