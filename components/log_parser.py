# components/log_parser.py

"""
Multi-format log file parser with chunked processing support
"""
import re
import pandas as pd
import numpy as np
from datetime import datetime
import json
import gzip
from io import StringIO, BytesIO
from typing import Union, Dict, List, Optional, Iterator, Callable
import streamlit as st
from config import LOG_PATTERNS, CHUNK_SIZE, MAX_FILE_SIZE_MB, ENABLE_SAMPLING, SAMPLE_RATE

class LogParser:
    """Universal log file parser supporting multiple formats with chunked processing"""

    def __init__(self):
        self.patterns = LOG_PATTERNS
        self.format_detected = None
        self.parse_errors = []
        self.chunk_size = CHUNK_SIZE
        self.max_file_size_mb = MAX_FILE_SIZE_MB

    def parse(self, file_object, chunked: bool = True, progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Parse log file and return DataFrame with optional chunked processing

        Args:
            file_object: Streamlit uploaded file object
            chunked: Whether to use chunked processing for large files
            progress_callback: Optional callback for progress updates

        Returns:
            pd.DataFrame: Parsed log data
        """
        # Check file size for chunked processing decision
        file_size_mb = len(file_object.read()) / (1024 * 1024)
        file_object.seek(0)  # Reset file pointer

        # Use chunked processing for large files
        if chunked and file_size_mb > 50:  # 50MB threshold
            return self._parse_chunked(file_object, progress_callback)

        # Read file content for smaller files
        content = self._read_file(file_object)

        # Detect format
        log_format = self._detect_format(content)

        if log_format == 'json':
            return self._parse_json(content)
        elif log_format:
            return self._parse_structured(content, log_format)
        else:
            return self._parse_csv(content)
    
    def _read_file(self, file_object) -> str:
        """Read file content, handling compression"""
        if file_object.name.endswith('.gz'):
            with gzip.open(BytesIO(file_object.read()), 'rt') as f:
                content = f.read()
        else:
            content = str(file_object.read(), 'utf-8', errors='ignore')

        return content

    def _parse_chunked(self, file_object, progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Parse large files using chunked processing

        Args:
            file_object: Streamlit uploaded file object
            progress_callback: Optional callback for progress updates

        Returns:
            pd.DataFrame: Parsed log data
        """
        # Get file size for progress tracking
        file_size = self._get_file_size(file_object)

        # Detect format from first chunk
        first_chunk = self._read_first_chunk(file_object)
        log_format = self._detect_format(first_chunk)

        # Reset file pointer
        file_object.seek(0)

        # Process file in chunks
        chunks = []
        total_processed = 0

        for chunk_content, chunk_size in self._chunk_file_generator(file_object, file_size):
            if progress_callback:
                progress = min(total_processed / file_size, 1.0)
                progress_callback(progress, f"Processing chunk... ({len(chunks) + 1})")

            # Parse chunk based on format
            if log_format == 'json':
                chunk_df = self._parse_json_chunk(chunk_content)
            elif log_format:
                chunk_df = self._parse_structured_chunk(chunk_content, log_format)
            else:
                chunk_df = self._parse_csv_chunk(chunk_content)

            if not chunk_df.empty:
                chunks.append(chunk_df)

            total_processed += chunk_size

            # Memory management: combine chunks periodically
            if len(chunks) >= 10:  # Combine every 10 chunks
                combined_df = pd.concat(chunks, ignore_index=True)
                chunks = [combined_df]

        # Final combination
        if chunks:
            final_df = pd.concat(chunks, ignore_index=True)
            return self._post_process(final_df)
        else:
            return pd.DataFrame()

    def _get_file_size(self, file_object) -> int:
        """Get file size without reading entire file"""
        current_pos = file_object.tell()
        file_object.seek(0, 2)  # Seek to end
        size = file_object.tell()
        file_object.seek(current_pos)  # Reset position
        return size

    def _read_first_chunk(self, file_object, chunk_size: int = 8192) -> str:
        """Read first chunk to detect format"""
        current_pos = file_object.tell()
        chunk = file_object.read(chunk_size)
        file_object.seek(current_pos)  # Reset position
        return str(chunk, 'utf-8', errors='ignore')

    def _chunk_file_generator(self, file_object, file_size: int, chunk_size: int = None) -> Iterator[tuple]:
        """
        Generate chunks from file

        Args:
            file_object: File object to read from
            file_size: Total file size
            chunk_size: Size of each chunk in bytes

        Yields:
            tuple: (chunk_content, chunk_size)
        """
        if chunk_size is None:
            chunk_size = min(self.chunk_size * 1024, file_size // 100)  # Adaptive chunk size

        bytes_read = 0

        while bytes_read < file_size:
            remaining = file_size - bytes_read
            current_chunk_size = min(chunk_size, remaining)

            chunk_data = file_object.read(current_chunk_size)
            if not chunk_data:
                break

            chunk_content = str(chunk_data, 'utf-8', errors='ignore')
            yield chunk_content, current_chunk_size

            bytes_read += current_chunk_size

    def _parse_json_chunk(self, content: str) -> pd.DataFrame:
        """Parse JSON chunk"""
        records = []
        lines = content.split('\n')

        for line in lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

        return pd.DataFrame(records) if records else pd.DataFrame()

    def _parse_structured_chunk(self, content: str, log_format: str) -> pd.DataFrame:
        """Parse structured log chunk"""
        pattern = re.compile(self.patterns[log_format])
        records = []
        lines = content.split('\n')

        for line in lines:
            if not line or line.startswith('#'):
                continue

            match = pattern.match(line)
            if match:
                groups = match.groups()
                record = self._extract_fields(groups, log_format)
                records.append(record)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def _parse_csv_chunk(self, content: str) -> pd.DataFrame:
        """Parse CSV chunk"""
        try:
            # For CSV, we need complete lines, so this is a simplified approach
            # In production, you'd want more sophisticated CSV chunking
            df = pd.read_csv(StringIO(content), error_bad_lines=False, warn_bad_lines=False)
            return df
        except Exception:
            return pd.DataFrame()
    
    def _detect_format(self, content: str) -> Optional[str]:
        """Detect log file format"""
        lines = content.split('\n')[:100]  # Check first 100 lines
        
        # Check for JSON
        if lines[0].strip().startswith('{'):
            try:
                json.loads(lines[0])
                self.format_detected = 'json'
                return 'json'
            except:
                pass
        
        # Check each pattern
        for format_name, pattern in self.patterns.items():
            regex = re.compile(pattern)
            matches = 0
            for line in lines:
                if line and regex.match(line):
                    matches += 1
            
            if matches > len(lines) * 0.5:  # If >50% lines match
                self.format_detected = format_name
                return format_name
        
        # Default to CSV
        self.format_detected = 'csv'
        return None
    
    def _parse_structured(self, content: str, log_format: str,
                          show_progress: bool = True) -> pd.DataFrame:
        """Parse structured log formats (Apache, Nginx, IIS, CloudFront)"""
        pattern = re.compile(self.patterns[log_format])

        records = []
        lines = content.split('\n')

        # Progress tracking (optional)
        progress_bar = None
        if show_progress and len(lines) > 1000:
            progress_bar = st.progress(0)

        for idx, line in enumerate(lines):
            if progress_bar and idx % 1000 == 0:
                progress_bar.progress(idx / len(lines))

            if not line or line.startswith('#'):
                continue

            match = pattern.match(line)
            if match:
                groups = match.groups()
                record = self._extract_fields(groups, log_format)
                records.append(record)
            else:
                self.parse_errors.append(f"Line {idx}: {line[:100]}")

        if progress_bar:
            progress_bar.empty()

        df = pd.DataFrame(records)
        return self._post_process(df)
    
    def _parse_json(self, content: str) -> pd.DataFrame:
        """Parse JSON formatted logs"""
        records = []
        lines = content.split('\n')
        
        for line in lines:
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    self.parse_errors.append(f"Invalid JSON: {line[:100]}")
        
        df = pd.DataFrame(records)
        return self._post_process(df)
    
    def _parse_csv(self, content: str) -> pd.DataFrame:
        """Parse CSV formatted logs"""
        try:
            df = pd.read_csv(StringIO(content), error_bad_lines=False, warn_bad_lines=False)
            return self._post_process(df)
        except Exception as e:
            st.error(f"CSV parsing error: {str(e)}")
            return pd.DataFrame()
    
    def _extract_fields(self, groups: tuple, log_format: str) -> Dict:
        """Extract fields based on log format"""
        record = {}
        
        if log_format in ['apache_common', 'apache_combined']:
            record['ip'] = groups[0]
            record['timestamp'] = self._parse_timestamp(groups[1])
            record['method'] = groups[2]
            record['url'] = groups[3]
            record['protocol'] = groups[4]
            record['status'] = int(groups[5])
            record['size'] = int(groups[6]) if groups[6] != '-' else 0
            
            if log_format == 'apache_combined':
                record['referrer'] = groups[7] if groups[7] != '-' else ''
                record['user_agent'] = groups[8]
        
        elif log_format == 'nginx':
            record['ip'] = groups[0]
            record['timestamp'] = self._parse_timestamp(groups[1])
            record['method'] = groups[2]
            record['url'] = groups[3]
            record['protocol'] = groups[4]
            record['status'] = int(groups[5])
            record['size'] = int(groups[6])
            record['referrer'] = groups[7] if groups[7] != '-' else ''
            record['user_agent'] = groups[8]
        
        elif log_format == 'iis':
            record['date'] = groups[0]
            record['time'] = groups[1]
            record['server_ip'] = groups[2]
            record['method'] = groups[3]
            record['url'] = groups[4]
            record['port'] = int(groups[5])
            record['username'] = groups[6]
            record['ip'] = groups[7]
            record['user_agent'] = groups[8]
            record['status'] = int(groups[9])
            record['timestamp'] = self._parse_timestamp(f"{groups[0]} {groups[1]}")
        
        elif log_format == 'cloudfront':
            record['date'] = groups[0]
            record['time'] = groups[1]
            record['edge_location'] = groups[2]
            record['size'] = int(groups[3])
            record['ip'] = groups[4]
            record['method'] = groups[5]
            record['host'] = groups[6]
            record['url'] = groups[7]
            record['status'] = int(groups[8])
            record['timestamp'] = self._parse_timestamp(f"{groups[0]} {groups[1]}")
        
        return record
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats"""
        formats = [
            '%d/%b/%Y:%H:%M:%S %z',  # Apache/Nginx
            '%Y-%m-%d %H:%M:%S',      # IIS
            '%Y-%m-%d\t%H:%M:%S',     # CloudFront
            '%Y-%m-%dT%H:%M:%S%z',    # ISO format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except:
                continue
        
        # Default to current time if parsing fails
        return datetime.now()
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process parsed DataFrame"""
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            if 'date' in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
            else:
                df['timestamp'] = datetime.now()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add calculated fields
        if 'status' in df.columns:
            df['status_category'] = df['status'].apply(lambda x: f"{x//100}xx")
        
        if 'size' in df.columns:
            df['size_mb'] = df['size'] / (1024 * 1024)
        
        # Add response time (if available in extended logs)
        if 'time_taken' in df.columns:
            df['response_time'] = pd.to_numeric(df['time_taken'], errors='coerce')
        elif 'response_time_ms' in df.columns:
            df['response_time'] = pd.to_numeric(df['response_time_ms'], errors='coerce')
        else:
            # Generate synthetic response times for demo
            df['response_time'] = np.random.lognormal(5, 1.5, len(df))
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Add session identification
        df = self._identify_sessions(df)
        
        return df
    
    def _identify_sessions(self, df: pd.DataFrame, timeout_minutes: int = 30) -> pd.DataFrame:
        """Identify user sessions based on IP and time gaps"""
        if 'ip' not in df.columns:
            return df
        
        df = df.sort_values(['ip', 'timestamp'])
        
        # Calculate time differences
        df['time_diff'] = df.groupby('ip')['timestamp'].diff()
        
        # Mark new sessions
        df['new_session'] = (df['time_diff'] > pd.Timedelta(minutes=timeout_minutes)) | df['time_diff'].isna()
        
        # Assign session IDs
        df['session_id'] = df.groupby('ip')['new_session'].cumsum()
        
        # Clean up
        df = df.drop(['time_diff', 'new_session'], axis=1)
        
        return df
    
    def get_sample_data(self, n_rows: int = 1000) -> pd.DataFrame:
        """Generate sample log data for testing"""
        np.random.seed(42)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=datetime.now() - pd.Timedelta(days=7),
            end=datetime.now(),
            periods=n_rows
        )
        
        # Generate IPs
        ips = [f"192.168.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}" 
               for _ in range(n_rows)]
        
        # Generate URLs
        pages = ['/home', '/products', '/about', '/contact', '/blog', '/api/v1/users',
                 '/api/v1/products', '/login', '/checkout', '/search']
        urls = np.random.choice(pages, n_rows)
        
        # Generate methods
        methods = np.random.choice(['GET', 'POST', 'PUT', 'DELETE'], n_rows, p=[0.8, 0.15, 0.03, 0.02])
        
        # Generate status codes
        statuses = np.random.choice([200, 301, 302, 404, 500], n_rows, p=[0.85, 0.05, 0.03, 0.05, 0.02])
        
        # Generate response times
        response_times = np.random.lognormal(5, 1.5, n_rows)
        
        # Generate user agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Safari/537.36',
            'Googlebot/2.1 (+http://www.google.com/bot.html)',
            'Mozilla/5.0 (compatible; bingbot/2.0)',
            'Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36',
        ]
        user_agent_list = np.random.choice(user_agents, n_rows, p=[0.4, 0.3, 0.15, 0.1, 0.05])
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'ip': ips,
            'method': methods,
            'url': urls,
            'status': statuses,
            'response_time': response_times,
            'user_agent': user_agent_list,
            'size': np.random.exponential(10000, n_rows),
            'referrer': np.random.choice(['https://google.com', 'https://facebook.com', '-'], n_rows),
        })
        
        # Add calculated fields
        df['status_category'] = df['status'].apply(lambda x: f"{x//100}xx")
        df['size_mb'] = df['size'] / (1024 * 1024)
        
        # Add bot detection
        df['is_bot'] = df['user_agent'].str.contains('bot|Bot|spider|Spider|crawl|Crawl', na=False)
        
        return df
