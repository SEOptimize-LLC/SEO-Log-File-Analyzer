# components/seo_analyzer.py

"""
SEO metrics analyzer
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict
import streamlit as st
from config import (
    CRAWL_BUDGET_EFFICIENCY_TARGET,
    ORPHAN_PAGE_THRESHOLD,
    MIN_CRAWL_FREQUENCY,
    MOBILE_FIRST_PRIORITY
)

class SEOAnalyzer:
    """Comprehensive SEO analysis from log data"""
    
    def __init__(self):
        self.crawl_budget_target = CRAWL_BUDGET_EFFICIENCY_TARGET
        self.orphan_threshold = ORPHAN_PAGE_THRESHOLD
        self.min_crawl_freq = MIN_CRAWL_FREQUENCY
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive SEO analysis
        
        Args:
            df: DataFrame with log data
            
        Returns:
            Dictionary with SEO metrics and insights
        """
        results = {}
        
        # Crawl budget analysis
        results['crawl_budget'] = self._analyze_crawl_budget(df)
        
        # Orphan pages detection
        results['orphan_pages'] = self._detect_orphan_pages(df)
        
        # Mobile vs Desktop analysis
        results['mobile_desktop'] = self._analyze_mobile_desktop(df)
        
        # JavaScript rendering
        results['js_rendering'] = self._analyze_js_rendering(df)
        
        # Internal linking
        results['internal_linking'] = self._analyze_internal_linking(df)
        
        # Crawl frequency
        results['crawl_frequency'] = self._analyze_crawl_frequency(df)
        
        # SEO errors
        results['seo_errors'] = self._analyze_seo_errors(df)
        
        # Page priorities
        results['page_priorities'] = self._calculate_page_priorities(df)
        
        return results
    
    def _analyze_crawl_budget(self, df: pd.DataFrame) -> Dict:
        """Analyze crawl budget efficiency"""
        if 'is_bot' not in df.columns or 'url' not in df.columns:
            return {}
        
        # Filter for search engine bots
        bot_df = df[df['is_bot']]
        search_bots = bot_df[bot_df['bot_type'].isin(['google', 'bing', 'yandex', 'baidu'])] if 'bot_type' in bot_df.columns else bot_df
        
        total_crawls = len(search_bots)
        unique_pages_crawled = search_bots['url'].nunique()
        
        # Calculate wasted crawl budget
        wasted_crawls = {
            '404_pages': (search_bots['status'] == 404).sum() if 'status' in search_bots.columns else 0,
            '5xx_errors': (search_bots['status'] >= 500).sum() if 'status' in search_bots.columns else 0,
            'redirects': search_bots['status'].isin([301, 302, 303, 307, 308]).sum() if 'status' in search_bots.columns else 0,
        }
        
        total_wasted = sum(wasted_crawls.values())
        efficiency = 1 - (total_wasted / total_crawls) if total_crawls > 0 else 0
        
        # Identify over-crawled pages
        page_crawl_counts = search_bots['url'].value_counts()
        over_crawled = page_crawl_counts[page_crawl_counts > page_crawl_counts.quantile(0.95)]
        
        # Identify under-crawled important pages
        all_pages = df['url'].unique()
        crawled_pages = search_bots['url'].unique()
        not_crawled = set(all_pages) - set(crawled_pages)
        
        return {
            'total_crawls': total_crawls,
            'unique_pages': unique_pages_crawled,
            'efficiency': efficiency,
            'wasted_crawls': wasted_crawls,
            'total_wasted': total_wasted,
            'over_crawled_pages': over_crawled.to_dict(),
            'not_crawled_pages': list(not_crawled)[:100],  # Limit to 100 for display
            'recommendation': self._get_crawl_budget_recommendation(efficiency)
        }
    
    def _detect_orphan_pages(self, df: pd.DataFrame) -> Dict:
        """Detect orphan pages (pages with traffic but no internal links)"""
        if 'url' not in df.columns:
            return {}
        
        # Pages that received traffic
        pages_with_traffic = df['url'].unique()
        
        # Build internal link graph from referrer data
        internal_links = defaultdict(set)
        if 'referrer' in df.columns:
            # Filter for internal referrers (simplified - in production, check domain)
            internal_df = df[df['referrer'].str.startswith('/', na=False)]
            for _, row in internal_df.iterrows():
                internal_links[row['referrer']].add(row['url'])
        
        # Find pages with no incoming internal links
        linked_pages = set()
        for targets in internal_links.values():
            linked_pages.update(targets)
        
        orphan_pages = set(pages_with_traffic) - linked_pages
        
        # Analyze orphan page traffic
        orphan_traffic = df[df['url'].isin(orphan_pages)]
        
        orphan_stats = []
        for page in list(orphan_pages)[:50]:  # Limit to top 50
            page_data = df[df['url'] == page]
            stats = {
                'url': page,
                'visits': len(page_data),
                'unique_visitors': page_data['ip'].nunique() if 'ip' in page_data.columns else 0,
                'avg_time_on_page': self._calculate_avg_time_on_page(page_data),
                'bounce_rate': self._calculate_bounce_rate(page_data)
            }
            orphan_stats.append(stats)
        
        # Sort by visits
        orphan_stats = sorted(orphan_stats, key=lambda x: x['visits'], reverse=True)
        
        return {
            'total_orphan_pages': len(orphan_pages),
            'orphan_pages': orphan_stats,
            'percentage_orphan': (len(orphan_pages) / len(pages_with_traffic) * 100) if pages_with_traffic else 0,
            'recommendation': self._get_orphan_recommendation(len(orphan_pages))
        }
    
    def _analyze_mobile_desktop(self, df: pd.DataFrame) -> Dict:
        """Analyze mobile vs desktop crawler behavior"""
        if 'user_agent' not in df.columns:
            return {}
        
        # Identify mobile vs desktop bots
        mobile_bots = df[df['user_agent'].str.contains('Mobile|Android', na=False, case=False)]
        desktop_bots = df[~df['user_agent'].str.contains('Mobile|Android', na=False, case=False)]
        
        # Filter for search engine bots only
        if 'is_bot' in df.columns:
            mobile_bots = mobile_bots[mobile_bots['is_bot']]
            desktop_bots = desktop_bots[desktop_bots['is_bot']]
        
        mobile_crawls = len(mobile_bots)
        desktop_crawls = len(desktop_bots)
        
        mobile_ratio = mobile_crawls / (mobile_crawls + desktop_crawls) if (mobile_crawls + desktop_crawls) > 0 else 0
        
        # Analyze page coverage
        mobile_pages = set(mobile_bots['url'].unique()) if 'url' in mobile_bots.columns else set()
        desktop_pages = set(desktop_bots['url'].unique()) if 'url' in desktop_bots.columns else set()
        
        mobile_only = mobile_pages - desktop_pages
        desktop_only = desktop_pages - mobile_pages
        both = mobile_pages & desktop_pages
        
        return {
            'mobile_crawls': mobile_crawls,
            'desktop_crawls': desktop_crawls,
            'mobile_ratio': mobile_ratio,
            'mobile_first_aligned': mobile_ratio > 0.6,
            'mobile_only_pages': len(mobile_only),
            'desktop_only_pages': len(desktop_only),
            'both_crawled': len(both),
            'recommendation': self._get_mobile_recommendation(mobile_ratio)
        }
    
    def _analyze_js_rendering(self, df: pd.DataFrame) -> Dict:
        """Analyze JavaScript rendering patterns"""
        if 'user_agent' not in df.columns:
            return {}
        
        # Identify Chrome/Chromium user agents (used for rendering)
        chrome_requests = df[df['user_agent'].str.contains('Chrome|Chromium', na=False, case=False)]
        
        # Filter for Googlebot Chrome
        googlebot_chrome = chrome_requests[chrome_requests['user_agent'].str.contains('Googlebot', na=False, case=False)]
        
        total_googlebot = df[df['user_agent'].str.contains('Googlebot', na=False, case=False)]
        
        rendering_ratio = len(googlebot_chrome) / len(total_googlebot) if len(total_googlebot) > 0 else 0
        
        # Identify pages that might need JS rendering (heuristic based on response times)
        if 'response_time' in df.columns:
            slow_pages = df[df['response_time'] > 2000]['url'].value_counts() if 'url' in df.columns else pd.Series()
        else:
            slow_pages = pd.Series()
        
        return {
            'total_renders': len(googlebot_chrome),
            'rendering_ratio': rendering_ratio,
            'potentially_js_heavy': slow_pages.head(20).to_dict(),
            'recommendation': self._get_js_recommendation(rendering_ratio)
        }
    
    def _analyze_internal_linking(self, df: pd.DataFrame) -> Dict:
        """Analyze internal linking structure"""
        if 'referrer' not in df.columns or 'url' not in df.columns:
            return {}
        
        # Build link graph
        G = nx.DiGraph()
        
        # Add edges from referrer to URL
        internal_df = df[df['referrer'].str.startswith('/', na=False)]
        for _, row in internal_df.iterrows():
            G.add_edge(row['referrer'], row['url'])
        
        if len(G.nodes()) == 0:
            return {'error': 'No internal linking data available'}
        
        # Calculate metrics
        pagerank = nx.pagerank(G)
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        # Find important pages
        top_pages_by_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        top_pages_by_links = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Find link issues
        dead_ends = [node for node in G.nodes() if G.out_degree(node) == 0]
        link_hubs = [node for node in G.nodes() if G.out_degree(node) > 50]
        
        return {
            'total_internal_links': G.number_of_edges(),
            'unique_pages_linked': G.number_of_nodes(),
            'avg_links_per_page': np.mean(list(out_degree.values())),
            'top_pages_pagerank': top_pages_by_pagerank,
            'top_pages_by_links': top_pages_by_links,
            'dead_end_pages': len(dead_ends),
            'link_hubs': len(link_hubs),
            'recommendation': self._get_linking_recommendation(G)
        }
    
    def _analyze_crawl_frequency(self, df: pd.DataFrame) -> Dict:
        """Analyze how frequently pages are crawled"""
        if 'url' not in df.columns or 'timestamp' not in df.columns:
            return {}
        
        # Filter for bot traffic
        if 'is_bot' in df.columns:
            bot_df = df[df['is_bot']]
        else:
            bot_df = df
        
        # Calculate crawl frequency per page
        crawl_freq = {}
        for url in bot_df['url'].unique():
            page_crawls = bot_df[bot_df['url'] == url]['timestamp'].sort_values()
            if len(page_crawls) > 1:
                # Calculate average time between crawls
                time_diffs = page_crawls.diff().dropna()
                avg_freq = time_diffs.mean().total_seconds() / 86400  # Convert to days
                crawl_freq[url] = {
                    'crawl_count': len(page_crawls),
                    'avg_frequency_days': avg_freq,
                    'last_crawled': page_crawls.iloc[-1]
                }
        
        # Sort by crawl count
        sorted_freq = sorted(crawl_freq.items(), key=lambda x: x[1]['crawl_count'], reverse=True)
        
        # Identify issues
        rarely_crawled = [url for url, data in crawl_freq.items() 
                         if data['avg_frequency_days'] > self.min_crawl_freq]
        
        return {
            'pages_analyzed': len(crawl_freq),
            'avg_crawl_frequency': np.mean([d['avg_frequency_days'] for d in crawl_freq.values()]) if crawl_freq else 0,
            'most_crawled': dict(sorted_freq[:10]),
            'rarely_crawled': rarely_crawled[:20],
            'recommendation': self._get_frequency_recommendation(crawl_freq)
        }
    
    def _analyze_seo_errors(self, df: pd.DataFrame) -> Dict:
        """Analyze SEO-related errors"""
        if 'status' not in df.columns:
            return {}
        
        errors = {
            '404_errors': df[df['status'] == 404],
            '5xx_errors': df[df['status'] >= 500],
            'redirect_chains': self._detect_redirect_chains(df),
            'timeout_errors': df[df['response_time'] > 10000] if 'response_time' in df.columns else pd.DataFrame()
        }
        
        # Analyze 404 errors
        if len(errors['404_errors']) > 0 and 'url' in errors['404_errors'].columns:
            top_404s = errors['404_errors']['url'].value_counts().head(20)
        else:
            top_404s = pd.Series()
        
        # Analyze 5xx errors
        if len(errors['5xx_errors']) > 0 and 'url' in errors['5xx_errors'].columns:
            top_5xx = errors['5xx_errors']['url'].value_counts().head(20)
        else:
            top_5xx = pd.Series()
        
        return {
            'total_404s': len(errors['404_errors']),
            'total_5xx': len(errors['5xx_errors']),
            'redirect_chains': len(errors['redirect_chains']),
            'timeout_errors': len(errors['timeout_errors']),
            'top_404_pages': top_404s.to_dict(),
            'top_5xx_pages': top_5xx.to_dict(),
            'error_rate': (len(errors['404_errors']) + len(errors['5xx_errors'])) / len(df) if len(df) > 0 else 0,
            'recommendation': self._get_error_recommendation(errors)
        }
    
    def _calculate_page_priorities(self, df: pd.DataFrame) -> List[Dict]:
        """Calculate priority scores for pages"""
        if 'url' not in df.columns:
            return []
        
        page_scores = {}
        
        for url in df['url'].unique():
            page_data = df[df['url'] == url]
            
            score = 0
            factors = {}
            
            # Traffic volume
            traffic = len(page_data)
            factors['traffic'] = traffic
            score += min(traffic / 100, 10)  # Max 10 points
            
            # Error rate
            if 'status' in page_data.columns:
                error_rate = (page_data['status'] >= 400).mean()
                factors['error_rate'] = error_rate
                score -= error_rate * 5  # Penalty for errors
            
            # Bot interest
            if 'is_bot' in page_data.columns:
                bot_ratio = page_data['is_bot'].mean()
                factors['bot_interest'] = bot_ratio
                score += bot_ratio * 3
            
            # Performance
            if 'response_time' in page_data.columns:
                avg_response = page_data['response_time'].mean()
                factors['avg_response_time'] = avg_response
                if avg_response < 1000:
                    score += 2
                elif avg_response > 3000:
                    score -= 2
            
            page_scores[url] = {
                'url': url,
                'priority_score': score,
                'factors': factors
            }
        
        # Sort by priority score
        sorted_pages = sorted(page_scores.values(), key=lambda x: x['priority_score'], reverse=True)
        
        return sorted_pages[:50]  # Return top 50
    
    def _detect_redirect_chains(self, df: pd.DataFrame) -> List:
        """Detect redirect chains in the data"""
        if 'status' not in df.columns:
            return []
        
        redirects = df[df['status'].isin([301, 302, 303, 307, 308])]
        
        # Simplified redirect chain detection
        # In production, would track actual redirect paths
        chains = []
        if 'url' in redirects.columns:
            for url in redirects['url'].unique():
                url_redirects = redirects[redirects['url'] == url]
                if len(url_redirects) > 1:
                    chains.append({
                        'url': url,
                        'redirect_count': len(url_redirects)
                    })
        
        return chains
    
    def _calculate_avg_time_on_page(self, page_data: pd.DataFrame) -> float:
        """Calculate average time on page"""
        # Simplified calculation - in production would use session data
        if 'session_id' in page_data.columns and 'timestamp' in page_data.columns:
            sessions = page_data.groupby('session_id')['timestamp'].agg(['min', 'max'])
            avg_time = (sessions['max'] - sessions['min']).mean().total_seconds()
            return avg_time
        return 0
    
    def _calculate_bounce_rate(self, page_data: pd.DataFrame) -> float:
        """Calculate bounce rate for a page"""
        # Simplified - in production would track actual user sessions
        if 'session_id' in page_data.columns:
            single_page_sessions = page_data.groupby('session_id').size() == 1
            bounce_rate = single_page_sessions.mean()
            return bounce_rate
        return 0
    
    # Recommendation methods
    def _get_crawl_budget_recommendation(self, efficiency: float) -> str:
        if efficiency >= self.crawl_budget_target:
            return "✅ Crawl budget is being used efficiently"
        elif efficiency >= 0.6:
            return "⚠️ Moderate crawl budget waste. Focus on fixing 404s and reducing redirects."
        else:
            return "🚨 Significant crawl budget waste. Urgent action needed to fix errors and optimize crawl paths."
    
    def _get_orphan_recommendation(self, orphan_count: int) -> str:
        if orphan_count == 0:
            return "✅ No orphan pages detected"
        elif orphan_count < 10:
            return "⚠️ Few orphan pages found. Add internal links to these pages."
        else:
            return f"🚨 {orphan_count} orphan pages detected. Review internal linking structure urgently."
    
    def _get_mobile_recommendation(self, mobile_ratio: float) -> str:
        if mobile_ratio >= 0.6:
            return "✅ Mobile-first indexing aligned. Good mobile crawler coverage."
        elif mobile_ratio >= 0.4:
            return "⚠️ Moderate mobile crawler activity. Ensure mobile optimization."
        else:
            return "🚨 Low mobile crawler activity. Check mobile accessibility and performance."
    
    def _get_js_recommendation(self, rendering_ratio: float) -> str:
        if rendering_ratio < 0.2:
            return "✅ Low JavaScript dependency. Good for SEO."
        elif rendering_ratio < 0.5:
            return "⚠️ Moderate JavaScript rendering. Consider server-side rendering for critical content."
        else:
            return "🚨 High JavaScript dependency. Implement SSR or static generation for better SEO."
    
    def _get_linking_recommendation(self, graph: nx.DiGraph) -> str:
        avg_degree = graph.number_of_edges() / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0
        if avg_degree >= 3:
            return "✅ Good internal linking structure"
        elif avg_degree >= 2:
            return "⚠️ Internal linking could be improved. Add more contextual links."
        else:
            return "🚨 Weak internal linking. Implement comprehensive linking strategy."
    
    def _get_frequency_recommendation(self, crawl_freq: Dict) -> str:
        if not crawl_freq:
            return "No crawl frequency data available"
        
        avg_freq = np.mean([d['avg_frequency_days'] for d in crawl_freq.values()])
        if avg_freq <= 7:
            return "✅ Pages are crawled frequently"
        elif avg_freq <= 14:
            return "⚠️ Moderate crawl frequency. Ensure important pages are updated regularly."
        else:
            return "🚨 Low crawl frequency. Improve content freshness and internal linking."
    
    def _get_error_recommendation(self, errors: Dict) -> str:
        total_errors = sum(len(e) if isinstance(e, pd.DataFrame) else e for e in errors.values())
        if total_errors == 0:
            return "✅ No significant errors detected"
        elif total_errors < 100:
            return "⚠️ Some errors detected. Review and fix 404s and server errors."
        else:
            return "🚨 High error rate detected. Urgent action needed to fix errors."
