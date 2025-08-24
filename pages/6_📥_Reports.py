"""
Export and Reports page for SEO Log Analyzer
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import base64
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import *
from utils.export_handler import ExportHandler
from components.visualizations import create_overview_charts

# Page config
st.set_page_config(
    page_title="Reports - SEO Log Analyzer",
    page_icon="📥",
    layout="wide"
)

def main():
    st.title("📥 Export & Reports")
    st.markdown("Generate comprehensive SEO reports and export data for further analysis")
    
    # Check for data
    if 'parsed_data' not in st.session_state or st.session_state.parsed_data is None:
        st.warning("⚠️ No data available. Please upload and process a log file first.")
        if st.button("Go to Main Dashboard"):
            st.switch_page("app.py")
        return
    
    data = st.session_state.parsed_data
    
    # Initialize export handler
    exporter = ExportHandler()
    
    # Report type selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        report_type = st.selectbox(
            "Select Report Type",
            [
                "Executive Summary",
                "Technical SEO Audit",
                "Bot Analysis Report",
                "Performance Report",
                "Crawl Budget Report",
                "Custom Report"
            ],
            help="Choose the type of report to generate"
        )
    
    with col2:
        export_format = st.selectbox(
            "Export Format",
            ["PDF", "Excel", "CSV", "JSON", "HTML"],
            help="Select the output format"
        )
    
    st.markdown("---")
    
    # Report configuration
    if report_type == "Executive Summary":
        generate_executive_summary(data, exporter, export_format)
    elif report_type == "Technical SEO Audit":
        generate_technical_audit(data, exporter, export_format)
    elif report_type == "Bot Analysis Report":
        generate_bot_report(data, exporter, export_format)
    elif report_type == "Performance Report":
        generate_performance_report(data, exporter, export_format)
    elif report_type == "Crawl Budget Report":
        generate_crawl_budget_report(data, exporter, export_format)
    else:
        generate_custom_report(data, exporter, export_format)

def generate_executive_summary(data, exporter, format):
    """Generate executive summary report"""
    st.header("📊 Executive Summary Report")
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        if 'timestamp' in data.columns:
            min_date = pd.to_datetime(data['timestamp']).min().date()
            max_date = pd.to_datetime(data['timestamp']).max().date()
            
            date_range = st.date_input(
                "Report Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="exec_date_range"
            )
    
    with col2:
        include_charts = st.checkbox("Include Visualizations", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
    
    # Report preview
    st.subheader("Report Preview")
    
    # Calculate summary metrics
    metrics = calculate_executive_metrics(data)
    
    # Display preview
    with st.expander("📈 Key Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", f"{metrics['total_requests']:,}")
            st.metric("Unique URLs", f"{metrics['unique_urls']:,}")
        
        with col2:
            st.metric("Bot Traffic", f"{metrics['bot_percentage']:.1f}%")
            st.metric("Error Rate", f"{metrics['error_rate']:.1f}%")
        
        with col3:
            st.metric("Avg Response Time", f"{metrics['avg_response_time']:.0f}ms")
            st.metric("Crawl Efficiency", f"{metrics['crawl_efficiency']:.1f}%")
        
        with col4:
            st.metric("SEO Score", f"{metrics['seo_score']}/100")
            st.metric("Critical Issues", metrics['critical_issues'])
    
    with st.expander("💡 Top Insights", expanded=True):
        insights = generate_executive_insights(metrics)
        for insight in insights:
            if insight['type'] == 'success':
                st.success(f"✅ {insight['text']}")
            elif insight['type'] == 'warning':
                st.warning(f"⚠️ {insight['text']}")
            else:
                st.error(f"🔴 {insight['text']}")
    
    # Generate report button
    if st.button("🚀 Generate Executive Summary", type="primary"):
        with st.spinner("Generating report..."):
            report_data = {
                'title': 'SEO Log Analysis - Executive Summary',
                'date_range': date_range if 'date_range' in locals() else None,
                'metrics': metrics,
                'insights': insights,
                'include_charts': include_charts,
                'include_recommendations': include_recommendations
            }
            
            # Generate report based on format
            if format == "PDF":
                pdf_buffer = exporter.generate_pdf_report(data, report_data)
                download_report(pdf_buffer, "executive_summary.pdf", "application/pdf")
            elif format == "Excel":
                excel_buffer = exporter.generate_excel_report(data, report_data)
                download_report(excel_buffer, "executive_summary.xlsx", "application/vnd.ms-excel")
            elif format == "HTML":
                html_content = exporter.generate_html_report(data, report_data)
                download_report(html_content, "executive_summary.html", "text/html")
            elif format == "JSON":
                json_content = json.dumps(report_data, default=str, indent=2)
                download_report(json_content, "executive_summary.json", "application/json")
            else:
                csv_buffer = exporter.generate_csv_report(data)
                download_report(csv_buffer, "executive_summary.csv", "text/csv")

def generate_technical_audit(data, exporter, format):
    """Generate technical SEO audit report"""
    st.header("🔧 Technical SEO Audit Report")
    
    # Audit configuration
    st.subheader("Audit Scope")
    
    col1, col2 = st.columns(2)
    
    with col1:
        audit_areas = st.multiselect(
            "Select Audit Areas",
            [
                "Crawlability",
                "Indexability",
                "Site Architecture",
                "Performance",
                "Mobile Optimization",
                "Security",
                "International SEO"
            ],
            default=["Crawlability", "Indexability", "Performance"]
        )
    
    with col2:
        severity_filter = st.multiselect(
            "Issue Severity",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"]
        )
    
    # Run audit
    if st.button("🔍 Run Technical Audit", type="primary"):
        with st.spinner("Running technical audit..."):
            audit_results = run_technical_audit(data, audit_areas)
            
            # Display results
            st.subheader("Audit Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Issues Found", audit_results['total_issues'])
            with col2:
                st.metric("Critical Issues", audit_results['critical_issues'])
            with col3:
                st.metric("Pages Affected", audit_results['pages_affected'])
            with col4:
                score_color = "🟢" if audit_results['health_score'] > 80 else "🟡" if audit_results['health_score'] > 60 else "🔴"
                st.metric("Health Score", f"{score_color} {audit_results['health_score']}/100")
            
            # Detailed issues
            st.subheader("Issues by Category")
            
            for category, issues in audit_results['issues_by_category'].items():
                with st.expander(f"{category} ({len(issues)} issues)"):
                    for issue in issues:
                        if issue['severity'] == 'Critical':
                            st.error(f"🔴 {issue['description']}")
                        elif issue['severity'] == 'High':
                            st.warning(f"🟠 {issue['description']}")
                        else:
                            st.info(f"🟡 {issue['description']}")
                        
                        if issue.get('affected_urls'):
                            st.write(f"Affected URLs: {len(issue['affected_urls'])}")
            
            # Export audit report
            if st.button("📥 Export Audit Report"):
                audit_buffer = exporter.generate_audit_report(audit_results, format)
                download_report(audit_buffer, f"technical_audit.{format.lower()}", get_mime_type(format))

def generate_bot_report(data, exporter, format):
    """Generate bot analysis report"""
    st.header("🤖 Bot Analysis Report")
    
    # Bot report configuration
    col1, col2 = st.columns(2)
    
    with col1:
        bot_types = st.multiselect(
            "Select Bot Types",
            ["Googlebot", "Bingbot", "Other Search Bots", "Malicious Bots", "Unknown"],
            default=["Googlebot", "Bingbot"]
        )
    
    with col2:
        analysis_depth = st.radio(
            "Analysis Depth",
            ["Summary", "Detailed", "Comprehensive"],
            index=1
        )
    
    # Generate bot report
    if st.button("🤖 Generate Bot Report", type="primary"):
        with st.spinner("Analyzing bot behavior..."):
            bot_analysis = analyze_bot_behavior(data, bot_types, analysis_depth)
            
            # Display bot metrics
            st.subheader("Bot Activity Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Bot Requests", f"{bot_analysis['total_bot_requests']:,}")
                st.metric("Unique Bot IPs", bot_analysis['unique_bot_ips'])
            
            with col2:
                st.metric("Verified Bots", f"{bot_analysis['verified_percentage']:.1f}%")
                st.metric("Crawl Frequency", f"{bot_analysis['avg_daily_crawls']:.0f}/day")
            
            with col3:
                st.metric("Bot Types Detected", bot_analysis['bot_types_count'])
                st.metric("Suspicious Activity", bot_analysis['suspicious_count'])
            
            # Bot behavior patterns
            if analysis_depth in ["Detailed", "Comprehensive"]:
                st.subheader("Bot Behavior Patterns")
                
                # Create bot timeline
                fig = create_bot_timeline(bot_analysis['timeline_data'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Bot comparison table
                st.subheader("Bot Comparison")
                st.dataframe(bot_analysis['comparison_table'], use_container_width=True)
            
            # Export options
            if st.button("📥 Export Bot Analysis"):
                bot_buffer = exporter.generate_bot_report(bot_analysis, format)
                download_report(bot_buffer, f"bot_analysis.{format.lower()}", get_mime_type(format))

def generate_performance_report(data, exporter, format):
    """Generate performance report"""
    st.header("⚡ Performance Report")
    
    # Performance metrics configuration
    st.subheader("Performance Metrics")
    
    metrics_to_include = st.multiselect(
        "Select Metrics",
        [
            "Response Times",
            "Page Load Speed",
            "Server Errors",
            "Slow Endpoints",
            "Resource Usage",
            "Cache Performance"
        ],
        default=["Response Times", "Page Load Speed", "Server Errors"]
    )
    
    # Generate performance report
    if st.button("⚡ Generate Performance Report", type="primary"):
        with st.spinner("Analyzing performance..."):
            perf_data = analyze_performance(data, metrics_to_include)
            
            # Display performance summary
            st.subheader("Performance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Avg Response Time",
                    f"{perf_data['avg_response_time']:.0f}ms",
                    delta=f"{perf_data['response_time_change']:.1f}%"
                )
            
            with col2:
                st.metric(
                    "P95 Response Time",
                    f"{perf_data['p95_response_time']:.0f}ms"
                )
            
            with col3:
                st.metric(
                    "Error Rate",
                    f"{perf_data['error_rate']:.2f}%",
                    delta=f"{perf_data['error_rate_change']:.1f}%",
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "Slow Pages",
                    perf_data['slow_pages_count'],
                    help="Pages > 3s response time"
                )
            
            # Performance charts
            if perf_data.get('charts'):
                for chart_name, chart_fig in perf_data['charts'].items():
                    st.plotly_chart(chart_fig, use_container_width=True)
            
            # Export performance report
            if st.button("📥 Export Performance Report"):
                perf_buffer = exporter.generate_performance_report(perf_data, format)
                download_report(perf_buffer, f"performance_report.{format.lower()}", get_mime_type(format))

def generate_crawl_budget_report(data, exporter, format):
    """Generate crawl budget optimization report"""
    st.header("🎯 Crawl Budget Report")
    
    # Crawl budget analysis
    st.subheader("Crawl Budget Analysis")
    
    # Run analysis
    if st.button("🎯 Analyze Crawl Budget", type="primary"):
        with st.spinner("Analyzing crawl budget..."):
            crawl_analysis = analyze_crawl_budget(data)
            
            # Display crawl budget metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Daily Crawl Budget", f"{crawl_analysis['daily_budget']:,}")
                st.metric("Budget Utilization", f"{crawl_analysis['utilization']:.1f}%")
            
            with col2:
                st.metric("Crawl Efficiency", f"{crawl_analysis['efficiency']:.1f}%")
                st.metric("Wasted Crawls", f"{crawl_analysis['wasted_percentage']:.1f}%")
            
            with col3:
                st.metric("Priority Pages Crawled", f"{crawl_analysis['priority_coverage']:.1f}%")
                st.metric("Optimization Potential", crawl_analysis['optimization_potential'])
            
            # Recommendations
            st.subheader("💡 Optimization Recommendations")
            
            for rec in crawl_analysis['recommendations']:
                if rec['impact'] == 'high':
                    st.error(f"🔴 High Impact: {rec['text']}")
                elif rec['impact'] == 'medium':
                    st.warning(f"🟡 Medium Impact: {rec['text']}")
                else:
                    st.info(f"🟢 Low Impact: {rec['text']}")
            
            # Export crawl budget report
            if st.button("📥 Export Crawl Budget Report"):
                crawl_buffer = exporter.generate_crawl_report(crawl_analysis, format)
                download_report(crawl_buffer, f"crawl_budget.{format.lower()}", get_mime_type(format))

def generate_custom_report(data, exporter, format):
    """Generate custom report with selected components"""
    st.header("🛠️ Custom Report Builder")
    
    # Report components selection
    st.subheader("Select Report Components")
    
    col1, col2 = st.columns(2)
    
    with col1:
        components = st.multiselect(
            "Report Sections",
            [
                "Overview Metrics",
                "Traffic Analysis",
                "Bot Detection",
                "Error Analysis",
                "Performance Metrics",
                "SEO Insights",
                "Crawl Budget",
                "Security Analysis"
            ],
            default=["Overview Metrics", "Traffic Analysis"]
        )
    
    with col2:
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        include_visualizations = st.checkbox("Include Charts", value=True)
        include_ai_insights = st.checkbox("Include AI Insights", value=True)
    
    # Time range
    if 'timestamp' in data.columns:
        st.subheader("Time Range")
        
        min_date = pd.to_datetime(data['timestamp']).min().date()
        max_date = pd.to_datetime(data['timestamp']).max().date()
        
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Generate custom report
    if st.button("🚀 Generate Custom Report", type="primary"):
        with st.spinner("Building custom report..."):
            custom_data = build_custom_report(
                data,
                components,
                include_raw_data,
                include_visualizations,
                include_ai_insights,
                date_range if 'date_range' in locals() else None
            )
            
            # Preview report structure
            st.subheader("Report Preview")
            
            st.json({
                "sections": components,
                "total_pages": custom_data.get('page_count', 'N/A'),
                "data_points": len(data),
                "format": format
            })
            
            # Export custom report
            if st.button("📥 Export Custom Report"):
                custom_buffer = exporter.generate_custom_report(custom_data, format)
                download_report(custom_buffer, f"custom_report.{format.lower()}", get_mime_type(format))

def calculate_executive_metrics(data):
    """Calculate executive summary metrics"""
    metrics = {}
    
    metrics['total_requests'] = len(data)
    metrics['unique_urls'] = data['url'].nunique() if 'url' in data.columns else 0
    
    if 'is_bot' in data.columns:
        metrics['bot_percentage'] = (data['is_bot'].sum() / len(data) * 100)
    else:
        metrics['bot_percentage'] = 0
    
    if 'status' in data.columns:
        error_codes = data[data['status'] >= 400]
        metrics['error_rate'] = (len(error_codes) / len(data) * 100)
    else:
        metrics['error_rate'] = 0
    
    if 'response_time' in data.columns:
        metrics['avg_response_time'] = data['response_time'].mean()
    else:
        metrics['avg_response_time'] = 0
    
    # Calculate crawl efficiency
    if 'is_bot' in data.columns and 'status' in data.columns:
        bot_data = data[data['is_bot'] == True]
        successful_crawls = bot_data[bot_data['status'] == 200]
        metrics['crawl_efficiency'] = (len(successful_crawls) / len(bot_data) * 100) if len(bot_data) > 0 else 0
    else:
        metrics['crawl_efficiency'] = 0
    
    # Calculate SEO score (simplified)
    seo_score = 100
    if metrics['error_rate'] > 5:
        seo_score -= 20
    if metrics['error_rate'] > 10:
        seo_score -= 20
    if metrics['crawl_efficiency'] < 80:
        seo_score -= 15
    if metrics['avg_response_time'] > 1000:
        seo_score -= 15
    
    metrics['seo_score'] = max(0, seo_score)
    
    # Count critical issues
    critical_issues = 0
    if metrics['error_rate'] > 10:
        critical_issues += 1
    if metrics['crawl_efficiency'] < 50:
        critical_issues += 1
    if metrics['avg_response_time'] > 3000:
        critical_issues += 1
    
    metrics['critical_issues'] = critical_issues
    
    return metrics

def generate_executive_insights(metrics):
    """Generate executive insights based on metrics"""
    insights = []
    
    if metrics['bot_percentage'] > 30:
        insights.append({
            'type': 'success',
            'text': f"Strong search engine interest with {metrics['bot_percentage']:.1f}% bot traffic"
        })
    elif metrics['bot_percentage'] < 10:
        insights.append({
            'type': 'warning',
            'text': f"Low search engine visibility - only {metrics['bot_percentage']:.1f}% bot traffic"
        })
    
    if metrics['error_rate'] > 5:
        insights.append({
            'type': 'error',
            'text': f"High error rate ({metrics['error_rate']:.1f}%) impacting user experience and SEO"
        })
    
    if metrics['crawl_efficiency'] < 80:
        insights.append({
            'type': 'warning',
            'text': f"Crawl efficiency at {metrics['crawl_efficiency']:.1f}% - optimize crawl budget"
        })
    
    if metrics['avg_response_time'] > 1000:
        insights.append({
            'type': 'warning',
            'text': f"Slow response times ({metrics['avg_response_time']:.0f}ms) may impact rankings"
        })
    
    if metrics['seo_score'] >= 80:
        insights.append({
            'type': 'success',
            'text': f"Strong SEO health score of {metrics['seo_score']}/100"
        })
    
    return insights

def run_technical_audit(data, audit_areas):
    """Run technical SEO audit"""
    results = {
        'total_issues': 0,
        'critical_issues': 0,
        'pages_affected': 0,
        'health_score': 100,
        'issues_by_category': {}
    }
    
    # Crawlability audit
    if "Crawlability" in audit_areas:
        crawl_issues = []
        
        if 'status' in data.columns:
            blocked_pages = data[data['status'] == 403]
            if len(blocked_pages) > 0:
                crawl_issues.append({
                    'severity': 'High',
                    'description': f"{len(blocked_pages)} pages blocked (403 Forbidden)",
                    'affected_urls': blocked_pages['url'].tolist() if 'url' in data.columns else []
                })
        
        results['issues_by_category']['Crawlability'] = crawl_issues
        results['total_issues'] += len(crawl_issues)
    
    # Add more audit logic for other areas...
    
    return results

def analyze_bot_behavior(data, bot_types, depth):
    """Analyze bot behavior patterns"""
    analysis = {
        'total_bot_requests': 0,
        'unique_bot_ips': 0,
        'verified_percentage': 0,
        'avg_daily_crawls': 0,
        'bot_types_count': len(bot_types),
        'suspicious_count': 0
    }
    
    if 'is_bot' in data.columns:
        bot_data = data[data['is_bot'] == True]
        analysis['total_bot_requests'] = len(bot_data)
        
        if 'ip' in bot_data.columns:
            analysis['unique_bot_ips'] = bot_data['ip'].nunique()
        
        if 'bot_verified' in bot_data.columns:
            verified = bot_data['bot_verified'].sum()
            analysis['verified_percentage'] = (verified / len(bot_data) * 100) if len(bot_data) > 0 else 0
        
        if 'timestamp' in bot_data.columns:
            days = (pd.to_datetime(bot_data['timestamp']).max() - pd.to_datetime(bot_data['timestamp']).min()).days
            analysis['avg_daily_crawls'] = len(bot_data) / max(days, 1)
    
    return analysis

def analyze_performance(data, metrics):
    """Analyze performance metrics"""
    perf_data = {
        'avg_response_time': 0,
        'p95_response_time': 0,
        'error_rate': 0,
        'slow_pages_count': 0,
        'response_time_change': 0,
        'error_rate_change': 0
    }
    
    if 'response_time' in data.columns:
        perf_data['avg_response_time'] = data['response_time'].mean()
        perf_data['p95_response_time'] = data['response_time'].quantile(0.95)
        perf_data['slow_pages_count'] = len(data[data['response_time'] > 3000])
    
    if 'status' in data.columns:
        errors = data[data['status'] >= 400]
        perf_data['error_rate'] = (len(errors) / len(data) * 100)
    
    return perf_data

def analyze_crawl_budget(data):
    """Analyze crawl budget utilization"""
    analysis = {
        'daily_budget': 0,
        'utilization': 0,
        'efficiency': 0,
        'wasted_percentage': 0,
        'priority_coverage': 0,
        'optimization_potential': 'Medium',
        'recommendations': []
    }
    
    if 'is_bot' in data.columns:
        bot_data = data[data['is_bot'] == True]
        
        if 'timestamp' in bot_data.columns:
            days = (pd.to_datetime(bot_data['timestamp']).max() - pd.to_datetime(bot_data['timestamp']).min()).days
            analysis['daily_budget'] = len(bot_data) / max(days, 1)
    
    # Add recommendations
    if analysis['wasted_percentage'] > 20:
        analysis['recommendations'].append({
            'impact': 'high',
            'text': 'Reduce crawl waste by blocking low-value pages'
        })
    
    return analysis

def build_custom_report(data, components, include_raw, include_viz, include_ai, date_range):
    """Build custom report with selected components"""
    report = {
        'components': components,
        'data': {},
        'visualizations': {},
        'insights': []
    }
    
    # Filter data by date range if specified
    if date_range and 'timestamp' in data.columns:
        mask = (pd.to_datetime(data['timestamp']).dt.date >= date_range[0]) & \
               (pd.to_datetime(data['timestamp']).dt.date <= date_range[1])
        filtered_data = data[mask]
    else:
        filtered_data = data
    
    # Add selected components
    for component in components:
        if component == "Overview Metrics":
            report['data']['overview'] = calculate_executive_metrics(filtered_data)
    
    if include_raw:
        report['raw_data'] = filtered_data.to_dict('records')
    
    return report

def create_bot_timeline(timeline_data):
    """Create bot activity timeline chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timeline_data.get('dates', []),
        y=timeline_data.get('counts', []),
        mode='lines+markers',
        name='Bot Activity',
        line=dict(color='#8b5cf6', width=2)
    ))
    
    fig.update_layout(
        title="Bot Activity Timeline",
        xaxis_title="Date",
        yaxis_title="Bot Requests",
        height=400
    )
    
    return fig

def download_report(content, filename, mime_type):
    """Create download button for report"""
    if isinstance(content, (io.BytesIO, io.StringIO)):
        content = content.getvalue()
    
    if isinstance(content, str):
        b64 = base64.b64encode(content.encode()).decode()
    else:
        b64 = base64.b64encode(content).decode()
    
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" class="btn btn-primary">📥 Download {filename}</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success(f"✅ Report generated successfully!")

def get_mime_type(format):
    """Get MIME type for file format"""
    mime_types = {
        'PDF': 'application/pdf',
        'Excel': 'application/vnd.ms-excel',
        'CSV': 'text/csv',
        'JSON': 'application/json',
        'HTML': 'text/html'
    }
    return mime_types.get(format, 'application/octet-stream')

if __name__ == "__main__":
    main()
