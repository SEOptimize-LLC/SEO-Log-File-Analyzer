# utils/export_handler.py

"""
Export functionality for SEO Log Analyzer
"""
import pandas as pd
import json
import io
from typing import Dict, List, Any, Optional
from datetime import datetime
import streamlit as st
import xlsxwriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
import plotly.io as pio
from config import EXPORT_FORMATS, MAX_EXPORT_ROWS

class ExportHandler:
    """Handle various export formats"""
    
    def __init__(self):
        self.formats = EXPORT_FORMATS
        self.max_rows = MAX_EXPORT_ROWS
    
    def export_data(self, data: pd.DataFrame, format: str, filename: str = None) -> bytes:
        """
        Export data in specified format
        
        Args:
            data: DataFrame to export
            format: Export format (CSV, Excel, JSON, PDF)
            filename: Optional filename
            
        Returns:
            Bytes data for download
        """
        if format.upper() == 'CSV':
            return self.export_csv(data)
        elif format.upper() == 'EXCEL':
            return self.export_excel(data)
        elif format.upper() == 'JSON':
            return self.export_json(data)
        elif format.upper() == 'PDF':
            return self.export_pdf(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_csv(self, data: pd.DataFrame) -> bytes:
        """Export DataFrame to CSV"""
        # Limit rows if needed
        if len(data) > self.max_rows:
            data = data.head(self.max_rows)
            st.warning(f"Data limited to {self.max_rows:,} rows for export")
        
        return data.to_csv(index=False).encode('utf-8')
    
    def export_excel(self, data: pd.DataFrame) -> bytes:
        """Export DataFrame to Excel with formatting"""
        output = io.BytesIO()
        
        # Limit rows if needed
        if len(data) > self.max_rows:
            data = data.head(self.max_rows)
            st.warning(f"Data limited to {self.max_rows:,} rows for export")
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write main data
            data.to_excel(writer, sheet_name='Log Data', index=False)
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Log Data']
            
            # Add formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BD',
                'border': 1
            })
            
            # Apply header format
            for col_num, value in enumerate(data.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths
            for idx, col in enumerate(data.columns):
                series = data[col]
                max_len = max(
                    series.astype(str).map(len).max(),
                    len(str(series.name))
                ) + 2
                worksheet.set_column(idx, idx, min(max_len, 50))
            
            # Add summary sheet if analysis results available
            if 'summary_stats' in st.session_state:
                self._add_summary_sheet(writer, st.session_state.summary_stats)
        
        return output.getvalue()
    
    def export_json(self, data: pd.DataFrame) -> bytes:
        """Export DataFrame to JSON"""
        # Limit rows if needed
        if len(data) > self.max_rows:
            data = data.head(self.max_rows)
            st.warning(f"Data limited to {self.max_rows:,} rows for export")
        
        # Convert datetime columns to string
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                data[col] = data[col].astype(str)
        
        return data.to_json(orient='records', indent=2).encode('utf-8')
    
    def export_pdf(self, data: Dict) -> bytes:
        """Export analysis report to PDF"""
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        story.append(Paragraph("SEO Log File Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Metadata
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        if 'summary' in data:
            for key, value in data['summary'].items():
                story.append(Paragraph(f"• {key}: {value}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Key Metrics
        story.append(Paragraph("Key Metrics", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        if 'metrics' in data:
            metrics_data = [['Metric', 'Value']]
            for key, value in data['metrics'].items():
                metrics_data.append([key, str(value)])
            
            metrics_table = Table(metrics_data)
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
        
        story.append(PageBreak())
        
        # Performance Analysis
        story.append(Paragraph("Performance Analysis", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        if 'performance' in data:
            for key, value in data['performance'].items():
                story.append(Paragraph(f"{key}: {value}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # SEO Insights
        story.append(Paragraph("SEO Insights", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        if 'seo_insights' in data:
            for insight in data['seo_insights']:
                story.append(Paragraph(f"• {insight}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        if 'recommendations' in data:
            for idx, rec in enumerate(data['recommendations'], 1):
                story.append(Paragraph(f"{idx}. {rec}", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return output.getvalue()
    
    def _add_summary_sheet(self, writer: pd.ExcelWriter, summary_data: Dict):
        """Add summary sheet to Excel export"""
        summary_df = pd.DataFrame([summary_data])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Summary']
        
        # Format summary sheet
        header_format = workbook.add_format({
            'bold': True,
            'fg_color': '#D7E4BD',
            'border': 1
        })
        
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)

def generate_pdf_report(
    analysis_results: Dict,
    performance_data: Dict,
    seo_data: Dict,
    bot_data: Dict
) -> bytes:
    """
    Generate comprehensive PDF report
    
    Args:
        analysis_results: General analysis results
        performance_data: Performance analysis data
        seo_data: SEO analysis data
        bot_data: Bot detection data
        
    Returns:
        PDF bytes
    """
    report_data = {
        'summary': {
            'Total Requests': analysis_results.get('total_requests', 0),
            'Unique Visitors': analysis_results.get('unique_visitors', 0),
            'Date Range': analysis_results.get('date_range', 'N/A'),
            'Bot Traffic %': f"{bot_data.get('bot_percentage', 0):.1f}%"
        },
        'metrics': {
            'Avg Response Time': f"{performance_data.get('avg_response_time', 0):.0f}ms",
            'P95 Response Time': f"{performance_data.get('p95_response_time', 0):.0f}ms",
            'Error Rate': f"{analysis_results.get('error_rate', 0):.2f}%",
            'Crawl Budget Efficiency': f"{seo_data.get('crawl_efficiency', 0):.1f}%"
        },
        'performance': performance_data,
        'seo_insights': [
            f"Crawl budget efficiency: {seo_data.get('crawl_efficiency', 0):.1f}%",
            f"Orphan pages detected: {seo_data.get('orphan_pages_count', 0)}",
            f"Mobile crawler ratio: {seo_data.get('mobile_ratio', 0):.1f}%",
            f"JavaScript rendering: {seo_data.get('js_rendering_ratio', 0):.1f}%"
        ],
        'recommendations': generate_recommendations(analysis_results, performance_data, seo_data)
    }
    
    handler = ExportHandler()
    return handler.export_pdf(report_data)

def export_to_excel(
    dataframes: Dict[str, pd.DataFrame],
    filename: str = "seo_analysis.xlsx"
) -> bytes:
    """
    Export multiple DataFrames to Excel with multiple sheets
    
    Args:
        dataframes: Dictionary of sheet_name: DataFrame
        filename: Output filename
        
    Returns:
        Excel file bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
            
            # Format sheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name[:31]]
            
            # Header format
            header_format = workbook.add_format({
                'bold': True,
                'fg_color': '#D7E4BD',
                'border': 1
            })
            
            # Apply formatting
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust columns
            for idx, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(idx, idx, min(max_len, 50))
    
    return output.getvalue()

def generate_recommendations(
    analysis_results: Dict,
    performance_data: Dict,
    seo_data: Dict
) -> List[str]:
    """
    Generate actionable recommendations based on analysis
    
    Args:
        analysis_results: General analysis results
        performance_data: Performance analysis data
        seo_data: SEO analysis data
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # Performance recommendations
    if performance_data.get('avg_response_time', 0) > 3000:
        recommendations.append("Critical: Average response time exceeds 3 seconds. Implement caching and optimize server performance.")
    elif performance_data.get('avg_response_time', 0) > 1000:
        recommendations.append("Warning: Response times are above optimal. Consider performance optimization.")
    
    # Error rate recommendations
    if analysis_results.get('error_rate', 0) > 5:
        recommendations.append("High error rate detected. Investigate and fix 404 and 5xx errors immediately.")
    
    # Crawl budget recommendations
    if seo_data.get('crawl_efficiency', 100) < 80:
        recommendations.append("Crawl budget is being wasted. Fix errors and optimize crawl paths.")
    
    # Orphan pages recommendations
    if seo_data.get('orphan_pages_count', 0) > 10:
        recommendations.append(f"Found {seo_data.get('orphan_pages_count', 0)} orphan pages. Add internal links to improve discoverability.")
    
    # Mobile recommendations
    if seo_data.get('mobile_ratio', 0) < 40:
        recommendations.append("Low mobile crawler activity detected. Ensure mobile optimization and test mobile usability.")
    
    # Bot traffic recommendations
    if seo_data.get('unverified_bot_percentage', 0) > 20:
        recommendations.append("High percentage of unverified bot traffic. Consider implementing bot verification.")
    
    # JavaScript recommendations
    if seo_data.get('js_rendering_ratio', 0) > 50:
        recommendations.append("High JavaScript dependency detected. Implement server-side rendering for critical content.")
    
    # If no issues found
    if not recommendations:
        recommendations.append("✅ No critical issues detected. Continue monitoring for optimal performance.")
    
    return recommendations

def create_download_button(
    data: bytes,
    filename: str,
    file_type: str,
    button_text: str = "Download"
) -> None:
    """
    Create a download button in Streamlit
    
    Args:
        data: File data in bytes
        filename: Name of the file
        file_type: MIME type
        button_text: Button label
    """
    st.download_button(
        label=button_text,
        data=data,
        file_name=filename,
        mime=file_type,
        help=f"Click to download {filename}"
    )
