# SEO Log File Analyzer

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced, AI-powered SEO log file analyzer built with Streamlit that provides comprehensive insights into server log data, search engine bot behavior, crawl budget optimization, and predictive analytics for improved SEO performance.

## 🚀 Features

### Core Analytics
- **Multi-format Log File Support**: Apache, Nginx, IIS, CloudFront, JSON structured logs
- **AI-Powered Bot Detection**: ML-based bot classification with DNS verification
- **Real-time Processing**: Efficient processing of large log files with sampling and chunking
- **Interactive Visualizations**: Plotly-powered charts and graphs for data exploration

### SEO Intelligence
- **Crawl Budget Analysis**: Identify crawl waste and optimization opportunities
- **Orphan Page Detection**: Find pages receiving traffic but lacking internal links
- **Internal Linking Analysis**: PageRank-based link structure evaluation
- **Mobile-First Indexing**: Mobile vs desktop crawler behavior analysis
- **International SEO**: Multi-language and geo-targeting insights

### Advanced Features
- **Predictive Analytics**: Prophet-based forecasting for crawl patterns and performance
- **Performance Monitoring**: Response time analysis and anomaly detection
- **Error Tracking**: Comprehensive 4xx/5xx error analysis and recommendations
- **Session Analysis**: User session identification and behavior patterns
- **Export Capabilities**: CSV, Excel, JSON, and PDF report generation

## 🏗️ Architecture

```
SEO-Log-File-Analyzer/
├── app.py                          # Main Streamlit application
├── config.py                       # Configuration settings and constants
├── requirements.txt                # Python dependencies
├── components/                     # Core components
│   ├── __init__.py
│   ├── bot_detector.py            # ML-powered bot detection
│   ├── cache_manager.py           # Data caching system
│   ├── log_parser.py              # Multi-format log parser
│   ├── performance_analyzer.py     # Performance metrics analysis
│   ├── seo_analyzer.py            # SEO-specific analysis
│   └── visualizations.py          # Chart and graph generation
├── models/                        # Machine learning models
│   ├── __init__.py
│   ├── ml_models.py               # Base ML models
│   └── predictive_analytics.py    # Forecasting models
├── pages/                         # Streamlit pages
│   ├── 1_📊_Overview.py           # Main dashboard
│   ├── 2_🤖_Bot_Analysis.py       # Bot behavior analysis
│   ├── 3_⚡_Performance.py        # Performance metrics
│   ├── 4_❌_Errors.py             # Error analysis
│   ├── 5_📈_SEO_Insights.py       # Advanced SEO analytics
│   └── 6_📥_Reports.py            # Report generation
└── utils/                         # Utility functions
    ├── __init__.py
    ├── data_processor.py          # Data processing utilities
    └── export_handler.py          # Export functionality
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM for large log file processing

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/SEOptimize-LLC/SEO-Log-File-Analyzer.git
cd SEO-Log-File-Analyzer
```

2. **Create a virtual environment**
```bash
python -m venv seo_analyzer_env
source seo_analyzer_env/bin/activate  # On Windows: seo_analyzer_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the application**
Open your browser and navigate to `http://localhost:8501`

## 📊 Usage

### Quick Start

1. **Upload Log File**: Drag and drop or select your server log file
2. **Process Data**: Click "Process File" to analyze the log data
3. **Explore Insights**: Navigate through different tabs for various analyses
4. **Generate Reports**: Export findings in your preferred format

### Supported Log Formats

#### Apache Common/Combined Log Format
```
192.168.1.1 - - [25/Dec/2023:10:00:00 +0000] "GET /page.html HTTP/1.1" 200 1234
```

#### Nginx Log Format
```
192.168.1.1 - user [25/Dec/2023:10:00:00 +0000] "GET /page.html HTTP/1.1" 200 1234 "http://example.com" "Mozilla/5.0..."
```

#### JSON Structured Logs
```json
{
  "timestamp": "2023-12-25T10:00:00Z",
  "ip": "192.168.1.1",
  "method": "GET",
  "url": "/page.html",
  "status": 200,
  "size": 1234,
  "user_agent": "Mozilla/5.0...",
  "response_time": 150
}
```

### Key Features by Page

#### 📊 Overview
- Traffic timeline and volume metrics
- Status code distribution
- Bot vs human traffic analysis
- Top pages and referrers

#### 🤖 Bot Analysis
- Search engine bot identification and verification
- Crawl frequency and patterns
- Bot behavior anomaly detection
- Fake bot identification using DNS verification

#### ⚡ Performance
- Response time analysis and percentiles
- Performance trend monitoring
- Slow page identification
- TTFB (Time to First Byte) analysis

#### 📈 SEO Insights
- **Crawl Budget Optimization**
  - Crawl waste identification
  - Over/under-crawled page analysis
  - Crawl efficiency recommendations

- **Content Discovery**
  - Orphan page detection
  - Content freshness impact
  - Discovery rate analysis

- **Internal Linking**
  - Link flow visualization
  - PageRank calculation
  - Dead-end page identification

- **Mobile vs Desktop Analysis**
  - Mobile-first indexing alignment
  - Device-specific crawl patterns
  - Coverage comparison

- **Predictive Analytics**
  - Crawl rate forecasting
  - SEO impact predictions
  - Trend analysis and alerts

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Application Settings
APP_NAME="SEO Log File Analyzer"
APP_VERSION="1.0.0"
DEBUG_MODE=False

# Processing Settings
MAX_FILE_SIZE_MB=1000
CHUNK_SIZE=10000
ENABLE_SAMPLING=True
SAMPLE_RATE=0.1

# Cache Settings
ENABLE_CACHE=True
CACHE_TTL=3600
MAX_CACHE_SIZE_MB=512

# Performance Thresholds
SLOW_RESPONSE_MS=1000
CRITICAL_RESPONSE_MS=3000
ERROR_RATE_THRESHOLD=0.05
```

### Advanced Configuration

Modify `config.py` to customize:
- Bot detection patterns
- Performance thresholds
- Alert configurations
- Export settings
- ML model parameters

## 🤖 Bot Detection

The application uses a multi-layered approach for bot detection:

1. **User Agent Pattern Matching**: Known bot signatures
2. **Behavioral Analysis**: Request patterns and frequency
3. **DNS Verification**: Reverse DNS lookup for claimed bots
4. **ML-based Detection**: Anomaly detection using Isolation Forest

### Supported Bots
- **Search Engines**: Googlebot, Bingbot, YandexBot, Baiduspider
- **Social Media**: FacebookBot, TwitterBot, LinkedInBot
- **AI Crawlers**: GPTBot, Claude-Web, ChatGPT-User
- **SEO Tools**: AhrefsBot, SemrushBot, MJ12bot

## 📈 Predictive Analytics

### Crawl Rate Forecasting
Uses Facebook Prophet for time series forecasting:
- Daily/weekly crawl patterns
- Seasonal trend analysis
- Confidence intervals
- Anomaly detection

### Performance Prediction
- Response time forecasting
- Error rate predictions
- Traffic volume projections
- Alert generation for threshold violations

## 🚨 Monitoring & Alerts

### Built-in Alert Types
- **Error Spike**: Sudden increase in 4xx/5xx errors
- **Bot Spike**: Unusual bot activity patterns
- **Performance Degradation**: Response time increases
- **Crawl Drop**: Significant decrease in crawl activity

### Custom Thresholds
Configure alert thresholds in `config.py`:
```python
ALERTS = {
    'error_spike': {'threshold': 0.1, 'window': '5min'},
    'bot_spike': {'threshold': 2.0, 'window': '1hour'},
    'performance_degradation': {'threshold': 1.5, 'window': '15min'},
    'crawl_drop': {'threshold': 0.5, 'window': '1day'},
}
```

## 📤 Export Options

### Available Formats
- **CSV**: Raw data export for further analysis
- **Excel**: Formatted spreadsheets with multiple sheets
- **JSON**: Structured data for API integration
- **PDF**: Executive summary reports

### Automated Reporting
Schedule automated reports using the export handler:
```python
from utils.export_handler import ExportHandler

exporter = ExportHandler()
exporter.generate_report(data, format='pdf', template='executive_summary')
```

## 🔒 Security & Privacy

- **Data Processing**: All processing happens locally
- **No External Calls**: Optional DNS verification only
- **Privacy First**: No data sent to external services
- **Secure Handling**: Temporary file cleanup and memory management

## 🧪 Testing

### Sample Data Generation
The application includes sample data generators for testing:
```python
from components.log_parser import LogParser

parser = LogParser()
sample_data = parser.get_sample_data(n_rows=1000)
```

### Performance Testing
- Tested with log files up to 10GB
- Memory optimization for large datasets
- Chunked processing for scalability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
flake8 .
```

## 📝 Changelog

### v1.0.0 (Current)
- Initial release with core functionality
- Multi-format log file support
- AI-powered bot detection
- Predictive analytics integration
- Interactive Streamlit interface

## 🐛 Known Issues

- Large log files (>5GB) may require increased memory
- DNS verification can slow down bot detection for large datasets
- Prophet forecasting requires minimum 7 days of data

## 📚 Documentation

### API Reference
Detailed documentation for all components available in `/docs`

### Tutorials
- [Getting Started Guide](docs/getting-started.md)
- [Advanced Configuration](docs/advanced-config.md)
- [Custom Bot Detection](docs/bot-detection.md)
- [Predictive Analytics](docs/predictive-analytics.md)

## 🙏 Acknowledgments

- **Streamlit** for the amazing app framework
- **Facebook Prophet** for time series forecasting
- **Plotly** for interactive visualizations
- **scikit-learn** for machine learning capabilities

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` folder
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Join our GitHub Discussions
- **Email**: support@seoptimize-llc.com

## 🔮 Roadmap

### v1.1.0 (Planned)
- [ ] Real-time log streaming support
- [ ] Advanced ML models for anomaly detection
- [ ] Integration with Google Search Console API
- [ ] Multi-site comparison features

### v1.2.0 (Future)
- [ ] Docker containerization
- [ ] REST API endpoints
- [ ] Cloud deployment templates
- [ ] Advanced alerting system

---

**Built with ❤️ by SEOptimize LLC**

*Making SEO data analysis accessible to everyone.*
