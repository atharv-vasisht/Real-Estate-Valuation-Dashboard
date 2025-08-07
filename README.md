# Real Estate Valuation & Forecasting Dashboard

A comprehensive real estate price forecasting system that combines machine learning models with an interactive web dashboard to predict housing prices across major US metropolitan areas from 2025 to 2050.

## 🎯 Project Objective

This project aims to provide accurate, long-term housing price forecasts for major US metropolitan areas using advanced machine learning techniques. The system combines:

- **LSTM Neural Networks** for capturing complex temporal patterns
- **Linear Regression** for trend analysis
- **Ensemble Methods** for improved accuracy
- **Interactive Dashboard** for easy data exploration and visualization

The forecasts help investors, real estate professionals, and policymakers understand potential housing market trends over the next 25 years.

## 📊 Data Sources

### Primary Dataset
- **Zillow Housing Data**: Cleaned dataset containing monthly median home prices for major US metropolitan areas
- **Time Period**: Historical data from 1996 to 2024
- **Coverage**: 100+ major metropolitan areas across the United States
- **Data Format**: Monthly median home prices in USD

### Data Processing
The raw Zillow data undergoes several preprocessing steps:
- Removal of missing values and outliers
- Normalization using MinMaxScaler
- Sequence preparation for LSTM training
- Feature engineering for growth rate calculations

## 🏗️ Architecture

### Backend (Python/Flask)
```
backend/
├── processing/
│   ├── future_forecast.py      # Core ML forecasting engine
│   ├── analyze_forecasts.py    # Visualization and analysis
│   └── data_processing.py      # Data preprocessing utilities
├── dashboard/
│   └── api.py                  # Flask REST API
└── data/
    └── zillow_housing_cleaned.csv  # Processed dataset
```

### Frontend (Next.js/React)
```
real-estate-forecast-2/
├── app/
│   ├── page.tsx               # Main dashboard component
│   ├── layout.tsx             # App layout
│   └── globals.css            # Global styles
└── components/
    └── ForecastChart.js       # Interactive chart component
```

## 🚀 How the Dashboard Works

### 1. Model Training & Forecasting
The `future_forecast.py` script implements a sophisticated forecasting pipeline:

**LSTM Architecture:**
- Input: 36-month sequences of normalized prices
- Hidden layers: 32 units with dropout (0.2)
- Output: Next month's predicted price
- Training: 50 epochs with early stopping

**Ensemble Method:**
- LSTM Weight: 70%
- Linear Regression Weight: 30%
- Combines both models for improved accuracy

**Post-Processing:**
- Exponential smoothing for trend adjustment
- Growth rate constraints to prevent unrealistic predictions
- Inflation adjustment to 2025 dollars (3% annual rate)

### 2. API Layer
The Flask API (`api.py`) provides three main endpoints:
- `/api/metros` - List all available metropolitan areas
- `/api/metro/<metro_name>` - Historical data for a specific metro
- `/api/forecast/<metro_name>` - Forecasted prices for a specific metro

### 3. Frontend Dashboard
The Next.js application provides:
- **Metro Selection**: Dropdown with all available metropolitan areas
- **Interactive Charts**: Real-time visualization using Recharts
- **Responsive Design**: Modern, fintech-inspired UI with Tailwind CSS
- **Real-time Updates**: Dynamic data fetching from the Flask API

## 🔧 Key Features

### Advanced Forecasting Techniques
1. **Bidirectional Growth Constraints**: Prevents unrealistic exponential growth or sudden drops
2. **Volatility-Based Penalties**: Reduces growth rates for historically volatile markets
3. **Inflated Price Penalties**: Applies higher constraints to already expensive markets
4. **Ensemble Modeling**: Combines LSTM and linear regression for robustness
5. **Exponential Smoothing**: Smooths predictions to reduce noise

### Performance Optimizations
- **Parallel Processing**: Multiprocessing for faster metro processing
- **GPU Acceleration**: TensorFlow GPU memory optimization
- **Memory Management**: Automatic cleanup to prevent memory leaks
- **Batch Processing**: Efficient training with batch size optimization

## 📈 Model Performance

The ensemble model addresses several common forecasting challenges:

- **Exponential Growth Mitigation**: Growth rate constraints prevent unrealistic price explosions
- **Volatility Handling**: Enhanced LSTM architecture with dropout layers
- **Market-Specific Adjustments**: Different constraints for different market types
- **Long-term Stability**: 25-year forecasts with realistic growth patterns

## ⚠️ Limitations & Assumptions

### Model Limitations
1. **Historical Data Dependency**: Forecasts assume historical patterns continue
2. **Market Shocks**: Cannot predict major economic disruptions or policy changes
3. **Local Factors**: Doesn't account for city-specific developments or zoning changes
4. **Data Quality**: Relies on Zillow's data accuracy and completeness

### Key Assumptions
1. **Inflation Rate**: Assumes 3% annual inflation for price adjustments
2. **Market Continuity**: Assumes no major structural changes in housing markets
3. **Economic Stability**: Assumes relatively stable economic conditions
4. **Population Growth**: Assumes continued population growth in major metros
5. **Interest Rates**: Assumes relatively stable mortgage interest rates

### Technical Assumptions
1. **Data Availability**: Assumes continued data availability from Zillow
2. **Model Stability**: Assumes LSTM patterns remain relevant over 25 years
3. **Computational Resources**: Assumes adequate processing power for model training

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd dashboard
python api.py
```

### Frontend Setup
```bash
cd real-estate-forecast-2
npm install
npm run dev
```

### Environment Configuration
Create `.env.local` in the frontend directory:
```
NEXT_PUBLIC_API_BASE_URL=http://your-ip:5002
```

## 📊 Usage

1. **Start the Backend**: Run the Flask API server
2. **Start the Frontend**: Run the Next.js development server
3. **Access Dashboard**: Navigate to `http://localhost:3000`
4. **Select Metro**: Choose a metropolitan area from the dropdown
5. **View Forecast**: Interactive chart shows historical and forecasted prices

## 🔮 Future Enhancements

- **Additional Models**: Integration of ARIMA, Prophet, or other time series models
- **Economic Indicators**: Incorporation of GDP, employment, and interest rate data
- **Geographic Granularity**: City-level and neighborhood-level forecasts
- **Risk Assessment**: Confidence intervals and uncertainty quantification
- **Real-time Updates**: Live data integration for current market conditions

## 📝 License

This project is for educational and research purposes. Please ensure compliance with data usage terms from Zillow and other data providers.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the forecasting models or dashboard functionality.

---

**Note**: This tool is designed for educational and research purposes. Real estate investments should be made based on comprehensive analysis including local market conditions, economic factors, and professional advice.
