# AI-Based Stock Price Prediction System

An intelligent stock analysis backend that combines **time-series forecasting**, 
**news sentiment analysis**, and **real-time financial APIs** to predict 
BSE stock prices.

## Features
- Time-series based price prediction for BSE-listed stocks
- News sentiment analysis to capture market mood
- Backtesting engine to validate model accuracy
- RESTful API with versioned endpoints
- Interactive data visualizations

## API Endpoints
| Endpoint | Description |
|---|---|
| `GET /api/v1/health` | Service health check |
| `GET /api/v1/analyze/<bse_code>` | Full analysis for a stock |
| `GET /api/v1/sentiment/<bse_code>` | News sentiment score |
| `GET /api/v1/price/<bse_code>` | Current & predicted price |

## Tech Stack
- **Backend:** Python, Flask
- **ML/AI:** Time-series forecasting, NLP sentiment analysis
- **Data:** Real-time financial APIs, historical OHLCV data
- **Other:** Backtesting engine, modular service architecture

## Project Structure
```
├── app.py               # Flask app entry point
├── api/routes.py        # API route definitions
├── services/            # Prediction, sentiment, price services
├── config/settings.py   # Configuration & stock mappings
├── utils/               # Helper utilities
├── backtest_engine.py   # Historical backtesting
└── run_backtest.py      # Backtest runner script
```

## Setup
```bash
pip install -r requirements.txt
python app.py
```
