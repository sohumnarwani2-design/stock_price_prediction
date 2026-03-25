import os
from typing import Dict, Any

class Config:
    # API Settings
    API_HOST = os.getenv('TRADING_API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('TRADING_API_PORT', '5000'))
    DEBUG = os.getenv('TRADING_DEBUG', 'False').lower() == 'true'
    
    # BSE API Settings
    BSE_RATE_LIMIT_DELAY = float(os.getenv('BSE_RATE_LIMIT_DELAY', '2.0'))
    BSE_MAX_RETRIES = int(os.getenv('BSE_MAX_RETRIES', '3'))
    
    # Sentiment Analysis
    SENTIMENT_LOOKBACK_DAYS = int(os.getenv('SENTIMENT_LOOKBACK_DAYS', '25'))
    POSITIVE_THRESHOLD = float(os.getenv('POSITIVE_THRESHOLD', '0.15'))
    NEGATIVE_THRESHOLD = float(os.getenv('NEGATIVE_THRESHOLD', '-0.15'))
    DECAY_RATE = float(os.getenv('DECAY_RATE', '0.25'))
    
    # Price Prediction
    PRICE_PREDICTION_PERIOD = os.getenv('PRICE_PREDICTION_PERIOD', '1y')
    PRICE_TRAIN_TEST_SPLIT = float(os.getenv('PRICE_TRAIN_TEST_SPLIT', '0.8'))
    
    # Signal Fusion
    SENTIMENT_WEIGHT = float(os.getenv('SENTIMENT_WEIGHT', '0.6'))
    TECHNICAL_WEIGHT = float(os.getenv('TECHNICAL_WEIGHT', '0.4'))
    
    # Available Stocks Mapping (BSE to Yahoo Finance)
    STOCK_MAPPING = {
        '532540': ('TCS.NS', 'Tata Consultancy Services'),
        '500209': ('INFY.NS', 'Infosys'),
        '500325': ('RELIANCE.NS', 'Reliance Industries'),
        '500180': ('HDFCBANK.NS', 'HDFC Bank'),
        '532174': ('ICICIBANK.NS', 'ICICI Bank'),
        '507685': ('WIPRO.NS', 'Wipro'),
        '500696': ('HINDUNILVR.NS', 'Hindustan Unilever'),
        '500875': ('ITC.NS', 'ITC Limited'),
        '500112': ('SBIN.NS', 'State Bank of India'),
        '500034': ('BAJFINANCE.NS', 'Bajaj Finance'),
        '532555': ('NTPC.NS', 'NTPC Limited')  
    }