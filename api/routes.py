# api/routes.py
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import logging
from services.sent_an import OptimizedSentimentAnalyzer
from services.price_pred import OptimizedPricePredictor
from services.signal_fuser import SignalFuser
from utils.helpers import get_bse_client

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize services function
def init_services(config):
    sentiment_analyzer = OptimizedSentimentAnalyzer(config)
    price_predictor = OptimizedPricePredictor(config)
    signal_fuser = SignalFuser(config)
    bse_client = get_bse_client(config)
    return sentiment_analyzer, price_predictor, signal_fuser, bse_client

@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'trading_backend',
        'version': '1.0.0'
    })

@api_bp.route('/analyze', methods=['POST'])
def analyze_stock():
    """Unified analysis endpoint - returns both sentiment and price prediction"""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'bse_code' not in data:
            return jsonify({'error': 'Missing bse_code in request body'}), 400
        
        bse_code = data['bse_code']
        
        # Get config from current_app
        config = current_app.config['TRADING_CONFIG']
        sentiment_analyzer, price_predictor, signal_fuser, bse_client = init_services(config)
        
        # Get parameters
        days = data.get('days', config.SENTIMENT_LOOKBACK_DAYS)
        
        # Check if stock is supported
        if bse_code not in config.STOCK_MAPPING:
            return jsonify({'error': f'Stock {bse_code} not supported'}), 404
        
        yahoo_symbol, company_name = config.STOCK_MAPPING[bse_code]
        
        # Get sentiment analysis
        announcements = bse_client.get_corporate_announcements(bse_code, days=days)
        sentiment_result = sentiment_analyzer.analyze_news_sentiment(announcements)
        
        # Get price prediction
        price_result = price_predictor.predict_next_day(yahoo_symbol)
        
        # Fuse signals
        if sentiment_result and price_result:
            fused_signal = signal_fuser.fuse_signals(sentiment_result, price_result)
        else:
            fused_signal = signal_fuser._get_neutral_signal()
        
        response = {
            'bse_code': bse_code,
            'company_name': company_name,
            'yahoo_symbol': yahoo_symbol,
            'analysis_timestamp': datetime.now().isoformat(),
            'lookback_days': days,
            'sentiment_analysis': sentiment_result,
            'price_prediction': price_result,
            'fused_signal': fused_signal,
            'announcements': announcements
        }
        
        return jsonify(response)
        
    except Exception as e:
        logging.error(f"Analysis error for {bse_code}: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'message': str(e)}), 500

@api_bp.route('/sentiment', methods=['POST'])
def sentiment_only():
    """Sentiment analysis only endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'bse_code' not in data:
            return jsonify({'error': 'Missing bse_code in request body'}), 400
        
        bse_code = data['bse_code']
        
        config = current_app.config['TRADING_CONFIG']
        sentiment_analyzer, _, _, bse_client = init_services(config)
        
        days = data.get('days', config.SENTIMENT_LOOKBACK_DAYS)
        announcements = bse_client.get_corporate_announcements(bse_code, days=days)
        result = sentiment_analyzer.analyze_news_sentiment(announcements)
        
        return jsonify({
            'bse_code': bse_code,
            'lookback_days': days,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_bp.route('/price', methods=['POST'])
def price_only():
    """Price prediction only endpoint"""
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'bse_code' not in data:
            return jsonify({'error': 'Missing bse_code in request body'}), 400
        
        bse_code = data['bse_code']
        
        config = current_app.config['TRADING_CONFIG']
        _, price_predictor, _, _ = init_services(config)
        
        if bse_code not in config.STOCK_MAPPING:
            return jsonify({'error': 'Stock not supported'}), 404
        
        yahoo_symbol, company_name = config.STOCK_MAPPING[bse_code]
        result = price_predictor.predict_next_day(yahoo_symbol)
        
        return jsonify({
            'bse_code': bse_code,
            'company_name': company_name,
            'yahoo_symbol': yahoo_symbol,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500