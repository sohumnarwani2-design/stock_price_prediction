# app.py
from flask import Flask
from config.settings import Config
from api.routes import api_bp
import logging

def create_app():
    app = Flask(__name__)
    
    # Configuration - create instance, not class
    app.config['TRADING_CONFIG'] = Config()  # Note the parentheses!
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    # Health check at root
    @app.route('/')
    def root():
        return {
            'service': 'Trading Analysis Backend',
            'version': '1.0.0',
            'endpoints': {
                'health': '/api/v1/health',
                'analyze': '/api/v1/analyze/<bse_code>',
                'sentiment': '/api/v1/sentiment/<bse_code>',
                'price': '/api/v1/price/<bse_code>'
            }
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    config = app.config['TRADING_CONFIG']
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Starting Trading Backend on {config.API_HOST}:{config.API_PORT}")
    logger.info(f"📊 Supported stocks: {len(config.STOCK_MAPPING)}")
    
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )