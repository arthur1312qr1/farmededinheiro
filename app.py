"""
Flask Application Factory for Farme de Dinheiro Trading Bot
Production-ready configuration with SocketIO and trading functionality
"""

import os
import logging
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# Configure logging
logger = logging.getLogger(__name__)

# Global SocketIO instance
socketio = SocketIO(
    cors_allowed_origins="*",
    async_mode='threading',
    logger=False,
    engineio_logger=False
)

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__)
    
    # Configuration
    app.config.update({
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production'),
        'DEBUG': os.environ.get('FLASK_DEBUG', 'False').lower() == 'true',
        'TESTING': False,
        
        # Trading Bot Configuration
        'PAPER_TRADING': os.environ.get('PAPER_TRADING', 'False').lower() == 'true',
        'SYMBOL': os.environ.get('SYMBOL', 'ethusdt_UMCBL'),
        'LEVERAGE': int(os.environ.get('LEVERAGE', '10')),
        'BASE_CURRENCY': os.environ.get('BASE_CURRENCY', 'USDT'),
        'TARGET_TRADES_PER_DAY': int(os.environ.get('TARGET_TRADES_PER_DAY', '200')),
        
        # API Keys
        'BITGET_API_KEY': os.environ.get('BITGET_API_KEY', ''),
        'BITGET_SECRET_KEY': os.environ.get('BITGET_SECRET_KEY', ''),
        'BITGET_PASSPHRASE': os.environ.get('BITGET_PASSPHRASE', ''),
        'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY', ''),
    })
    
    # Enable CORS for development
    CORS(app)
    
    # Initialize SocketIO with app
    socketio.init_app(app)
    
    # Register routes
    from server.routes import register_routes
    register_routes(app, socketio)
    
    # Initialize trading bot
    try:
        from server.trading_bot import TradingBot
        from server.socketio_manager import SocketIOManager
        
        # Create trading bot instance
        trading_bot = TradingBot(app.config)
        socketio_manager = SocketIOManager(socketio, trading_bot)
        
        # Store instances in app context
        app.trading_bot = trading_bot
        app.socketio_manager = socketio_manager
        
        logger.info("✅ Trading bot inicializado com sucesso")
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar trading bot: {e}")
        # Continue without trading bot for debugging
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return {
            'status': 'healthy',
            'trading_active': hasattr(app, 'trading_bot'),
            'paper_trading': app.config.get('PAPER_TRADING', True)
        }
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Endpoint não encontrado'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Erro interno: {error}")
        return {'error': 'Erro interno do servidor'}, 500
    
    logger.info("🔧 Flask app configurada com sucesso")
    return app

# For Gunicorn compatibility
app = create_app()

if __name__ == '__main__':
    # Development server
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
