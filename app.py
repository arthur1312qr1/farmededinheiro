import os
import logging
from flask import Flask
from flask_socketio import SocketIO
import eventlet
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev-secret-key')

# Initialize SocketIO with eventlet
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    logger=True,
    engineio_logger=True
)

# Initialize trading bot
trading_bot = None

def create_trading_bot():
    """Create and configure trading bot"""
    global trading_bot
    if trading_bot is None:
        from trading_bot import TradingBot
        trading_bot = TradingBot(socketio)
        logger.info("🤖 Trading Bot created and ready")
        # Iniciar o bot automaticamente
        if not trading_bot.is_running:
            trading_bot.start()
    return trading_bot

# Import routes after app creation to avoid circular imports
from routes import *
from websocket_handler import *

# Chamar a função para criar e iniciar o bot
# Isso garante que ele esteja pronto quando o servidor iniciar
create_trading_bot()

# Entry point moved to main.py for proper deployment
