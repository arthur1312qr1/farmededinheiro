import os
import logging
from flask import Flask
from flask_socketio import SocketIO
import threading

# Patch standard libraries to cooperate with eventlet
# This is crucial and must be done at the very beginning
from eventlet import monkey_patch
monkey_patch()

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

# Import routes and websocket handlers
from routes import *
from websocket_handler import *

# Define the trading bot variable
trading_bot = None

# Function to create and run the trading bot
def start_bot():
    global trading_bot
    if trading_bot is None:
        from trading_bot import TradingBot
        trading_bot = TradingBot(socketio)
        logger.info("🤖 Trading Bot created and ready")
        # Start the bot, which should run in a separate thread
        bot_thread = threading.Thread(target=trading_bot.run)
        bot_thread.daemon = True
        bot_thread.start()
        logger.info("🚀 Trading Bot started - REAL MONEY MODE")

# Create a custom Gunicorn hook to run the bot when the worker starts
def on_starting(server):
    logger.info("Server is starting. Initializing bot...")
    with app.app_context():
        start_bot()

# This part is only for local development, it will be ignored by Gunicorn
if __name__ == '__main__':
    with app.app_context():
        start_bot()
    socketio.run(app, port=os.environ.get('PORT', 5000))

# To get Gunicorn to recognize the hook, you need to use a config file.
# However, for simplicity on Render, we can rely on a different approach
# by checking for the Gunicorn worker process.
# We will use this in combination with the `create_trading_bot` function
# inside a specific route or event handler, but the simplest approach is to
# create a separate worker script.

# Since you are using Gunicorn, the best practice is to separate the web server
# from the trading bot logic. The bot should run as a background task.
# We'll use a `threading.Thread` to run the bot.
# The previous solution of using a `main.py` is more robust.
