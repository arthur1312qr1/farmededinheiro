import os
from dotenv import load_dotenv

# Load environment variables from a .env file (for local development)
load_dotenv()

class Config:
    # Render.com environment variables
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'false').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    LEVERAGE = int(os.getenv('LEVERAGE', 10))
    MIN_LEVERAGE = int(os.getenv('MIN_LEVERAGE', 10))
    MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', 10))
    SYMBOL = os.getenv('SYMBOL', 'ethusdt_UMCBL')
    MIN_MARGIN_USAGE_PERCENT = float(os.getenv('MIN_MARGIN_USAGE_PERCENT', 80.0))
    POLL_INTERVAL = float(os.getenv('POLL_INTERVAL', 1.0))
    DRAWDOWN_CLOSE_PCT = float(os.getenv('DRAWDOWN_CLOSE_PCT', 0.03))
    LIQ_DIST_THRESHOLD = float(os.getenv('LIQ_DIST_THRESHOLD', 0.03))
    MAX_CONSECUTIVE_LOSSES = int(os.getenv('MAX_CONSECUTIVE_LOSSES', 5))
    MIN_BALANCE_USDT = float(os.getenv('MIN_BALANCE_USDT', 50.0))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
    RETRY_DELAY = float(os.getenv('RETRY_DELAY', 5.0))
    
    # API Keys
    BITGET_API_KEY = os.getenv('BITGET_API_KEY')
    BITGET_API_SECRET = os.getenv('BITGET_API_SECRET')
    BITGET_PASSPHRASE = os.getenv('BITGET_PASSPHRASE')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    SESSION_SECRET = os.getenv('SESSION_SECRET', 'dev-secret-key')

    # Important: Base URL for Bitget API
    BITGET_BASE_URL = "https://api.bitget.com"
    
    # Internal bot configurations
    ANALYSIS_INTERVAL = 30.0  # seconds
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.05
