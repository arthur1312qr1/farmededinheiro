import os
import logging

logger = logging.getLogger(__name__)

class Config:
    # Configurações básicas
    SECRET_KEY = os.environ.get('SECRET_KEY', 'trading-bot-secret-key-2024')
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    # Configurações do Render que você definiu
    PAPER_TRADING = os.environ.get('PAPER_TRADING', 'false').lower() == 'true'
    PORT = int(os.environ.get('PORT', 5000))
    
    # Trading
    LEVERAGE = int(os.environ.get('LEVERAGE', 10))
    MIN_LEVERAGE = int(os.environ.get('MIN_LEVERAGE', 10))
    MAX_LEVERAGE = int(os.environ.get('MAX_LEVERAGE', 10))
    SYMBOL = os.environ.get('SYMBOL', 'ethusdt_UMCBL')
    MIN_MARGIN_USAGE_PERCENT = float(os.environ.get('MIN_MARGIN_USAGE_PERCENT', 80.0))
    POLL_INTERVAL = float(os.environ.get('POLL_INTERVAL', 1.0))
    DRAWDOWN_CLOSE_PCT = float(os.environ.get('DRAWDOWN_CLOSE_PCT', 0.03))
    LIQ_DIST_THRESHOLD = float(os.environ.get('LIQ_DIST_THRESHOLD', 0.03))
    MAX_CONSECUTIVE_LOSSES = int(os.environ.get('MAX_CONSECUTIVE_LOSSES', 5))
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))
    RETRY_DELAY = float(os.environ.get('RETRY_DELAY', 5.0))
    
    # API Keys
    BITGET_API_KEY = os.environ.get('BITGET_API_KEY', '')
    BITGET_API_SECRET = os.environ.get('BITGET_API_SECRET', '')
    BITGET_PASSPHRASE = os.environ.get('BITGET_PASSPHRASE', '')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    
    # Configurações de scalping otimizadas
    MIN_CONFIDENCE_SCORE = 60.0  # 60% para mais trades
    MIN_TRADE_INTERVAL = 30.0    # 30 segundos
    MAX_DAILY_TRADES = 200       # 200 trades por dia
    
    def __init__(self):
        logger.info("⚙️ Configuração carregada")
        logger.info(f"📊 Paper Trading: {self.PAPER_TRADING}")
        logger.info(f"⚡ Símbolo: {self.SYMBOL}")
        logger.info(f"📈 Alavancagem: {self.LEVERAGE}x")
