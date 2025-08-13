import os
from dotenv import load_dotenv

# Carrega .env apenas se existir (para desenvolvimento local)
load_dotenv()

class Config:
    # Configurações básicas
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-apenas'
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://')
    
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Bot settings
    PAPER_TRADING = os.environ.get('PAPER_TRADING', 'true').lower() == 'true'
    
    # API Keys
    BITGET_API_KEY = os.environ.get('BITGET_API_KEY', '')
    BITGET_API_SECRET = os.environ.get('BITGET_API_SECRET', '')
    BITGET_PASSPHRASE = os.environ.get('BITGET_PASSPHRASE', '')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    
    # Trading - ALAVANCAGEM FIXA 10X
    SYMBOL = os.environ.get('SYMBOL', 'ethusdt_UMCBL')
    LEVERAGE = 10
    MIN_LEVERAGE = 10
    MAX_LEVERAGE = 10
    MIN_MARGIN_USAGE_PERCENT = float(os.environ.get('MIN_MARGIN_USAGE_PERCENT', '80.0'))
    POLL_INTERVAL = float(os.environ.get('POLL_INTERVAL', '1.0'))
    DRAWDOWN_CLOSE_PCT = float(os.environ.get('DRAWDOWN_CLOSE_PCT', '0.03'))
    LIQ_DIST_THRESHOLD = float(os.environ.get('LIQ_DIST_THRESHOLD', '0.03'))
    MAX_CONSECUTIVE_LOSSES = int(os.environ.get('MAX_CONSECUTIVE_LOSSES', '5'))
    MIN_BALANCE_USDT = float(os.environ.get('MIN_BALANCE_USDT', '50.0'))
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
    RETRY_DELAY = float(os.environ.get('RETRY_DELAY', '5.0'))
    
    # Server - PORTA FIXA 5000
    PORT = int(os.environ.get('PORT', 5000))
    HOST = '0.0.0.0'
    
    @staticmethod
    def get_port():
        """Retorna sempre porta 5000"""
        return int(os.environ.get('PORT', 5000))
