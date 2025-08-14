import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'trading-bot-scalping-2024'
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://')
    
    SQLALCHEMY_DATABASE_URI = DATABASE_URL or 'sqlite:///scalping_bot.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ===== TRADING REAL - SUAS CONFIGURAÇÕES =====
    PAPER_TRADING = os.environ.get('PAPER_TRADING', 'false').lower() == 'true'
    
    # API Keys
    BITGET_API_KEY = os.environ.get('BITGET_API_KEY', '')
    BITGET_API_SECRET = os.environ.get('BITGET_API_SECRET', '')
    BITGET_PASSPHRASE = os.environ.get('BITGET_PASSPHRASE', '')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
    
    # Suas configurações do Render
    SYMBOL = os.environ.get('SYMBOL', 'ethusdt_UMCBL')
    LEVERAGE = int(os.environ.get('LEVERAGE', 10))
    MIN_LEVERAGE = int(os.environ.get('MIN_LEVERAGE', 10))
    MAX_LEVERAGE = int(os.environ.get('MAX_LEVERAGE', 10))
    MIN_MARGIN_USAGE_PERCENT = float(os.environ.get('MIN_MARGIN_USAGE_PERCENT', '80.0'))
    POLL_INTERVAL = float(os.environ.get('POLL_INTERVAL', '1.0'))
    DRAWDOWN_CLOSE_PCT = float(os.environ.get('DRAWDOWN_CLOSE_PCT', '0.03'))
    LIQ_DIST_THRESHOLD = float(os.environ.get('LIQ_DIST_THRESHOLD', '0.03'))
    MAX_CONSECUTIVE_LOSSES = int(os.environ.get('MAX_CONSECUTIVE_LOSSES', '5'))
    MAX_RETRIES = int(os.environ.get('MAX_RETRIES', '3'))
    RETRY_DELAY = float(os.environ.get('RETRY_DELAY', '5.0'))
    
    # ===== SCALPING AGRESSIVO - CONFIGURAÇÕES OTIMIZADAS =====
    # TRADES CONSTANTES - BAIXOU A CONFIANÇA MÍNIMA
    MIN_CONFIDENCE_SCORE = float(os.environ.get('MIN_CONFIDENCE_SCORE', '60.0'))  # 60% = MAIS TRADES
    REQUIRED_INDICATORS_AGREEMENT = int(os.environ.get('REQUIRED_INDICATORS_AGREEMENT', '4'))  # 4 de 10 = MAIS FLEXÍVEL
    AI_CONFIRMATION_REQUIRED = os.environ.get('AI_CONFIRMATION_REQUIRED', 'false').lower() == 'true'  # NÃO OBRIGATÓRIO
    
    # FREQUÊNCIA ALTA DE TRADES
    MIN_TRADE_INTERVAL = float(os.environ.get('MIN_TRADE_INTERVAL', '30.0'))  # 30 segundos entre trades
    MAX_DAILY_TRADES = int(os.environ.get('MAX_DAILY_TRADES', 200))  # ATÉ 200 TRADES POR DIA
    ENABLE_SCALPING_MODE = True  # SEMPRE ATIVO
    
    # POSIÇÕES MENORES MAS MAIS FREQUENTES
    MIN_BALANCE_USDT = float(os.environ.get('MIN_BALANCE_USDT', '50.0'))
    MAX_POSITION_SIZE_PCT = float(os.environ.get('MAX_POSITION_SIZE_PCT', '15.0'))  # 15% por trade (menor)
    
    # SCALPING - LUCROS E PERDAS MENORES MAS RÁPIDOS
    STOP_LOSS_PCT = float(os.environ.get('STOP_LOSS_PCT', '0.8'))  # 0.8% stop loss
    TAKE_PROFIT_PCT = float(os.environ.get('TAKE_PROFIT_PCT', '1.2'))  # 1.2% take profit
    
    # ANÁLISE TÉCNICA RÁPIDA PARA SCALPING
    TECHNICAL_INDICATORS = [
        'RSI_FAST',      # RSI 7 períodos
        'EMA_FAST',      # EMA 9
        'MACD_SIGNAL',   # Sinal MACD
        'BOLLINGER',     # Bandas de Bollinger
        'VOLUME_SPIKE',  # Picos de volume
        'PRICE_ACTION'   # Ação do preço
    ]
    
    # TIMEFRAMES PARA SCALPING
    PRIMARY_TIMEFRAME = '1m'    # 1 minuto principal
    SECONDARY_TIMEFRAME = '5m'  # 5 minutos confirmação
    
    # Server
    PORT = int(os.environ.get('PORT', 5000))
    HOST = '0.0.0.0'
    DEBUG = False
    
    @classmethod
    def validate_config(cls):
        required_keys = ['BITGET_API_KEY', 'BITGET_API_SECRET', 'BITGET_PASSPHRASE']
        missing = [key for key in required_keys if not getattr(cls, key)]
        
        if missing and not cls.PAPER_TRADING:
            raise ValueError(f"Configurações obrigatórias ausentes: {missing}")
        
        return True
