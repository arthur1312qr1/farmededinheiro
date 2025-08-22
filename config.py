# config.py - Configuração otimizada para alta precisão e lucratividade

import os
from dotenv import load_dotenv

load_dotenv()

# Configurações da API
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
BASE_URL = 'https://testnet.binancefuture.com'  # Testnet para testes

# Configurações de Trading - OTIMIZADAS
SYMBOL = 'BTCUSDT'
TIMEFRAME = '5m'  # Timeframe mais responsivo para day trading
QUANTITY = 0.001  # Quantidade por trade

# Configurações de Risk Management - RIGOROSAS
TAKE_PROFIT_PERCENT = 1.2  # TP mínimo de 1.2% (acima do mínimo de 0.7%)
STOP_LOSS_PERCENT = 0.4   # SL agressivo para preservar capital
MAX_DAILY_LOSS = 0.10     # Máximo 10% de perda diária
DAILY_PROFIT_TARGET = 0.50 # Meta de 50% de lucro diário

# Configurações de Controle de Trades
MAX_CONCURRENT_TRADES = 1  # Apenas 1 trade por vez
COOLDOWN_MINUTES = 15      # 15 min entre trades para evitar overtrading

# Configurações do Modelo de ML - ALTA PRECISÃO
LOOKBACK_PERIOD = 200      # Mais dados históricos para melhor precisão
CONFIDENCE_THRESHOLD = 0.85 # Alta confiança necessária (85%)
MIN_PREDICTION_STRENGTH = 0.75 # Força mínima da previsão

# Configurações de Indicadores Técnicos - MÚLTIPLOS SINAIS
RSI_PERIOD = 14
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 25

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

BB_PERIOD = 20
BB_STD = 2

# Configurações de Filtros Avançados
VOLUME_THRESHOLD = 1.5     # Volume mínimo 1.5x da média
VOLATILITY_MIN = 0.02      # Volatilidade mínima para trade
TREND_STRENGTH_MIN = 0.6   # Força mínima da tendência

# Configurações de Backtesting
BACKTEST_DAYS = 30
TRAIN_TEST_SPLIT = 0.8

# Configurações de Logging
LOG_LEVEL = 'INFO'
LOG_FILE = 'trading_bot.log'

# Configurações de Notificação
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Horários de Trading (UTC)
TRADING_START_HOUR = 6   # 6:00 UTC
TRADING_END_HOUR = 22    # 22:00 UTC

# Configurações de Validação de Sinal
MIN_SIGNALS_AGREEMENT = 4  # Mínimo 4 indicadores devem concordar
SIGNAL_WEIGHTS = {
    'ml_prediction': 0.35,
    'rsi': 0.20,
    'macd': 0.20, 
    'bollinger': 0.15,
    'volume': 0.10
}
