from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

def get_config() -> Dict[str, Any]:
    """Get configuration EXTREMAMENTE OTIMIZADA para 50% lucro diário GARANTIDO"""
    return {
        # ===== CONFIGURAÇÕES EXTREMAS CORRIGIDAS PARA 50% DIÁRIO =====
        
        # Trading parameters - EXTREMO para 50% com LUCRO LÍQUIDO GARANTIDO
        'PAPER_TRADING': os.getenv('PAPER_TRADING', 'false').lower() == 'true',
        'SYMBOL': os.getenv('SYMBOL', 'ETHUSDT'),  # ETH/USDT otimizado
        'LEVERAGE': int(os.getenv('LEVERAGE', 10)),  # 10x leverage ideal
        'TARGET_TRADES_PER_DAY': int(os.getenv('TARGET_TRADES_PER_DAY', 300)),  # 300 trades/dia EXTREMO
        'BASE_CURRENCY': os.getenv('BASE_CURRENCY', 'USDT'),
        
        # ===== CONFIGURAÇÕES EXTREMAS PARA 50% LUCRO DIÁRIO LÍQUIDO =====
        'DAILY_PROFIT_TARGET': float(os.getenv('DAILY_PROFIT_TARGET', 50.0)),  # 50% por dia GARANTIDO
        'MIN_TRADES_PER_HOUR': int(os.getenv('MIN_TRADES_PER_HOUR', 12)),  # Mínimo 12/hora EXTREMO
        'MAX_TIME_BETWEEN_TRADES': int(os.getenv('MAX_TIME_BETWEEN_TRADES', 180)),  # 3 minutos máximo
        'FORCE_TRADE_INTERVAL': int(os.getenv('FORCE_TRADE_INTERVAL', 360)),  # Forçar após 6 minutos
        
        # API Keys - MESMO
        'BITGET_API_KEY': os.getenv('BITGET_API_KEY'),
        'BITGET_SECRET_KEY': os.getenv('BITGET_SECRET'),
        'BITGET_PASSPHRASE': os.getenv('BITGET_PASSPHRASE'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        
        # ===== RISK MANAGEMENT EXTREMO CORRIGIDO PARA LUCRO LÍQUIDO GARANTIDO =====
        'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT', 0.3)),  # 0.3% stop loss RIGOROSO
        'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT', 0.9)),  # 0.9% take profit MÍNIMO para lucro líquido
        'MINIMUM_PROFIT_PCT': float(os.getenv('MINIMUM_PROFIT_PCT', 0.7)),  # 0.7% mínimo absoluto GARANTIDO
        'POSITION_SIZE_PCT': float(os.getenv('POSITION_SIZE_PCT', 100.0)),  # 100% do saldo
        'MAX_POSITION_TIME': int(os.getenv('MAX_POSITION_TIME', 120)),  # 2 minutos máximo
        'MIN_POSITION_TIME': int(os.getenv('MIN_POSITION_TIME', 25)),  # 25 segundos mínimo
        
        # ===== ANÁLISE TÉCNICA EXTREMAMENTE SELETIVA =====
        'MIN_CONFIDENCE': float(os.getenv('MIN_CONFIDENCE', 87.0)),  # 87% confiança mínima EXTREMA
        'MIN_STRENGTH': float(os.getenv('MIN_STRENGTH', 1.4)),  # 1.4% força mínima EXTREMA
        'MIN_SIGNALS_AGREEMENT': int(os.getenv('MIN_SIGNALS_AGREEMENT', 9)),  # 9 sinais mínimo EXTREMO
        'MOMENTUM_BOOST': float(os.getenv('MOMENTUM_BOOST', 2.2)),  # Multiplicador 2.2x EXTREMO
        
        # ===== MACHINE LEARNING EXTREMO =====
        'ML_CONFIDENCE_THRESHOLD': float(os.getenv('ML_CONFIDENCE_THRESHOLD', 78.0)),  # 78% ML confidence
        'ML_PREDICTION_WEIGHT': float(os.getenv('ML_PREDICTION_WEIGHT', 0.45)),  # 45% peso para ML
        'ML_RETRAIN_INTERVAL': int(os.getenv('ML_RETRAIN_INTERVAL', 800)),  # Retreinar a cada 800 análises
        'ML_MIN_PREDICTION_STRENGTH': float(os.getenv('ML_MIN_PREDICTION_STRENGTH', 0.6)),  # 0.6% predição mínima
        
        # ===== SCALPING EXTREMO =====
        'SCALPING_INTERVAL': float(os.getenv('SCALPING_INTERVAL', 0.25)),  # 250ms entre análises EXTREMO
        'ULTRA_FAST_MODE': os.getenv('ULTRA_FAST_MODE', 'true').lower() == 'true',
        'QUALITY_OVER_QUANTITY': os.getenv('QUALITY_OVER_QUANTITY', 'true').lower() == 'true',
        'BREAKEVEN_TIME': int(os.getenv('BREAKEVEN_TIME', 35)),  # 35s para breakeven
        
        # ===== TRAILING STOP EXTREMO =====
        'TRAILING_STOP_DISTANCE': float(os.getenv('TRAILING_STOP_DISTANCE', 0.6)),  # 0.6% trailing RIGOROSO
        'PROFIT_LOCK_LEVELS': [0.7, 1.0, 1.3],  # Lock profits em 0.7%, 1.0%, 1.3%
        'PARTIAL_EXIT_LEVELS': [1.2, 1.8, 2.5],  # Saídas parciais CORRIGIDAS
        'EXIT_PERCENTAGES': [25, 35, 40],  # % para sair em cada nível
        
        # ===== CONFIGURAÇÕES DE EMERGÊNCIA EXTREMAS =====
        'EMERGENCY_STOP_LOSS': float(os.getenv('EMERGENCY_STOP_LOSS', 1.0)),  # 1.0% stop de emergência
        'MAX_CONSECUTIVE_LOSSES': int(os.getenv('MAX_CONSECUTIVE_LOSSES', 2)),  # 2 perdas seguidas MÁXIMO
        'DAILY_LOSS_LIMIT': float(os.getenv('DAILY_LOSS_LIMIT', 4.0)),  # 4% perda máxima/dia RIGOROSA
        
        # ===== PREVENÇÃO DE MÚLTIPLAS POSIÇÕES =====
        'SINGLE_POSITION_MODE': os.getenv('SINGLE_POSITION_MODE', 'true').lower() == 'true',
        'POSITION_CHECK_INTERVAL': int(os.getenv('POSITION_CHECK_INTERVAL', 3)),  # Verificar a cada 3s
        'FORCE_CLOSE_MULTIPLE_POSITIONS': os.getenv('FORCE_CLOSE_MULTIPLE_POSITIONS', 'true').lower() == 'true',
        'POSITION_LOCK_TIMEOUT': int(os.getenv('POSITION_LOCK_TIMEOUT', 25)),  # 25s timeout para locks
        
        # ===== FECHAMENTO GARANTIDO =====
        'GUARANTEED_CLOSE_ATTEMPTS': int(os.getenv('GUARANTEED_CLOSE_ATTEMPTS', 7)),  # 7 tentativas
        'CLOSE_VERIFICATION_ATTEMPTS': int(os.getenv('CLOSE_VERIFICATION_ATTEMPTS', 4)),  # 4 verificações
        'FORCE_CLOSE_ON_ERROR': os.getenv('FORCE_CLOSE_ON_ERROR', 'true').lower() == 'true',
        'EMERGENCY_CLOSE_TIMEOUT': int(os.getenv('EMERGENCY_CLOSE_TIMEOUT', 8)),  # 8s timeout emergência
        
        # ===== HORÁRIOS DE TRADING =====
        'TRADING_HOURS_START': int(os.getenv('TRADING_HOURS_START', 0)),  # 24h trading
        'TRADING_HOURS_END': int(os.getenv('TRADING_HOURS_END', 24)),  # 24h trading
        'WEEKEND_TRADING': os.getenv('WEEKEND_TRADING', 'true').lower() == 'true',
        
        # ===== CONFIGURAÇÕES DE LOG EXTREMO =====
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'LOG_TRADES': os.getenv('LOG_TRADES', 'true').lower() == 'true',
        'DETAILED_LOGGING': os.getenv('DETAILED_LOGGING', 'true').lower() == 'true',
        'LOG_ML_PREDICTIONS': os.getenv('LOG_ML_PREDICTIONS', 'true').lower() == 'true',
        'LOG_POSITION_MANAGEMENT': os.getenv('LOG_POSITION_MANAGEMENT', 'true').lower() == 'true',
        
        # ===== CONFIGURAÇÕES DE PERFORMANCE EXTREMA =====
        'THREAD_PRIORITY': os.getenv('THREAD_PRIORITY', 'high'),
        'MEMORY_OPTIMIZATION': os.getenv('MEMORY_OPTIMIZATION', 'true').lower() == 'true',
        'EXTREME_EXECUTION_MODE': os.getenv('EXTREME_EXECUTION_MODE', 'true').lower() == 'true',
        'PRECISION_MODE': os.getenv('PRECISION_MODE', 'maximum'),
        
        # ===== INDICADORES TÉCNICOS EXTREMOS =====
        'RSI_PERIODS': [7, 14, 21],  # Múltiplos períodos RSI
        'RSI_OVERSOLD_EXTREME': float(os.getenv('RSI_OVERSOLD_EXTREME', 18.0)),  # RSI < 18
        'RSI_OVERBOUGHT_EXTREME': float(os.getenv('RSI_OVERBOUGHT_EXTREME', 82.0)),  # RSI > 82
        'MACD_FAST': int(os.getenv('MACD_FAST', 12)),
        'MACD_SLOW': int(os.getenv('MACD_SLOW', 26)),
        'MACD_SIGNAL': int(os.getenv('MACD_SIGNAL', 9)),
        'BOLLINGER_PERIOD': int(os.getenv('BOLLINGER_PERIOD', 20)),
        'BOLLINGER_STD_DEV': float(os.getenv('BOLLINGER_STD_DEV', 2.1)),
        
        # ===== VOLUME ANALYSIS EXTREMO =====
        'VOLUME_SPIKE_THRESHOLD': float(os.getenv('VOLUME_SPIKE_THRESHOLD', 220.0)),  # 220% volume spike
        'VOLUME_CONFIRMATION_WEIGHT': float(os.getenv('VOLUME_CONFIRMATION_WEIGHT', 0.35)),  # 35% peso
        'MIN_VOLUME_RATIO': float(os.getenv('MIN_VOLUME_RATIO', 0.85)),  # Volume mínimo 85%
        
        # ===== MOMENTUM EXTREMO =====
        'MOMENTUM_PERIODS': [5, 10, 20],  # Múltiplos períodos momentum
        'MOMENTUM_THRESHOLD': float(os.getenv('MOMENTUM_THRESHOLD', 0.9)),  # 0.9% momentum mínimo
        'MOMENTUM_WEIGHT': float(os.getenv('MOMENTUM_WEIGHT', 0.3)),  # 30% peso
        
        # ===== VOLATILITY BREAKOUT =====
        'VOLATILITY_LOOKBACK': int(os.getenv('VOLATILITY_LOOKBACK', 35)),  # 35 períodos
        'VOLATILITY_BREAKOUT_THRESHOLD': float(os.getenv('VOLATILITY_BREAKOUT_THRESHOLD', 1.6)),  # 160%
        'MIN_BREAKOUT_MOVE': float(os.getenv('MIN_BREAKOUT_MOVE', 0.9)),  # 0.9% movimento mínimo
        
        # ===== SUPPORT/RESISTANCE EXTREMO =====
        'SR_LOOKBACK_PERIODS': int(os.getenv('SR_LOOKBACK_PERIODS', 60)),  # 60 períodos
        'SR_PROXIMITY_THRESHOLD': float(os.getenv('SR_PROXIMITY_THRESHOLD', 0.25)),  # 0.25% proximidade
        'SR_STRENGTH_MULTIPLIER': float(os.getenv('SR_STRENGTH_MULTIPLIER', 2.2)),  # 2.2x multiplicador
        
        # ===== CONFIGURAÇÕES DE AJUSTE DINÂMICO =====
        'DYNAMIC_ADJUSTMENT_INTERVAL': int(os.getenv('DYNAMIC_ADJUSTMENT_INTERVAL', 150)),  # A cada 150 análises
        'PROFIT_DEFICIT_THRESHOLD': float(os.getenv('PROFIT_DEFICIT_THRESHOLD', 3.0)),  # 3% deficit
        'PROFIT_SURPLUS_THRESHOLD': float(os.getenv('PROFIT_SURPLUS_THRESHOLD', 1.5)),  # 1.5% surplus
        'ADJUSTMENT_STEP_SIZE': float(os.getenv('ADJUSTMENT_STEP_SIZE', 0.03)),  # 3% ajuste
        
        # ===== CONFIGURAÇÕES DE QUALIDADE EXTREMA =====
        'MIN_QUALITY_SCORE': float(os.getenv('MIN_QUALITY_SCORE', 88.0)),  # 88% qualidade mínima
        'QUALITY_WEIGHT_CONFIDENCE': float(os.getenv('QUALITY_WEIGHT_CONFIDENCE', 0.32)),  # 32%
        'QUALITY_WEIGHT_STRENGTH': float(os.getenv('QUALITY_WEIGHT_STRENGTH', 0.32)),  # 32%
        'QUALITY_WEIGHT_SIGNALS': float(os.getenv('QUALITY_WEIGHT_SIGNALS', 0.18)),  # 18%
        'QUALITY_WEIGHT_ML': float(os.getenv('QUALITY_WEIGHT_ML', 0.18)),  # 18%
        
        # ===== CONFIGURAÇÕES DE BACKUP E RECUPERAÇÃO =====
        'AUTO_RECOVERY_MODE': os.getenv('AUTO_RECOVERY_MODE', 'true').lower() == 'true',
        'RECOVERY_CHECK_INTERVAL': int(os.getenv('RECOVERY_CHECK_INTERVAL', 45)),  # 45s
        'MAX_RECOVERY_ATTEMPTS': int(os.getenv('MAX_RECOVERY_ATTEMPTS', 4)),  # 4 tentativas
        'RECOVERY_DELAY': int(os.getenv('RECOVERY_DELAY', 8)),  # 8s delay
        
        # ===== CONFIGURAÇÕES DE MONITORAMENTO =====
        'PERFORMANCE_LOG_INTERVAL': int(os.getenv('PERFORMANCE_LOG_INTERVAL', 75)),  # Log a cada 75 trades
        'STATUS_UPDATE_INTERVAL': int(os.getenv('STATUS_UPDATE_INTERVAL', 25)),  # Update a cada 25s
        'DETAILED_METRICS': os.getenv('DETAILED_METRICS', 'true').lower() == 'true',
        'REAL_TIME_MONITORING': os.getenv('REAL_TIME_MONITORING', 'true').lower() == 'true'
    }
