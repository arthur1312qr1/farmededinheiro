from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

def get_config() -> Dict[str, Any]:
    """Configuração CORRIGIDA para trading consistente com 0.9% take profit"""
    return {
        # ===== CONFIGURAÇÕES BÁSICAS =====
        'PAPER_TRADING': os.getenv('PAPER_TRADING', 'false').lower() == 'true',
        'SYMBOL': os.getenv('SYMBOL', 'ETHUSDT'),
        'LEVERAGE': int(os.getenv('LEVERAGE', 10)),
        'BASE_CURRENCY': os.getenv('BASE_CURRENCY', 'USDT'),
        
        # ===== CONFIGURAÇÕES REALISTAS CORRIGIDAS =====
        'DAILY_PROFIT_TARGET': float(os.getenv('DAILY_PROFIT_TARGET', 5.0)),  # 5% por dia REALISTA
        'TARGET_TRADES_PER_DAY': int(os.getenv('TARGET_TRADES_PER_DAY', 50)),  # 50 trades/dia REALISTA
        'MIN_TRADES_PER_HOUR': int(os.getenv('MIN_TRADES_PER_HOUR', 2)),  # 2/hora REALISTA
        'MAX_TIME_BETWEEN_TRADES': int(os.getenv('MAX_TIME_BETWEEN_TRADES', 600)),  # 10 minutos
        
        # API Keys
        'BITGET_API_KEY': os.getenv('BITGET_API_KEY'),
        'BITGET_SECRET_KEY': os.getenv('BITGET_SECRET'),
        'BITGET_PASSPHRASE': os.getenv('BITGET_PASSPHRASE'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        
        # ===== RISK MANAGEMENT CORRIGIDO - TAKE PROFIT 0.9% =====
        'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT', 0.4)),  # 0.4% stop loss
        'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT', 0.9)),  # 0.9% take profit CORRIGIDO
        'MINIMUM_PROFIT_PCT': float(os.getenv('MINIMUM_PROFIT_PCT', 0.5)),  # 0.5% mínimo
        'POSITION_SIZE_PCT': float(os.getenv('POSITION_SIZE_PCT', 95.0)),  # 95% do saldo
        'MAX_POSITION_TIME': int(os.getenv('MAX_POSITION_TIME', 300)),  # 5 minutos máximo
        'MIN_POSITION_TIME': int(os.getenv('MIN_POSITION_TIME', 15)),  # 15 segundos mínimo
        'BREAKEVEN_TIME': int(os.getenv('BREAKEVEN_TIME', 60)),  # 60s para breakeven
        
        # ===== ANÁLISE TÉCNICA REALISTA =====
        'MIN_CONFIDENCE': float(os.getenv('MIN_CONFIDENCE', 70.0)),  # 70% confiança mínima
        'MIN_STRENGTH': float(os.getenv('MIN_STRENGTH', 0.8)),  # 0.8% força mínima
        'MIN_SIGNALS_AGREEMENT': int(os.getenv('MIN_SIGNALS_AGREEMENT', 3)),  # 3 sinais mínimo
        'MOMENTUM_BOOST': float(os.getenv('MOMENTUM_BOOST', 1.2)),  # Multiplicador 1.2x
        
        # ===== MACHINE LEARNING MODERADO =====
        'ML_CONFIDENCE_THRESHOLD': float(os.getenv('ML_CONFIDENCE_THRESHOLD', 60.0)),  # 60% ML
        'ML_PREDICTION_WEIGHT': float(os.getenv('ML_PREDICTION_WEIGHT', 0.3)),  # 30% peso
        'ML_RETRAIN_INTERVAL': int(os.getenv('ML_RETRAIN_INTERVAL', 500)),  # Retreinar a cada 500
        'ML_MIN_PREDICTION_STRENGTH': float(os.getenv('ML_MIN_PREDICTION_STRENGTH', 0.3)),
        
        # ===== SCALPING REALISTA =====
        'SCALPING_INTERVAL': float(os.getenv('SCALPING_INTERVAL', 2.0)),  # 2s entre análises
        'ULTRA_FAST_MODE': os.getenv('ULTRA_FAST_MODE', 'false').lower() == 'true',
        'QUALITY_OVER_QUANTITY': os.getenv('QUALITY_OVER_QUANTITY', 'true').lower() == 'true',
        
        # ===== TRAILING STOP MODERADO =====
        'TRAILING_STOP_DISTANCE': float(os.getenv('TRAILING_STOP_DISTANCE', 0.3)),  # 0.3% trailing
        'PROFIT_LOCK_LEVELS': [0.5, 0.7, 0.9],  # Lock em 0.5%, 0.7%, 0.9%
        'PARTIAL_EXIT_LEVELS': [0.7, 0.9, 1.2],  # Saídas parciais
        'EXIT_PERCENTAGES': [30, 40, 30],  # % para sair em cada nível
        
        # ===== CONFIGURAÇÕES DE EMERGÊNCIA =====
        'EMERGENCY_STOP_LOSS': float(os.getenv('EMERGENCY_STOP_LOSS', 1.0)),  # 1% stop emergência
        'MAX_CONSECUTIVE_LOSSES': int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3)),  # 3 perdas seguidas
        'DAILY_LOSS_LIMIT': float(os.getenv('DAILY_LOSS_LIMIT', 2.0)),  # 2% perda máxima/dia
        
        # ===== PREVENÇÃO DE MÚLTIPLAS POSIÇÕES =====
        'SINGLE_POSITION_MODE': os.getenv('SINGLE_POSITION_MODE', 'true').lower() == 'true',
        'POSITION_CHECK_INTERVAL': int(os.getenv('POSITION_CHECK_INTERVAL', 5)),
        'FORCE_CLOSE_MULTIPLE_POSITIONS': os.getenv('FORCE_CLOSE_MULTIPLE_POSITIONS', 'true').lower() == 'true',
        'POSITION_LOCK_TIMEOUT': int(os.getenv('POSITION_LOCK_TIMEOUT', 30)),
        
        # ===== FECHAMENTO GARANTIDO =====
        'GUARANTEED_CLOSE_ATTEMPTS': int(os.getenv('GUARANTEED_CLOSE_ATTEMPTS', 5)),
        'CLOSE_VERIFICATION_ATTEMPTS': int(os.getenv('CLOSE_VERIFICATION_ATTEMPTS', 3)),
        'FORCE_CLOSE_ON_ERROR': os.getenv('FORCE_CLOSE_ON_ERROR', 'true').lower() == 'true',
        'EMERGENCY_CLOSE_TIMEOUT': int(os.getenv('EMERGENCY_CLOSE_TIMEOUT', 10)),
        
        # ===== HORÁRIOS DE TRADING =====
        'TRADING_HOURS_START': int(os.getenv('TRADING_HOURS_START', 0)),  # 24h
        'TRADING_HOURS_END': int(os.getenv('TRADING_HOURS_END', 24)),  # 24h
        'WEEKEND_TRADING': os.getenv('WEEKEND_TRADING', 'true').lower() == 'true',
        
        # ===== CONFIGURAÇÕES DE LOG =====
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'LOG_TRADES': os.getenv('LOG_TRADES', 'true').lower() == 'true',
        'DETAILED_LOGGING': os.getenv('DETAILED_LOGGING', 'true').lower() == 'true',
        'LOG_ML_PREDICTIONS': os.getenv('LOG_ML_PREDICTIONS', 'false').lower() == 'true',
        'LOG_POSITION_MANAGEMENT': os.getenv('LOG_POSITION_MANAGEMENT', 'true').lower() == 'true',
        
        # ===== CONFIGURAÇÕES DE PERFORMANCE =====
        'THREAD_PRIORITY': os.getenv('THREAD_PRIORITY', 'normal'),
        'MEMORY_OPTIMIZATION': os.getenv('MEMORY_OPTIMIZATION', 'true').lower() == 'true',
        'EXTREME_EXECUTION_MODE': os.getenv('EXTREME_EXECUTION_MODE', 'false').lower() == 'true',
        'PRECISION_MODE': os.getenv('PRECISION_MODE', 'high'),
        
        # ===== INDICADORES TÉCNICOS =====
        'RSI_PERIODS': [14, 21],  # RSI padrão
        'RSI_OVERSOLD_EXTREME': float(os.getenv('RSI_OVERSOLD_EXTREME', 30.0)),  # RSI < 30
        'RSI_OVERBOUGHT_EXTREME': float(os.getenv('RSI_OVERBOUGHT_EXTREME', 70.0)),  # RSI > 70
        'MACD_FAST': int(os.getenv('MACD_FAST', 12)),
        'MACD_SLOW': int(os.getenv('MACD_SLOW', 26)),
        'MACD_SIGNAL': int(os.getenv('MACD_SIGNAL', 9)),
        'BOLLINGER_PERIOD': int(os.getenv('BOLLINGER_PERIOD', 20)),
        'BOLLINGER_STD_DEV': float(os.getenv('BOLLINGER_STD_DEV', 2.0)),
        
        # ===== VOLUME ANALYSIS =====
        'VOLUME_SPIKE_THRESHOLD': float(os.getenv('VOLUME_SPIKE_THRESHOLD', 150.0)),  # 150% volume
        'VOLUME_CONFIRMATION_WEIGHT': float(os.getenv('VOLUME_CONFIRMATION_WEIGHT', 0.2)),  # 20%
        'MIN_VOLUME_RATIO': float(os.getenv('MIN_VOLUME_RATIO', 0.5)),  # Volume mínimo 50%
        
        # ===== MOMENTUM =====
        'MOMENTUM_PERIODS': [10, 20],  # Períodos momentum
        'MOMENTUM_THRESHOLD': float(os.getenv('MOMENTUM_THRESHOLD', 0.5)),  # 0.5% momentum
        'MOMENTUM_WEIGHT': float(os.getenv('MOMENTUM_WEIGHT', 0.2)),  # 20% peso
        
        # ===== VOLATILITY BREAKOUT =====
        'VOLATILITY_LOOKBACK': int(os.getenv('VOLATILITY_LOOKBACK', 30)),
        'VOLATILITY_BREAKOUT_THRESHOLD': float(os.getenv('VOLATILITY_BREAKOUT_THRESHOLD', 1.2)),
        'MIN_BREAKOUT_MOVE': float(os.getenv('MIN_BREAKOUT_MOVE', 0.5)),  # 0.5% movimento
        
        # ===== SUPPORT/RESISTANCE =====
        'SR_LOOKBACK_PERIODS': int(os.getenv('SR_LOOKBACK_PERIODS', 50)),
        'SR_PROXIMITY_THRESHOLD': float(os.getenv('SR_PROXIMITY_THRESHOLD', 0.2)),  # 0.2%
        'SR_STRENGTH_MULTIPLIER': float(os.getenv('SR_STRENGTH_MULTIPLIER', 1.5)),
        
        # ===== AJUSTE DINÂMICO =====
        'DYNAMIC_ADJUSTMENT_INTERVAL': int(os.getenv('DYNAMIC_ADJUSTMENT_INTERVAL', 100)),
        'PROFIT_DEFICIT_THRESHOLD': float(os.getenv('PROFIT_DEFICIT_THRESHOLD', 2.0)),  # 2%
        'PROFIT_SURPLUS_THRESHOLD': float(os.getenv('PROFIT_SURPLUS_THRESHOLD', 1.0)),  # 1%
        'ADJUSTMENT_STEP_SIZE': float(os.getenv('ADJUSTMENT_STEP_SIZE', 0.02)),  # 2%
        
        # ===== QUALIDADE =====
        'MIN_QUALITY_SCORE': float(os.getenv('MIN_QUALITY_SCORE', 70.0)),  # 70% qualidade
        'QUALITY_WEIGHT_CONFIDENCE': float(os.getenv('QUALITY_WEIGHT_CONFIDENCE', 0.3)),
        'QUALITY_WEIGHT_STRENGTH': float(os.getenv('QUALITY_WEIGHT_STRENGTH', 0.3)),
        'QUALITY_WEIGHT_SIGNALS': float(os.getenv('QUALITY_WEIGHT_SIGNALS', 0.2)),
        'QUALITY_WEIGHT_ML': float(os.getenv('QUALITY_WEIGHT_ML', 0.2)),
        
        # ===== BACKUP E RECUPERAÇÃO =====
        'AUTO_RECOVERY_MODE': os.getenv('AUTO_RECOVERY_MODE', 'true').lower() == 'true',
        'RECOVERY_CHECK_INTERVAL': int(os.getenv('RECOVERY_CHECK_INTERVAL', 60)),
        'MAX_RECOVERY_ATTEMPTS': int(os.getenv('MAX_RECOVERY_ATTEMPTS', 3)),
        'RECOVERY_DELAY': int(os.getenv('RECOVERY_DELAY', 10)),
        
        # ===== MONITORAMENTO =====
        'PERFORMANCE_LOG_INTERVAL': int(os.getenv('PERFORMANCE_LOG_INTERVAL', 50)),
        'STATUS_UPDATE_INTERVAL': int(os.getenv('STATUS_UPDATE_INTERVAL', 30)),
        'DETAILED_METRICS': os.getenv('DETAILED_METRICS', 'true').lower() == 'true',
        'REAL_TIME_MONITORING': os.getenv('REAL_TIME_MONITORING', 'true').lower() == 'true',
        
        # ===== TAXAS E LUCRO LÍQUIDO =====
        'EXCHANGE_FEE_PCT': float(os.getenv('EXCHANGE_FEE_PCT', 0.1)),  # 0.1% taxa Bitget
        'SLIPPAGE_BUFFER_PCT': float(os.getenv('SLIPPAGE_BUFFER_PCT', 0.05)),  # 0.05% slippage
        'MIN_LIQUID_PROFIT_MARGIN': float(os.getenv('MIN_LIQUID_PROFIT_MARGIN', 0.2)),  # 0.2% margem
        'FEE_ADJUSTED_TARGETS': os.getenv('FEE_ADJUSTED_TARGETS', 'true').lower() == 'true',
        
        # ===== EXECUÇÃO AVANÇADA =====
        'ORDER_EXECUTION_TIMEOUT': int(os.getenv('ORDER_EXECUTION_TIMEOUT', 10)),
        'MAX_ORDER_ATTEMPTS': int(os.getenv('MAX_ORDER_ATTEMPTS', 3)),
        'ORDER_RETRY_DELAY': float(os.getenv('ORDER_RETRY_DELAY', 1.0)),
        'GUARANTEED_EXECUTION_MODE': os.getenv('GUARANTEED_EXECUTION_MODE', 'true').lower() == 'true',
        
        # ===== PROTEÇÃO DE CAPITAL =====
        'MAX_DAILY_TRADES': int(os.getenv('MAX_DAILY_TRADES', 100)),  # Máximo 100 trades/dia
        'CAPITAL_PROTECTION_MODE': os.getenv('CAPITAL_PROTECTION_MODE', 'true').lower() == 'true',
        'PROGRESSIVE_RISK_REDUCTION': os.getenv('PROGRESSIVE_RISK_REDUCTION', 'true').lower() == 'true',
        'LOSS_STREAK_PROTECTION': os.getenv('LOSS_STREAK_PROTECTION', 'true').lower() == 'true',
        
        # ===== OTIMIZAÇÃO REALISTA =====
        'TARGET_ACHIEVEMENT_TRACKING': os.getenv('TARGET_ACHIEVEMENT_TRACKING', 'true').lower() == 'true',
        'HOURLY_PROFIT_TARGETS': os.getenv('HOURLY_PROFIT_TARGETS', 'false').lower() == 'true',
        'ADAPTIVE_STRATEGY_MODE': os.getenv('ADAPTIVE_STRATEGY_MODE', 'true').lower() == 'true',
        'PROFIT_ACCELERATION_MODE': os.getenv('PROFIT_ACCELERATION_MODE', 'false').lower() == 'true',
        
        # ===== VALIDAÇÃO E SEGURANÇA =====
        'DOUBLE_CHECK_POSITIONS': os.getenv('DOUBLE_CHECK_POSITIONS', 'true').lower() == 'true',
        'VALIDATE_ORDERS_BEFORE_EXECUTION': os.getenv('VALIDATE_ORDERS_BEFORE_EXECUTION', 'true').lower() == 'true',
        'SAFETY_CHECKS_ENABLED': os.getenv('SAFETY_CHECKS_ENABLED', 'true').lower() == 'true',
        'PARANOID_MODE': os.getenv('PARANOID_MODE', 'false').lower() == 'true',
        
        # ===== PERFORMANCE FINAL - REALISTA =====
        'EXTREME_PRECISION_MODE': os.getenv('EXTREME_PRECISION_MODE', 'false').lower() == 'true',
        'MAXIMUM_EFFICIENCY_MODE': os.getenv('MAXIMUM_EFFICIENCY_MODE', 'true').lower() == 'true',
        'REALISTIC_PROFIT_MODE': os.getenv('REALISTIC_PROFIT_MODE', 'true').lower() == 'true',
        'CONSISTENT_TRADING_MODE': os.getenv('CONSISTENT_TRADING_MODE', 'true').lower() == 'true'
    }
