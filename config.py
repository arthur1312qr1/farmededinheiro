from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

def get_config() -> Dict[str, Any]:
    """Get configuration optimized for 50% daily profit target"""
    return {
        # ===== CONFIGURAÇÕES ULTRA AGRESSIVAS PARA 50% DIÁRIO =====
        
        # Trading parameters - ULTRA AGGRESSIVE FOR 50% DAILY
        'PAPER_TRADING': os.getenv('PAPER_TRADING', 'false').lower() == 'true',
        'SYMBOL': os.getenv('SYMBOL', 'ETHUSDT'),  # Manter ETH/USDT
        'LEVERAGE': int(os.getenv('LEVERAGE', 10)),  # Manter 10x (não alterar)
        'TARGET_TRADES_PER_DAY': int(os.getenv('TARGET_TRADES_PER_DAY', 800)),  # 800 trades/dia
        'BASE_CURRENCY': os.getenv('BASE_CURRENCY', 'USDT'),
        
        # ===== CONFIGURAÇÕES PARA 50% LUCRO DIÁRIO =====
        'DAILY_PROFIT_TARGET': float(os.getenv('DAILY_PROFIT_TARGET', 50.0)),  # 50% por dia
        'MIN_TRADES_PER_HOUR': int(os.getenv('MIN_TRADES_PER_HOUR', 35)),  # Mínimo 35/hora
        'MAX_TIME_BETWEEN_TRADES': int(os.getenv('MAX_TIME_BETWEEN_TRADES', 30)),  # 30s máximo
        'FORCE_TRADE_INTERVAL': int(os.getenv('FORCE_TRADE_INTERVAL', 60)),  # Forçar após 60s
        
        # API Keys - MESMO
        'BITGET_API_KEY': os.getenv('BITGET_API_KEY'),
        'BITGET_SECRET_KEY': os.getenv('BITGET_SECRET'),
        'BITGET_PASSPHRASE': os.getenv('BITGET_PASSPHRASE'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        
        # ===== RISK MANAGEMENT ULTRA AGRESSIVO =====
        'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT', 1.2)),  # 1.2% stop loss
        'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT', 0.8)),  # 0.8% take profit
        'MICRO_PROFIT_PCT': float(os.getenv('MICRO_PROFIT_PCT', 0.3)),  # 0.3% micro profit
        'POSITION_SIZE_PCT': float(os.getenv('POSITION_SIZE_PCT', 100.0)),  # 100% do saldo
        'MAX_POSITION_TIME': int(os.getenv('MAX_POSITION_TIME', 45)),  # 45s máximo
        
        # ===== ANÁLISE TÉCNICA ULTRA AGRESSIVA =====
        'MIN_CONFIDENCE': float(os.getenv('MIN_CONFIDENCE', 35.0)),  # 35% confiança mínima
        'MIN_STRENGTH': float(os.getenv('MIN_STRENGTH', 0.1)),  # 0.1% força mínima
        'MIN_SIGNALS_AGREEMENT': int(os.getenv('MIN_SIGNALS_AGREEMENT', 3)),  # 3 sinais mínimo
        'MOMENTUM_BOOST': float(os.getenv('MOMENTUM_BOOST', 2.0)),  # Multiplicador 2x
        
        # ===== SCALPING EXTREMO =====
        'SCALPING_INTERVAL': float(os.getenv('SCALPING_INTERVAL', 0.1)),  # 100ms entre análises
        'ULTRA_FAST_MODE': os.getenv('ULTRA_FAST_MODE', 'true').lower() == 'true',
        'MICRO_MOVEMENTS_TRADING': os.getenv('MICRO_MOVEMENTS_TRADING', 'true').lower() == 'true',
        'BREAKEVEN_TIME': int(os.getenv('BREAKEVEN_TIME', 15)),  # 15s para breakeven
        
        # ===== TRAILING STOP AGRESSIVO =====
        'TRAILING_STOP_DISTANCE': float(os.getenv('TRAILING_STOP_DISTANCE', 0.2)),  # 0.2% trailing
        'PROFIT_LOCK_LEVELS': [0.2, 0.4, 0.6],  # Lock profits em 0.2%, 0.4%, 0.6%
        'PARTIAL_EXIT_LEVELS': [0.4, 0.6, 0.8],  # Saídas parciais
        'EXIT_PERCENTAGES': [30, 40, 30],  # % para sair em cada nível
        
        # ===== CONFIGURAÇÕES DE EMERGÊNCIA =====
        'EMERGENCY_STOP_LOSS': float(os.getenv('EMERGENCY_STOP_LOSS', 2.0)),  # 2% stop de emergência
        'MAX_CONSECUTIVE_LOSSES': int(os.getenv('MAX_CONSECUTIVE_LOSSES', 5)),  # 5 perdas seguidas
        'DAILY_LOSS_LIMIT': float(os.getenv('DAILY_LOSS_LIMIT', 10.0)),  # 10% perda máxima/dia
        
        # ===== HORÁRIOS DE TRADING =====
        'TRADING_HOURS_START': int(os.getenv('TRADING_HOURS_START', 0)),  # 24h trading
        'TRADING_HOURS_END': int(os.getenv('TRADING_HOURS_END', 24)),  # 24h trading
        'WEEKEND_TRADING': os.getenv('WEEKEND_TRADING', 'true').lower() == 'true',
        
        # ===== CONFIGURAÇÕES DE LOG =====
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'LOG_TRADES': os.getenv('LOG_TRADES', 'true').lower() == 'true',
        'DETAILED_LOGGING': os.getenv('DETAILED_LOGGING', 'true').lower() == 'true',
        
        # ===== CONFIGURAÇÕES DE PERFORMANCE =====
        'THREAD_PRIORITY': os.getenv('THREAD_PRIORITY', 'high'),
        'MEMORY_OPTIMIZATION': os.getenv('MEMORY_OPTIMIZATION', 'true').lower() == 'true',
        'FAST_EXECUTION_MODE': os.getenv('FAST_EXECUTION_MODE', 'true').lower() == 'true'
    }
