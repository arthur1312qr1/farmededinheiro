from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

def get_config() -> Dict[str, Any]:
    """Get configuration optimized for 50% daily profit target - CORRIGIDO"""
    return {
        # ===== CONFIGURAÇÕES CORRIGIDAS PARA 50% DIÁRIO REAL =====
        
        # Trading parameters - REALISTA para 50% com LUCRO REAL
        'PAPER_TRADING': os.getenv('PAPER_TRADING', 'false').lower() == 'true',
        'SYMBOL': os.getenv('SYMBOL', 'ETHUSDT'),  # Manter ETH/USDT
        'LEVERAGE': int(os.getenv('LEVERAGE', 10)),  # Manter 10x
        'TARGET_TRADES_PER_DAY': int(os.getenv('TARGET_TRADES_PER_DAY', 200)),  # 200 trades/dia REALISTA
        'BASE_CURRENCY': os.getenv('BASE_CURRENCY', 'USDT'),
        
        # ===== CONFIGURAÇÕES PARA 50% LUCRO DIÁRIO REAL =====
        'DAILY_PROFIT_TARGET': float(os.getenv('DAILY_PROFIT_TARGET', 50.0)),  # 50% por dia
        'MIN_TRADES_PER_HOUR': int(os.getenv('MIN_TRADES_PER_HOUR', 8)),  # Mínimo 8/hora REALISTA
        'MAX_TIME_BETWEEN_TRADES': int(os.getenv('MAX_TIME_BETWEEN_TRADES', 300)),  # 5 minutos máximo
        'FORCE_TRADE_INTERVAL': int(os.getenv('FORCE_TRADE_INTERVAL', 600)),  # Forçar após 10 minutos
        
        # API Keys - MESMO
        'BITGET_API_KEY': os.getenv('BITGET_API_KEY'),
        'BITGET_SECRET_KEY': os.getenv('BITGET_SECRET'),
        'BITGET_PASSPHRASE': os.getenv('BITGET_PASSPHRASE'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        
        # ===== RISK MANAGEMENT CORRIGIDO PARA LUCRO REAL =====
        'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT', 0.8)),  # 0.8% stop loss
        'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT', 1.5)),  # 1.5% take profit MÍNIMO
        'MINIMUM_PROFIT_PCT': float(os.getenv('MINIMUM_PROFIT_PCT', 1.0)),  # 1.0% mínimo absoluto
        'POSITION_SIZE_PCT': float(os.getenv('POSITION_SIZE_PCT', 100.0)),  # 100% do saldo
        'MAX_POSITION_TIME': int(os.getenv('MAX_POSITION_TIME', 180)),  # 3 minutos máximo
        'MIN_POSITION_TIME': int(os.getenv('MIN_POSITION_TIME', 45)),  # 45 segundos mínimo
        
        # ===== ANÁLISE TÉCNICA SELETIVA PARA QUALIDADE =====
        'MIN_CONFIDENCE': float(os.getenv('MIN_CONFIDENCE', 70.0)),  # 70% confiança mínima
        'MIN_STRENGTH': float(os.getenv('MIN_STRENGTH', 1.0)),  # 1.0% força mínima
        'MIN_SIGNALS_AGREEMENT': int(os.getenv('MIN_SIGNALS_AGREEMENT', 6)),  # 6 sinais mínimo
        'MOMENTUM_BOOST': float(os.getenv('MOMENTUM_BOOST', 1.5)),  # Multiplicador 1.5x
        
        # ===== SCALPING PROFISSIONAL =====
        'SCALPING_INTERVAL': float(os.getenv('SCALPING_INTERVAL', 0.5)),  # 500ms entre análises
        'ULTRA_FAST_MODE': os.getenv('ULTRA_FAST_MODE', 'true').lower() == 'true',
        'QUALITY_OVER_QUANTITY': os.getenv('QUALITY_OVER_QUANTITY', 'true').lower() == 'true',
        'BREAKEVEN_TIME': int(os.getenv('BREAKEVEN_TIME', 60)),  # 60s para breakeven
        
        # ===== TRAILING STOP PROFISSIONAL =====
        'TRAILING_STOP_DISTANCE': float(os.getenv('TRAILING_STOP_DISTANCE', 0.5)),  # 0.5% trailing
        'PROFIT_LOCK_LEVELS': [1.0, 1.5, 2.0],  # Lock profits em 1.0%, 1.5%, 2.0%
        'PARTIAL_EXIT_LEVELS': [1.2, 1.8, 2.5],  # Saídas parciais
        'EXIT_PERCENTAGES': [40, 35, 25],  # % para sair em cada nível
        
        # ===== CONFIGURAÇÕES DE EMERGÊNCIA =====
        'EMERGENCY_STOP_LOSS': float(os.getenv('EMERGENCY_STOP_LOSS', 2.5)),  # 2.5% stop de emergência
        'MAX_CONSECUTIVE_LOSSES': int(os.getenv('MAX_CONSECUTIVE_LOSSES', 3)),  # 3 perdas seguidas
        'DAILY_LOSS_LIMIT': float(os.getenv('DAILY_LOSS_LIMIT', 8.0)),  # 8% perda máxima/dia
        
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
        'QUALITY_EXECUTION_MODE': os.getenv('QUALITY_EXECUTION_MODE', 'true').lower() == 'true'
    }
