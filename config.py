from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

def get_config() -> Dict[str, Any]:
    """Get configuration from environment variables"""
    return {
        # Trading parameters
        'PAPER_TRADING': os.getenv('PAPER_TRADING', 'false').lower() == 'true',
        'SYMBOL': os.getenv('SYMBOL', 'ETHUSDT'),
        'LEVERAGE': int(os.getenv('LEVERAGE', 10)),
        'TARGET_TRADES_PER_DAY': int(os.getenv('TARGET_TRADES_PER_DAY', 200)),
        'BASE_CURRENCY': os.getenv('BASE_CURRENCY', 'USDT'),
        
        # API Keys
        'BITGET_API_KEY': os.getenv('BITGET_API_KEY'),
        'BITGET_SECRET_KEY': os.getenv('BITGET_SECRET'),
        'BITGET_PASSPHRASE': os.getenv('BITGET_PASSPHRASE'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        
        # Risk management
        'STOP_LOSS_PCT': float(os.getenv('STOP_LOSS_PCT', 0.02)),
        'TAKE_PROFIT_PCT': float(os.getenv('TAKE_PROFIT_PCT', 0.01)),
        'POSITION_SIZE_PCT': float(os.getenv('POSITION_SIZE_PCT', 0.8))  # 80%
    }
