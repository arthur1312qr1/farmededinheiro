import os
from typing import Optional

class Config:
    """Configuration management for the trading bot"""
    
    def __init__(self):
        # API Keys
        self.BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
        self.BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
        self.BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
        
        self.NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
        self.ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
        self.COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        
        # Trading Configuration
        self.PAPER_TRADING = os.getenv("PAPER_TRADING", "false").lower() == "true"
        self.SYMBOL = os.getenv("SYMBOL", "ETHUSDT_UMCBL")  # Correct format for Bitget USDT-M
        
        # Leverage and Risk Management
        self.MIN_LEVERAGE = int(os.getenv("MIN_LEVERAGE", "9"))
        self.MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "60"))
        self.MIN_MARGIN_USAGE_PERCENT = float(os.getenv("MIN_MARGIN_USAGE_PERCENT", "80.0"))
        
        # Bot Behavior
        self.POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1.0"))
        self.DRAWDOWN_CLOSE_PCT = float(os.getenv("DRAWDOWN_CLOSE_PCT", "0.03"))
        self.LIQ_DIST_THRESHOLD = float(os.getenv("LIQ_DIST_THRESHOLD", "0.03"))
        self.MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))
        
        # Files
        self.STATE_FILE = "bot_state.json"
        self.LOG_FILE = "bot.log"
        
        # Bitget API Configuration
        self.BITGET_BASE_URL = "https://api.bitget.com"
        self.BITGET_FEE_RATE = 0.0006  # Taker fee rate
        
        # Product type for Futures USDT-M
        self.PRODUCT_TYPE = "USDT-FUTURES"  # Bitget product type for USDT-M futures
        
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are present"""
        required_keys = [
            self.BITGET_API_KEY,
            self.BITGET_API_SECRET, 
            self.BITGET_PASSPHRASE
        ]
        
        missing_keys = [key for key in required_keys if not key or key.strip() == ""]
        
        if missing_keys:
            print(f"Missing required Bitget API keys: {len(missing_keys)} keys missing")
            return False
            
        # Optional but recommended keys
        optional_keys = {
            "GEMINI_API_KEY": self.GEMINI_API_KEY,
            "NEWSAPI_KEY": self.NEWSAPI_KEY,
            "COINGECKO_API_KEY": self.COINGECKO_API_KEY,
            "ETHERSCAN_API_KEY": self.ETHERSCAN_API_KEY
        }
        
        for key_name, key_value in optional_keys.items():
            if not key_value:
                print(f"Warning: {key_name} not set - some features may be limited")
        
        return True
    
    def get_trading_config(self) -> dict:
        """Get trading configuration as dictionary"""
        return {
            "symbol": self.SYMBOL,
            "paper_trading": self.PAPER_TRADING,
            "min_leverage": self.MIN_LEVERAGE,
            "max_leverage": self.MAX_LEVERAGE,
            "min_margin_usage": self.MIN_MARGIN_USAGE_PERCENT,
            "poll_interval": self.POLL_INTERVAL,
            "drawdown_close_pct": self.DRAWDOWN_CLOSE_PCT,
            "liq_dist_threshold": self.LIQ_DIST_THRESHOLD,
            "max_consecutive_losses": self.MAX_CONSECUTIVE_LOSSES
        }
