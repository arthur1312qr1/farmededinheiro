import os
import logging

logger = logging.getLogger(__name__)

class RailwayConfig:
    """Configuration class optimized for Railway deployment"""
    
    def __init__(self):
        # Railway environment settings
        self.PORT = int(os.getenv("PORT", 5000))
        self.RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT", "production")
        
        # API Keys (set these in Railway environment variables)
        self.BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
        self.BITGET_API_SECRET = os.getenv("BITGET_API_SECRET", "")
        self.BITGET_PASSPHRASE = os.getenv("BITGET_PASSPHRASE", "")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        
        # Trading configuration
        self.PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.SYMBOL = os.getenv("SYMBOL", "ETHUSDT")
        self.POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "3.0"))
        self.MIN_BALANCE_USDT = float(os.getenv("MIN_BALANCE_USDT", "20.0"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        
        # API URLs
        self.BITGET_BASE_URL = "https://api.bitget.com"
        
        logger.info(f"Configuration loaded - Paper Trading: {self.PAPER_TRADING}")
        
    def validate_api_keys(self) -> bool:
        """Validate that required API keys are present"""
        if self.PAPER_TRADING:
            logger.info("Paper trading mode - API keys not required")
            return True
            
        required_keys = [
            self.BITGET_API_KEY,
            self.BITGET_API_SECRET,
            self.BITGET_PASSPHRASE
        ]
        
        missing_keys = [key for key in required_keys if not key or key.strip() == ""]
        
        if missing_keys:
            logger.error(f"Missing {len(missing_keys)} required Bitget API keys")
            return False
            
        logger.info("All required API keys are present")
        return True
        
    def get_display_config(self):
        """Get configuration for display purposes (without sensitive data)"""
        return {
            'environment': self.RAILWAY_ENVIRONMENT,
            'paper_trading': self.PAPER_TRADING,
            'symbol': self.SYMBOL,
            'poll_interval': self.POLL_INTERVAL,
            'min_balance': self.MIN_BALANCE_USDT,
            'api_keys_configured': self.validate_api_keys()
        }
