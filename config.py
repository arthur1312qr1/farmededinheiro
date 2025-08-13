import os

class Config:
    # Bitget API Configuration
    BITGET_API_KEY = os.getenv('BITGET_API_KEY', '')
    BITGET_API_SECRET = os.getenv('BITGET_API_SECRET', '')
    BITGET_PASSPHRASE = os.getenv('BITGET_PASSPHRASE', '')
    BITGET_BASE_URL = "https://api.bitget.com"
    
    # Trading Configuration
    SYMBOL = "ETHUSDT"
    LEVERAGE = 10.0
    MAX_POSITION_PERCENT = 80.0  # 80% of balance for maximum aggression
    STOP_LOSS_PERCENT = 5.0  # 5% stop loss
    TAKE_PROFIT_PERCENT = 5.0  # 5% take profit as requested
    MIN_BALANCE = 0.0  # No minimum balance required
    
    # Risk Management - Aggressive settings for high frequency
    MAX_CONSECUTIVE_LOSSES = 999999  # No limit on consecutive losses
    MAX_DAILY_TRADES = 999999  # No limit on daily trades
    ANALYSIS_INTERVAL = 15  # Analyze every 15 seconds for higher frequency
    
    # Technical Analysis Settings
    RSI_PERIODS = [14, 21]
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BB_PERIOD = 20
    BB_STD = 2
    EMA_PERIODS = [9, 21, 50]
    
    # Timeframes for multi-timeframe analysis
    TIMEFRAMES = ['1m', '5m', '15m', '1h']
    
    # Signal Confidence Thresholds - More aggressive for frequent trading
    MIN_SIGNAL_CONFIDENCE = 0.55  # Lower threshold for more frequent trades
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    
    # WebSocket Settings
    PRICE_UPDATE_INTERVAL = 1  # Update price every second
    
    # Paper Trading (set to False for real trading)
    PAPER_TRADING = os.getenv('PAPER_TRADING', 'True').lower() == 'true'
    
    @classmethod
    def validate_api_keys(cls):
        """Validate that API keys are present for real trading"""
        if not cls.PAPER_TRADING:
            required_keys = [cls.BITGET_API_KEY, cls.BITGET_API_SECRET, cls.BITGET_PASSPHRASE]
            if not all(required_keys):
                raise ValueError("API keys are required for real trading mode")
        return True
