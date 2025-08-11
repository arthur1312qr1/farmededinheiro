import logging
import time
from datetime import datetime
from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingBot:
    """Simplified trading bot for Railway deployment"""
    
    def __init__(self, config, gemini_handler):
        self.config = config
        self.gemini_handler = gemini_handler
        self.bitget_api = None
        self.running = False
        self.emergency_stop = False
        
        # Cache for balance info
        self.last_balance_check = None
        self.cached_balance = None
        
        # Initialize Bitget API if not in paper trading mode
        if not config.PAPER_TRADING and config.validate_api_keys():
            try:
                self.bitget_api = BitgetAPI(
                    config.BITGET_API_KEY,
                    config.BITGET_API_SECRET,
                    config.BITGET_PASSPHRASE
                )
                logger.info("Bitget API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Bitget API: {e}")
        
        # Bot state tracking
        self.bot_state = {
            'balance': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'consecutive_losses': 0,
            'emergency_stop': False,
            'last_updated': datetime.now().isoformat(),
            'current_positions': {},
            'trading_history': []
        }
        
        logger.info(f"Trading bot initialized for {config.SYMBOL}")
    
    def get_balance_info(self):
        """Get current balance information"""
        try:
            # Paper trading mode
            if self.config.PAPER_TRADING or not self.bitget_api:
                return {
                    'available_balance': 1000.0,  # Simulated balance
                    'total_equity': 1000.0,
                    'unrealized_pnl': 0,
                    'currency': 'USDT',
                    'source': 'paper_trading',
                    'last_updated': datetime.now().isoformat(),
                    'paper_trading': True,
                    'sufficient_balance': True,
                    'success': True
                }
            
            # Check cache (30 second cache)
            now = time.time()
            if (self.cached_balance and self.last_balance_check and 
                now - self.last_balance_check < 30):
                return self.cached_balance
            
            # Get real balance from API
            balance_info = self.bitget_api.get_balance()
            
            if balance_info.get('success'):
                balance_info['paper_trading'] = False
                balance_info['sufficient_balance'] = (
                    balance_info.get('available_balance', 0) >= self.config.MIN_BALANCE_USDT
                )
                
                # Cache the result
                self.cached_balance = balance_info
                self.last_balance_check = now
                
                return balance_info
            else:
                return {
                    'error': balance_info.get('error', 'Unknown error'),
                    'available_balance': 0,
                    'currency': 'USDT',
                    'last_updated': datetime.now().isoformat(),
                    'paper_trading': False,
                    'sufficient_balance': False,
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Error getting balance info: {e}")
            return {
                'error': str(e),
                'available_balance': 0,
                'currency': 'USDT',
                'last_updated': datetime.now().isoformat(),
                'success': False
            }
    
    def get_state(self):
        """Get current bot state"""
        # Update balance in state
        balance_info = self.get_balance_info()
        if balance_info.get('success'):
            self.bot_state['balance'] = balance_info.get('available_balance', 0)
        
        self.bot_state['last_updated'] = datetime.now().isoformat()
        return self.bot_state.copy()
    
    def start(self):
        """Start the trading bot"""
        self.running = True
        self.emergency_stop = False
        logger.info("Trading bot started")
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        logger.info("Trading bot stopped")
    
    def execute_trading_cycle(self):
        """Execute one trading cycle"""
        if not self.running or self.emergency_stop:
            return
        
        try:
            # Get current balance
            balance_info = self.get_balance_info()
            
            if balance_info.get('error'):
                logger.warning(f"Balance check failed: {balance_info['error']}")
                return
            
            # Update bot state
            self.bot_state['balance'] = balance_info.get('available_balance', 0)
            self.bot_state['last_updated'] = datetime.now().isoformat()
            
            # Check if we have sufficient balance
            if not balance_info.get('sufficient_balance', False):
                logger.warning(f"Insufficient balance: ${balance_info.get('available_balance', 0):.2f}")
                return
            
            # Get market analysis from Gemini
            try:
                market_analysis = self.gemini_handler.analyze_market_conditions()
                
                # Log analysis results
                if market_analysis:
                    signal = market_analysis.get('signal', 'hold')
                    confidence = market_analysis.get('confidence', 'low')
                    logger.info(f"Market analysis - Signal: {signal}, Confidence: {confidence}")
                
            except Exception as e:
                logger.error(f"Market analysis failed: {e}")
                market_analysis = {'signal': 'hold', 'confidence': 'low'}
            
            # In paper trading mode, just log the cycle
            if balance_info.get('paper_trading'):
                logger.debug(f"Paper trading cycle completed - Balance: ${balance_info.get('available_balance'):.2f}")
            else:
                logger.info(f"Trading cycle completed - Balance: ${balance_info.get('available_balance'):.2f} USDT")
            
            # Here you would implement actual trading logic based on market analysis
            # For now, we just track the state
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            
            # Use Gemini to analyze the error
            try:
                error_analysis = self.gemini_handler.analyze_and_fix_error(str(e))
                if "STOP_TRADING" in error_analysis:
                    logger.critical(f"Emergency stop triggered: {error_analysis}")
                    self.emergency_stop = True
                    self.running = False
                else:
                    logger.info(f"Error analysis: {error_analysis}")
            except Exception as analysis_error:
                logger.error(f"Error analysis failed: {analysis_error}")
