import threading
import time
import logging
from datetime import datetime
from bitget_api import BitgetAPI
from risk_manager import RiskManager
from technical_analysis import TechnicalAnalysis
from gemini_handler import GeminiHandler
from config import Config
import eventlet # Importado para uso assíncrono

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, socketio=None):
        self.api = BitgetAPI()
        self.risk_manager = RiskManager()
        self.tech_analysis = TechnicalAnalysis()
        self.gemini_handler = GeminiHandler()
        self.socketio = socketio
        
        # Bot state
        self.is_running = False
        self.current_position = None
        self.consecutive_losses = 0
        self.last_balance = 0
        self.bot_thread = None
        
        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        
        logger.info("🤖 Trading Bot initialized for REAL TRADING")

    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Bot is already running")
            return False
            
        self.is_running = True
        self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self.bot_thread.start()
        
        logger.info("🚀 Trading Bot started - REAL MONEY MODE")
        self._emit_status("started")
        return True

    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        if self.bot_thread:
            self.bot_thread.join(timeout=5.0)
        
        logger.info("⏹️ Trading Bot stopped")
        self._emit_status("stopped")

    def _run_bot(self):
        """Main bot trading loop - EXECUTA TRADES REAIS"""
        self._initialize_bot()
        
        last_analysis_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                self._update_data()
                
                if current_time - last_analysis_time >= Config.ANALYSIS_INTERVAL:
                    self._perform_analysis()
                    last_analysis_time = current_time
                
                self._check_trading_signals()
                self._check_risk_conditions()
                
                time.sleep(Config.POLL_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in bot loop: {e}")
                eventlet.sleep(5)

    def _initialize_bot(self):
        # ... (código existente)
        pass

    def _update_data(self):
        """Update balance and price data"""
        try:
            balance_result = self.api.get_account_balance()
            if balance_result['success']:
                current_balance = balance_result['data']['available']
                mode = balance_result['data'].get('mode', 'unknown')
                
                # Emitir de forma assíncrona
                eventlet.spawn(self._emit_data, 'balance_update', {
                    'balance': current_balance,
                    'total': balance_result['data']['total'],
                    'mode': mode,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.last_balance = current_balance
            
            price_result = self.api.get_current_price(Config.SYMBOL)
            if price_result['success']:
                current_price = price_result['price']
                
                # Emitir de forma assíncrona
                eventlet.spawn(self._emit_data, 'price_update', {
                    'symbol': Config.SYMBOL,
                    'price': current_price,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")

    def _check_trading_signals(self):
        # ... (código existente)
        pass

    def _execute_real_trade(self, signals):
        # ... (código existente)
        trade_data = {
            'id': order_result['data']['orderId'],
            'side': side,
            'size': position_size,
            'price': order_result['data'].get('price', 0),
            'timestamp': datetime.now().isoformat(),
            'signal_strength': signals.get('confidence', 0),
            'mode': 'REAL'
        }
        eventlet.spawn(self._emit_data, 'trade_executed', trade_data)
        # ... (código existente)
        
    def _set_stop_loss_take_profit(self, trade_data):
        # ... (código existente)
        pass

    def _calculate_position_size(self, available_balance):
        # ... (código existente)
        pass

    def _can_trade(self):
        # ... (código existente)
        pass

    def _check_risk_conditions(self):
        # ... (código existente)
        pass

    def _emergency_stop(self):
        logger.critical("🛑 EMERGENCY STOP TRIGGERED")
        self.stop()
        
        eventlet.spawn(self._emit_data, 'emergency_stop', {
            'reason': 'High drawdown detected',
            'timestamp': datetime.now().isoformat()
        })

    def get_statistics(self):
        # ... (código existente)
        pass

    def _emit_status(self, status):
        """Emit bot status update"""
        if self.socketio:
            eventlet.spawn(self.socketio.emit, 'bot_status', {'status': status})

    def _emit_data(self, event, data):
        """Emit data update"""
        if self.socketio:
            eventlet.spawn(self.socketio.emit, event, data)

    def _perform_analysis(self):
        # ... (código existente)
        pass
