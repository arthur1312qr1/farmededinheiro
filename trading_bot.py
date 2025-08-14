import threading
import time
import logging
from datetime import datetime
from bitget_api import BitgetAPI
from risk_manager import RiskManager
from technical_analysis import TechnicalAnalysis
from gemini_handler import GeminiHandler
from config import Config

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
        # Initialize bot
        self._initialize_bot()
        
        last_analysis_time = 0
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Update balance and price
                self._update_data()
                
                # Perform analysis every 30 seconds
                if current_time - last_analysis_time >= Config.ANALYSIS_INTERVAL:
                    self._perform_analysis()
                    last_analysis_time = current_time
                
                # Check for trading signals - EXECUTA TRADES REAIS
                self._check_trading_signals()
                
                # Risk management checks
                self._check_risk_conditions()
                
                time.sleep(Config.POLL_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in bot loop: {e}")
                time.sleep(5)

    def _initialize_bot(self):
        """Initialize bot settings"""
        try:
            logger.info(f"🔥 INITIALIZING BOT FOR REAL TRADING - PAPER_TRADING={Config.PAPER_TRADING}")
            
            # Set leverage
            leverage_result = self.api.set_leverage(Config.LEVERAGE)
            if leverage_result['success']:
                logger.info(f"✅ Leverage set to {Config.LEVERAGE}x")
            else:
                logger.error(f"❌ Failed to set leverage: {leverage_result.get('error')}")
            
            # Get initial balance
            balance_result = self.api.get_account_balance()
            if balance_result['success']:
                self.last_balance = balance_result['data']['available']
                mode = balance_result['data'].get('mode', 'unknown')
                logger.info(f"✅ Initial balance: ${self.last_balance:.2f} ({mode} mode)")
            
        except Exception as e:
            logger.error(f"Error initializing bot: {e}")

    def _update_data(self):
        """Update balance and price data"""
        try:
            # Get current balance
            balance_result = self.api.get_account_balance()
            if balance_result['success']:
                current_balance = balance_result['data']['available']
                mode = balance_result['data'].get('mode', 'unknown')
                
                # Emit balance update
                self._emit_data('balance_update', {
                    'balance': current_balance,
                    'total': balance_result['data']['total'],
                    'mode': mode,
                    'timestamp': datetime.now().isoformat()
                })
                
                self.last_balance = current_balance
            
            # Get current price
            price_result = self.api.get_current_price(Config.SYMBOL)
            if price_result['success']:
                current_price = price_result['price']
                
                # Emit price update
                self._emit_data('price_update', {
                    'symbol': Config.SYMBOL,
                    'price': current_price,
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            logger.error(f"Error updating data: {e}")

    def _check_trading_signals(self):
        """Check for trading signals and execute REAL trades"""
        try:
            # Get latest klines
            klines_result = self.api.get_klines(Config.SYMBOL, '1m', 50)
            if not klines_result['success']:
                return
            
            klines = klines_result['data']
            
            # Get technical signals for 99% accuracy
            signals = self.tech_analysis.get_high_accuracy_signals(klines)
            
            if signals['action'] != 'hold' and signals['confidence'] >= 0.99:
                # Check risk conditions
                if self._can_trade():
                    logger.info(f"🎯 HIGH ACCURACY SIGNAL: {signals['action']} - Confidence: {signals['confidence']*100:.1f}%")
                    self._execute_real_trade(signals)
                
        except Exception as e:
            logger.error(f"Error checking trading signals: {e}")

    def _execute_real_trade(self, signals):
        """Execute a REAL trading order - NOT PAPER TRADING"""
        try:
            side = signals['action']  # 'buy' or 'sell'
            
            logger.info(f"🔥 EXECUTING REAL TRADE: {side}")
            
            # Calculate position size
            balance_result = self.api.get_account_balance()
            if not balance_result['success']:
                logger.error("Failed to get balance for trade")
                return
            
            available_balance = balance_result['data']['available']
            position_size = self._calculate_position_size(available_balance)
            
            if position_size <= 0:
                logger.warning("Position size too small, skipping trade")
                return
            
            # Place REAL order
            order_result = self.api.place_real_order(side, position_size)
            
            if order_result['success']:
                self.total_trades += 1
                
                trade_data = {
                    'id': order_result['data']['orderId'],
                    'side': side,
                    'size': position_size,
                    'price': order_result['data'].get('price', 0),
                    'timestamp': datetime.now().isoformat(),
                    'signal_strength': signals.get('confidence', 0),
                    'mode': 'REAL'
                }
                
                # Emit trade update
                self._emit_data('trade_executed', trade_data)
                
                logger.info(f"✅ REAL TRADE EXECUTED: {side} {position_size} @ ${trade_data['price']}")
            
                # Set stop loss and take profit
                self._set_stop_loss_take_profit(trade_data)
                
            else:
                logger.error(f"❌ Trade failed: {order_result.get('error')}")
        
        except Exception as e:
            logger.error(f"Error executing trade: {e}")

    def _set_stop_loss_take_profit(self, trade_data):
        """Set stop loss (2%) and take profit (5%) for the trade"""
        try:
            entry_price = trade_data['price']
            side = trade_data['side']
            
            if side == 'buy':
                stop_loss_price = entry_price * (1 - Config.STOP_LOSS_PCT)
                take_profit_price = entry_price * (1 + Config.TAKE_PROFIT_PCT)
            else:
                stop_loss_price = entry_price * (1 + Config.STOP_LOSS_PCT)
                take_profit_price = entry_price * (1 - Config.TAKE_PROFIT_PCT)
            
            # Place stop loss order
            self.api.place_stop_loss_order(trade_data['id'], stop_loss_price)
            
            # Place take profit order
            self.api.place_take_profit_order(trade_data['id'], take_profit_price)
            
            logger.info(f"📊 SL/TP Set - SL: ${stop_loss_price:.4f}, TP: ${take_profit_price:.4f}")
            
        except Exception as e:
            logger.error(f"Error setting SL/TP: {e}")

    def _calculate_position_size(self, available_balance):
        """Calculate position size based on 80% margin usage"""
        # Use 80% of available balance as specified
        margin_usage = available_balance * (Config.MIN_MARGIN_USAGE_PERCENT / 100)
        
        # Get current price
        price_result = self.api.get_current_price(Config.SYMBOL)
        if not price_result['success']:
            return 0
        
        current_price = price_result['price']
        position_size = margin_usage / current_price
        
        # Apply 10x leverage
        position_size *= Config.LEVERAGE
        
        return round(position_size, 6)

    def _can_trade(self):
        """Check if bot can trade based on risk conditions"""
        # Check consecutive losses
        if self.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            logger.warning(f"Max consecutive losses reached: {self.consecutive_losses}")
            return False
        
        # Check minimum balance
        if self.last_balance < Config.MIN_BALANCE_USDT:
            logger.warning(f"Balance too low: ${self.last_balance}")
            return False
        
        return True

    def _check_risk_conditions(self):
        """Check and handle risk conditions"""
        try:
            # Check drawdown
            balance_result = self.api.get_account_balance()
            if balance_result['success']:
                current_balance = balance_result['data']['available']
                
                if self.last_balance > 0:
                    drawdown = (self.last_balance - current_balance) / self.last_balance
                    
                    if drawdown >= Config.DRAWDOWN_CLOSE_PCT:
                        logger.warning(f"⚠️ High drawdown detected: {drawdown*100:.2f}%")
                        self._emergency_stop()
            
        except Exception as e:
            logger.error(f"Error checking risk conditions: {e}")

    def _emergency_stop(self):
        """Emergency stop all trading"""
        logger.critical("🛑 EMERGENCY STOP TRIGGERED")
        self.stop()
        
        self._emit_data('emergency_stop', {
            'reason': 'High drawdown detected',
            'timestamp': datetime.now().isoformat()
        })

    def get_statistics(self):
        """Get bot statistics"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'is_running': self.is_running,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(self.total_pnl, 2),
            'consecutive_losses': self.consecutive_losses,
            'last_balance': round(self.last_balance, 2)
        }

    def _emit_status(self, status):
        """Emit bot status update"""
        if self.socketio:
            self.socketio.emit('bot_status', {'status': status})

    def _emit_data(self, event, data):
        """Emit data update"""
        if self.socketio:
            self.socketio.emit(event, data)

    def _perform_analysis(self):
        """Perform technical and AI analysis"""
        try:
            # Get klines data
            klines_result = self.api.get_klines(Config.SYMBOL, '5m', 100)
            if not klines_result['success']:
                logger.error("Failed to get klines for analysis")
                return
            
            klines = klines_result['data']
            
            # Technical analysis
            tech_signals = self.tech_analysis.analyze(klines)
            
            # AI analysis using Gemini
            ai_analysis = self.gemini_handler.analyze_market(klines)
            
            # Combined analysis
            analysis_data = {
                'technical': tech_signals,
                'ai': ai_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            # Emit analysis update
            self._emit_data('analysis_update', analysis_data)
            
            logger.info(f"📊 Analysis completed - Tech Score: {tech_signals.get('score', 0)}")
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
