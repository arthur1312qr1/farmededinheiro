import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import requests

from config import Config
from bitget_api import BitgetAPI
from gemini_handler import GeminiErrorHandler
from api_integrations import MarketDataAggregator

logger = logging.getLogger(__name__)

class TradingBot:
    """Enhanced cryptocurrency trading bot with AI error correction"""
    
    def __init__(self, config: Config, gemini_handler: GeminiErrorHandler):
        """
        Initialize trading bot
        
        Args:
            config: Bot configuration
            gemini_handler: Gemini AI error handler
        """
        self.config = config
        self.gemini_handler = gemini_handler
        self.state_lock = threading.Lock()
        self.running = False
        
        # Initialize Bitget API
        if config.validate_api_keys():
            self.bitget_api = BitgetAPI(
                api_key=config.BITGET_API_KEY,
                api_secret=config.BITGET_API_SECRET,
                passphrase=config.BITGET_PASSPHRASE,
                testnet=config.PAPER_TRADING
            )
        else:
            self.bitget_api = None
            logger.error("Cannot initialize Bitget API - missing credentials")
        
        # Initialize market data aggregator
        self.market_aggregator = MarketDataAggregator(config)
        
        # Load or create bot state
        self.state = self._load_state()
        
        # Trading parameters
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.emergency_stop = False
        self.balance = 0.0
        self.total_trades = 0
        
        logger.info(f"Trading bot initialized for {config.SYMBOL} (Paper trading: {config.PAPER_TRADING})")
    
    def _load_state(self) -> Dict[str, Any]:
        """Load bot state from file"""
        try:
            with open(self.config.STATE_FILE, 'r') as f:
                state = json.load(f)
                logger.info("Loaded bot state from file")
                return state
        except FileNotFoundError:
            logger.info("No existing state file found, creating default state")
            return self._create_default_state()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid state file format: {e}")
            return self._create_default_state()
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return self._create_default_state()
    
    def _create_default_state(self) -> Dict[str, Any]:
        """Create default bot state"""
        default_state = {
            'balance': 0.0,
            'total_profit': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'current_positions': {},
            'last_prices': {},
            'trading_history': [],
            'consecutive_losses': 0,
            'last_trade_time': None,
            'emergency_stop': False,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        self._save_state(default_state)
        return default_state
    
    def _save_state(self, state: Optional[Dict] = None) -> None:
        """Save bot state to file"""
        if state is None:
            state = self.state
        
        with self.state_lock:
            try:
                state['last_updated'] = datetime.now().isoformat()
                with open(self.config.STATE_FILE, 'w') as f:
                    json.dump(state, f, indent=2, default=str)
                logger.debug("Bot state saved successfully")
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current bot state"""
        with self.state_lock:
            return self.state.copy()
    
    def get_balance_info(self) -> Dict[str, Any]:
        """Get current balance information"""
        try:
            if not self.bitget_api:
                return {
                    'error': 'API not available',
                    'paper_trading': self.config.PAPER_TRADING,
                    'available_balance': self.state.get('balance', 0.0)
                }
            
            if self.config.PAPER_TRADING:
                return {
                    'paper_trading': True,
                    'available_balance': self.state.get('balance', 1000.0),
                    'total_equity': self.state.get('balance', 1000.0),
                    'unrealized_pnl': 0.0,
                    'currency': 'USDT',
                    'last_updated': datetime.now().isoformat()
                }
            else:
                balance_info = self.bitget_api.get_futures_balance()
                balance_info['paper_trading'] = False
                return balance_info
                
        except Exception as e:
            logger.error(f"Error getting balance info: {e}")
            return {
                'error': str(e),
                'paper_trading': self.config.PAPER_TRADING,
                'available_balance': 0.0
            }
    
    def execute_trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        try:
            logger.debug("Starting trading cycle")
            
            # Check emergency stop
            if self.emergency_stop:
                logger.warning("Emergency stop is active - skipping trading cycle")
                return
            
            # Check consecutive losses
            if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
                logger.warning(f"Maximum consecutive losses reached ({self.consecutive_losses})")
                self._handle_max_losses()
                return
            
            # Get market data
            market_data = self._get_market_data()
            if not market_data:
                logger.warning("No market data available - skipping cycle")
                return
            
            # Get current positions
            positions = self._get_current_positions()
            
            # Analyze market conditions
            signal = self._analyze_market_conditions(market_data, positions)
            
            # Execute trading logic
            if signal:
                self._execute_trading_signal(signal, market_data)
            
            # Update state
            self._update_state(market_data, positions)
            
            logger.debug("Trading cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            
            # Use Gemini AI for error analysis
            try:
                context = {
                    'function': 'execute_trading_cycle',
                    'symbol': self.config.SYMBOL,
                    'paper_trading': self.config.PAPER_TRADING,
                    'consecutive_losses': self.consecutive_losses
                }
                
                error_analysis = self.gemini_handler.analyze_and_fix_error(str(e), context)
                logger.info(f"Gemini error analysis: {error_analysis}")
                
                # Apply automatic fixes based on analysis
                if "RESTART_REQUIRED" in error_analysis:
                    logger.warning("Gemini recommends restart - setting emergency stop")
                    self.emergency_stop = True
                elif "STOP_TRADING" in error_analysis:
                    logger.warning("Gemini recommends stopping trading")
                    self.stop()
                    
            except Exception as gemini_error:
                logger.error(f"Gemini error analysis failed: {gemini_error}")
            
            raise  # Re-raise the original exception
    
    def _get_market_data(self) -> Optional[Dict[str, Any]]:
        """Get current market data using all integrated APIs"""
        try:
            if self.config.PAPER_TRADING or not self.bitget_api:
                # Use external sources for paper trading
                return self._get_enhanced_market_data()
            else:
                # Combine Bitget data with external APIs
                bitget_data = self.bitget_api.get_market_data(self.config.SYMBOL)
                enhanced_data = self._get_enhanced_market_data()
                
                # Merge data for complete analysis
                if bitget_data and enhanced_data:
                    bitget_data.update({
                        'news': enhanced_data.get('news', []),
                        'market_sentiment': enhanced_data.get('sentiment', {}),
                        'gas_fees': enhanced_data.get('gas_fees', {}),
                        'general_trend': enhanced_data.get('summary', {}).get('trend', 'UNKNOWN')
                    })
                
                return bitget_data or enhanced_data
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def _get_enhanced_market_data(self) -> Optional[Dict[str, Any]]:
        """Get enhanced market data using multi-API aggregator"""
        try:
            # Use market data aggregator for complete analysis
            complete_analysis = self.market_aggregator.get_complete_market_analysis()
            
            if complete_analysis:
                price_data = complete_analysis.get('price_data', {})
                
                # Format data for compatibility with bot
                market_data = {
                    'symbol': self.config.SYMBOL,
                    'last_price': price_data.get('current_price', 0),
                    'change_percent_24h': price_data.get('change_24h', 0),
                    'volume_24h': price_data.get('volume_24h', 0),
                    'timestamp': datetime.now().isoformat(),
                    'source': price_data.get('source', 'aggregated')
                }
                
                # Add extra data from complete analysis
                market_data.update({
                    'news': complete_analysis.get('news', []),
                    'sentiment': complete_analysis.get('sentiment', {}),
                    'gas_fees': complete_analysis.get('gas_fees', {}),
                    'summary': complete_analysis.get('summary', {}),
                    'enhanced_analysis': True
                })
                
                logger.info(f"Enhanced data obtained: price ${market_data['last_price']}, 24h change: {market_data['change_percent_24h']:.2f}%")
                return market_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting enhanced market data: {e}")
            return None
    
    def _get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current trading positions"""
        try:
            if self.config.PAPER_TRADING or not self.bitget_api:
                # Return paper trading positions from state
                return list(self.state.get('current_positions', {}).values())
            else:
                return self.bitget_api.get_positions(self.config.SYMBOL)
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _analyze_market_conditions(self, market_data: Dict, positions: List) -> Optional[Dict[str, Any]]:
        """Analyze market conditions using AI and generate trading signals"""
        try:
            current_price = market_data.get('last_price', 0)
            if current_price <= 0:
                logger.warning("Invalid current price")
                return None
            
            # Basic momentum analysis
            price_change_24h = market_data.get('change_percent_24h', 0)
            
            # Check for open positions
            has_long_position = any(pos.get('side', '').lower() == 'long' for pos in positions)
            has_short_position = any(pos.get('side', '').lower() == 'short' for pos in positions)
            
            signal = None
            
            # Use Gemini AI for advanced analysis if available
            if self.gemini_handler and market_data.get('enhanced_analysis'):
                try:
                    # Prepare context for AI analysis
                    trading_context = {
                        'current_price': current_price,
                        'change_24h': price_change_24h,
                        'volume_24h': market_data.get('volume_24h', 0),
                        'news': market_data.get('news', [])[:3],
                        'sentiment': market_data.get('sentiment', {}),
                        'gas_fees': market_data.get('gas_fees', {}),
                        'open_positions': {
                            'long': has_long_position,
                            'short': has_short_position
                        },
                        'available_balance': self.balance,
                        'symbol': self.config.SYMBOL,
                        'total_trades': self.total_trades,
                        'consecutive_losses': self.consecutive_losses
                    }
                    
                    # Request AI analysis
                    ai_analysis = self.gemini_handler.analyze_trading_performance(trading_context)
                    
                    if ai_analysis:
                        # Parse AI response to extract trading signal
                        ai_upper = ai_analysis.upper()
                        
                        if 'BUY' in ai_upper or 'LONG' in ai_upper:
                            if not has_long_position:
                                signal = {
                                    'action': 'open_long',
                                    'reason': f'Gemini AI recommends BUY: {ai_analysis[:100]}...',
                                    'confidence': 0.8,
                                    'ai_analysis': True
                                }
                        elif 'SELL' in ai_upper or 'SHORT' in ai_upper:
                            if not has_short_position:
                                signal = {
                                    'action': 'open_short',
                                    'reason': f'Gemini AI recommends SELL: {ai_analysis[:100]}...',
                                    'confidence': 0.8,
                                    'ai_analysis': True
                                }
                        elif 'CLOSE' in ai_upper:
                            if has_long_position:
                                signal = {
                                    'action': 'close_long',
                                    'reason': f'Gemini AI recommends close position: {ai_analysis[:100]}...',
                                    'confidence': 0.9,
                                    'ai_analysis': True
                                }
                            elif has_short_position:
                                signal = {
                                    'action': 'close_short',
                                    'reason': f'Gemini AI recommends close position: {ai_analysis[:100]}...',
                                    'confidence': 0.9,
                                    'ai_analysis': True
                                }
                        
                        # If AI generated signal, use it; otherwise use basic analysis
                        if signal:
                            logger.info(f"AI generated signal: {signal}")
                            return signal
                        
                except Exception as gemini_error:
                    logger.warning(f"AI analysis error, using basic analysis: {gemini_error}")
            
            # Basic analysis as fallback
            # Entry conditions
            if not has_long_position and not has_short_position:
                if price_change_24h > 2.0:  # Strong upward momentum
                    signal = {
                        'action': 'open_long',
                        'reason': f'Strong upward momentum: {price_change_24h:.2f}%',
                        'confidence': min(abs(price_change_24h) / 10.0, 1.0),
                        'ai_analysis': False
                    }
                elif price_change_24h < -2.0:  # Strong downward momentum
                    signal = {
                        'action': 'open_short',
                        'reason': f'Strong downward momentum: {price_change_24h:.2f}%',
                        'confidence': min(abs(price_change_24h) / 10.0, 1.0),
                        'ai_analysis': False
                    }
            
            # Exit conditions
            elif has_long_position and price_change_24h < -1.0:
                signal = {
                    'action': 'close_long',
                    'reason': f'Long position stop loss: {price_change_24h:.2f}%',
                    'confidence': 0.8,
                    'ai_analysis': False
                }
            elif has_short_position and price_change_24h > 1.0:
                signal = {
                    'action': 'close_short',
                    'reason': f'Short position stop loss: {price_change_24h:.2f}%',
                    'confidence': 0.8,
                    'ai_analysis': False
                }
            
            if signal:
                logger.info(f"Trading signal generated: {signal}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return None
    
    def _execute_trading_signal(self, signal: Dict[str, Any], market_data: Dict) -> None:
        """Execute trading signal"""
        try:
            action = signal.get('action')
            confidence = signal.get('confidence', 0.5)
            reason = signal.get('reason', 'No reason provided')
            
            logger.info(f"Executing trading signal: {action} (confidence: {confidence:.2f}) - {reason}")
            
            # Calculate position size based on available balance and risk management
            position_size = self._calculate_position_size(confidence)
            if position_size <= 0:
                logger.warning("Position size is zero or negative - skipping trade")
                return
            
            # Execute the trade
            if action == 'open_long':
                self._open_position('long', position_size, market_data)
            elif action == 'open_short':
                self._open_position('short', position_size, market_data)
            elif action == 'close_long':
                self._close_position('long', market_data)
            elif action == 'close_short':
                self._close_position('short', market_data)
            else:
                logger.warning(f"Unknown trading action: {action}")
            
        except Exception as e:
            logger.error(f"Error executing trading signal: {e}")
            raise
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on risk management"""
        try:
            balance_info = self.get_balance_info()
            available_balance = balance_info.get('available_balance', 0)
            
            if available_balance <= 0:
                return 0
            
            # Risk management: use percentage of available balance
            risk_percentage = 0.02 * confidence  # 2% max risk, scaled by confidence
            risk_amount = available_balance * risk_percentage
            
            # Convert to position size (simplified calculation)
            # In reality, this would depend on the current price and leverage
            position_size = risk_amount * 10  # Assuming 10x leverage
            
            logger.debug(f"Calculated position size: {position_size} (balance: {available_balance}, risk: {risk_percentage:.2%})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _open_position(self, side: str, size: float, market_data: Dict) -> None:
        """Open a new position"""
        try:
            current_price = market_data.get('last_price', 0)
            
            if self.config.PAPER_TRADING:
                # Paper trading logic
                position = {
                    'symbol': self.config.SYMBOL,
                    'side': side,
                    'size': size,
                    'entry_price': current_price,
                    'timestamp': datetime.now().isoformat(),
                    'leverage': self.config.MIN_LEVERAGE
                }
                
                with self.state_lock:
                    self.state['current_positions'][side] = position
                    self._save_state()
                
                logger.info(f"[PAPER] Opened {side} position: size={size}, price={current_price}")
                
            else:
                # Real trading
                if not self.bitget_api:
                    raise Exception("Bitget API not available")
                
                trade_side = 'buy' if side == 'long' else 'sell'
                result = self.bitget_api.place_market_order(
                    symbol=self.config.SYMBOL,
                    side=trade_side,
                    size=size,
                    leverage=self.config.MIN_LEVERAGE
                )
                
                logger.info(f"Opened {side} position: {result}")
                
                # Update state
                with self.state_lock:
                    self.state['total_trades'] += 1
                    self.state['last_trade_time'] = datetime.now().isoformat()
                    self._save_state()
            
        except Exception as e:
            logger.error(f"Error opening {side} position: {e}")
            raise
    
    def _close_position(self, side: str, market_data: Dict) -> None:
        """Close an existing position"""
        try:
            current_price = market_data.get('last_price', 0)
            
            if self.config.PAPER_TRADING:
                # Paper trading logic
                with self.state_lock:
                    position = self.state['current_positions'].get(side)
                    if position:
                        entry_price = position.get('entry_price', 0)
                        size = position.get('size', 0)
                        
                        # Calculate P&L
                        if side == 'long':
                            pnl = (current_price - entry_price) * size / entry_price
                        else:
                            pnl = (entry_price - current_price) * size / entry_price
                        
                        self.state['total_profit'] += pnl
                        
                        if pnl > 0:
                            self.state['winning_trades'] += 1
                            self.consecutive_losses = 0
                        else:
                            self.state['losing_trades'] += 1
                            self.consecutive_losses += 1
                        
                        # Remove position
                        del self.state['current_positions'][side]
                        
                        # Update balance
                        self.state['balance'] = self.state.get('balance', 1000.0) + pnl
                        
                        self._save_state()
                        
                        logger.info(f"[PAPER] Closed {side} position: P&L={pnl:.2f}, price={current_price}")
                        
            else:
                # Real trading
                if not self.bitget_api:
                    raise Exception("Bitget API not available")
                
                result = self.bitget_api.close_position(
                    symbol=self.config.SYMBOL,
                    side=side
                )
                
                logger.info(f"Closed {side} position: {result}")
                
                # Update state
                with self.state_lock:
                    self.state['last_trade_time'] = datetime.now().isoformat()
                    self._save_state()
            
        except Exception as e:
            logger.error(f"Error closing {side} position: {e}")
            raise
    
    def _update_state(self, market_data: Dict, positions: List) -> None:
        """Update bot state with current market data and positions"""
        try:
            with self.state_lock:
                self.state['last_prices'][self.config.SYMBOL] = market_data.get('last_price', 0)
                self.state['last_updated'] = datetime.now().isoformat()
                self.state['consecutive_losses'] = self.consecutive_losses
                self._save_state()
                
        except Exception as e:
            logger.error(f"Error updating state: {e}")
    
    def _handle_max_losses(self) -> None:
        """Handle maximum consecutive losses reached"""
        try:
            logger.warning("Handling maximum consecutive losses")
            
            # Close all positions
            positions = self._get_current_positions()
            for position in positions:
                side = position.get('side', '').lower()
                if side in ['long', 'short']:
                    self._close_position(side, {'last_price': position.get('mark_price', 0)})
            
            # Set emergency stop
            self.emergency_stop = True
            
            # Use Gemini AI for analysis
            try:
                context = {
                    'consecutive_losses': self.consecutive_losses,
                    'max_allowed': self.config.MAX_CONSECUTIVE_LOSSES,
                    'total_trades': self.state.get('total_trades', 0),
                    'win_rate': self._calculate_win_rate()
                }
                
                analysis = self.gemini_handler.analyze_trading_performance(context)
                logger.info(f"Gemini performance analysis after max losses: {analysis}")
                
            except Exception as e:
                logger.error(f"Error in Gemini analysis after max losses: {e}")
            
            # Reset consecutive losses counter after a waiting period
            # This could be enhanced with more sophisticated recovery logic
            
        except Exception as e:
            logger.error(f"Error handling max losses: {e}")
    
    def _calculate_win_rate(self) -> float:
        """Calculate current win rate"""
        try:
            winning = self.state.get('winning_trades', 0)
            losing = self.state.get('losing_trades', 0)
            total = winning + losing
            
            if total == 0:
                return 0.0
            
            return winning / total
            
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def stop(self) -> None:
        """Stop the trading bot"""
        try:
            logger.info("Stopping trading bot")
            self.running = False
            self.emergency_stop = True
            
            # Close all open positions (optional)
            # positions = self._get_current_positions()
            # for position in positions:
            #     side = position.get('side', '').lower()
            #     if side in ['long', 'short']:
            #         self._close_position(side, {'last_price': position.get('mark_price', 0)})
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
