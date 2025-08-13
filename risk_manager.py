import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
from models import BotStatus, Trade
from app import db
from config import Config

logger = logging.getLogger(__name__)

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self):
        self.config = Config
    
    def check_trading_conditions(self) -> Dict:
        """
        Check if trading conditions are met
        Returns: {'allowed': bool, 'reason': str}
        """
        try:
            # Get bot status
            bot_status = BotStatus.query.first()
            if not bot_status:
                return {'allowed': False, 'reason': 'Bot status not initialized'}
            
            # Check if bot is active
            if not bot_status.is_active:
                return {'allowed': False, 'reason': 'Bot is not active'}
            
            # Skip minimum balance check since MIN_BALANCE is 0
            # Balance check removed for aggressive trading
            
            # Check consecutive losses (more aggressive - allow up to 5)
            if bot_status.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
                return {'allowed': False, 'reason': f'Max consecutive losses reached: {bot_status.consecutive_losses}'}
            
            # All checks passed
            return {'allowed': True, 'reason': 'All conditions met'}
            
        except Exception as e:
            logger.error(f"Error checking trading conditions: {e}")
            return {'allowed': False, 'reason': f'Error: {str(e)}'}
    
    def calculate_position_size(self, balance: float, confidence: float) -> float:
        """
        Calculate position size based on balance and signal confidence
        More aggressive sizing for high frequency trading
        """
        try:
            # Base position size (up to 80% of balance for high frequency)
            base_percentage = self.config.MAX_POSITION_PERCENT
            
            # Adjust based on confidence (higher confidence = larger position)
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
            
            position_percentage = base_percentage * confidence_multiplier
            position_size = balance * (position_percentage / 100)
            
            # Apply leverage
            position_size *= self.config.LEVERAGE
            
            logger.info(f"Position size calculated: ${position_size:.2f} ({position_percentage:.1f}% of balance)")
            
            return max(10, position_size)  # Minimum $10 position
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 10.0
    
    def calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price"""
        try:
            stop_loss_percentage = self.config.STOP_LOSS_PERCENT / 100
            
            if side == 'long':
                return entry_price * (1 - stop_loss_percentage)
            else:  # short
                return entry_price * (1 + stop_loss_percentage)
                
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price
    
    def calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price"""
        try:
            take_profit_percentage = self.config.TAKE_PROFIT_PERCENT / 100
            
            if side == 'long':
                return entry_price * (1 + take_profit_percentage)
            else:  # short
                return entry_price * (1 - take_profit_percentage)
                
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return entry_price
    
    def should_close_position(self, trade: Trade, current_price: float) -> Dict:
        """
        Check if position should be closed
        Returns: {'should_close': bool, 'reason': str}
        """
        try:
            if trade.status != 'open':
                return {'should_close': False, 'reason': 'Trade not open'}
            
            # Check stop loss
            if trade.side == 'long' and current_price <= trade.stop_loss:
                return {'should_close': True, 'reason': 'Stop loss hit'}
            elif trade.side == 'short' and current_price >= trade.stop_loss:
                return {'should_close': True, 'reason': 'Stop loss hit'}
            
            # Check take profit
            if trade.side == 'long' and current_price >= trade.take_profit:
                return {'should_close': True, 'reason': 'Take profit reached'}
            elif trade.side == 'short' and current_price <= trade.take_profit:
                return {'should_close': True, 'reason': 'Take profit reached'}
            
            # Check for early exit on trend reversal (intelligent stop loss)
            reversal_check = self._check_trend_reversal(trade, current_price)
            if reversal_check['should_exit']:
                return {'should_close': True, 'reason': reversal_check['reason']}
            
            return {'should_close': False, 'reason': 'All conditions normal'}
            
        except Exception as e:
            logger.error(f"Error checking position closure: {e}")
            return {'should_close': False, 'reason': f'Error: {str(e)}'}
    
    def _check_trend_reversal(self, trade: Trade, current_price: float) -> Dict:
        """Check for trend reversal for early exit"""
        try:
            # Calculate current P&L percentage
            if trade.side == 'long':
                pnl_percentage = ((current_price - trade.entry_price) / trade.entry_price) * 100
            else:
                pnl_percentage = ((trade.entry_price - current_price) / trade.entry_price) * 100
            
            # Apply leverage effect
            pnl_percentage *= trade.leverage
            
            # Intelligent early exit on trend reversal - more aggressive
            if pnl_percentage < -2.0:
                return {'should_exit': True, 'reason': 'Trend reversal detected - immediate exit'}
            
            # Take profits on good moves
            if pnl_percentage > 8.0:
                return {'should_exit': True, 'reason': 'Good profit - secure gains'}
            
            return {'should_exit': False, 'reason': 'No reversal detected'}
            
        except Exception as e:
            logger.error(f"Error checking trend reversal: {e}")
            return {'should_exit': False, 'reason': 'Error in reversal check'}
    
    def update_trade_pnl(self, trade: Trade, current_price: float):
        """Update trade P&L"""
        try:
            if trade.side == 'long':
                price_diff = current_price - trade.entry_price
            else:
                price_diff = trade.entry_price - current_price
            
            # Calculate P&L with leverage
            pnl = (price_diff / trade.entry_price) * trade.quantity * trade.leverage
            pnl_percentage = (price_diff / trade.entry_price) * trade.leverage * 100
            
            trade.pnl = pnl
            trade.pnl_percentage = pnl_percentage
            
            db.session.commit()
            
        except Exception as e:
            logger.error(f"Error updating trade P&L: {e}")
    
    def record_trade_result(self, trade: Trade, exit_price: float, reason: str):
        """Record trade result and update statistics"""
        try:
            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = datetime.utcnow()
            trade.status = 'closed'
            
            # Calculate final P&L
            if trade.side == 'long':
                price_diff = exit_price - trade.entry_price
            else:
                price_diff = trade.entry_price - exit_price
            
            trade.pnl = (price_diff / trade.entry_price) * trade.quantity * trade.leverage
            trade.pnl_percentage = (price_diff / trade.entry_price) * trade.leverage * 100
            
            # Update bot statistics
            bot_status = BotStatus.query.first()
            if bot_status:
                bot_status.total_trades += 1
                bot_status.daily_pnl += trade.pnl
                bot_status.balance += trade.pnl
                
                if trade.pnl > 0:
                    bot_status.winning_trades += 1
                    bot_status.consecutive_losses = 0  # Reset on win
                else:
                    bot_status.losing_trades += 1
                    bot_status.consecutive_losses += 1
                
                bot_status.last_updated = datetime.utcnow()
            
            db.session.commit()
            
            logger.info(f"Trade closed: {trade.side} {trade.symbol} P&L: ${trade.pnl:.2f} ({trade.pnl_percentage:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error recording trade result: {e}")
            db.session.rollback()
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        try:
            bot_status = BotStatus.query.first()
            if not bot_status:
                return {'error': 'No bot status found'}
            
            # Get recent trades for additional metrics
            recent_trades = Trade.query.filter(
                Trade.entry_time >= datetime.utcnow() - timedelta(days=1),
                Trade.status == 'closed'
            ).all()
            
            # Calculate additional metrics
            total_pnl = sum(trade.pnl for trade in recent_trades if trade.pnl)
            avg_win = np.mean([trade.pnl for trade in recent_trades if trade.pnl and trade.pnl > 0]) if recent_trades else 0
            avg_loss = np.mean([trade.pnl for trade in recent_trades if trade.pnl and trade.pnl < 0]) if recent_trades else 0
            
            return {
                'balance': bot_status.balance,
                'daily_pnl': bot_status.daily_pnl,
                'total_trades': bot_status.total_trades,
                'win_rate': (bot_status.winning_trades / max(bot_status.total_trades, 1)) * 100,
                'consecutive_losses': bot_status.consecutive_losses,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf'),
                'trades_today': len(recent_trades)
            }
            
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {'error': str(e)}
