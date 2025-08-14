import logging
from config import Config
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)
class RiskManager:
    def __init__(self):
        self.max_drawdown = Config.DRAWDOWN_CLOSE_PCT
        self.max_consecutive_losses = Config.MAX_CONSECUTIVE_LOSSES
        self.min_balance = Config.MIN_BALANCE_USDT
        self.liq_distance_threshold = Config.LIQ_DIST_THRESHOLD
        # Risk tracking
        self.consecutive_losses = 0
        self.peak_balance = 0
        self.daily_trades = 0
        self.last_trade_time = None
        logger.info("🛡️ Risk Manager initialized for REAL TRADING")

    def check_balance_risk(self, current_balance, peak_balance):
        """Check if balance is at risk"""
        if current_balance < self.min_balance:
            return {
                'risk': True,
                'level': 'high',
                'reason': f'Balance below minimum: ${current_balance:.2f} < ${self.min_balance:.2f}'
            }
        if peak_balance > 0:
            drawdown = (peak_balance - current_balance) / peak_balance
            if drawdown >= self.max_drawdown:
                return {
                    'risk': True,
                    'level': 'critical',
                    'reason': f'High drawdown: {drawdown*100:.2f}% >= {self.max_drawdown*100:.2f}%'
                }
        return {'risk': False}

    def check_consecutive_losses(self, losses_count):
        """Check consecutive losses risk"""
        if losses_count >= self.max_consecutive_losses:
            return {
                'risk': True,
                'level': 'high',
                'reason': f'Max consecutive losses reached: {losses_count}'
            }
        return {'risk': False}

    def check_trading_frequency(self):
        """Check if trading too frequently"""
        current_time = datetime.now()
        if self.last_trade_time:
            time_diff = current_time - self.last_trade_time
            if time_diff < timedelta(minutes=5):
                return {
                    'risk': True,
                    'level': 'medium',
                    'reason': 'Trading too frequently'
                }
        return {'risk': False}

    def calculate_position_size(self, balance, leverage, risk_percent=0.8):
        """Calculate safe position size using 80% margin"""
        # Use 80% of balance as specified by user
        max_risk_amount = balance * risk_percent
        # Account for leverage
        position_value = max_risk_amount * leverage
        return {
            'recommended_size': position_value,
            'max_loss': max_risk_amount,
            'leverage_used': leverage
        }

    def check_liquidation_risk(self, position_size, entry_price, current_price, leverage):
        """Check liquidation distance"""
        try:
            # Calculate liquidation price (simplified)
            maintenance_margin_rate = 0.005 # 0.5% for most crypto futures
            liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
            # Calculate distance to liquidation
            distance_to_liq = abs(current_price - liquidation_price) / current_price
            if distance_to_liq <= self.liq_distance_threshold:
                return {
                    'risk': True,
                    'level': 'critical',
                    'reason': f'Close to liquidation: {distance_to_liq*100:.2f}%',
                    'liquidation_price': liquidation_price
                }
            return {
                'risk': False,
                'liquidation_price': liquidation_price,
                'distance': distance_to_liq
            }
        except Exception as e:
            logger.error(f"Error calculating liquidation risk: {e}")
            return {'risk': False}

    def update_trade_result(self, is_profitable):
        """Update risk tracking with trade result"""
        self.last_trade_time = datetime.now()
        if is_profitable:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
        self.daily_trades += 1
        logger.info(f"Trade result updated - Consecutive losses: {self.consecutive_losses}")

    def get_risk_summary(self, balance, peak_balance):
        """Get comprehensive risk summary"""
        risks = []
        balance_risk = self.check_balance_risk(balance, peak_balance)
        if balance_risk['risk']:
            risks.append(balance_risk)
        losses_risk = self.check_consecutive_losses(self.consecutive_losses)
        if losses_risk['risk']:
            risks.append(losses_risk)
        frequency_risk = self.check_trading_frequency()
        if frequency_risk['risk']:
            risks.append(frequency_risk)
        return {
            'total_risks': len(risks),
            'risks': risks,
            'safe_to_trade': len(risks) == 0
        }
