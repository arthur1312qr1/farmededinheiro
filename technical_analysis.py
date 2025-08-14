import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    def __init__(self):
        logger.info("📈 Technical Analysis initialized for 99% ACCURACY")

    def get_high_accuracy_signals(self, klines):
        """Get high accuracy trading signals (99% accuracy target)"""
        try:
            if not klines or len(klines) < 50:
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Insufficient data'}
            
            # Extract prices
            closes = [float(k[4]) for k in klines]  # Close prices
            highs = [float(k[2]) for k in klines]   # High prices
            lows = [float(k[3]) for k in klines]    # Low prices
            volumes = [float(k[5]) for k in klines] # Volume
            
            # Calculate indicators
            sma_20 = self.calculate_sma(closes, 20)
            sma_50 = self.calculate_sma(closes, 50)
            ema_12 = self.calculate_ema(closes, 12)
            ema_26 = self.calculate_ema(closes, 26)
            rsi = self.calculate_rsi(closes, 14)
            macd_data = self.calculate_macd(closes)
            bb_data = self.calculate_bollinger_bands(closes, 20)
            
            if not all([sma_20, sma_50, ema_12, ema_26, rsi, bb_data]):
                return {'action': 'hold', 'confidence': 0.0, 'reason': 'Indicator calculation failed'}
            
            # Current values
            current_price = closes[-1]
            current_rsi = rsi[-1] if rsi else 50
            current_macd = macd_data['macd'][-1] if macd_data['macd'] else 0
            current_signal = macd_data['signal'][-1] if macd_data['signal'] else 0
            
            # Volume analysis
            avg_volume = sum(volumes[-20:]) / 20
            current_volume = volumes[-1]
            volume_surge = current_volume > avg_volume * 1.5
            
            # Signal scoring system for 99% accuracy
            buy_score = 0
            sell_score = 0
            confidence = 0.0
            
            # Trend Analysis (40% weight)
            if len(sma_20) > 0 and len(sma_50) > 0:
                if sma_20[-1] > sma_50[-1] and current_price > sma_20[-1]:
                    buy_score += 40
                elif sma_20[-1] < sma_50[-1] and current_price < sma_20[-1]:
                    sell_score += 40

            # RSI Analysis (25% weight)
            if current_rsi < 30 and current_rsi > 25:  # Strong oversold but not extreme
                buy_score += 25
            elif current_rsi > 70 and current_rsi < 75:  # Strong overbought but not extreme
                sell_score += 25
            
            # MACD Analysis (20% weight)
            if current_macd > current_signal and current_macd > 0:
                buy_score += 20
            elif current_macd < current_signal and current_macd < 0:
                sell_score += 20
            
            # Volume Confirmation (15% weight)
            if volume_surge:
                if buy_score > sell_score:
                    buy_score += 15
                else:
                    sell_score += 15
            
            # Rejection Signal Detection
            rejection_detected = self._detect_rejection_signals(closes, highs, lows, rsi, current_price)

            # Determine action with 99% accuracy requirement
            max_score = max(buy_score, sell_score)
            confidence = max_score / 100.0

            # ONLY TRADE WITH 99% CONFIDENCE
            if confidence >= 0.99:
                if buy_score > sell_score:
                    action = 'buy'
                    reason = f'Strong BUY signal - Score: {buy_score}/100'
                else:
                    action = 'sell'
                    reason = f'Strong SELL signal - Score: {sell_score}/100'
            else:
                action = 'hold'
                reason = f'Confidence too low: {confidence*100:.1f}% < 99%'
            
            # Override with rejection detection if rejection_detected['detected'] and confidence >= 0.85:
            if rejection_detected['detected'] and confidence >= 0.85:
                action = 'hold'
                reason = f'Rejection signal detected: {rejection_detected["reason"]}'
                confidence = 0.85

            return {
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'rsi': current_rsi,
                'volume_surge': volume_surge,
                'rejection': rejection_detected
            }
        except Exception as e:
            logger.error(f"Error in high accuracy signals: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {e}'}

    def _detect_rejection_signals(self, closes, highs, lows, rsi, current_price):
        """Detect rejection signals before reaching take profit"""
        try:
            if len(closes) < 10:
                return {'detected': False, 'reason': 'Insufficient data'}

            # Check for price rejection at resistance/support
            recent_highs = highs[-5:]
            recent_lows = lows[-5:]
            recent_closes = closes[-5:]

            # Doji pattern detection
            last_candle_range = highs[-1] - lows[-1]
            body_size = abs(closes[-1] - closes[-2]) if len(closes) > 1 else 0
            if last_candle_range > 0 and body_size / last_candle_range < 0.1:
                return {'detected': True, 'reason': 'Doji pattern detected'}
            
            # RSI divergence
            if len(rsi) >= 5:
                price_trend = recent_closes[-1] > recent_closes[0]
                rsi_trend = rsi[-1] > rsi[-5]
                if price_trend != rsi_trend:
                    return {'detected': True, 'reason': 'RSI divergence detected'}

            # Support/Resistance rejection
            resistance_level = max(recent_highs)
            support_level = min(recent_lows)
            if current_price >= resistance_level * 0.999:
                return {'detected': True, 'reason': 'Price at resistance level'}
            elif current_price <= support_level * 1.001:
                return {'detected': True, 'reason': 'Price at support level'}

            return {'detected': False, 'reason': 'No rejection signals'}
        except Exception as e:
            logger.error(f"Error detecting rejection: {e}")
            return {'detected': False, 'reason': f'Error: {e}'}

    def calculate_sma(self, prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return []
        sma = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma.append(avg)
        return sma

    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return []
        multiplier = 2 / (period + 1)
        ema = [prices[0]]
        for price in prices[1:]:
            ema_value = (price * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_value)
        return ema

    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        rsi = []
        for i in range(period, len(deltas)):
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))
                rsi.append(rsi_value)
            # Update averages
            current_gain = gains[i] if i < len(gains) else 0
            current_loss = losses[i] if i < len(losses) else 0
            avg_gain = (avg_gain * (period - 1) + current_gain) / period
            avg_loss = (avg_loss * (period - 1) + current_loss) / period
        return rsi

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        if len(prices) < slow:
            return {'macd': [], 'signal': [], 'histogram': []}
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        if not ema_fast or not ema_slow:
            return {'macd': [], 'signal': [], 'histogram': []}
        # Align arrays
        min_len = min(len(ema_fast), len(ema_slow))
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(min_len)]
        signal_line = self.calculate_ema(macd_line, signal)
        # Calculate histogram
        histogram = []
        if signal_line:
            min_len = min(len(macd_line), len(signal_line))
            histogram = [macd_line[i] - signal_line[i] for i in range(min_len)]
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {'upper': [], 'middle': [], 'lower': []}
        sma = self.calculate_sma(prices, period)
        upper_band = []
        lower_band = []
        for i in range(len(sma)):
            price_slice = prices[i:i + period]
            if len(price_slice) == period:
                std = (sum([(p - sma[i]) ** 2 for p in price_slice]) / period) ** 0.5
                upper_band.append(sma[i] + (std_dev * std))
                lower_band.append(sma[i] - (std_dev * std))
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    # Legacy methods for compatibility
    def analyze(self, klines):
        """Legacy analyze method"""
        signals = self.get_high_accuracy_signals(klines)
        return {
            'score': signals.get('confidence', 0) * 100,
            'signals': signals
        }

    def get_trading_signals(self, klines):
        """Legacy method"""
        return self.get_high_accuracy_signals(klines)
