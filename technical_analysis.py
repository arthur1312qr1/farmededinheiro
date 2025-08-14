import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    def __init__(self):
        logger.info("📈 Technical Analysis initialized")

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
        
        # Align EMAs
        if len(ema_fast) > len(ema_slow):
            ema_fast = ema_fast[-len(ema_slow):]
        elif len(ema_slow) > len(ema_fast):
            ema_slow = ema_slow[-len(ema_fast):]
        
        macd_line = [fast_val - slow_val for fast_val, slow_val in zip(ema_fast, ema_slow)]
        signal_line = self.calculate_ema(macd_line, signal)
        
        # Align MACD and signal
        if len(macd_line) > len(signal_line):
            macd_aligned = macd_line[-len(signal_line):]
        else:
            macd_aligned = macd_line
            
        histogram = [macd_val - signal_val for macd_val, signal_val in zip(macd_aligned, signal_line)]
        
        return {
            'macd': macd_aligned,
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
            # Get the subset of prices for standard deviation calculation
            price_subset = prices[i:i + period]
            if len(price_subset) == period:
                std = np.std(price_subset)
                upper_band.append(sma[i] + (std * std_dev))
                lower_band.append(sma[i] - (std * std_dev))
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    def detect_support_resistance(self, klines: List[Dict]) -> Dict:
        """Detect support and resistance levels"""
        if len(klines) < 20:
            return {'support': [], 'resistance': []}
        
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        
        # Find local peaks and troughs
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(highs) - 2):
            # Resistance (local high)
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
            
            # Support (local low)
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        # Filter and sort levels
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:5]
        support_levels = sorted(list(set(support_levels)))[-5:]
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def analyze(self, klines: List[Dict]) -> Dict:
        """Perform comprehensive technical analysis"""
        if len(klines) < 30:
            return {'error': 'Insufficient data for analysis'}
        
        # Extract prices
        closes = [k['close'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        
        # Calculate indicators
        sma_20 = self.calculate_sma(closes, 20)
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        rsi = self.calculate_rsi(closes)
        macd_data = self.calculate_macd(closes)
        bb_data = self.calculate_bollinger_bands(closes)
        sr_levels = self.detect_support_resistance(klines)
        
        current_price = closes[-1]
        
        # Analysis score (0-100)
        score = 50  # Neutral starting point
        signals = []
        
        # RSI Analysis
        if rsi and len(rsi) > 0:
            current_rsi = rsi[-1]
            if current_rsi < 30:
                score += 15
                signals.append("RSI oversold (bullish)")
            elif current_rsi > 70:
                score -= 15
                signals.append("RSI overbought (bearish)")
        
        # Moving Average Analysis
        if sma_20 and len(sma_20) > 0:
            if current_price > sma_20[-1]:
                score += 10
                signals.append("Price above SMA20 (bullish)")
            else:
                score -= 10
                signals.append("Price below SMA20 (bearish)")
        
        # EMA Crossover
        if len(ema_12) > 1 and len(ema_26) > 1:
            if ema_12[-1] > ema_26[-1] and ema_12[-2] <= ema_26[-2]:
                score += 20
                signals.append("Golden cross (bullish)")
            elif ema_12[-1] < ema_26[-1] and ema_12[-2] >= ema_26[-2]:
                score -= 20
                signals.append("Death cross (bearish)")
        
        # MACD Analysis
        if macd_data['macd'] and macd_data['signal'] and len(macd_data['histogram']) > 1:
            current_histogram = macd_data['histogram'][-1]
            prev_histogram = macd_data['histogram'][-2]
            
            if current_histogram > 0 and prev_histogram <= 0:
                score += 15
                signals.append("MACD bullish crossover")
            elif current_histogram < 0 and prev_histogram >= 0:
                score -= 15
                signals.append("MACD bearish crossover")
        
        # Bollinger Bands Analysis
        if bb_data['upper'] and bb_data['lower'] and len(bb_data['upper']) > 0:
            if current_price <= bb_data['lower'][-1]:
                score += 10
                signals.append("Price at lower Bollinger Band (bullish)")
            elif current_price >= bb_data['upper'][-1]:
                score -= 10
                signals.append("Price at upper Bollinger Band (bearish)")
        
        # Constrain score to 0-100
        score = max(0, min(100, score))
        
        return {
            'score': score,
            'signals': signals,
            'indicators': {
                'rsi': rsi[-1] if rsi else None,
                'sma_20': sma_20[-1] if sma_20 else None,
                'macd': macd_data['macd'][-1] if macd_data['macd'] else None,
                'signal': macd_data['signal'][-1] if macd_data['signal'] else None
            },
            'support_resistance': sr_levels,
            'current_price': current_price
        }

    def get_trading_signals(self, klines: List[Dict]) -> Dict:
        """Get trading signals based on analysis"""
        analysis = self.analyze(klines)
        
        if 'error' in analysis:
            return {'action': 'hold', 'strength': 0, 'reason': 'Insufficient data'}
        
        score = analysis['score']
        
        # Determine action based on score
        if score >= 70:
            action = 'buy'
            strength = (score - 50) / 50  # 0.4 to 1.0
        elif score <= 30:
            action = 'sell' 
            strength = (50 - score) / 50  # 0.4 to 1.0
        else:
            action = 'hold'
            strength = 0.5
        
        return {
            'action': action,
            'strength': round(strength, 2),
            'score': score,
            'reason': f"Technical score: {score}/100",
            'signals': analysis['signals']
        }
