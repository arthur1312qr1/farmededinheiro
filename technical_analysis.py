import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from config import Config

logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Advanced technical analysis with multiple indicators"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return []
        
        df = pd.DataFrame({'price': prices})
        delta = df['price'].diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).tolist()
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return {'macd': [], 'signal': [], 'histogram': []}
        
        df = pd.DataFrame({'price': prices})
        
        ema_fast = df['price'].ewm(span=fast).mean()
        ema_slow = df['price'].ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd.fillna(0).tolist(),
            'signal': signal_line.fillna(0).tolist(),
            'histogram': histogram.fillna(0).tolist()
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std: int = 2) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {'upper': [], 'middle': [], 'lower': [], 'width': []}
        
        df = pd.DataFrame({'price': prices})
        
        middle = df['price'].rolling(window=period).mean()
        std_dev = df['price'].rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        width = ((upper - lower) / middle) * 100
        
        return {
            'upper': upper.fillna(0).tolist(),
            'middle': middle.fillna(0).tolist(),
            'lower': lower.fillna(0).tolist(),
            'width': width.fillna(0).tolist()
        }
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return [prices[-1]] * len(prices) if prices else []
        
        df = pd.DataFrame({'price': prices})
        ema = df['price'].ewm(span=period).mean()
        
        return ema.fillna(prices[0] if prices else 0).tolist()
    
    @staticmethod
    def calculate_support_resistance(highs: List[float], lows: List[float], 
                                   closes: List[float], window: int = 20) -> Dict:
        """Calculate dynamic support and resistance levels"""
        if len(closes) < window:
            return {'support': 0, 'resistance': 0, 'pivot': 0}
        
        recent_highs = highs[-window:]
        recent_lows = lows[-window:]
        recent_closes = closes[-window:]
        
        # Calculate pivot points
        pivot = (max(recent_highs) + min(recent_lows) + recent_closes[-1]) / 3
        resistance = (2 * pivot) - min(recent_lows)
        support = (2 * pivot) - max(recent_highs)
        
        return {
            'support': support,
            'resistance': resistance,
            'pivot': pivot
        }
    
    @staticmethod
    def analyze_volume_profile(volumes: List[float], prices: List[float], 
                              bins: int = 20) -> Dict:
        """Analyze volume profile for key levels"""
        if len(volumes) < bins or len(prices) != len(volumes):
            return {'poc': 0, 'vah': 0, 'val': 0}  # Point of Control, Value Area High/Low
        
        # Create price bins
        price_range = max(prices) - min(prices)
        bin_size = price_range / bins
        
        volume_at_price = {}
        for i, price in enumerate(prices):
            bin_key = int((price - min(prices)) / bin_size)
            if bin_key not in volume_at_price:
                volume_at_price[bin_key] = 0
            volume_at_price[bin_key] += volumes[i]
        
        # Find Point of Control (highest volume)
        poc_bin = max(volume_at_price.keys(), key=lambda k: volume_at_price[k])
        poc_price = min(prices) + (poc_bin * bin_size)
        
        # Calculate Value Area (70% of volume)
        total_volume = sum(volume_at_price.values())
        value_area_volume = total_volume * 0.7
        
        # Sort bins by volume and find value area
        sorted_bins = sorted(volume_at_price.keys(), 
                           key=lambda k: volume_at_price[k], reverse=True)
        
        va_volume = 0
        va_bins = []
        for bin_key in sorted_bins:
            va_volume += volume_at_price[bin_key]
            va_bins.append(bin_key)
            if va_volume >= value_area_volume:
                break
        
        vah = min(prices) + (max(va_bins) * bin_size)  # Value Area High
        val = min(prices) + (min(va_bins) * bin_size)  # Value Area Low
        
        return {
            'poc': poc_price,
            'vah': vah,
            'val': val
        }
    
    @classmethod
    def comprehensive_analysis(cls, klines: List[Dict]) -> Dict:
        """Perform comprehensive technical analysis on kline data"""
        if not klines or len(klines) < 50:
            return cls._empty_analysis()
        
        # Extract price data
        opens = [k['open'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        closes = [k['close'] for k in klines]
        volumes = [k['volume'] for k in klines]
        
        try:
            # Calculate all indicators
            analysis = {
                'rsi': {},
                'macd': cls.calculate_macd(closes, Config.MACD_FAST, 
                                         Config.MACD_SLOW, Config.MACD_SIGNAL),
                'bollinger': cls.calculate_bollinger_bands(closes, Config.BB_PERIOD, Config.BB_STD),
                'ema': {},
                'support_resistance': cls.calculate_support_resistance(highs, lows, closes),
                'volume_profile': cls.analyze_volume_profile(volumes, closes),
                'price_action': cls._analyze_price_action(opens, highs, lows, closes),
                'current_price': closes[-1],
                'timestamp': klines[-1]['timestamp']
            }
            
            # Calculate RSI for different periods
            for period in Config.RSI_PERIODS:
                analysis['rsi'][f'rsi_{period}'] = cls.calculate_rsi(closes, period)
            
            # Calculate EMA for different periods
            for period in Config.EMA_PERIODS:
                analysis['ema'][f'ema_{period}'] = cls.calculate_ema(closes, period)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return cls._empty_analysis()
    
    @staticmethod
    def _analyze_price_action(opens: List[float], highs: List[float], 
                            lows: List[float], closes: List[float]) -> Dict:
        """Analyze price action patterns"""
        if len(closes) < 3:
            return {'trend': 'neutral', 'momentum': 0, 'volatility': 0}
        
        # Calculate trend
        short_trend = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        long_trend = (closes[-1] - closes[-20]) / closes[-20] if len(closes) >= 20 else 0
        
        trend = 'bullish' if short_trend > 0.02 else 'bearish' if short_trend < -0.02 else 'neutral'
        
        # Calculate momentum
        momentum = short_trend * 100
        
        # Calculate volatility (average true range)
        atr_values = []
        for i in range(1, min(14, len(closes))):
            true_range = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]),
                abs(lows[-i] - closes[-i-1])
            )
            atr_values.append(true_range)
        
        volatility = np.mean(atr_values) / closes[-1] * 100 if atr_values else 0
        
        return {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility,
            'short_trend': short_trend,
            'long_trend': long_trend
        }
    
    @staticmethod
    def _empty_analysis() -> Dict:
        """Return empty analysis structure"""
        return {
            'rsi': {f'rsi_{p}': [] for p in Config.RSI_PERIODS},
            'macd': {'macd': [], 'signal': [], 'histogram': []},
            'bollinger': {'upper': [], 'middle': [], 'lower': [], 'width': []},
            'ema': {f'ema_{p}': [] for p in Config.EMA_PERIODS},
            'support_resistance': {'support': 0, 'resistance': 0, 'pivot': 0},
            'volume_profile': {'poc': 0, 'vah': 0, 'val': 0},
            'price_action': {'trend': 'neutral', 'momentum': 0, 'volatility': 0},
            'current_price': 0,
            'timestamp': 0
        }
