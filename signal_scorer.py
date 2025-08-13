import logging
from typing import Dict, List
from config import Config

logger = logging.getLogger(__name__)

class SignalScorer:
    """Advanced signal scoring system with multiple indicator confluence"""
    
    @staticmethod
    def score_signal(analysis: Dict, timeframe: str) -> Dict:
        """Score trading signal based on technical analysis"""
        try:
            if not analysis or not analysis.get('current_price'):
                return {'signal': 'hold', 'confidence': 0.0, 'reasons': ['No analysis data']}
            
            bullish_signals = []
            bearish_signals = []
            confidence_factors = []
            
            # RSI Analysis (25% weight)
            rsi_score = SignalScorer._analyze_rsi(analysis.get('rsi', {}), bullish_signals, bearish_signals)
            confidence_factors.append(('rsi', rsi_score, 0.25))
            
            # MACD Analysis (25% weight)
            macd_score = SignalScorer._analyze_macd(analysis.get('macd', {}), bullish_signals, bearish_signals)
            confidence_factors.append(('macd', macd_score, 0.25))
            
            # Bollinger Bands Analysis (20% weight)
            bb_score = SignalScorer._analyze_bollinger_bands(
                analysis.get('bollinger', {}), analysis['current_price'], bullish_signals, bearish_signals
            )
            confidence_factors.append(('bollinger', bb_score, 0.20))
            
            # EMA Analysis (20% weight)
            ema_score = SignalScorer._analyze_ema(
                analysis.get('ema', {}), analysis['current_price'], bullish_signals, bearish_signals
            )
            confidence_factors.append(('ema', ema_score, 0.20))
            
            # Price Action Analysis (10% weight)
            pa_score = SignalScorer._analyze_price_action(
                analysis.get('price_action', {}), bullish_signals, bearish_signals
            )
            confidence_factors.append(('price_action', pa_score, 0.10))
            
            # Calculate weighted confidence
            bullish_confidence = sum(score * weight for name, score, weight in confidence_factors if score > 0)
            bearish_confidence = sum(abs(score) * weight for name, score, weight in confidence_factors if score < 0)
            
            # Apply timeframe multiplier for higher frequency trading
            timeframe_multiplier = SignalScorer._get_timeframe_multiplier(timeframe)
            bullish_confidence *= timeframe_multiplier
            bearish_confidence *= timeframe_multiplier
            
            # Generate final signal
            if bullish_confidence > Config.MIN_SIGNAL_CONFIDENCE and bullish_confidence > bearish_confidence:
                return {
                    'signal': 'buy',
                    'confidence': min(bullish_confidence, 1.0),
                    'reasons': bullish_signals,
                    'timeframe': timeframe
                }
            elif bearish_confidence > Config.MIN_SIGNAL_CONFIDENCE and bearish_confidence > bullish_confidence:
                return {
                    'signal': 'sell',
                    'confidence': min(bearish_confidence, 1.0),
                    'reasons': bearish_signals,
                    'timeframe': timeframe
                }
            else:
                return {
                    'signal': 'hold',
                    'confidence': max(bullish_confidence, bearish_confidence),
                    'reasons': bullish_signals + bearish_signals + ['Insufficient confluence'],
                    'timeframe': timeframe
                }
                
        except Exception as e:
            logger.error(f"Error scoring signal: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'reasons': ['Error in analysis']}
    
    @staticmethod
    def _analyze_rsi(rsi_data: Dict, bullish_signals: List, bearish_signals: List) -> float:
        """Analyze RSI indicators with enhanced sensitivity"""
        if not rsi_data:
            return 0.0
        
        score = 0.0
        
        for rsi_key, rsi_values in rsi_data.items():
            if not rsi_values or len(rsi_values) < 2:
                continue
            
            current_rsi = rsi_values[-1]
            previous_rsi = rsi_values[-2]
            
            # Enhanced RSI analysis - more aggressive thresholds
            if current_rsi < 35:  # Lowered from 30 for more frequent signals
                score += 0.8
                bullish_signals.append(f"RSI oversold: {current_rsi:.1f}")
            elif current_rsi < 45 and current_rsi > previous_rsi:
                score += 0.4
                bullish_signals.append(f"RSI recovering from oversold: {current_rsi:.1f}")
            elif current_rsi > 65:  # Lowered from 70
                score -= 0.8
                bearish_signals.append(f"RSI overbought: {current_rsi:.1f}")
            elif current_rsi > 55 and current_rsi < previous_rsi:
                score -= 0.4
                bearish_signals.append(f"RSI declining from overbought: {current_rsi:.1f}")
            
            # RSI divergence detection
            if len(rsi_values) >= 5:
                rsi_trend = rsi_values[-1] - rsi_values[-5]
                if rsi_trend > 8:
                    score += 0.3
                    bullish_signals.append("RSI showing strong upward momentum")
                elif rsi_trend < -8:
                    score -= 0.3
                    bearish_signals.append("RSI showing strong downward momentum")
        
        return score / len(rsi_data) if rsi_data else 0.0
    
    @staticmethod
    def _analyze_macd(macd_data: Dict, bullish_signals: List, bearish_signals: List) -> float:
        """Analyze MACD with enhanced signal detection"""
        if not macd_data or not all(k in macd_data for k in ['macd', 'signal', 'histogram']):
            return 0.0
        
        macd_line = macd_data['macd']
        signal_line = macd_data['signal']
        histogram = macd_data['histogram']
        
        if len(macd_line) < 3 or len(signal_line) < 3 or len(histogram) < 3:
            return 0.0
        
        score = 0.0
        
        # Current values
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        previous_histogram = histogram[-2]
        
        # MACD crossover analysis
        if current_macd > current_signal and previous_histogram < 0 < current_histogram:
            score += 1.0
            bullish_signals.append("MACD bullish crossover")
        elif current_macd < current_signal and previous_histogram > 0 > current_histogram:
            score -= 1.0
            bearish_signals.append("MACD bearish crossover")
        
        # MACD momentum analysis
        if current_histogram > previous_histogram and current_histogram > 0:
            score += 0.5
            bullish_signals.append("MACD histogram increasing (bullish momentum)")
        elif current_histogram < previous_histogram and current_histogram < 0:
            score -= 0.5
            bearish_signals.append("MACD histogram decreasing (bearish momentum)")
        
        # Zero line analysis
        if current_macd > 0 and current_signal > 0:
            score += 0.2
            bullish_signals.append("MACD above zero line")
        elif current_macd < 0 and current_signal < 0:
            score -= 0.2
            bearish_signals.append("MACD below zero line")
        
        return score
    
    @staticmethod
    def _analyze_bollinger_bands(bb_data: Dict, current_price: float, 
                               bullish_signals: List, bearish_signals: List) -> float:
        """Analyze Bollinger Bands with enhanced sensitivity"""
        if not bb_data or not all(k in bb_data for k in ['upper', 'middle', 'lower', 'width']):
            return 0.0
        
        upper_band = bb_data['upper']
        middle_band = bb_data['middle']
        lower_band = bb_data['lower']
        width = bb_data['width']
        
        if not upper_band or not lower_band or not middle_band:
            return 0.0
        
        current_upper = upper_band[-1]
        current_lower = lower_band[-1]
        current_middle = middle_band[-1]
        current_width = width[-1] if width else 0
        
        score = 0.0
        
        # Price position relative to bands - more aggressive
        band_position = (current_price - current_lower) / (current_upper - current_lower)
        
        if band_position < 0.15:  # Near lower band - more aggressive threshold
            score += 0.8
            bullish_signals.append(f"Price near lower BB ({band_position:.2f})")
        elif band_position < 0.3:
            score += 0.4
            bullish_signals.append(f"Price in lower BB zone ({band_position:.2f})")
        elif band_position > 0.85:  # Near upper band
            score -= 0.8
            bearish_signals.append(f"Price near upper BB ({band_position:.2f})")
        elif band_position > 0.7:
            score -= 0.4
            bearish_signals.append(f"Price in upper BB zone ({band_position:.2f})")
        
        # Bollinger Band squeeze/expansion
        if len(width) >= 20:
            avg_width = sum(width[-20:]) / 20
            if current_width < avg_width * 0.7:
                score += 0.3
                bullish_signals.append("BB squeeze detected - potential breakout")
            elif current_width > avg_width * 1.5:
                score += 0.2
                bullish_signals.append("BB expansion - trend confirmation")
        
        # Middle band as support/resistance
        if current_price > current_middle:
            score += 0.2
            bullish_signals.append("Price above BB middle line")
        else:
            score -= 0.2
            bearish_signals.append("Price below BB middle line")
        
        return score
    
    @staticmethod
    def _analyze_ema(ema_data: Dict, current_price: float,
                    bullish_signals: List, bearish_signals: List) -> float:
        """Analyze EMA with enhanced trend detection"""
        if not ema_data:
            return 0.0
        
        score = 0.0
        emas = {}
        
        # Get current EMA values
        for ema_key, ema_values in ema_data.items():
            if ema_values:
                period = int(ema_key.split('_')[1])
                emas[period] = ema_values[-1]
        
        if not emas:
            return 0.0
        
        # EMA alignment analysis (more aggressive)
        if 9 in emas and 21 in emas and 50 in emas:
            if emas[9] > emas[21] > emas[50]:
                score += 0.8
                bullish_signals.append("EMAs in bullish alignment")
            elif emas[9] < emas[21] < emas[50]:
                score -= 0.8
                bearish_signals.append("EMAs in bearish alignment")
        
        # Price vs EMA analysis
        for period, ema_value in emas.items():
            distance = (current_price - ema_value) / ema_value * 100
            
            if period == 9:  # Short-term EMA
                if distance > 0.5:  # More sensitive threshold
                    score += 0.5
                    bullish_signals.append(f"Price above EMA{period} by {distance:.1f}%")
                elif distance < -0.5:
                    score -= 0.5
                    bearish_signals.append(f"Price below EMA{period} by {abs(distance):.1f}%")
            
            elif period == 21:  # Medium-term EMA
                if distance > 1.0:
                    score += 0.3
                    bullish_signals.append(f"Price above EMA{period}")
                elif distance < -1.0:
                    score -= 0.3
                    bearish_signals.append(f"Price below EMA{period}")
        
        # EMA slope analysis
        for period, ema_values in ema_data.items():
            if len(ema_values) >= 5:
                period_num = int(period.split('_')[1])
                slope = (ema_values[-1] - ema_values[-5]) / ema_values[-5] * 100
                
                if slope > 0.5 and period_num <= 21:
                    score += 0.3
                    bullish_signals.append(f"EMA{period_num} trending up")
                elif slope < -0.5 and period_num <= 21:
                    score -= 0.3
                    bearish_signals.append(f"EMA{period_num} trending down")
        
        return score
    
    @staticmethod
    def _analyze_price_action(pa_data: Dict, bullish_signals: List, bearish_signals: List) -> float:
        """Analyze price action with enhanced pattern recognition"""
        if not pa_data:
            return 0.0
        
        score = 0.0
        
        trend = pa_data.get('trend', 'neutral')
        momentum = pa_data.get('momentum', 0)
        volatility = pa_data.get('volatility', 0)
        
        # Trend analysis - more sensitive
        if trend == 'bullish':
            score += 0.7
            bullish_signals.append("Bullish price action trend")
        elif trend == 'bearish':
            score -= 0.7
            bearish_signals.append("Bearish price action trend")
        
        # Momentum analysis - enhanced sensitivity
        if momentum > 1.0:  # Lowered threshold
            score += 0.5
            bullish_signals.append(f"Strong bullish momentum: {momentum:.1f}%")
        elif momentum > 0.3:
            score += 0.3
            bullish_signals.append(f"Moderate bullish momentum: {momentum:.1f}%")
        elif momentum < -1.0:
            score -= 0.5
            bearish_signals.append(f"Strong bearish momentum: {abs(momentum):.1f}%")
        elif momentum < -0.3:
            score -= 0.3
            bearish_signals.append(f"Moderate bearish momentum: {abs(momentum):.1f}%")
        
        # Volatility analysis
        if volatility > 2.0:
            score += 0.2  # High volatility can indicate breakout potential
            bullish_signals.append("High volatility - potential breakout")
        elif volatility < 0.5:
            score -= 0.1  # Low volatility might indicate consolidation
            bearish_signals.append("Low volatility - consolidation phase")
        
        return score
    
    @staticmethod
    def _get_timeframe_multiplier(timeframe: str) -> float:
        """Get confidence multiplier based on timeframe for higher frequency trading"""
        multipliers = {
            '1m': 1.2,   # Higher weight for short-term signals
            '5m': 1.1,   # Slightly increased
            '15m': 1.0,  # Base multiplier
            '1h': 0.9,   # Lower weight for longer timeframes in HF trading
            '4h': 0.8,
            '1d': 0.7
        }
        return multipliers.get(timeframe, 1.0)
