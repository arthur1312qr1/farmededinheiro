import logging
from typing import Dict, List, Tuple
from config import Config

logger = logging.getLogger(__name__)

class SignalScorer:
    """Advanced signal scoring system for trade decision making"""
    
    # Scoring weights for different indicators
    WEIGHTS = {
        'rsi': 0.20,
        'macd': 0.25,
        'bollinger': 0.15,
        'ema': 0.20,
        'support_resistance': 0.10,
        'volume': 0.05,
        'price_action': 0.05
    }
    
    @classmethod
    def score_signal(cls, analysis: Dict, timeframe: str) -> Dict:
        """
        Score trading signal based on technical analysis
        Returns: {'signal': 'buy'/'sell'/'hold', 'confidence': float, 'reasons': list}
        """
        if not analysis or not analysis.get('current_price'):
            return {'signal': 'hold', 'confidence': 0.0, 'reasons': ['Insufficient data']}
        
        try:
            scores = cls._calculate_individual_scores(analysis)
            
            # Calculate weighted final score
            final_score = sum(scores[indicator] * cls.WEIGHTS[indicator] 
                            for indicator in scores)
            
            # Apply timeframe multiplier
            timeframe_multiplier = cls._get_timeframe_multiplier(timeframe)
            final_score *= timeframe_multiplier
            
            # Determine signal
            signal_result = cls._determine_signal(final_score, scores, analysis)
            
            # Add confluence check
            confluence = cls._check_confluence(scores)
            signal_result['confluence'] = confluence
            signal_result['confidence'] *= confluence
            
            logger.info(f"Signal scored: {signal_result['signal']} with confidence {signal_result['confidence']:.2f}")
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error in signal scoring: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'reasons': ['Scoring error']}
    
    @classmethod
    def _calculate_individual_scores(cls, analysis: Dict) -> Dict:
        """Calculate individual indicator scores"""
        scores = {}
        
        # RSI Score
        scores['rsi'] = cls._score_rsi(analysis.get('rsi', {}))
        
        # MACD Score
        scores['macd'] = cls._score_macd(analysis.get('macd', {}))
        
        # Bollinger Bands Score
        scores['bollinger'] = cls._score_bollinger(analysis.get('bollinger', {}), 
                                                  analysis.get('current_price', 0))
        
        # EMA Score
        scores['ema'] = cls._score_ema(analysis.get('ema', {}), 
                                      analysis.get('current_price', 0))
        
        # Support/Resistance Score
        scores['support_resistance'] = cls._score_support_resistance(
            analysis.get('support_resistance', {}), 
            analysis.get('current_price', 0)
        )
        
        # Volume Score
        scores['volume'] = cls._score_volume(analysis.get('volume_profile', {}))
        
        # Price Action Score
        scores['price_action'] = cls._score_price_action(analysis.get('price_action', {}))
        
        return scores
    
    @staticmethod
    def _score_rsi(rsi_data: Dict) -> float:
        """Score RSI indicators (-1 to 1)"""
        if not rsi_data:
            return 0.0
        
        total_score = 0.0
        count = 0
        
        for rsi_key, rsi_values in rsi_data.items():
            if not rsi_values or len(rsi_values) == 0:
                continue
                
            current_rsi = rsi_values[-1]
            count += 1
            
            if current_rsi <= 30:
                # Oversold - bullish signal
                total_score += min(1.0, (30 - current_rsi) / 30)
            elif current_rsi >= 70:
                # Overbought - bearish signal
                total_score += max(-1.0, (70 - current_rsi) / 30)
            else:
                # Neutral zone
                if current_rsi < 45:
                    total_score += 0.3  # Slightly bullish
                elif current_rsi > 55:
                    total_score += -0.3  # Slightly bearish
        
        return total_score / count if count > 0 else 0.0
    
    @staticmethod
    def _score_macd(macd_data: Dict) -> float:
        """Score MACD indicator (-1 to 1)"""
        macd_line = macd_data.get('macd', [])
        signal_line = macd_data.get('signal', [])
        histogram = macd_data.get('histogram', [])
        
        if not all([macd_line, signal_line, histogram]) or len(macd_line) < 2:
            return 0.0
        
        score = 0.0
        
        # MACD line above/below signal line
        if macd_line[-1] > signal_line[-1]:
            score += 0.5
        else:
            score -= 0.5
        
        # MACD histogram trend
        if len(histogram) >= 2:
            if histogram[-1] > histogram[-2]:
                score += 0.3  # Increasing momentum
            else:
                score -= 0.3  # Decreasing momentum
        
        # MACD line trend
        if len(macd_line) >= 2:
            if macd_line[-1] > macd_line[-2]:
                score += 0.2
            else:
                score -= 0.2
        
        return max(-1.0, min(1.0, score))
    
    @staticmethod
    def _score_bollinger(bb_data: Dict, current_price: float) -> float:
        """Score Bollinger Bands (-1 to 1)"""
        upper = bb_data.get('upper', [])
        lower = bb_data.get('lower', [])
        middle = bb_data.get('middle', [])
        
        if not all([upper, lower, middle]) or current_price <= 0:
            return 0.0
        
        upper_val = upper[-1]
        lower_val = lower[-1]
        middle_val = middle[-1]
        
        # Calculate position within bands
        band_width = upper_val - lower_val
        if band_width == 0:
            return 0.0
        
        position = (current_price - lower_val) / band_width
        
        if position <= 0.1:
            # Near lower band - bullish
            return 0.8
        elif position >= 0.9:
            # Near upper band - bearish
            return -0.8
        elif position < 0.4:
            # Below middle - slightly bullish
            return 0.3
        elif position > 0.6:
            # Above middle - slightly bearish
            return -0.3
        else:
            # Around middle - neutral
            return 0.0
    
    @staticmethod
    def _score_ema(ema_data: Dict, current_price: float) -> float:
        """Score EMA alignment (-1 to 1)"""
        if not ema_data or current_price <= 0:
            return 0.0
        
        ema_values = {}
        for ema_key, ema_list in ema_data.items():
            if ema_list:
                period = int(ema_key.split('_')[1])
                ema_values[period] = ema_list[-1]
        
        if len(ema_values) < 2:
            return 0.0
        
        sorted_periods = sorted(ema_values.keys())
        score = 0.0
        
        # Check EMA alignment
        bullish_alignment = all(ema_values[sorted_periods[i]] > ema_values[sorted_periods[i+1]] 
                              for i in range(len(sorted_periods)-1))
        bearish_alignment = all(ema_values[sorted_periods[i]] < ema_values[sorted_periods[i+1]] 
                               for i in range(len(sorted_periods)-1))
        
        if bullish_alignment:
            score += 0.6
        elif bearish_alignment:
            score -= 0.6
        
        # Price vs fastest EMA
        if sorted_periods:
            fastest_ema = ema_values[sorted_periods[0]]
            if current_price > fastest_ema:
                score += 0.4
            else:
                score -= 0.4
        
        return max(-1.0, min(1.0, score))
    
    @staticmethod
    def _score_support_resistance(sr_data: Dict, current_price: float) -> float:
        """Score support/resistance levels (-1 to 1)"""
        support = sr_data.get('support', 0)
        resistance = sr_data.get('resistance', 0)
        
        if not all([support, resistance, current_price]) or support >= resistance:
            return 0.0
        
        # Distance from support/resistance
        support_distance = (current_price - support) / support
        resistance_distance = (resistance - current_price) / current_price
        
        if support_distance < 0.01:  # Very close to support
            return 0.7
        elif resistance_distance < 0.01:  # Very close to resistance
            return -0.7
        elif support_distance < 0.02:  # Near support
            return 0.4
        elif resistance_distance < 0.02:  # Near resistance
            return -0.4
        else:
            return 0.0
    
    @staticmethod
    def _score_volume(volume_data: Dict) -> float:
        """Score volume profile (-1 to 1)"""
        # Simplified volume scoring
        # In a real implementation, you'd analyze volume trends
        return 0.0
    
    @staticmethod
    def _score_price_action(pa_data: Dict) -> float:
        """Score price action patterns (-1 to 1)"""
        trend = pa_data.get('trend', 'neutral')
        momentum = pa_data.get('momentum', 0)
        
        score = 0.0
        
        if trend == 'bullish':
            score += 0.5
        elif trend == 'bearish':
            score -= 0.5
        
        # Add momentum component
        score += max(-0.5, min(0.5, momentum / 5))
        
        return score
    
    @staticmethod
    def _get_timeframe_multiplier(timeframe: str) -> float:
        """Get multiplier based on timeframe importance"""
        multipliers = {
            '1m': 0.8,   # Lower weight for noise
            '5m': 1.0,   # Base weight
            '15m': 1.2,  # Higher weight
            '1h': 1.4    # Highest weight
        }
        return multipliers.get(timeframe, 1.0)
    
    @staticmethod
    def _determine_signal(final_score: float, scores: Dict, analysis: Dict) -> Dict:
        """Determine final signal based on score"""
        confidence = abs(final_score)
        reasons = []
        
        # Generate reasons based on scores
        for indicator, score in scores.items():
            if abs(score) > 0.3:
                direction = "bullish" if score > 0 else "bearish"
                reasons.append(f"{indicator}: {direction} ({score:.2f})")
        
        if final_score > Config.MIN_SIGNAL_CONFIDENCE:
            signal = 'buy'
        elif final_score < -Config.MIN_SIGNAL_CONFIDENCE:
            signal = 'sell'
        else:
            signal = 'hold'
            reasons.append(f"Low confidence: {confidence:.2f}")
        
        return {
            'signal': signal,
            'confidence': min(1.0, confidence),
            'reasons': reasons,
            'score': final_score
        }
    
    @staticmethod
    def _check_confluence(scores: Dict) -> float:
        """Check confluence between indicators (0 to 1)"""
        positive_scores = sum(1 for score in scores.values() if score > 0.2)
        negative_scores = sum(1 for score in scores.values() if score < -0.2)
        total_scores = len(scores)
        
        if positive_scores > total_scores * 0.6:
            return min(1.0, positive_scores / total_scores + 0.2)
        elif negative_scores > total_scores * 0.6:
            return min(1.0, negative_scores / total_scores + 0.2)
        else:
            return 0.5  # Mixed signals
