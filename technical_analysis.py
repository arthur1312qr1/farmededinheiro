import numpy as np
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ScalpingAnalysis:
    def __init__(self):
        self.lookback_period = 20  # Análise rápida
        
    async def quick_analysis(self, prices: List[float], volumes: List[float], market_data: Dict) -> Dict:
        """Análise técnica ultra-rápida para scalping"""
        try:
            if len(prices) < 14:
                return {}
            
            # Converter para arrays numpy
            price_array = np.array(prices)
            volume_array = np.array(volumes)
            
            # Análises rápidas
            signals = {
                'rsi': self.fast_rsi(price_array, period=7),  # RSI rápido
                'ema_direction': self.ema_direction(price_array),
                'ema_trend': self.ema_trend_strength(price_array),
                'macd_signal': self.fast_macd_signal(price_array),
                'bb_position': self.bollinger_position(price_array),
                'volume_spike': self.volume_spike(volume_array),
                'momentum': self.price_momentum(price_array),
                'support_resistance': self.quick_sr_levels(price_array)
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Erro na análise técnica: {e}")
            return {}
    
    def fast_rsi(self, prices: np.array, period: int = 7) -> float:
        """RSI rápido para scalping"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def ema_direction(self, prices: np.array) -> str:
        """Direção da EMA"""
        if len(prices) < 9:
            return 'neutral'
        
        # EMA 9
        ema9 = self.calculate_ema(prices, 9)
        current_ema = ema9[-1]
        prev_ema = ema9[-2]
        
        if current_ema > prev_ema:
            return 'up'
        elif current_ema < prev_ema:
            return 'down'
        else:
            return 'neutral'
    
    def ema_trend_strength(self, prices: np.array) -> str:
        """Força da tendência EMA"""
        if len(prices) < 21:
            return 'weak'
        
        ema9 = self.calculate_ema(prices, 9)[-1]
        ema21 = self.calculate_ema(prices, 21)[-1]
        current_price = prices[-1]
        
        # Alinhamento das EMAs
        if current_price > ema9 > ema21:
            return 'strong'  # Tendência de alta forte
        elif current_price < ema9 < ema21:
            return 'strong'  # Tendência de baixa forte
        else:
            return 'weak'
    
    def fast_macd_signal(self, prices: np.array) -> str:
        """MACD rápido"""
        if len(prices) < 26:
            return 'neutral'
        
        # MACD (12, 26, 9)
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        
        macd_line = ema12 - ema26
        signal_line = self.calculate_ema(macd_line, 9)
        
        # Últimos 2 valores
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        
        # Cruzamentos
        if current_macd > current_signal and prev_macd <= prev_signal:
            return 'bullish'
        elif current_macd < current_signal and prev_macd >= prev_signal:
            return 'bearish'
        else:
            return 'neutral'
    
    def bollinger_position(self, prices: np.array, period: int = 20) -> str:
        """Posição nas Bandas de Bollinger"""
        if len(prices) < period:
            return 'middle'
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        current_price = prices[-1]
        
        if current_price <= lower_band:
            return 'lower'
        elif current_price >= upper_band:
            return 'upper'
        else:
            return 'middle'
    
    def volume_spike(self, volumes: np.array) -> bool:
        """Detecta pico de volume"""
        if len(volumes) < 10:
            return False
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:-1])
        
        return current_volume > (avg_volume * 1.5)  # 50% acima da média
    
    def price_momentum(self, prices: np.array) -> str:
        """Momentum do preço"""
        if len(prices) < 5:
            return 'neutral'
        
        # Comparar últimos 3 preços
        recent_change = (prices[-1] - prices[-4]) / prices[-4]
        
        if recent_change > 0.002:  # 0.2% para cima
            return 'up'
        elif recent_change < -0.002:  # 0.2% para baixo
            return 'down'
        else:
            return 'neutral'
    
    def quick_sr_levels(self, prices: np.array) -> Dict:
        """Níveis rápidos de suporte e resistência"""
        if len(prices) < 20:
            return {'support': 0, 'resistance': 0}
        
        recent_prices = prices[-20:]
        support = np.min(recent_prices)
        resistance = np.max(recent_prices)
        
        return {
            'support': support,
            'resistance': resistance,
            'current': prices[-1]
        }
    
    def calculate_ema(self, prices: np.array, period: int) -> np.array:
        """Calcula EMA"""
        if len(prices) < period:
            return prices
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
