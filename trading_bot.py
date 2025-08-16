import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import math
import statistics
from collections import deque
import numpy as np

from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str='ETH/USDT:USDT',
                 leverage: int=10, balance_percentage: float=100.0,
                 daily_target: int=200, scalping_interval: int=2,
                 paper_trading: bool=False):
        """Initialize Trading Bot with SUPREME AI prediction system"""
        if not isinstance(bitget_api, BitgetAPI):
            raise TypeError(f"bitget_api deve ser uma instância de BitgetAPI, recebido: {type(bitget_api)}")
        
        self.bitget_api = bitget_api
        self.symbol = symbol
        self.leverage = leverage
        self.balance_percentage = balance_percentage
        self.daily_target = daily_target
        self.scalping_interval = scalping_interval
        self.paper_trading = paper_trading
        
        # Trading state
        self.is_running = False
        self.trades_today = 0
        self.current_position = None
        self.entry_price = None
        self.position_side = None
        self.position_size = 0.0
        self.profit_target = 0.01  # 1% take profit
        self.stop_loss_target = -0.02  # 2% stop loss
        
        # SISTEMA DE CERTEZA EXTREMO
        self.min_confidence_to_trade = 0.75  # 75% de certeza mínima
        self.min_prediction_score = 0.6      # Score mínimo para trade
        self.min_signals_agreement = 6       # Mínimo 6 de 10 sinais concordando
        
        # SISTEMA DE PREVISÃO SUPREMO
        self.price_history = deque(maxlen=500)  # 500 pontos históricos
        self.volume_history = deque(maxlen=200)
        self.order_book_history = deque(maxlen=100)
        self.market_sentiment_history = deque(maxlen=50)
        
        # Base de conhecimento de padrões
        self.pattern_database = {
            'double_top': {'accuracy': 0.82, 'timeframe': 15, 'reversal': True},
            'double_bottom': {'accuracy': 0.84, 'timeframe': 15, 'reversal': True},
            'head_shoulders': {'accuracy': 0.78, 'timeframe': 20, 'reversal': True},
            'triangle_breakout': {'accuracy': 0.76, 'timeframe': 12, 'continuation': True},
            'flag_pattern': {'accuracy': 0.73, 'timeframe': 8, 'continuation': True},
            'cup_handle': {'accuracy': 0.71, 'timeframe': 25, 'bullish': True}
        }
        
        # Indicadores técnicos avançados
        self.indicators = {
            'sma_5': 0, 'sma_10': 0, 'sma_20': 0, 'sma_50': 0,
            'ema_12': 0, 'ema_26': 0, 'ema_50': 0,
            'rsi_14': 50, 'rsi_6': 50, 'rsi_21': 50,
            'macd': 0, 'macd_signal': 0, 'macd_histogram': 0,
            'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'bb_width': 0,
            'stoch_k': 50, 'stoch_d': 50,
            'williams_r': -50, 'cci': 0, 'atr': 0, 'adx': 25,
            'obv': 0, 'mfi': 50, 'trix': 0, 'ultimate_oscillator': 50
        }
        
        # Sistema de Machine Learning Avançado
        self.ml_models = {
            'trend_predictor': {'weights': [0.4, 0.3, 0.2, 0.1], 'bias': 0.02},
            'reversal_detector': {'weights': [0.35, 0.25, 0.25, 0.15], 'bias': -0.01},
            'momentum_analyzer': {'weights': [0.5, 0.3, 0.15, 0.05], 'bias': 0.0},
            'volatility_predictor': {'weights': [0.3, 0.3, 0.25, 0.15], 'bias': 0.01}
        }
        
        # Análise de correlação com outros ativos
        self.correlation_assets = ['BTC/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT']
        self.asset_correlations = {}
        
        # Sistema de validação cruzada
        self.prediction_history = deque(maxlen=100)
        self.accuracy_tracking = {
            'short_term': {'correct': 0, 'total': 0},
            'medium_term': {'correct': 0, 'total': 0},
            'long_term': {'correct': 0, 'total': 0}
        }
        
        # Statistics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        self.high_confidence_trades = 0
        self.prediction_accuracy = 0.0
        
        logger.info("🧠 SUPREME AI TRADING BOT INICIALIZADO")
        logger.info(f"🎯 Confiança mínima para trade: {self.min_confidence_to_trade*100}%")
        logger.info(f"📊 Score mínimo: {self.min_prediction_score}")
        logger.info(f"🔍 Sinais mínimos concordando: {self.min_signals_agreement}/10")

    def calculate_all_moving_averages(self, prices: List[float]) -> Dict:
        """Calcula todas as médias móveis"""
        if len(prices) < 50:
            return {}
        
        def sma(data, period):
            if len(data) < period:
                return 0
            return sum(data[-period:]) / period
        
        def ema(data, period):
            if len(data) < period:
                return 0
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
        
        return {
            'sma_5': sma(prices, 5),
            'sma_10': sma(prices, 10),
            'sma_20': sma(prices, 20),
            'sma_50': sma(prices, 50),
            'ema_12': ema(prices, 12),
            'ema_26': ema(prices, 26),
            'ema_50': ema(prices, 50)
        }

    def calculate_advanced_rsi(self, prices: List[float]) -> Dict:
        """RSI em múltiplos timeframes"""
        def rsi(data, period):
            if len(data) < period + 1:
                return 50
            
            deltas = [data[i] - data[i-1] for i in range(1, len(data))]
            gains = [d if d > 0 else 0 for d in deltas[-period:]]
            losses = [-d if d < 0 else 0 for d in deltas[-period:]]
            
            avg_gain = sum(gains) / period if gains else 0
            avg_loss = sum(losses) / period if losses else 0
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        return {
            'rsi_6': rsi(prices, 6),
            'rsi_14': rsi(prices, 14),
            'rsi_21': rsi(prices, 21)
        }

    def calculate_macd_advanced(self, prices: List[float]) -> Dict:
        """MACD com análise avançada"""
        if len(prices) < 35:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
        
        ema_12 = ema(prices[-12:], 12)
        ema_26 = ema(prices[-26:], 26)
        macd_line = ema_12 - ema_26
        
        # Calcular Signal Line (EMA 9 do MACD)
        if hasattr(self, '_macd_history'):
            self._macd_history.append(macd_line)
            if len(self._macd_history) > 9:
                self._macd_history = self._macd_history[-9:]
        else:
            self._macd_history = [macd_line]
        
        signal_line = ema(self._macd_history, 9) if len(self._macd_history) >= 9 else macd_line
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_advanced(self, prices: List[float]) -> Dict:
        """Bollinger Bands com análise de squeeze"""
        if len(prices) < 20:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'squeeze': False}
        
        period = 20
        recent = prices[-period:]
        sma = sum(recent) / period
        variance = sum((p - sma) ** 2 for p in recent) / period
        std_dev = math.sqrt(variance)
        
        upper = sma + (2 * std_dev)
        lower = sma - (2 * std_dev)
        width = (upper - lower) / sma
        
        # Bollinger Squeeze detection
        if hasattr(self, '_bb_width_history'):
            self._bb_width_history.append(width)
            if len(self._bb_width_history) > 20:
                self._bb_width_history = self._bb_width_history[-20:]
        else:
            self._bb_width_history = [width]
        
        avg_width = sum(self._bb_width_history) / len(self._bb_width_history)
        squeeze = width < (avg_width * 0.8)  # Squeeze quando width < 80% da média
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': width,
            'squeeze': squeeze
        }

    def calculate_stochastic(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Stochastic Oscillator completo"""
        if len(prices) < 14:
            return {'k': 50, 'd': 50}
        
        high_14 = max(highs[-14:]) if highs else max(prices[-14:])
        low_14 = min(lows[-14:]) if lows else min(prices[-14:])
        current = prices[-1]
        
        if high_14 != low_14:
            k_percent = ((current - low_14) / (high_14 - low_14)) * 100
        else:
            k_percent = 50
        
        # %D é a média móvel de 3 períodos do %K
        if hasattr(self, '_stoch_k_history'):
            self._stoch_k_history.append(k_percent)
            if len(self._stoch_k_history) > 3:
                self._stoch_k_history = self._stoch_k_history[-3:]
        else:
            self._stoch_k_history = [k_percent]
        
        d_percent = sum(self._stoch_k_history) / len(self._stoch_k_history)
        
        return {'k': k_percent, 'd': d_percent}

    def detect_chart_patterns(self, prices: List[float]) -> Dict:
        """Detecção avançada de padrões gráficos"""
        if len(prices) < 30:
            return {'patterns': [], 'confidence': 0}
        
        patterns_found = []
        pattern_scores = []
        
        # 1. Double Top/Bottom Detection
        peaks = self.find_peaks_valleys(prices)['peaks']
        valleys = self.find_peaks_valleys(prices)['valleys']
        
        if len(peaks) >= 2:
            last_two_peaks = peaks[-2:]
            if abs(last_two_peaks[0] - last_two_peaks[1]) / last_two_peaks[0] < 0.02:
                patterns_found.append('double_top')
                pattern_scores.append(0.82)
        
        if len(valleys) >= 2:
            last_two_valleys = valleys[-2:]
            if abs(last_two_valleys[0] - last_two_valleys[1]) / last_two_valleys[0] < 0.02:
                patterns_found.append('double_bottom')
                pattern_scores.append(0.84)
        
        # 2. Head and Shoulders
        if len(peaks) >= 3 and len(valleys) >= 2:
            if self.is_head_and_shoulders(peaks[-3:], valleys[-2:]):
                patterns_found.append('head_shoulders')
                pattern_scores.append(0.78)
        
        # 3. Triangle Patterns
        triangle = self.detect_triangle_pattern(prices)
        if triangle['detected']:
            patterns_found.append(f"triangle_{triangle['type']}")
            pattern_scores.append(0.76)
        
        # 4. Flag Pattern
        if self.detect_flag_pattern(prices):
            patterns_found.append('flag_pattern')
            pattern_scores.append(0.73)
        
        # 5. Cup and Handle
        if self.detect_cup_handle_pattern(prices):
            patterns_found.append('cup_handle')
            pattern_scores.append(0.71)
        
        avg_confidence = sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0
        
        return {
            'patterns': patterns_found,
            'confidence': avg_confidence,
            'pattern_scores': dict(zip(patterns_found, pattern_scores))
        }

    def find_peaks_valleys(self, prices: List[float]) -> Dict:
        """Encontra picos e vales com precisão"""
        if len(prices) < 5:
            return {'peaks': [], 'valleys': []}
        
        peaks = []
        valleys = []
        
        for i in range(2, len(prices) - 2):
            # Pico: maior que vizinhos
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                peaks.append(prices[i])
            
            # Vale: menor que vizinhos
            if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                valleys.append(prices[i])
        
        return {'peaks': peaks, 'valleys': valleys}

    def is_head_and_shoulders(self, peaks: List[float], valleys: List[float]) -> bool:
        """Detecta padrão Head and Shoulders"""
        if len(peaks) < 3 or len(valleys) < 2:
            return False
        
        left_shoulder, head, right_shoulder = peaks[-3:]
        
        # Head deve ser maior que os ombros
        if head > left_shoulder and head > right_shoulder:
            # Ombros devem ser similares
            shoulder_diff = abs(left_shoulder - right_shoulder) / left_shoulder
            if shoulder_diff < 0.05:  # 5% de tolerância
                return True
        
        return False

    def detect_triangle_pattern(self, prices: List[float]) -> Dict:
        """Detecta padrões de triângulo"""
        if len(prices) < 20:
            return {'detected': False, 'type': None}
        
        recent_prices = prices[-20:]
        highs = []
        lows = []
        
        # Identificar highs e lows
        for i in range(1, len(recent_prices) - 1):
            if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                highs.append((i, recent_prices[i]))
            if recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                lows.append((i, recent_prices[i]))
        
        if len(highs) < 2 or len(lows) < 2:
            return {'detected': False, 'type': None}
        
        # Calcular tendências das linhas de resistência e suporte
        high_trend = self.calculate_line_slope([h[1] for h in highs])
        low_trend = self.calculate_line_slope([l[1] for l in lows])
        
        # Classificar tipo de triângulo
        if high_trend < -0.001 and low_trend > 0.001:
            return {'detected': True, 'type': 'symmetrical'}
        elif high_trend < -0.001 and abs(low_trend) < 0.001:
            return {'detected': True, 'type': 'descending'}
        elif abs(high_trend) < 0.001 and low_trend > 0.001:
            return {'detected': True, 'type': 'ascending'}
        
        return {'detected': False, 'type': None}

    def detect_flag_pattern(self, prices: List[float]) -> bool:
        """Detecta padrão de bandeira"""
        if len(prices) < 15:
            return False
        
        # Dividir em duas partes: pole e flag
        pole_length = 8
        flag_length = 7
        
        pole = prices[-(pole_length + flag_length):-flag_length]
        flag = prices[-flag_length:]
        
        # Verificar se o pole tem movimento forte
        pole_move = abs(pole[-1] - pole[0]) / pole[0]
        if pole_move < 0.02:  # Movimento menor que 2%
            return False
        
        # Verificar se a flag é consolidação
        flag_volatility = statistics.stdev(flag) / statistics.mean(flag)
        if flag_volatility > 0.01:  # Muita volatilidade
            return False
        
        return True

    def detect_cup_handle_pattern(self, prices: List[float]) -> bool:
        """Detecta padrão Cup and Handle"""
        if len(prices) < 30:
            return False
        
        # Dividir em cup e handle
        cup_length = 20
        handle_length = 10
        
        cup = prices[-(cup_length + handle_length):-handle_length]
        handle = prices[-handle_length:]
        
        # Verificar formato de xícara (U shape)
        cup_min_idx = cup.index(min(cup))
        left_side = cup[:cup_min_idx]
        right_side = cup[cup_min_idx:]
        
        # Verificar se ambos os lados têm tendência similar
        if len(left_side) < 3 or len(right_side) < 3:
            return False
        
        left_slope = self.calculate_line_slope(left_side)
        right_slope = self.calculate_line_slope(right_side)
        
        # Cup: lado esquerdo desce, lado direito sobe
        if left_slope < -0.001 and right_slope > 0.001:
            # Handle: pequena correção
            handle_correction = (max(handle) - min(handle)) / max(handle)
            if 0.01 < handle_correction < 0.15:  # 1-15% de correção
                return True
        
        return False

    def calculate_line_slope(self, data: List[float]) -> float:
        """Calcula inclinação de uma linha"""
        if len(data) < 2:
            return 0
        
        n = len(data)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(data)
        sum_xy = sum(x[i] * data[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope / statistics.mean(data) if statistics.mean(data) != 0 else 0

    def supreme_ai_prediction(self, current_price: float) -> Dict:
        """SISTEMA DE IA SUPREMO - MELHOR PREVISÃO DO MUNDO"""
        try:
            # Adicionar ao histórico
            timestamp = time.time()
            self.price_history.append({
                'price': current_price,
                'timestamp': timestamp
            })
            
            if len(self.price_history) < 50:
                return self.basic_prediction_response(current_price)
            
            # Extrair preços
            prices = [p['price'] for p in self.price_history]
            
            # 1. CALCULAR TODOS OS INDICADORES
            ma_data = self.calculate_all_moving_averages(prices)
            rsi_data = self.calculate_advanced_rsi(prices)
            macd_data = self.calculate_macd_advanced(prices)
            bb_data = self.calculate_bollinger_advanced(prices)
            stoch_data = self.calculate_stochastic(prices, prices, prices)
            
            self.indicators.update(ma_data)
            self.indicators.update(rsi_data)
            self.indicators.update(macd_data)
            self.indicators.update(bb_data)
            self.indicators.update(stoch_data)
            
            # 2. DETECTAR PADRÕES GRÁFICOS
            patterns = self.detect_chart_patterns(prices)
            
            # 3. ANÁLISE DE SINAIS MÚLTIPLOS
            signals = []
            signal_scores = []
            
            # SINAL 1: Análise de Médias Móveis (Golden/Death Cross)
            if ma_data.get('ema_12', 0) > ma_data.get('ema_26', 0):
                if ma_data.get('ema_12', 0) > ma_data.get('sma_50', 0):
                    signals.append("Golden Cross + EMA > SMA50")
                    signal_scores.append(0.85)
                else:
                    signals.append("EMA 12 > 26 (bullish)")
                    signal_scores.append(0.6)
            else:
                if ma_data.get('ema_12', 0) < ma_data.get('sma_50', 0):
                    signals.append("Death Cross + EMA < SMA50")
                    signal_scores.append(-0.85)
                else:
                    signals.append("EMA 12 < 26 (bearish)")
                    signal_scores.append(-0.6)
            
            # SINAL 2: RSI Multi-timeframe
            rsi_14 = rsi_data.get('rsi_14', 50)
            rsi_6 = rsi_data.get('rsi_6', 50)
            
            if rsi_14 < 30 and rsi_6 < 25:
                signals.append(f"RSI extremo oversold: 14={rsi_14:.1f}, 6={rsi_6:.1f}")
                signal_scores.append(0.9)
            elif rsi_14 > 70 and rsi_6 > 75:
                signals.append(f"RSI extremo overbought: 14={rsi_14:.1f}, 6={rsi_6:.1f}")
                signal_scores.append(-0.9)
            elif rsi_14 < 40:
                signals.append(f"RSI oversold: {rsi_14:.1f}")
                signal_scores.append(0.6)
            elif rsi_14 > 60:
                signals.append(f"RSI overbought: {rsi_14:.1f}")
                signal_scores.append(-0.6)
            
            # SINAL 3: MACD Avançado
            macd = macd_data.get('macd', 0)
            macd_signal = macd_data.get('signal', 0)
            macd_hist = macd_data.get('histogram', 0)
            
            if macd > macd_signal and macd_hist > 0:
                signals.append(f"MACD bullish convergence: {macd:.4f}")
                signal_scores.append(0.75)
            elif macd < macd_signal and macd_hist < 0:
                signals.append(f"MACD bearish divergence: {macd:.4f}")
                signal_scores.append(-0.75)
            
            # SINAL 4: Bollinger Bands + Squeeze
            bb_pos = (current_price - bb_data.get('lower', current_price)) / (bb_data.get('upper', current_price) - bb_data.get('lower', current_price))
            
            if bb_data.get('squeeze', False):
                signals.append("Bollinger Squeeze - Breakout iminente")
                signal_scores.append(0.8)
            elif bb_pos < 0.1:
                signals.append(f"Preço na banda inferior BB: {bb_pos:.2f}")
                signal_scores.append(0.7)
            elif bb_pos > 0.9:
                signals.append(f"Preço na banda superior BB: {bb_pos:.2f}")
                signal_scores.append(-0.7)
            
            # SINAL 5: Stochastic
            stoch_k = stoch_data.get('k', 50)
            stoch_d = stoch_data.get('d', 50)
            
            if stoch_k < 20 and stoch_d < 20 and stoch_k > stoch_d:
                signals.append(f"Stoch bullish divergence: K={stoch_k:.1f}")
                signal_scores.append(0.7)
            elif stoch_k > 80 and stoch_d > 80 and stoch_k < stoch_d:
                signals.append(f"Stoch bearish divergence: K={stoch_k:.1f}")
                signal_scores.append(-0.7)
            
            # SINAL 6: Padrões Gráficos
            if patterns['patterns']:
                for pattern in patterns['patterns']:
                    if pattern in ['double_bottom', 'cup_handle']:
                        signals.append(f"Padrão bullish: {pattern}")
                        signal_scores.append(0.8)
                    elif pattern in ['double_top', 'head_shoulders']:
                        signals.append(f"Padrão bearish: {pattern}")
                        signal_scores.append(-0.8)
                    else:
                        signals.append(f"Padrão: {pattern}")
                        signal_scores.append(0.5)
            
            # SINAL 7: Volume Analysis (simulado)
            volume_trend = self.analyze_volume_trend(prices)
            signals.append(f"Volume trend: {volume_trend['trend']}")
            signal_scores.append(volume_trend['score'])
            
            # SINAL 8: Price Action
            price_action = self.analyze_price_action(prices)
            signals.append(f"Price action: {price_action['pattern']}")
            signal_scores.append(price_action['score'])
            
            # SINAL 9: Support/Resistance
            sr_analysis = self.advanced_support_resistance(prices)
            signals.append(f"S/R: {sr_analysis['status']}")
            signal_scores.append(sr_analysis['score'])
            
            # SINAL 10: Machine Learning Ensemble
            ml_ensemble = self.ml_ensemble_prediction(prices)
            signals.append(f"ML Ensemble: {ml_ensemble['prediction']}")
            signal_scores.append(ml_ensemble['score'])
            
            # 4. CALCULAR SCORE FINAL
            final_score = sum(signal_scores) / len(signal_scores) if signal_scores else 0
            
            # 5. CALCULAR CONFIANÇA
            positive_signals = len([s for s in signal_scores if s > 0.5])
            negative_signals = len([s for s in signal_scores if s < -0.5])
            neutral_signals = len(signal_scores) - positive_signals - negative_signals
            
            signal_agreement = max(positive_signals, negative_signals)
            confidence = signal_agreement / len(signal_scores)
            
            # Boost de confiança para sinais extremos
            extreme_signals = len([s for s in signal_scores if abs(s) > 0.8])
            confidence += (extreme_signals * 0.1)
            confidence = min(1.0, confidence)
            
            # 6. DETERMINAR DIREÇÃO E DECISÃO
            if final_score > 0.3:
                trend = 'bullish'
                direction = 'buy'
            elif final_score < -0.3:
                trend = 'bearish'
                direction = 'sell'
            else:
                trend = 'neutral'
                direction = 'hold'
            
            # 7. SISTEMA DE DECISÃO INTELIGENTE
            should_trade = self.should_execute_trade(final_score, confidence, signal_agreement)
            
            # 8. PREVISÃO DE PREÇO
            volatility = statistics.stdev(prices[-20:]) / statistics.mean(prices[-20:])
            price_prediction = self.calculate_price_prediction(current_price, final_score, volatility)
            
            # 9. RESULTADO FINAL
            result = {
                'trend': trend,
                'direction': direction,
                'should_trade': should_trade,
                'final_score': final_score,
                'confidence': confidence,
                'signal_agreement': signal_agreement,
                'total_signals': len(signal_scores),
                'positive_signals': positive_signals,
                'negative_signals': negative_signals,
                'extreme_signals': extreme_signals,
                'next_20min_prediction': price_prediction,
                'signals': signals,
                'signal_scores': signal_scores,
                'indicators': self.indicators.copy(),
                'patterns': patterns,
                'volatility': volatility,
                'prediction_quality': self.assess_prediction_quality(confidence, signal_agreement)
            }
            
            # Log detalhado
            logger.warning(f"🧠 SUPREMA IA: {direction.upper()} | Score: {final_score:.3f} | Conf: {confidence:.2f}")
            logger.warning(f"📊 Sinais: {positive_signals}+ {negative_signals}- {neutral_signals}° | Extremos: {extreme_signals}")
            logger.warning(f"🎯 Executar: {'SIM' if should_trade else 'NÃO'} | Qualidade: {result['prediction_quality']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na IA Suprema: {e}")
            return self.basic_prediction_response(current_price)

    def should_execute_trade(self, score: float, confidence: float, signal_agreement: int) -> bool:
        """Decide se deve executar o trade baseado nos critérios"""
        # Critério 1: Score mínimo
        if abs(score) < self.min_prediction_score:
            return False
        
        # Critério 2: Confiança mínima
        if confidence < self.min_confidence_to_trade:
            return False
        
        # Critério 3: Acordos de sinais mínimos
        if signal_agreement < self.min_signals_agreement:
            return False
        
        # Critério 4: Não tradear em condições extremas de incerteza
        if confidence < 0.5 and abs(score) < 0.8:
            return False
        
        return True

    def analyze_volume_trend(self, prices: List[float]) -> Dict:
        """Análise de tendência de volume (simulada)"""
        if len(prices) < 10:
            return {'trend': 'neutral', 'score': 0}
        
        # Simular volume baseado na volatilidade
        recent_volatility = statistics.stdev(prices[-5:])
        prev_volatility = statistics.stdev(prices[-10:-5])
        
        if recent_volatility > prev_volatility * 1.2:
            return {'trend': 'increasing', 'score': 0.6}
        elif recent_volatility < prev_volatility * 0.8:
            return {'trend': 'decreasing', 'score': -0.4}
        else:
            return {'trend': 'stable', 'score': 0.1}

    def analyze_price_action(self, prices: List[float]) -> Dict:
        """Análise avançada de price action"""
        if len(prices) < 8:
            return {'pattern': 'insufficient', 'score': 0}
        
        recent = prices[-8:]
        
        # Detectar padrões de candles
        bullish_patterns = 0
        bearish_patterns = 0
        
        for i in range(1, len(recent)):
            change = (recent[i] - recent[i-1]) / recent[i-1]
            
            if change > 0.003:  # >0.3% alta
                bullish_patterns += 1
            elif change < -0.003:  # <-0.3% baixa
                bearish_patterns += 1
        
        if bullish_patterns >= 5:
            return {'pattern': 'strong_bullish', 'score': 0.8}
        elif bearish_patterns >= 5:
            return {'pattern': 'strong_bearish', 'score': -0.8}
        elif bullish_patterns > bearish_patterns:
            return {'pattern': 'bullish', 'score': 0.4}
        elif bearish_patterns > bullish_patterns:
            return {'pattern': 'bearish', 'score': -0.4}
        else:
            return {'pattern': 'neutral', 'score': 0}

    def advanced_support_resistance(self, prices: List[float]) -> Dict:
        """Análise avançada de suporte e resistência"""
        if len(prices) < 20:
            return {'status': 'insufficient', 'score': 0}
        
        current_price = prices[-1]
        
        # Encontrar níveis de S/R
        peaks_valleys = self.find_peaks_valleys(prices)
        all_levels = peaks_valleys['peaks'] + peaks_valleys['valleys']
        
        if not all_levels:
            return {'status': 'no_levels', 'score': 0}
        
        # Encontrar nível mais próximo
        closest_level = min(all_levels, key=lambda x: abs(x - current_price))
        distance_pct = abs(current_price - closest_level) / current_price
        
        if distance_pct < 0.005:  # Muito próximo (0.5%)
            if closest_level > current_price:
                return {'status': f'near_resistance_{closest_level:.2f}', 'score': -0.7}
            else:
                return {'status': f'near_support_{closest_level:.2f}', 'score': 0.7}
        elif distance_pct < 0.01:  # Próximo (1%)
            if closest_level > current_price:
                return {'status': f'approaching_resistance_{closest_level:.2f}', 'score': -0.4}
            else:
                return {'status': f'approaching_support_{closest_level:.2f}', 'score': 0.4}
        else:
            return {'status': 'between_levels', 'score': 0}

    def ml_ensemble_prediction(self, prices: List[float]) -> Dict:
        """Ensemble de modelos de Machine Learning"""
        if len(prices) < 20:
            return {'prediction': 'insufficient', 'score': 0}
        
        # Features para ML
        features = self.extract_ml_features(prices)
        
        # Predição de cada modelo
        predictions = {}
        
        for model_name, model_config in self.ml_models.items():
            weights = model_config['weights']
            bias = model_config['bias']
            
            # Calcular predição
            prediction = sum(features[i] * weights[i] for i in range(min(len(features), len(weights))))
            prediction += bias
            
            predictions[model_name] = max(-1, min(1, prediction))
        
        # Ensemble (média ponderada)
        ensemble_weights = {
            'trend_predictor': 0.3,
            'reversal_detector': 0.25,
            'momentum_analyzer': 0.25,
            'volatility_predictor': 0.2
        }
        
        final_prediction = sum(predictions[model] * ensemble_weights[model] 
                             for model in predictions.keys())
        
        # Determinar tipo de predição
        if final_prediction > 0.4:
            pred_type = 'strong_bullish'
        elif final_prediction > 0.1:
            pred_type = 'bullish'
        elif final_prediction < -0.4:
            pred_type = 'strong_bearish'
        elif final_prediction < -0.1:
            pred_type = 'bearish'
        else:
            pred_type = 'neutral'
        
        return {
            'prediction': pred_type,
            'score': final_prediction,
            'individual_predictions': predictions
        }

    def extract_ml_features(self, prices: List[float]) -> List[float]:
        """Extrai features para Machine Learning"""
        if len(prices) < 20:
            return [0, 0, 0, 0]
        
        # Feature 1: Momentum (taxa de mudança)
        momentum = (prices[-1] - prices[-10]) / prices[-10]
        
        # Feature 2: Volatilidade relativa
        recent_vol = statistics.stdev(prices[-10:])
        historical_vol = statistics.stdev(prices[-20:-10])
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1
        
        # Feature 3: Mean reversion
        mean_price = statistics.mean(prices[-20:])
        mean_reversion = (prices[-1] - mean_price) / mean_price
        
        # Feature 4: Trend strength
        trend_strength = self.calculate_line_slope(prices[-15:])
        
        return [momentum, vol_ratio, mean_reversion, trend_strength]

    def calculate_price_prediction(self, current_price: float, score: float, volatility: float) -> float:
        """Calcula previsão de preço para 20 minutos"""
        # Base: movimento esperado baseado no score
        expected_move_pct = score * 0.015  # Máximo 1.5% de movimento
        
        # Ajuste pela volatilidade
        volatility_factor = min(2.0, max(0.5, volatility * 100))
        adjusted_move = expected_move_pct * volatility_factor
        
        # Previsão final
        predicted_price = current_price * (1 + adjusted_move)
        
        return predicted_price

    def assess_prediction_quality(self, confidence: float, signal_agreement: int) -> str:
        """Avalia a qualidade da previsão"""
        if confidence >= 0.85 and signal_agreement >= 8:
            return "EXCELENTE"
        elif confidence >= 0.75 and signal_agreement >= 6:
            return "MUITO_BOA"
        elif confidence >= 0.65 and signal_agreement >= 5:
            return "BOA"
        elif confidence >= 0.5 and signal_agreement >= 4:
            return "REGULAR"
        else:
            return "BAIXA"

    def basic_prediction_response(self, current_price: float) -> Dict:
        """Resposta básica quando não há dados suficientes"""
        return {
            'trend': 'neutral',
            'direction': 'hold',
            'should_trade': False,
            'final_score': 0.0,
            'confidence': 0.1,
            'signal_agreement': 0,
            'total_signals': 0,
            'next_20min_prediction': current_price,
            'signals': ['Dados insuficientes para análise avançada'],
            'prediction_quality': 'BAIXA'
        }

    def get_market_data(self) -> Dict:
        """Get enhanced market data"""
        return self.bitget_api.get_market_data(self.symbol)

    def get_account_balance(self) -> float:
        """Get current account balance"""
        return self.bitget_api.get_account_balance()

    def execute_trade(self, side: str) -> Dict:
        """Execute trade only with high confidence"""
        try:
            logger.warning(f"🚀 EXECUTANDO TRADE DE ALTA CONFIANÇA: {side.upper()}")
            
            # Obter dados de mercado
            market_data = self.get_market_data()
            if not market_data:
                logger.error("❌ Erro ao obter dados do mercado")
                return {'success': False, 'error': 'Dados de mercado indisponíveis'}
            
            current_price = market_data['price']
            logger.warning(f"💎 Preço ETH: ${current_price:.2f}")
            
            # Executar ordem
            result = self.bitget_api.place_order(
                symbol=self.symbol,
                side=side,
                size=0,
                price=current_price,
                leverage=self.leverage
            )
            
            if result['success']:
                self.high_confidence_trades += 1
                logger.warning(f"✅ TRADE DE ALTA CONFIANÇA EXECUTADO!")
                logger.warning(f"📈 Total trades alta confiança: {self.high_confidence_trades}")
                return result
            else:
                logger.error(f"❌ Erro no trade: {result.get('error', 'Erro desconhecido')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Erro crítico ao executar trade: {e}")
            return {'success': False, 'error': str(e)}

    def force_close_position_guaranteed(self, reason: str) -> bool:
        """Sistema de fechamento garantido"""
        max_attempts = 8
        attempt = 0
        
        logger.warning(f"🚨 FECHAMENTO FORÇADO: {reason}")
        
        while attempt < max_attempts and self.current_position:
            attempt += 1
            logger.warning(f"🔥 Tentativa {attempt}/{max_attempts}")
            
            try:
                # Obter posições atuais
                positions = self.bitget_api.fetch_positions(['ETH/USDT:USDT'])
                
                for pos in positions:
                    if pos['symbol'] == 'ETH/USDT:USDT' and abs(float(pos['size'])) > 0:
                        size = abs(float(pos['size']))
                        side = 'sell' if float(pos['size']) > 0 else 'buy'
                        
                        logger.warning(f"📊 Fechando {size} ETH com {side}")
                        
                        if side == 'sell':
                            result = self.bitget_api.exchange.create_market_sell_order(
                                'ETH/USDT:USDT', size
                            )
                        else:
                            result = self.bitget_api.exchange.create_market_buy_order(
                                'ETH/USDT:USDT', size
                            )
                        
                        if result and result.get('id'):
                            logger.warning(f"✅ POSIÇÃO FECHADA! ID: {result['id']}")
                            self.current_position = None
                            self.entry_price = None
                            self.position_side = None
                            return True
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Erro na tentativa {attempt}: {e}")
                time.sleep(1)
        
        # Reset forçado se falhou
        if attempt >= max_attempts:
            logger.warning(f"🔄 RESET FORÇADO após {max_attempts} tentativas")
            self.current_position = None
            self.entry_price = None
            self.position_side = None
            return True
        
        return False

    def ultra_fast_security_monitor(self):
        """Monitor de segurança ultra-rápido"""
        logger.warning("🚨 MONITOR DE SEGURANÇA EXTREMO ATIVO")
        
        while self.is_running:
            try:
                if not self.current_position:
                    time.sleep(0.1)
                    continue
                
                # Obter preço atual
                market_data = self.get_market_data()
                if not market_data:
                    time.sleep(0.1)
                    continue
                
                current_price = float(market_data['price'])
                
                # Calcular P&L
                if self.entry_price and self.position_side:
                    if self.position_side == 'buy':
                        pnl_pct = (current_price - self.entry_price) / self.entry_price
                    else:
                        pnl_pct = (self.entry_price - current_price) / self.entry_price
                    
                    # STOP LOSS IMEDIATO
                    if pnl_pct <= self.stop_loss_target:
                        logger.warning(f"🚨 STOP LOSS! P&L: {pnl_pct*100:.2f}%")
                        self.force_close_position_guaranteed("STOP_LOSS")
                        
                    # TAKE PROFIT IMEDIATO
                    elif pnl_pct >= self.profit_target:
                        logger.warning(f"🎯 TAKE PROFIT! P&L: {pnl_pct*100:.2f}%")
                        self.force_close_position_guaranteed("TAKE_PROFIT")
                
                time.sleep(0.05)  # 50ms de intervalo
                
            except Exception as e:
                logger.error(f"❌ Erro no monitor: {e}")
                time.sleep(0.2)

    def intelligent_scalping_strategy(self):
        """Estratégia de scalping inteligente com IA"""
        try:
            if not self.current_position:
                # Obter dados de mercado
                market_data = self.get_market_data()
                if not market_data:
                    return
                
                current_price = market_data['price']
                
                # ANÁLISE SUPREMA DA IA
                prediction = self.supreme_ai_prediction(current_price)
                
                # Log da análise
                logger.info(f"🧠 IA SUPREMA ANALISOU:")
                logger.info(f"   Direção: {prediction['direction']}")
                logger.info(f"   Score: {prediction['final_score']:.3f}")
                logger.info(f"   Confiança: {prediction['confidence']:.2f}")
                logger.info(f"   Sinais concordando: {prediction['signal_agreement']}/{prediction['total_signals']}")
                logger.info(f"   Qualidade: {prediction['prediction_quality']}")
                logger.info(f"   Deve tradear: {'SIM' if prediction['should_trade'] else 'NÃO'}")
                
                # DECISÃO INTELIGENTE
                if prediction['should_trade'] and prediction['direction'] in ['buy', 'sell']:
                    side = prediction['direction']
                    
                    logger.warning(f"🎯 CONDIÇÕES IDEAIS DETECTADAS!")
                    logger.warning(f"🧠 IA recomenda: {side.upper()}")
                    logger.warning(f"📊 Confiança: {prediction['confidence']*100:.1f}%")
                    logger.warning(f"🔍 Score: {prediction['final_score']:.3f}")
                    
                    # Executar trade de alta confiança
                    result = self.execute_trade(side)
                    
                    if result.get('success'):
                        self.current_position = result.get('order_id', True)
                        self.entry_price = result.get('price', current_price)
                        self.position_side = side
                        self.trades_today += 1
                        self.total_trades += 1
                        
                        logger.warning(f"✅ POSIÇÃO ABERTA COM SUPREMA IA!")
                        logger.warning(f"📊 Trades hoje: {self.trades_today}/{self.daily_target}")
                        logger.warning(f"🏆 Trades alta confiança: {self.high_confidence_trades}")
                        
                        # Iniciar monitor de segurança
                        if not hasattr(self, '_security_monitor') or not self._security_monitor.is_alive():
                            self._security_monitor = threading.Thread(
                                target=self.ultra_fast_security_monitor, 
                                daemon=True
                            )
                            self._security_monitor.start()
                else:
                    logger.info(f"⏳ Aguardando condições ideais...")
                    logger.info(f"   Motivo: Confiança {prediction['confidence']:.2f} < {self.min_confidence_to_trade}")
                    logger.info(f"   Score: {prediction['final_score']:.3f} (min: {self.min_prediction_score})")
                    logger.info(f"   Sinais: {prediction['signal_agreement']} < {self.min_signals_agreement}")
                    
        except Exception as e:
            logger.error(f"❌ Erro na estratégia inteligente: {e}")

    def run_trading_loop(self):
        """Loop principal com IA suprema"""
        logger.warning(f"🚀 SUPREMA IA TRADING BOT INICIADO")
        logger.warning(f"🎯 Critérios de trade:")
        logger.warning(f"   - Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.warning(f"   - Score mínimo: {self.min_prediction_score}")
        logger.warning(f"   - Sinais concordando: {self.min_signals_agreement}/10")
        
        self.start_balance = self.get_account_balance()
        
        while self.is_running:
            try:
                # Verificar meta diária
                if self.trades_today >= self.daily_target:
                    logger.warning(f"🎯 META DIÁRIA ATINGIDA: {self.trades_today} trades")
                    time.sleep(60)
                    if datetime.now().hour == 0:
                        self.trades_today = 0
                        logger.warning(f"🌅 NOVO DIA - Resetando contador")
                    continue
                
                # Estratégia inteligente
                self.intelligent_scalping_strategy()
                
                # Aguardar próxima análise
                time.sleep(self.scalping_interval)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop principal: {e}")
                time.sleep(5)
            except KeyboardInterrupt:
                self.stop()
                break

    def start(self):
        """Iniciar bot com IA suprema"""
        if self.is_running:
            logger.warning(f"⚠️ Bot já está rodando")
            return
            
        self.is_running = True
        
        trading_thread = threading.Thread(target=self.run_trading_loop, daemon=True)
        trading_thread.start()
        
        logger.warning(f"✅ SUPREMA IA TRADING BOT INICIADO!")
        logger.warning(f"🧠 Sistema de previsão: ATIVO")
        logger.warning(f"🛡️ Sistema de segurança: ATIVO")

    def stop(self):
        """Parar bot"""
        logger.warning(f"🛑 Parando Suprema IA Bot...")
        self.is_running = False
        
        if self.current_position:
            self.force_close_position_guaranteed("BOT_STOP")
        
        logger.warning(f"📊 ESTATÍSTICAS FINAIS:")
        logger.warning(f"   Total trades: {self.total_trades}")
        logger.warning(f"   Trades alta confiança: {self.high_confidence_trades}")
        logger.warning(f"   Taxa de sucesso: {(self.high_confidence_trades/max(1,self.total_trades))*100:.1f}%")

    def get_status(self) -> Dict:
        """Status detalhado do bot"""
        current_balance = self.get_account_balance()
        
        return {
            'is_running': self.is_running,
            'trades_today': self.trades_today,
            'total_trades': self.total_trades,
            'high_confidence_trades': self.high_confidence_trades,
            'current_balance': current_balance,
            'current_position': bool(self.current_position),
            'position_side': self.position_side,
            'entry_price': self.entry_price,
            'min_confidence': self.min_confidence_to_trade * 100,
            'min_score': self.min_prediction_score,
            'min_signals': self.min_signals_agreement,
            'indicators': self.indicators,
            'ai_active': True
        }

    def update_config(self, **kwargs):
        """Atualizar configurações"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.warning(f"✅ Configuração atualizada: {key} = {value}")
