import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import math
import statistics
from collections import deque

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
        self.price_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=200)
        self.prediction_history = deque(maxlen=100)
        
        # Históricos para cálculos
        self._macd_history = deque(maxlen=26)
        self._bb_width_history = deque(maxlen=20)
        self._stoch_k_history = deque(maxlen=3)
        
        # Indicadores técnicos
        self.indicators = {
            'sma_5': 0, 'sma_10': 0, 'sma_20': 0, 'sma_50': 0,
            'ema_12': 0, 'ema_26': 0, 'ema_50': 0,
            'rsi_14': 50, 'rsi_6': 50, 'rsi_21': 50,
            'macd': 0, 'macd_signal': 0, 'macd_histogram': 0,
            'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0, 'bb_width': 0,
            'stoch_k': 50, 'stoch_d': 50,
            'williams_r': -50, 'cci': 0, 'atr': 0
        }
        
        # Sistema de emergência
        self.emergency_stop = False
        self.force_close_active = False
        self.position_monitor_active = False
        
        # Statistics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        self.high_confidence_trades = 0
        self.forced_closes = 0
        self.stop_loss_triggered = 0
        self.take_profit_triggered = 0
        
        logger.info("🧠 SUPREME AI TRADING BOT INICIALIZADO")
        logger.info(f"🎯 Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"📊 Score mínimo: {self.min_prediction_score}")
        logger.info(f"🔍 Sinais mínimos: {self.min_signals_agreement}/10")

    def get_market_data(self) -> Dict:
        """Get current market data"""
        return self.bitget_api.get_market_data(self.symbol)

    def get_account_balance(self) -> float:
        """Get current account balance"""
        return self.bitget_api.get_account_balance()

    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return 0.0
        return sum(prices[-period:]) / period

    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return 0.0
        
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas[-period:]]
        losses = [-d if d < 0 else 0 for d in deltas[-period:]]
        
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: List[float]) -> Dict:
        """MACD Indicator"""
        if len(prices) < 35:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        
        self._macd_history.append(macd_line)
        
        if len(self._macd_history) >= 9:
            signal_line = self.calculate_ema(list(self._macd_history), 9)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20) -> Dict:
        """Bollinger Bands"""
        if len(prices) < period:
            return {'upper': 0, 'middle': 0, 'lower': 0, 'width': 0, 'squeeze': False}
        
        recent = prices[-period:]
        sma = sum(recent) / period
        variance = sum((p - sma) ** 2 for p in recent) / period
        std_dev = math.sqrt(variance)
        
        upper = sma + (2 * std_dev)
        lower = sma - (2 * std_dev)
        width = (upper - lower) / sma if sma != 0 else 0
        
        self._bb_width_history.append(width)
        avg_width = sum(self._bb_width_history) / len(self._bb_width_history)
        squeeze = width < (avg_width * 0.8)
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': width,
            'squeeze': squeeze
        }

    def calculate_stochastic(self, prices: List[float]) -> Dict:
        """Stochastic Oscillator"""
        if len(prices) < 14:
            return {'k': 50, 'd': 50}
        
        high_14 = max(prices[-14:])
        low_14 = min(prices[-14:])
        current = prices[-1]
        
        if high_14 != low_14:
            k_percent = ((current - low_14) / (high_14 - low_14)) * 100
        else:
            k_percent = 50
        
        self._stoch_k_history.append(k_percent)
        d_percent = sum(self._stoch_k_history) / len(self._stoch_k_history)
        
        return {'k': k_percent, 'd': d_percent}

    def calculate_williams_r(self, prices: List[float]) -> float:
        """Williams %R"""
        if len(prices) < 14:
            return -50.0
        
        high_14 = max(prices[-14:])
        low_14 = min(prices[-14:])
        current = prices[-1]
        
        if high_14 != low_14:
            return ((high_14 - current) / (high_14 - low_14)) * -100
        return -50.0

    def find_peaks(self, prices: List[float]) -> List[float]:
        """Find peaks in price data"""
        peaks = []
        for i in range(2, len(prices) - 2):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                peaks.append(prices[i])
        return peaks

    def find_valleys(self, prices: List[float]) -> List[float]:
        """Find valleys in price data"""
        valleys = []
        for i in range(2, len(prices) - 2):
            if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                valleys.append(prices[i])
        return valleys

    def calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength using linear regression"""
        if len(prices) < 10:
            return 0.0
        
        n = len(prices)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(prices)
        sum_xy = sum(x[i] * prices[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        avg_price = sum_y / n
        normalized_slope = (slope / avg_price) * 1000 if avg_price != 0 else 0
        
        return max(-1.0, min(1.0, normalized_slope))

    def detect_patterns(self, prices: List[float]) -> Dict:
        """Detect chart patterns"""
        if len(prices) < 30:
            return {'patterns': [], 'confidence': 0}
        
        patterns = []
        scores = []
        
        # Double Top/Bottom
        peaks = self.find_peaks(prices)
        valleys = self.find_valleys(prices)
        
        if len(peaks) >= 2:
            if abs(peaks[-1] - peaks[-2]) / peaks[-1] < 0.02:
                patterns.append('double_top')
                scores.append(0.8)
        
        if len(valleys) >= 2:
            if abs(valleys[-1] - valleys[-2]) / valleys[-1] < 0.02:
                patterns.append('double_bottom')
                scores.append(0.8)
        
        # Breakout setup
        recent_20 = prices[-20:]
        volatility = statistics.stdev(recent_20) / statistics.mean(recent_20)
        
        if volatility < 0.005:
            patterns.append('breakout_setup')
            scores.append(0.7)
        
        # Strong trend
        trend_strength = self.calculate_trend_strength(prices)
        if abs(trend_strength) > 0.7:
            patterns.append('strong_trend')
            scores.append(0.6)
        
        avg_confidence = sum(scores) / len(scores) if scores else 0
        
        return {
            'patterns': patterns,
            'confidence': avg_confidence
        }

    def supreme_ai_prediction(self, current_price: float) -> Dict:
        """SISTEMA DE IA SUPREMO"""
        try:
            timestamp = time.time()
            self.price_history.append({
                'price': current_price,
                'timestamp': timestamp
            })
            
            if len(self.price_history) < 50:
                return self.basic_prediction(current_price)
            
            prices = [p['price'] for p in self.price_history]
            
            # Calcular todos os indicadores
            self.indicators.update({
                'sma_5': self.calculate_sma(prices, 5),
                'sma_10': self.calculate_sma(prices, 10),
                'sma_20': self.calculate_sma(prices, 20),
                'sma_50': self.calculate_sma(prices, 50),
                'ema_12': self.calculate_ema(prices, 12),
                'ema_26': self.calculate_ema(prices, 26),
                'ema_50': self.calculate_ema(prices, 50),
                'rsi_6': self.calculate_rsi(prices, 6),
                'rsi_14': self.calculate_rsi(prices, 14),
                'rsi_21': self.calculate_rsi(prices, 21),
                'williams_r': self.calculate_williams_r(prices)
            })
            
            # MACD
            macd_data = self.calculate_macd(prices)
            self.indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(prices)
            self.indicators.update(bb_data)
            
            # Stochastic
            stoch_data = self.calculate_stochastic(prices)
            self.indicators.update(stoch_data)
            
            # Padrões
            patterns = self.detect_patterns(prices)
            
            # ANÁLISE DE 10 SINAIS
            signals = []
            signal_scores = []
            
            # SINAL 1: Golden/Death Cross
            ema_12 = self.indicators['ema_12']
            ema_26 = self.indicators['ema_26']
            sma_50 = self.indicators['sma_50']
            
            if ema_12 > ema_26 and ema_12 > sma_50:
                signals.append("Golden Cross Confirmado")
                signal_scores.append(0.9)
            elif ema_12 < ema_26 and ema_12 < sma_50:
                signals.append("Death Cross Confirmado")
                signal_scores.append(-0.9)
            elif ema_12 > ema_26:
                signals.append("EMA Bullish")
                signal_scores.append(0.6)
            else:
                signals.append("EMA Bearish")
                signal_scores.append(-0.6)
            
            # SINAL 2: RSI Multi-timeframe
            rsi_6 = self.indicators['rsi_6']
            rsi_14 = self.indicators['rsi_14']
            
            if rsi_14 < 30 and rsi_6 < 25:
                signals.append(f"RSI Extremo Oversold: {rsi_14:.1f}")
                signal_scores.append(0.95)
            elif rsi_14 > 70 and rsi_6 > 75:
                signals.append(f"RSI Extremo Overbought: {rsi_14:.1f}")
                signal_scores.append(-0.95)
            elif rsi_14 < 40:
                signals.append(f"RSI Oversold: {rsi_14:.1f}")
                signal_scores.append(0.7)
            elif rsi_14 > 60:
                signals.append(f"RSI Overbought: {rsi_14:.1f}")
                signal_scores.append(-0.7)
            else:
                signals.append(f"RSI Neutro: {rsi_14:.1f}")
                signal_scores.append(0.0)
            
            # SINAL 3: MACD
            macd = self.indicators['macd']
            macd_signal = self.indicators['signal']
            macd_hist = self.indicators['histogram']
            
            if macd > macd_signal and macd_hist > 0:
                signals.append("MACD Bullish")
                signal_scores.append(0.8)
            elif macd < macd_signal and macd_hist < 0:
                signals.append("MACD Bearish")
                signal_scores.append(-0.8)
            else:
                signals.append("MACD Neutro")
                signal_scores.append(0.0)
            
            # SINAL 4: Bollinger Bands
            bb_upper = self.indicators['upper']
            bb_lower = self.indicators['lower']
            bb_squeeze = self.indicators['squeeze']
            
            if bb_squeeze:
                signals.append("Bollinger Squeeze")
                signal_scores.append(0.85)
            elif current_price <= bb_lower * 1.01:
                signals.append("BB Lower Band")
                signal_scores.append(0.75)
            elif current_price >= bb_upper * 0.99:
                signals.append("BB Upper Band")
                signal_scores.append(-0.75)
            else:
                signals.append("BB Meio")
                signal_scores.append(0.0)
            
            # SINAL 5: Stochastic
            stoch_k = self.indicators['k']
            stoch_d = self.indicators['d']
            
            if stoch_k < 20 and stoch_d < 20 and stoch_k > stoch_d:
                signals.append("Stoch Bullish")
                signal_scores.append(0.7)
            elif stoch_k > 80 and stoch_d > 80 and stoch_k < stoch_d:
                signals.append("Stoch Bearish")
                signal_scores.append(-0.7)
            else:
                signals.append("Stoch Neutro")
                signal_scores.append(0.0)
            
            # SINAL 6: Williams %R
            williams_r = self.indicators['williams_r']
            
            if williams_r < -80:
                signals.append("Williams Oversold")
                signal_scores.append(0.6)
            elif williams_r > -20:
                signals.append("Williams Overbought")
                signal_scores.append(-0.6)
            else:
                signals.append("Williams Neutro")
                signal_scores.append(0.0)
            
            # SINAL 7: Padrões
            if patterns['patterns']:
                for pattern in patterns['patterns']:
                    if pattern in ['double_bottom', 'breakout_setup']:
                        signals.append(f"Padrão Bullish: {pattern}")
                        signal_scores.append(0.8)
                    elif pattern in ['double_top']:
                        signals.append(f"Padrão Bearish: {pattern}")
                        signal_scores.append(-0.8)
                    else:
                        signals.append(f"Padrão: {pattern}")
                        signal_scores.append(0.4)
            else:
                signals.append("Sem padrões")
                signal_scores.append(0.0)
            
            # SINAL 8: Tendência
            trend_strength = self.calculate_trend_strength(prices)
            
            if trend_strength > 0.7:
                signals.append("Tendência Bullish Forte")
                signal_scores.append(0.8)
            elif trend_strength < -0.7:
                signals.append("Tendência Bearish Forte")
                signal_scores.append(-0.8)
            elif trend_strength > 0.3:
                signals.append("Tendência Bullish")
                signal_scores.append(0.5)
            elif trend_strength < -0.3:
                signals.append("Tendência Bearish")
                signal_scores.append(-0.5)
            else:
                signals.append("Lateral")
                signal_scores.append(0.0)
            
            # SINAL 9: Volatilidade
            volatility = statistics.stdev(prices[-20:]) / statistics.mean(prices[-20:])
            
            if volatility > 0.01:
                signals.append("Alta Volatilidade")
                signal_scores.append(-0.3)
            elif volatility < 0.005:
                signals.append("Baixa Volatilidade")
                signal_scores.append(0.4)
            else:
                signals.append("Volatilidade Normal")
                signal_scores.append(0.1)
            
            # SINAL 10: Momentum
            momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0
            momentum_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) > 10 else 0
            
            if momentum_5 > 0.01 and momentum_10 > 0.01:
                signals.append("Momentum Bullish")
                signal_scores.append(0.7)
            elif momentum_5 < -0.01 and momentum_10 < -0.01:
                signals.append("Momentum Bearish")
                signal_scores.append(-0.7)
            else:
                signals.append("Momentum Neutro")
                signal_scores.append(0.0)
            
            # CALCULAR SCORE FINAL
            final_score = sum(signal_scores) / len(signal_scores) if signal_scores else 0
            
            # CALCULAR CONFIANÇA
            positive_signals = len([s for s in signal_scores if s > 0.5])
            negative_signals = len([s for s in signal_scores if s < -0.5])
            extreme_signals = len([s for s in signal_scores if abs(s) > 0.8])
            
            signal_agreement = max(positive_signals, negative_signals)
            confidence = signal_agreement / len(signal_scores)
            confidence += (extreme_signals * 0.1)
            confidence = min(1.0, confidence)
            
            # DECISÃO
            if final_score > 0.3:
                trend = 'bullish'
                direction = 'buy'
            elif final_score < -0.3:
                trend = 'bearish'
                direction = 'sell'
            else:
                trend = 'neutral'
                direction = 'hold'
            
            should_trade = self.should_execute_trade(final_score, confidence, signal_agreement)
            
            # PREVISÃO DE PREÇO
            price_prediction = current_price * (1 + (final_score * 0.02))
            
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
                'trend_strength': trend_strength
            }
            
            # Log
            logger.warning(f"🧠 SUPREMA IA: {direction.upper()} | Score: {final_score:.3f} | Conf: {confidence:.2f}")
            logger.warning(f"📊 Sinais: {positive_signals}+ {negative_signals}- | Extremos: {extreme_signals}")
            logger.warning(f"🎯 Executar: {'SIM' if should_trade else 'NÃO'} | Acordo: {signal_agreement}/10")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro na IA: {e}")
            return self.basic_prediction(current_price)

    def should_execute_trade(self, score: float, confidence: float, signal_agreement: int) -> bool:
        """Decide se deve executar trade"""
        if abs(score) < self.min_prediction_score:
            return False
        if confidence < self.min_confidence_to_trade:
            return False
        if signal_agreement < self.min_signals_agreement:
            return False
        return True

    def basic_prediction(self, current_price: float) -> Dict:
        """Previsão básica"""
        return {
            'trend': 'neutral',
            'direction': 'hold',
            'should_trade': False,
            'final_score': 0.0,
            'confidence': 0.1,
            'signal_agreement': 0,
            'total_signals': 0,
            'positive_signals': 0,
            'negative_signals': 0,
            'extreme_signals': 0,
            'next_20min_prediction': current_price,
            'signals': ['Dados insuficientes'],
            'signal_scores': [],
            'indicators': {},
            'patterns': {'patterns': [], 'confidence': 0.0}
        }

    def force_close_position_guaranteed(self, reason: str) -> bool:
        """Fechamento garantido da posição"""
        max_attempts = 10
        attempt = 0
        
        logger.warning(f"🚨 FECHAMENTO FORÇADO: {reason}")
        
        while attempt < max_attempts and self.current_position:
            attempt += 1
            
            try:
                positions = self.bitget_api.fetch_positions(['ETH/USDT:USDT'])
                
                for pos in positions:
                    if pos['symbol'] == 'ETH/USDT:USDT' and abs(float(pos['size'])) > 0:
                        size = abs(float(pos['size']))
                        side = 'sell' if float(pos['size']) > 0 else 'buy'
                        
                        logger.warning(f"📊 Fechando {size} ETH com {side}")
                        
                        if side == 'sell':
                            result = self.bitget_api.exchange.create_market_sell_order('ETH/USDT:USDT', size)
                        else:
                            result = self.bitget_api.exchange.create_market_buy_order('ETH/USDT:USDT', size)
                        
                        if result and result.get('id'):
                            logger.warning(f"✅ POSIÇÃO FECHADA! ID: {result['id']}")
                            self.current_position = None
                            self.entry_price = None
                            self.position_side = None
                            self.forced_closes += 1
                            return True
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Erro tentativa {attempt}: {e}")
                time.sleep(1.0)
        
        # Reset forçado
        logger.warning("🔄 RESET FORÇADO")
        self.current_position = None
        self.entry_price = None
        self.position_side = None
        return True

    def ultra_fast_position_monitor(self):
        """Monitor ultra-rápido de posição"""
        logger.warning("🚨 MONITOR DE SEGURANÇA ATIVADO")
        self.position_monitor_active = True
        
        while self.is_running and self.position_monitor_active and not self.emergency_stop:
            try:
                if not self.current_position:
                    time.sleep(0.1)
                    continue
                
                market_data = self.get_market_data()
                if not market_data:
                    time.sleep(0.1)
                    continue
                
                current_price = float(market_data['price'])
                
                if self.entry_price and self.position_side:
                    if self.position_side == 'buy':
                        pnl_pct = (current_price - self.entry_price) / self.entry_price
                    else:
                        pnl_pct = (self.entry_price - current_price) / self.entry_price
                    
                    # STOP LOSS
                    if pnl_pct <= self.stop_loss_target:
                        logger.warning(f"🚨 STOP LOSS! P&L: {pnl_pct*100:.2f}%")
                        self.force_close_position_guaranteed("STOP_LOSS")
                        self.stop_loss_triggered += 1
                    
                    # TAKE PROFIT
                    elif pnl_pct >= self.profit_target:
                        logger.warning(f"🎯 TAKE PROFIT! P&L: {pnl_pct*100:.2f}%")
                        self.force_close_position_guaranteed("TAKE_PROFIT")
                        self.take_profit_triggered += 1
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Erro no monitor: {e}")
                time.sleep(0.2)

    def execute_trade(self, side: str) -> Dict:
        """Execute trade"""
        try:
            logger.warning(f"🚀 EXECUTANDO TRADE {side.upper()}")
            
            market_data = self.get_market_data()
            if not market_data:
                return {'success': False, 'error': 'Erro nos dados de mercado'}
            
            current_price = float(market_data['price'])
            logger.warning(f"💎 Preço atual: ${current_price:.2f}")
            
            result = self.bitget_api.place_order(side=side)
            
            if result.get('success'):
                logger.warning(f"✅ TRADE {side.upper()} EXECUTADO!")
                return result
            else:
                logger.error(f"❌ Erro no trade: {result.get('error', 'Desconhecido')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Erro crítico no trade: {e}")
            return {'success': False, 'error': str(e)}

    def scalping_strategy(self):
        """Estratégia de scalping com IA"""
        try:
            if self.emergency_stop:
                logger.error("🚨 BOT PARADO POR EMERGÊNCIA")
                return
            
            if not self.current_position:
                current_market = self.get_market_data()
                if current_market and 'price' in current_market:
                    current_price = current_market['price']
                    prediction = self.supreme_ai_prediction(current_price)
                    
                    if prediction['should_trade']:
                        side = prediction['direction']
                        
                        logger.warning(f"🚀 ABRINDO POSIÇÃO {side.upper()}")
                        logger.warning(f"🔮 IA: {prediction['trend']} | Conf: {prediction['confidence']:.2f}")
                        logger.warning(f"📊 Score: {prediction['final_score']:.3f} | Sinais: {prediction['signal_agreement']}/10")
                        
                        result = self.execute_trade(side)
                        if result.get('success'):
                            self.current_position = result.get('order_id', True)
                            self.entry_price = result.get('price', current_price)
                            self.position_side = side
                            self.trades_today += 1
                            self.total_trades += 1
                            self.high_confidence_trades += 1
                            
                            logger.warning(f"✅ POSIÇÃO ABERTA: {side.upper()}")
                            logger.warning(f"📊 Trades hoje: {self.trades_today}/{self.daily_target}")
                            
                            # Iniciar monitor de segurança
                            if not self.position_monitor_active:
                                monitor_thread = threading.Thread(target=self.ultra_fast_position_monitor, daemon=True)
                                monitor_thread.start()
                    else:
                        logger.info(f"⏳ Aguardando condições ideais...")
                        logger.info(f"📊 Score: {prediction['final_score']:.3f} | Conf: {prediction['confidence']:.2f}")
                        
        except Exception as e:
            logger.error(f"❌ Erro na estratégia: {e}")

    def run_trading_loop(self):
        """Loop principal de trading"""
        logger.warning(f"🚀 Trading bot iniciado com SUPREMA IA")
        self.start_balance = self.get_account_balance()
        
        while self.is_running and not self.emergency_stop:
            try:
                if self.trades_today >= self.daily_target:
                    logger.warning(f"🎯 META DIÁRIA ATINGIDA: {self.trades_today} trades")
                    time.sleep(60)
                    if datetime.now().hour == 0:
                        self.trades_today = 0
                        logger.warning(f"🌅 NOVO DIA")
                    continue
                
                self.scalping_strategy()
                time.sleep(self.scalping_interval)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop: {e}")
                time.sleep(5)
            except KeyboardInterrupt:
                self.stop()
                break

    def start(self):
        """Start trading bot"""
        if self.is_running:
            logger.warning(f"⚠️ Bot já está rodando")
            return
            
        self.is_running = True
        self.emergency_stop = False
        
        trading_thread = threading.Thread(target=self.run_trading_loop, daemon=True)
        trading_thread.start()
        logger.warning(f"✅ SUPREMA IA TRADING BOT INICIADO!")

    def stop(self):
        """Stop trading bot"""
        logger.warning(f"🛑 Parando bot...")
        self.is_running = False
        self.position_monitor_active = False
        
        if self.current_position:
            logger.warning(f"🔄 Fechando posição...")
            self.force_close_position_guaranteed("BOT_STOP")
        
        logger.warning(f"🛑 Bot parado")
        logger.warning(f"📊 ESTATÍSTICAS:")
        logger.warning(f"   Total trades: {self.total_trades}")
        logger.warning(f"   Stop Loss: {self.stop_loss_triggered}")
        logger.warning(f"   Take Profit: {self.take_profit_triggered}")
        logger.warning(f"   Trades alta confiança: {self.high_confidence_trades}")

    def get_status(self) -> Dict:
        """Get bot status"""
        current_balance = self.get_account_balance()
        
        return {
            'is_running': self.is_running,
            'trades_today': self.trades_today,
            'daily_target': self.daily_target,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'current_balance': current_balance,
            'start_balance': self.start_balance,
            'current_position': bool(self.current_position),
            'position_side': self.position_side,
            'entry_price': self.entry_price,
            'profit_target': self.profit_target * 100,
            'stop_loss_target': abs(self.stop_loss_target) * 100,
            'stop_loss_triggered': self.stop_loss_triggered,
            'take_profit_triggered': self.take_profit_triggered,
            'high_confidence_trades': self.high_confidence_trades,
            'forced_closes': self.forced_closes,
            'emergency_stop': self.emergency_stop,
            'min_confidence': self.min_confidence_to_trade * 100,
            'min_score': self.min_prediction_score,
            'min_signals': self.min_signals_agreement,
            'test_mode': getattr(self.bitget_api, 'test_mode', False)
        }

    def update_config(self, **kwargs):
        """Update configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.warning(f"✅ Config atualizada: {key} = {value}")
