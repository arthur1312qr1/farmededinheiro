import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import math
import statistics
from collections import deque
import pytz

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
        
        # TIMEZONE BRASILEIRO
        self.brazil_tz = pytz.timezone('America/Sao_Paulo')
        
        # Trading state
        self.is_running = False
        self.trades_today = 0
        self.current_position = None
        self.entry_price = None
        self.position_side = None
        self.position_size = 0.0
        self.profit_target = 0.01  # 1% take profit
        self.stop_loss_target = -0.02  # 2% stop loss
        
        # SISTEMA DE CERTEZA MAIS SENSÍVEL PARA TESTES
        self.min_confidence_to_trade = 0.60  # Reduzido de 75% para 60%
        self.min_prediction_score = 0.4      # Reduzido de 0.6 para 0.4
        self.min_signals_agreement = 4       # Reduzido de 6 para 4 sinais
        
        # SISTEMA DE PREVISÃO SUPREMO
        self.price_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=200)
        self.prediction_history = deque(maxlen=100)
        
        # Históricos para cálculos
        self._macd_history = deque(maxlen=26)
        self._bb_width_history = deque(maxlen=20)
        self._stoch_k_history = deque(maxlen=3)
        
        # Cache de dados para evitar calls desnecessárias
        self.last_price_update = 0
        self.last_market_data = None
        self.price_update_interval = 1.0  # Atualiza a cada 1 segundo
        
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
        self.debug_mode = True  # ATIVAR DEBUG
        
        # Statistics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        self.high_confidence_trades = 0
        self.forced_closes = 0
        self.stop_loss_triggered = 0
        self.take_profit_triggered = 0
        
        # INICIAR MONITOR AUTOMATICAMENTE
        self.monitor_thread = None
        
        logger.info("🧠 SUPREME AI TRADING BOT INICIALIZADO")
        logger.info(f"🎯 Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"📊 Score mínimo: {self.min_prediction_score}")
        logger.info(f"🔍 Sinais mínimos: {self.min_signals_agreement}/10")
        logger.info(f"🇧🇷 Timezone: {self.brazil_tz}")

    def get_brazil_time(self):
        """Retorna horário atual do Brasil"""
        utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)
        brazil_time = utc_now.astimezone(self.brazil_tz)
        return brazil_time

    def log_brazil_time(self, message: str, level: str = "info"):
        """Log com horário brasileiro"""
        brazil_time = self.get_brazil_time()
        time_str = brazil_time.strftime("%d/%m/%Y %H:%M:%S")
        full_message = f"[{time_str} BR] {message}"
        
        if level == "warning":
            logger.warning(full_message)
        elif level == "error":
            logger.error(full_message)
        else:
            logger.info(full_message)

    def get_market_data(self) -> Dict:
        """Get current market data with cache busting"""
        current_time = time.time()
        
        # Forçar atualização se passou do intervalo
        if current_time - self.last_price_update > self.price_update_interval:
            try:
                # Cache busting com timestamp
                fresh_data = self.bitget_api.get_market_data(self.symbol, cache_bust=current_time)
                
                if fresh_data and 'price' in fresh_data:
                    self.last_market_data = fresh_data
                    self.last_price_update = current_time
                    
                    if self.debug_mode:
                        self.log_brazil_time(f"💰 Preço atualizado: ${fresh_data['price']}", "info")
                
                return fresh_data
                
            except Exception as e:
                self.log_brazil_time(f"❌ Erro ao buscar dados: {e}", "error")
                return self.last_market_data
        
        return self.last_market_data

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
        """SISTEMA DE IA SUPREMO COM DEBUG"""
        try:
            timestamp = time.time()
            self.price_history.append({
                'price': current_price,
                'timestamp': timestamp
            })
            
            if len(self.price_history) < 20:  # Reduzido de 50 para 20
                if self.debug_mode:
                    self.log_brazil_time(f"🔄 Coletando dados... {len(self.price_history)}/20", "info")
                return self.basic_prediction(current_price)
            
            prices = [p['price'] for p in self.price_history]
            
            # Calcular todos os indicadores
            self.indicators.update({
                'sma_5': self.calculate_sma(prices, 5),
                'sma_10': self.calculate_sma(prices, 10),
                'sma_20': self.calculate_sma(prices, 20),
                'ema_12': self.calculate_ema(prices, 12),
                'ema_26': self.calculate_ema(prices, 26),
                'rsi_6': self.calculate_rsi(prices, 6),
                'rsi_14': self.calculate_rsi(prices, 14),
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
            
            # ANÁLISE DE 10 SINAIS COM DEBUG
            signals = []
            signal_scores = []
            
            # SINAL 1: Golden/Death Cross
            ema_12 = self.indicators['ema_12']
            ema_26 = self.indicators['ema_26']
            
            if ema_12 > ema_26:
                signals.append("EMA Bullish")
                signal_scores.append(0.6)
            else:
                signals.append("EMA Bearish")
                signal_scores.append(-0.6)
            
            # SINAL 2: RSI
            rsi_14 = self.indicators['rsi_14']
            
            if rsi_14 < 35:  # Mais sensível
                signals.append(f"RSI Oversold: {rsi_14:.1f}")
                signal_scores.append(0.8)
            elif rsi_14 > 65:  # Mais sensível
                signals.append(f"RSI Overbought: {rsi_14:.1f}")
                signal_scores.append(-0.8)
            else:
                signals.append(f"RSI Neutro: {rsi_14:.1f}")
                signal_scores.append(0.0)
            
            # SINAL 3: MACD
            macd = self.indicators['macd']
            macd_signal = self.indicators['signal']
            
            if macd > macd_signal:
                signals.append("MACD Bullish")
                signal_scores.append(0.5)
            else:
                signals.append("MACD Bearish")
                signal_scores.append(-0.5)
            
            # SINAL 4: Bollinger Bands
            bb_upper = self.indicators['upper']
            bb_lower = self.indicators['lower']
            
            if current_price <= bb_lower * 1.02:  # Mais sensível
                signals.append("BB Lower Band")
                signal_scores.append(0.7)
            elif current_price >= bb_upper * 0.98:  # Mais sensível
                signals.append("BB Upper Band")
                signal_scores.append(-0.7)
            else:
                signals.append("BB Meio")
                signal_scores.append(0.0)
            
            # SINAL 5: Momentum simples
            if len(prices) > 5:
                momentum = (prices[-1] - prices[-5]) / prices[-5]
                if momentum > 0.005:  # Mais sensível
                    signals.append("Momentum Positivo")
                    signal_scores.append(0.5)
                elif momentum < -0.005:
                    signals.append("Momentum Negativo")
                    signal_scores.append(-0.5)
                else:
                    signals.append("Momentum Neutro")
                    signal_scores.append(0.0)
            else:
                signals.append("Momentum Neutro")
                signal_scores.append(0.0)
            
            # Completar com 5 sinais neutros para ter 10 total
            for i in range(5):
                signals.append(f"Sinal {i+6}: Neutro")
                signal_scores.append(0.0)
            
            # CALCULAR SCORE FINAL
            final_score = sum(signal_scores) / len(signal_scores) if signal_scores else 0
            
            # CALCULAR CONFIANÇA
            positive_signals = len([s for s in signal_scores if s > 0.3])
            negative_signals = len([s for s in signal_scores if s < -0.3])
            
            signal_agreement = max(positive_signals, negative_signals)
            confidence = signal_agreement / len(signal_scores)
            confidence = min(1.0, confidence + 0.1)  # Boost de confiança
            
            # DECISÃO
            if final_score > 0.2:  # Mais sensível
                trend = 'bullish'
                direction = 'buy'
            elif final_score < -0.2:  # Mais sensível
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
                'next_20min_prediction': price_prediction,
                'signals': signals,
                'signal_scores': signal_scores,
                'indicators': self.indicators.copy(),
                'patterns': patterns
            }
            
            # DEBUG LOG DETALHADO
            if self.debug_mode:
                self.log_brazil_time(f"🧠 IA: {direction.upper()} | Score: {final_score:.3f} | Conf: {confidence:.2f}", "warning")
                self.log_brazil_time(f"📊 +{positive_signals} -{negative_signals} | Executar: {'SIM' if should_trade else 'NÃO'}", "warning")
                for i, signal in enumerate(signals[:5]):  # Mostrar só 5 principais
                    score = signal_scores[i]
                    self.log_brazil_time(f"  📍 {signal}: {score:.2f}", "info")
            
            return result
            
        except Exception as e:
            self.log_brazil_time(f"❌ Erro na IA: {e}", "error")
            return self.basic_prediction(current_price)

    def should_execute_trade(self, score: float, confidence: float, signal_agreement: int) -> bool:
        """Decide se deve executar trade - MAIS SENSÍVEL"""
        if self.debug_mode:
            self.log_brazil_time(f"🔍 Checando: Score={score:.3f} Conf={confidence:.3f} Sinais={signal_agreement}", "info")
        
        if abs(score) < self.min_prediction_score:
            if self.debug_mode:
                self.log_brazil_time(f"❌ Score muito baixo: {score:.3f} < {self.min_prediction_score}", "info")
            return False
        
        if confidence < self.min_confidence_to_trade:
            if self.debug_mode:
                self.log_brazil_time(f"❌ Confiança baixa: {confidence:.3f} < {self.min_confidence_to_trade}", "info")
            return False
        
        if signal_agreement < self.min_signals_agreement:
            if self.debug_mode:
                self.log_brazil_time(f"❌ Poucos sinais: {signal_agreement} < {self.min_signals_agreement}", "info")
            return False
        
        if self.debug_mode:
            self.log_brazil_time(f"✅ TRADE APROVADO!", "warning")
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
        
        self.log_brazil_time(f"🚨 FECHAMENTO FORÇADO: {reason}", "warning")
        
        while attempt < max_attempts and self.current_position:
            attempt += 1
            
            try:
                positions = self.bitget_api.fetch_positions(['ETH/USDT:USDT'])
                
                for pos in positions:
                    if pos['symbol'] == 'ETH/USDT:USDT' and abs(float(pos['size'])) > 0:
                        size = abs(float(pos['size']))
                        side = 'sell' if float(pos['size']) > 0 else 'buy'
                        
                        self.log_brazil_time(f"📊 Fechando {size} ETH com {side}", "warning")
                        
                        if side == 'sell':
                            result = self.bitget_api.exchange.create_market_sell_order('ETH/USDT:USDT', size)
                        else:
                            result = self.bitget_api.exchange.create_market_buy_order('ETH/USDT:USDT', size)
                        
                        if result and result.get('id'):
                            self.log_brazil_time(f"✅ POSIÇÃO FECHADA! ID: {result['id']}", "warning")
                            self.current_position = None
                            self.entry_price = None
                            self.position_side = None
                            self.forced_closes += 1
                            return True
                
                time.sleep(0.5)
                
            except Exception as e:
                self.log_brazil_time(f"❌ Erro tentativa {attempt}: {e}", "error")
                time.sleep(1.0)
        
        # Reset forçado
        self.log_brazil_time("🔄 RESET FORÇADO", "warning")
        self.current_position = None
        self.entry_price = None
        self.position_side = None
        return True

    def start_position_monitor(self):
        """Iniciar monitor de posição em thread separada"""
        if not self.position_monitor_active:
            self.position_monitor_active = True
            self.monitor_thread = threading.Thread(target=self.ultra_fast_position_monitor, daemon=True)
            self.monitor_thread.start()
            self.log_brazil_time("🚨 MONITOR DE SEGURANÇA INICIADO", "warning")

    def ultra_fast_position_monitor(self):
        """Monitor ultra-rápido de posição"""
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
                        self.log_brazil_time(f"🚨 STOP LOSS! P&L: {pnl_pct*100:.2f}%", "warning")
                        self.force_close_position_guaranteed("STOP_LOSS")
                        self.stop_loss_triggered += 1
                    
                    # TAKE PROFIT
                    elif pnl_pct >= self.profit_target:
                        self.log_brazil_time(f"🎯 TAKE PROFIT! P&L: {pnl_pct*100:.2f}%", "warning")
                        self.force_close_position_guaranteed("TAKE_PROFIT")
                        self.take_profit_triggered += 1
                
                time.sleep(0.1)
                
            except Exception as e:
                self.log_brazil_time(f"❌ Erro no monitor: {e}", "error")
                time.sleep(0.2)

    def execute_trade(self, side: str) -> Dict:
        """Execute trade"""
        try:
            self.log_brazil_time(f"🚀 EXECUTANDO TRADE {side.upper()}", "warning")
            
            market_data = self.get_market_data()
            if not market_data:
                return {'success': False, 'error': 'Erro nos dados de mercado'}
            
            current_price = float(market_data['price'])
            self.log_brazil_time(f"💎 Preço atual: ${current_price:.2f}", "warning")
            
            result = self.bitget_api.place_order(side=side)
            
            if result.get('success'):
                self.log_brazil_time(f"✅ TRADE {side.upper()} EXECUTADO!", "warning")
                
                # Registrar posição
                self.current_position = result
                self.entry_price = current_price
                self.position_side = side
                
                # INICIAR MONITOR SE NÃO ESTIVER ATIVO
                if not self.position_monitor_active:
                    self.start_position_monitor()
                
                return result
            else:
                self.log_brazil_time(f"❌ Erro no trade: {result.get('error', 'Desconhecido')}", "error")
                return result
                
        except Exception as e:
            self.log_brazil_time(f"❌ Erro crítico no trade: {e}", "error")
            return {'success': False, 'error': str(e)}

    def scalping_strategy(self):
        """Estratégia de scalping com IA MAIS ATIVA"""
        try:
            if self.emergency_stop:
                self.log_brazil_time("🚨 BOT PARADO POR EMERGÊNCIA", "error")
                return
            
            # ATIVAR MONITOR NA PRIMEIRA EXECUÇÃO
            if not self.position_monitor_active:
                self.start_position_monitor()
            
            if not self.current_position:
                current_market = self.get_market_data()
                if current_market and 'price' in current_market:
                    current_price = current_market['price']
                    prediction = self.supreme_ai_prediction(current_price)
                    
                    if prediction['should_trade']:
                        side = prediction['direction']
                        
                        self.log_brazil_time(f"🚀 EXECUTANDO {side.upper()}", "warning")
                        self.log_brazil_time(f"🔮 Confiança: {prediction['confidence']:.2f}", "warning")
                        
                        result = self.execute_trade(side)
                        
                        if result.get('success'):
                            self.high_confidence_trades += 1
                            self.total_trades += 1
                    
                    elif self.debug_mode:
                        self.log_brazil_time(f"⏸️ Aguardando sinal melhor...", "info")
                        
            else:
                if self.debug_mode:
                    self.log_brazil_time(f"📊 Posição ativa: {self.position_side}", "info")
                
        except Exception as e:
            self.log_brazil_time(f"❌ Erro na estratégia: {e}", "error")

    def start_trading(self):
        """Start the trading bot"""
        self.is_running = True
        self.start_balance = self.get_account_balance()
        self.log_brazil_time(f"🚀 TRADING BOT INICIADO!", "warning")
        self.log_brazil_time(f"💰 Saldo inicial: ${self.start_balance:.2f}", "warning")
        
        # INICIAR MONITOR IMEDIATAMENTE
        self.start_position_monitor()
        
        while self.is_running:
            try:
                self.scalping_strategy()
                time.sleep(self.scalping_interval)
            except KeyboardInterrupt:
                self.log_brazil_time("⏹️ Bot interrompido pelo usuário", "warning")
                break
            except Exception as e:
                self.log_brazil_time(f"❌ Erro no loop principal: {e}", "error")
                time.sleep(5)

    def stop_trading(self):
        """Stop the trading bot"""
        self.is_running = False
        self.position_monitor_active = False
        self.emergency_stop = True
        
        # Fechar posições abertas
        if self.current_position:
            self.force_close_position_guaranteed("BOT_STOPPED")
        
        self.log_brazil_time("⏹️ TRADING BOT PARADO", "warning")

    def get_stats(self) -> Dict:
        """Get trading statistics"""
        current_balance = self.get_account_balance()
        profit_loss = current_balance - self.start_balance if self.start_balance > 0 else 0
        
        win_rate = (self.profitable_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': win_rate,
            'start_balance': self.start_balance,
            'current_balance': current_balance,
            'profit_loss': profit_loss,
            'stop_loss_triggered': self.stop_loss_triggered,
            'take_profit_triggered': self.take_profit_triggered,
            'forced_closes': self.forced_closes,
            'high_confidence_trades': self.high_confidence_trades
        }
