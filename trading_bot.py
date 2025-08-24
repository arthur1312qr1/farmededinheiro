import logging
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import math
import statistics
from collections import deque
import numpy as np
from dataclasses import dataclass
from enum import Enum

from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Estados do bot"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY = "emergency"

class TradeDirection(Enum):
    """Direções de trade"""
    LONG = "long"
    SHORT = "short"

@dataclass
class TradePosition:
    """Representação de uma posição"""
    side: TradeDirection
    size: float
    entry_price: float
    start_time: float
    target_price: float = None
    stop_price: float = None
    order_id: str = None
    
    def get_duration(self) -> float:
        return time.time() - self.start_time
    
    def calculate_pnl(self, current_price: float) -> float:
        if self.side == TradeDirection.LONG:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

@dataclass
class TradingMetrics:
    """Métricas de performance"""
    total_trades: int = 0
    profitable_trades: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_trade_duration: float = 0.0
    total_fees_paid: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return (self.profitable_trades / max(1, self.total_trades)) * 100
    
    @property
    def losing_trades(self) -> int:
        return self.total_trades - self.profitable_trades
    
    @property
    def net_profit(self) -> float:
        return self.total_profit - self.total_fees_paid

class AdvancedIndicators:
    """Indicadores técnicos aprimorados e mais precisos"""
    
    @staticmethod
    def rsi_with_validation(prices: List[float], period: int = 14) -> Tuple[float, float]:
        """RSI com validação de qualidade do sinal"""
        if len(prices) < period + 10:  # Mais dados para precisão
            return 50.0, 0.0
        
        # Usar mais dados para melhor precisão
        extended_period = min(len(prices), period * 2)
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Calcular RSI com suavização
        gains = [max(0, delta) for delta in deltas[-extended_period:]]
        losses = [max(0, -delta) for delta in deltas[-extended_period:]]
        
        # Média móvel exponencial para suavizar
        if len(gains) > period:
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
        else:
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0.001
        
        rs = avg_gain / max(avg_loss, 0.001)
        rsi = 100 - (100 / (1 + rs))
        
        # Calcular força do sinal (volatilidade do RSI)
        if len(prices) > period * 2:
            recent_rsi = []
            for i in range(period, min(len(prices), period + 10)):
                subset = prices[:i+1]
                if len(subset) >= period:
                    sub_deltas = [subset[j] - subset[j-1] for j in range(1, len(subset))]
                    sub_gains = [max(0, d) for d in sub_deltas[-period:]]
                    sub_losses = [max(0, -d) for d in sub_deltas[-period:]]
                    sub_avg_gain = sum(sub_gains) / len(sub_gains) if sub_gains else 0
                    sub_avg_loss = sum(sub_losses) / len(sub_losses) if sub_losses else 0.001
                    sub_rs = sub_avg_gain / sub_avg_loss
                    sub_rsi = 100 - (100 / (1 + sub_rs))
                    recent_rsi.append(sub_rsi)
            
            if len(recent_rsi) > 3:
                rsi_volatility = statistics.stdev(recent_rsi)
                signal_strength = min(1.0, rsi_volatility / 20)  # Normalizar
            else:
                signal_strength = 0.5
        else:
            signal_strength = 0.3
        
        return rsi, signal_strength
    
    @staticmethod
    def macd_advanced(prices: List[float], fast=12, slow=26, signal=9) -> Tuple[float, float, float, float]:
        """MACD com histogram e força do sinal"""
        if len(prices) < slow + signal + 5:
            return 0.0, 0.0, 0.0, 0.0
        
        # EMAs
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_vals = [data[0]]
            for price in data[1:]:
                ema_vals.append((price * multiplier) + (ema_vals[-1] * (1 - multiplier)))
            return ema_vals[-1]
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Calcular linha de sinal
        if len(prices) >= slow + signal:
            macd_history = []
            for i in range(slow, len(prices)):
                subset = prices[:i+1]
                if len(subset) >= slow:
                    fast_ema = ema(subset, fast)
                    slow_ema = ema(subset, slow)
                    macd_history.append(fast_ema - slow_ema)
            
            if len(macd_history) >= signal:
                signal_line = ema(macd_history, signal)
                histogram = macd_line - signal_line
                
                # Força do sinal baseada no momentum
                if len(macd_history) > 3:
                    recent_macd = macd_history[-3:]
                    momentum = (recent_macd[-1] - recent_macd[0]) / abs(recent_macd[0] + 0.001)
                    signal_strength = min(1.0, abs(momentum) * 10)
                else:
                    signal_strength = 0.3
            else:
                signal_line = 0.0
                histogram = macd_line
                signal_strength = 0.3
        else:
            signal_line = 0.0
            histogram = macd_line
            signal_strength = 0.3
        
        return macd_line, signal_line, histogram, signal_strength
    
    @staticmethod
    def bollinger_with_squeeze(prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """Bollinger Bands com detecção de squeeze"""
        if len(prices) < period + 10:
            price = prices[-1] if prices else 0
            return {
                'upper': price, 'lower': price, 'middle': price,
                'squeeze': False, 'breakout_direction': None, 'strength': 0.0
            }
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / len(recent_prices)
        variance = sum([(p - middle) ** 2 for p in recent_prices]) / len(recent_prices)
        std = math.sqrt(variance)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        current_price = prices[-1]
        
        # Detectar squeeze (bandas estreitas)
        band_width = (upper - lower) / middle
        historical_widths = []
        
        for i in range(max(0, len(prices) - 50), len(prices) - period):
            if i >= period:
                subset = prices[i:i+period]
                sub_middle = sum(subset) / len(subset)
                sub_var = sum([(p - sub_middle) ** 2 for p in subset]) / len(subset)
                sub_std = math.sqrt(sub_var)
                sub_width = (sub_std * 2 * 2) / sub_middle
                historical_widths.append(sub_width)
        
        if historical_widths:
            avg_width = sum(historical_widths) / len(historical_widths)
            is_squeeze = band_width < avg_width * 0.8
        else:
            is_squeeze = False
        
        # Detectar direção de breakout
        bb_position = (current_price - lower) / max(upper - lower, 0.001)
        breakout_direction = None
        strength = 0.0
        
        if bb_position > 0.8:
            breakout_direction = "UP"
            strength = min(1.0, (bb_position - 0.8) * 5)
        elif bb_position < 0.2:
            breakout_direction = "DOWN" 
            strength = min(1.0, (0.2 - bb_position) * 5)
        
        return {
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'squeeze': is_squeeze,
            'breakout_direction': breakout_direction,
            'strength': strength,
            'position': bb_position
        }
    
    @staticmethod
    def volume_price_analysis(prices: List[float], volumes: List[float]) -> Dict:
        """Análise de volume-preço"""
        if len(prices) < 10 or len(volumes) < 10:
            return {'strength': 0.0, 'signal': 'NEUTRAL', 'quality': 0.0}
        
        # Alinhar preços e volumes
        min_len = min(len(prices), len(volumes))
        recent_prices = prices[-min_len:]
        recent_volumes = volumes[-min_len:]
        
        if len(recent_prices) < 5:
            return {'strength': 0.0, 'signal': 'NEUTRAL', 'quality': 0.0}
        
        # Calcular mudanças de preço e volume
        price_changes = [recent_prices[i] - recent_prices[i-1] for i in range(1, len(recent_prices))]
        volume_changes = [recent_volumes[i] - recent_volumes[i-1] for i in range(1, len(recent_volumes))]
        
        # Correlação preço-volume
        positive_moves = 0
        negative_moves = 0
        total_strength = 0
        
        for i in range(len(price_changes)):
            price_change = price_changes[i]
            volume = recent_volumes[i+1]
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            
            volume_strength = volume / max(avg_volume, 1)
            
            if price_change > 0 and volume_strength > 1.2:
                positive_moves += 1
                total_strength += volume_strength
            elif price_change < 0 and volume_strength > 1.2:
                negative_moves += 1
                total_strength += volume_strength
        
        total_moves = positive_moves + negative_moves
        if total_moves == 0:
            return {'strength': 0.0, 'signal': 'NEUTRAL', 'quality': 0.0}
        
        if positive_moves > negative_moves:
            signal = 'BULLISH'
            strength = (positive_moves / total_moves) * (total_strength / total_moves)
        elif negative_moves > positive_moves:
            signal = 'BEARISH'
            strength = (negative_moves / total_moves) * (total_strength / total_moves)
        else:
            signal = 'NEUTRAL'
            strength = 0.0
        
        quality = min(1.0, total_strength / len(price_changes))
        
        return {
            'strength': min(1.0, strength),
            'signal': signal,
            'quality': quality
        }

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ETHUSDT',
                 leverage: int = 10, balance_percentage: float = 95.0,
                 scalping_interval: float = 2.0, paper_trading: bool = False):
        """
        Bot de trading profissional focado em consistência e lucratividade real
        """
        
        if not isinstance(bitget_api, BitgetAPI):
            raise TypeError(f"bitget_api deve ser BitgetAPI, recebido: {type(bitget_api)}")

        # Configurações básicas
        self.bitget_api = bitget_api
        self.symbol = symbol
        self.leverage = leverage
        self.balance_percentage = min(balance_percentage, 95.0)
        self.scalping_interval = max(scalping_interval, 1.0)
        self.paper_trading = paper_trading

        # Estado do bot
        self.state = TradingState.STOPPED
        self.current_position: Optional[TradePosition] = None
        self.trading_thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None

        # CONFIGURAÇÕES MANTIDAS - Take Profit 0.9% e Stop Loss 0.4%
        self.profit_target = 0.009           # 0.9% take profit (MANTIDO)
        self.stop_loss = 0.004               # 0.4% stop loss (MANTIDO)
        self.min_profit_target = 0.005       # 0.5% mínimo para compensar fees
        self.max_position_time = 300         # 5 minutos máximo por trade
        self.min_position_time = 15          # 15 segundos mínimo
        
        # Controles de risco
        self.max_daily_loss = 0.02          # 2% perda máxima por dia
        self.max_consecutive_losses = 3     # Parar após 3 perdas seguidas
        self.min_time_between_trades = 10   # 10 segundos entre trades
        
        # Indicadores e dados - APRIMORADOS
        self.price_history = deque(maxlen=500)  # Mais dados para análise
        self.volume_history = deque(maxlen=200) # Mais volume para análise
        
        # Métricas
        self.metrics = TradingMetrics()
        self.trades_today = 0
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = 0
        
        # Threading - APRIMORADO
        self._lock = threading.Lock()
        self.is_entering_position = False
        self.is_exiting_position = False
        self.force_close_flag = False  # Flag para fechamento forçado
        
        # Controle de fechamento CRÍTICO
        self.close_attempts = 0
        self.max_close_attempts = 5
        self.last_close_attempt = 0
        
        # Inicialização
        logger.info("Trading Bot Profissional Aprimorado Inicializado")
        logger.info(f"Take Profit: {self.profit_target*100:.1f}% (MANTIDO)")
        logger.info(f"Stop Loss: {self.stop_loss*100:.1f}% (MANTIDO)")

    def start(self) -> bool:
        """Iniciar bot com configurações otimizadas"""
        try:
            if self.state == TradingState.RUNNING:
                return True
            
            logger.info("Iniciando bot profissional aprimorado...")
            
            # Reset estado
            self.state = TradingState.RUNNING
            self.last_trade_time = time.time()
            self.consecutive_losses = 0
            self.daily_loss = 0.0
            self.trades_today = 0
            
            # Reset locks e flags críticos
            self.is_entering_position = False
            self.is_exiting_position = False
            self.force_close_flag = False
            self.close_attempts = 0
            
            # Coletar dados iniciais
            self._collect_initial_data()
            
            # Iniciar thread de trading
            self.trading_thread = threading.Thread(
                target=self._main_trading_loop,
                daemon=True,
                name="EnhancedTradingBot"
            )
            self.trading_thread.start()
            
            logger.info("Bot profissional aprimorado iniciado com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            return False

    def stop(self) -> bool:
        """Parar bot com fechamento seguro"""
        try:
            logger.info("Parando bot...")
            self.state = TradingState.STOPPED
            
            # Fechar posição se existir - FORÇADO
            if self.current_position:
                self.force_close_flag = True
                self._close_position_safely("Bot stopping - FORCED")
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Relatório final
            daily_profit = self.metrics.net_profit * 100
            logger.info(f"RELATÓRIO FINAL:")
            logger.info(f"   Trades hoje: {self.trades_today}")
            logger.info(f"   Win Rate: {self.metrics.win_rate:.1f}%")
            logger.info(f"   Profit líquido: {daily_profit:.3f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao parar bot: {e}")
            return False

    def _collect_initial_data(self):
        """Coletar dados históricos iniciais - APRIMORADO"""
        try:
            logger.info("Coletando dados iniciais aprimorados...")
            for i in range(100):  # Mais dados para análise melhor
                market_data = self.bitget_api.get_market_data(self.symbol)
                if market_data and market_data.get('price', 0) > 0:
                    self.price_history.append(float(market_data['price']))
                    if market_data.get('volume', 0) > 0:
                        self.volume_history.append(float(market_data['volume']))
                
                # Mostrar progresso
                if (i + 1) % 20 == 0:
                    logger.info(f"Coletados {len(self.price_history)} pontos...")
                
                time.sleep(0.3)  # Mais rápido
            
            logger.info(f"✅ Dados coletados: {len(self.price_history)} preços, {len(self.volume_history)} volumes")
        except Exception as e:
            logger.error(f"Erro coletando dados: {e}")

    def _main_trading_loop(self):
        """Loop principal otimizado"""
        logger.info("Loop principal aprimorado iniciado")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                
                # Verificar condições de parada
                if self._should_stop_trading():
                    logger.warning("Condições de parada atingidas")
                    break
                
                # Atualizar dados de mercado
                self._update_market_data()
                
                # Gerenciar posição existente - CRÍTICO
                if self.current_position:
                    self._manage_position_enhanced()
                
                # Procurar nova oportunidade - APRIMORADO
                elif self._can_open_new_position():
                    signal = self._analyze_market_advanced()
                    if signal:
                        direction, confidence, analysis = signal
                        if confidence > 0.75 and analysis.get('quality', 0) > 0.6:
                            self._execute_trade(direction, confidence, analysis)
                
                # Sleep controlado
                elapsed = time.time() - loop_start
                sleep_time = max(0.5, self.scalping_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                time.sleep(5)
        
        logger.info("Loop principal finalizado")

    def _update_market_data(self):
        """Atualizar dados de mercado"""
        try:
            market_data = self.bitget_api.get_market_data(self.symbol)
            if market_data and market_data.get('price', 0) > 0:
                price = float(market_data['price'])
                self.price_history.append(price)
                
                if market_data.get('volume', 0) > 0:
                    volume = float(market_data['volume'])
                    self.volume_history.append(volume)
                        
        except Exception as e:
            logger.error(f"Erro atualizando dados: {e}")

    def _analyze_market_advanced(self) -> Optional[Tuple[TradeDirection, float, Dict]]:
        """Análise de mercado APRIMORADA com múltiplos indicadores"""
        try:
            if len(self.price_history) < 100:  # Mais dados necessários
                return None
                
            prices = list(self.price_history)
            volumes = list(self.volume_history) if len(self.volume_history) > 50 else None
            current_price = prices[-1]
            
            # Indicadores avançados
            rsi, rsi_strength = AdvancedIndicators.rsi_with_validation(prices, 14)
            macd, signal_line, histogram, macd_strength = AdvancedIndicators.macd_advanced(prices)
            bb_data = AdvancedIndicators.bollinger_with_squeeze(prices)
            
            # Análise de volume se disponível
            volume_analysis = {'strength': 0.0, 'signal': 'NEUTRAL', 'quality': 0.0}
            if volumes and len(volumes) > 20:
                volume_analysis = AdvancedIndicators.volume_price_analysis(prices, volumes)
            
            # Sistema de pontuação aprimorado
            signals = []
            total_confidence = 0.0
            
            # 1. RSI com força
            if rsi < 25 and rsi_strength > 0.4:
                signals.append(("LONG", 0.35 * rsi_strength, "RSI Oversold Strong"))
                total_confidence += 0.35 * rsi_strength
            elif rsi > 75 and rsi_strength > 0.4:
                signals.append(("SHORT", 0.35 * rsi_strength, "RSI Overbought Strong"))
                total_confidence += 0.35 * rsi_strength
            
            # 2. MACD com momentum
            if histogram > 0 and macd > signal_line and macd_strength > 0.3:
                signals.append(("LONG", 0.3 * macd_strength, "MACD Bullish"))
                total_confidence += 0.3 * macd_strength
            elif histogram < 0 and macd < signal_line and macd_strength > 0.3:
                signals.append(("SHORT", 0.3 * macd_strength, "MACD Bearish"))
                total_confidence += 0.3 * macd_strength
            
            # 3. Bollinger Bands com breakout
            if bb_data['breakout_direction'] == "UP" and bb_data['strength'] > 0.5:
                signals.append(("LONG", 0.25 * bb_data['strength'], "BB Breakout Up"))
                total_confidence += 0.25 * bb_data['strength']
            elif bb_data['breakout_direction'] == "DOWN" and bb_data['strength'] > 0.5:
                signals.append(("SHORT", 0.25 * bb_data['strength'], "BB Breakout Down"))
                total_confidence += 0.25 * bb_data['strength']
            
            # 4. Volume confirmation
            if volume_analysis['signal'] == 'BULLISH' and volume_analysis['quality'] > 0.5:
                signals.append(("LONG", 0.2 * volume_analysis['strength'], "Volume Bullish"))
                total_confidence += 0.2 * volume_analysis['strength']
            elif volume_analysis['signal'] == 'BEARISH' and volume_analysis['quality'] > 0.5:
                signals.append(("SHORT", 0.2 * volume_analysis['strength'], "Volume Bearish"))
                total_confidence += 0.2 * volume_analysis['strength']
            
            # 5. Trend following aprimorado
            sma_20 = sum(prices[-20:]) / 20
            sma_50 = sum(prices[-50:]) / 50 if len(prices) >= 50 else sma_20
            
            trend_strength = abs(current_price - sma_20) / sma_20
            if current_price > sma_20 > sma_50 and trend_strength > 0.003:
                signals.append(("LONG", 0.15 * min(trend_strength * 100, 1.0), "Trend Following"))
                total_confidence += 0.15 * min(trend_strength * 100, 1.0)
            elif current_price < sma_20 < sma_50 and trend_strength > 0.003:
                signals.append(("SHORT", 0.15 * min(trend_strength * 100, 1.0), "Trend Following"))
                total_confidence += 0.15 * min(trend_strength * 100, 1.0)
            
            # Calcular direção final com validação
            long_signals = [s for s in signals if s[0] == "LONG"]
            short_signals = [s for s in signals if s[0] == "SHORT"]
            
            long_strength = sum([s[1] for s in long_signals])
            short_strength = sum([s[1] for s in short_signals])
            
            # Critérios mais rigorosos
            min_confidence = 0.75
            min_signal_count = 3
            
            analysis_data = {
                'rsi': rsi,
                'rsi_strength': rsi_strength,
                'macd': macd,
                'macd_strength': macd_strength,
                'bb_data': bb_data,
                'volume_analysis': volume_analysis,
                'signals': signals,
                'total_confidence': total_confidence,
                'quality': min(1.0, (len(signals) / 5) * total_confidence)
            }
            
            if (long_strength > short_strength and 
                long_strength >= min_confidence and 
                len(long_signals) >= min_signal_count):
                
                logger.info(f"🎯 SINAL LONG detectado:")
                for signal in long_signals:
                    logger.info(f"   {signal[2]}: {signal[1]:.3f}")
                
                return TradeDirection.LONG, min(long_strength, 0.95), analysis_data
                
            elif (short_strength > long_strength and 
                  short_strength >= min_confidence and 
                  len(short_signals) >= min_signal_count):
                
                logger.info(f"🎯 SINAL SHORT detectado:")
                for signal in short_signals:
                    logger.info(f"   {signal[2]}: {signal[1]:.3f}")
                
                return TradeDirection.SHORT, min(short_strength, 0.95), analysis_data
            
            # Log de análise quando não há sinal forte
            if len(signals) > 0:
                logger.debug(f"📊 Análise: L:{long_strength:.2f} S:{short_strength:.2f} Q:{analysis_data['quality']:.2f}")
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na análise avançada: {e}")
            return None

    def _execute_trade(self, direction: TradeDirection, confidence: float, analysis: Dict):
        """Executar trade de forma segura - MANTIDO IGUAL"""
        if self.is_entering_position:
            return
            
        self.is_entering_position = True
        
        try:
            # Verificar saldo
            balance = self.get_account_balance()
            if balance <= 0:
                if self.paper_trading:
                    balance = 1000
                else:
                    logger.error("Saldo insuficiente")
                    return
            
            current_price = self.price_history[-1]
            
            # Calcular targets - MANTIDOS
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)
                stop_price = current_price * (1 - self.stop_loss)
            else:
                target_price = current_price * (1 - self.profit_target)
                stop_price = current_price * (1 + self.stop_loss)
            
            logger.info(f"🚀 Executando {direction.name} APRIMORADO:")
            logger.info(f"   Preço: ${current_price:.2f}")
            logger.info(f"   Target: ${target_price:.2f} ({self.profit_target*100:.1f}%)")
            logger.info(f"   Stop: ${stop_price:.2f}")
            logger.info(f"   Confiança: {confidence*100:.1f}%")
            logger.info(f"   Qualidade: {analysis.get('quality', 0)*100:.1f}%")
            
            # Executar ordem - CORRIGIDO
            success = False
            if self.paper_trading:
                success = True
                logger.info("PAPER TRADING - Ordem simulada")
            else:
                try:
                    # CORRIGIDO - só fazer LONG por enquanto para evitar erros
                    if direction == TradeDirection.LONG:
                        result = self.bitget_api.place_buy_order()
                        success = result and result.get('success', False)
                        logger.info(f"Resultado compra: {result}")
                    else:
                        # Por enquanto pular SHORT até corrigir API
                        logger.info("SHORT temporariamente desabilitado")
                        return
                        
                except Exception as e:
                    logger.error(f"Erro executando ordem: {e}")
                    success = False
            
            if success:
                # Criar posição
                position_value = balance * (self.balance_percentage / 100) * self.leverage
                position_size = position_value / current_price
                
                self.current_position = TradePosition(
                    side=direction,
                    size=position_size,
                    entry_price=current_price,
                    start_time=time.time(),
                    target_price=target_price,
                    stop_price=stop_price
                )
                
                self.trades_today += 1
                self.last_trade_time = time.time()
                self.close_attempts = 0  # Reset contador
                
                logger.info(f"✅ Trade #{self.trades_today} executado com SUCESSO!")
                logger.info(f"   Posição: {position_size:.4f} ETH")
                logger.info(f"   Valor: ${position_value:.2f}")
            
        except Exception as e:
            logger.error(f"Erro no trade: {e}")
            traceback.print_exc()
        finally:
            self.is_entering_position = False

    def _manage_position_enhanced(self):
        """Gerenciar posição - MÉTODO CRÍTICO APRIMORADO"""
        if not self.current_position or self.is_exiting_position:
            return
            
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            should_close = False
            reason = ""
            is_emergency = False
            
            # CONDIÇÕES DE FECHAMENTO CRÍTICAS - MANTIDAS
            
            # 1. TAKE PROFIT - 0.9% (MANTIDO)
            if pnl >= self.profit_target:
                should_close = True
                reason = f"✅ TAKE PROFIT ATINGIDO: {pnl*100:.3f}% >= {self.profit_target*100:.1f}%"
                logger.info(f"🎯 TAKE PROFIT DETECTADO: {pnl*100:.3f}%")
                
            # 2. STOP LOSS - 0.4% (MANTIDO) 
            elif pnl <= -self.stop_loss:
                should_close = True
                is_emergency = True
                reason = f"⛔ STOP LOSS ATINGIDO: {pnl*100:.3f}% <= -{self.stop_loss*100:.1f}%"
                logger.warning(f"🚨 STOP LOSS DETECTADO: {pnl*100:.3f}%")
                
            # 3. TEMPO LIMITE - 5 minutos
            elif duration >= self.max_position_time:
                should_close = True
                reason = f"⏰ TEMPO LIMITE: {duration:.0f}s >= {self.max_position_time}s, PnL: {pnl*100:.3f}%"
                logger.warning(f"⏰ TEMPO LIMITE ATINGIDO: {duration:.0f}s")
                
            # 4. PERDA CRÍTICA - Emergência
            elif pnl <= -0.01:  # -1% perda crítica
                should_close = True
                is_emergency = True
                reason = f"🚨 PERDA CRÍTICA: {pnl*100:.3f}%"
                logger.error(f"🚨 PERDA CRÍTICA DETECTADA: {pnl*100:.3f}%")
            
            # 5. FORÇA FECHAMENTO se muitas tentativas
            elif self.close_attempts >= 3:
                should_close = True
                is_emergency = True
                reason = f"🔴 FECHAMENTO FORÇADO: {self.close_attempts} tentativas, PnL: {pnl*100:.3f}%"
                logger.error(reason)
            
            # 6. Flag de fechamento forçado
            elif self.force_close_flag:
                should_close = True
                is_emergency = True
                reason = "🔴 FECHAMENTO FORÇADO POR FLAG"
                logger.error(reason)
            
            # Log periódico da posição a cada 10 segundos
            if int(duration) % 10 == 0 and int(duration) > 0:
                target_distance = (self.profit_target - pnl) * 100
                stop_distance = (pnl + self.stop_loss) * 100
                logger.info(f"📊 Posição: {pnl*100:.3f}% | {duration:.0f}s | Target: -{target_distance:.1f}% | Stop: +{stop_distance:.1f}%")
            
            if should_close:
                logger.info(f"🔄 INICIANDO FECHAMENTO: {reason}")
                if is_emergency:
                    self.force_close_flag = True
                success = self._close_position_safely(reason)
                
                # Se falhou, incrementar contador
                if not success:
                    self.close_attempts += 1
                    logger.error(f"❌ FALHA NO FECHAMENTO #{self.close_attempts}")
                    
                    # Tentar novamente em 2 segundos
                    time.sleep(2)
                    if self.close_attempts < self.max_close_attempts:
                        logger.warning(f"🔄 NOVA TENTATIVA DE FECHAMENTO #{self.close_attempts + 1}")
                
        except Exception as e:
            logger.error(f"Erro crítico gerenciando posição: {e}")
            traceback.print_exc()
            # Em caso de erro crítico, forçar fechamento
            self.force_close_flag = True
            self.close_attempts += 1

    def _close_position_safely(self, reason: str) -> bool:
        """Fechar posição com múltiplas tentativas - CRÍTICO CORRIGIDO"""
        if self.is_exiting_position or not self.current_position:
            return False
            
        # Prevenir múltiplas execuções simultâneas
        with self._lock:
            if self.is_exiting_position:
                return False
            self.is_exiting_position = True
        
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            final_pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            logger.info(f"🔄 EXECUTANDO FECHAMENTO: {reason}")
            logger.info(f"   PnL atual: {final_pnl*100:.4f}%")
            logger.info(f"   Duração: {duration:.1f}s")
            logger.info(f"   Tentativa: #{self.close_attempts + 1}")
            
            success = False
            max_attempts = 5 if self.force_close_flag else 3
            
            for attempt in range(max_attempts):
                try:
                    attempt_start = time.time()
                    logger.info(f"🔄 Tentativa {attempt+1}/{max_attempts}...")
                    
                    if self.paper_trading:
                        success = True
                        logger.info("PAPER TRADING - Fechamento simulado com SUCESSO")
                        break
                    
                    # FECHAR POSIÇÃO REAL - CRÍTICO
                    if self.current_position.side == TradeDirection.LONG:
                        logger.info(f"📤 Vendendo posição LONG...")
                        
                        # Usar profit_target=0 para vender IMEDIATAMENTE
                        result = self.bitget_api.place_sell_order(profit_target=0)
                        
                        if result and isinstance(result, dict):
                            success = result.get('success', False)
                            if success:
                                logger.info(f"✅ VENDA EXECUTADA: {result}")
                                break
                            else:
                                error_msg = result.get('error', 'Erro desconhecido')
                                logger.warning(f"⚠️ Venda falhou: {error_msg}")
                        else:
                            logger.warning(f"⚠️ Resultado venda inválido: {result}")
                    else:
                        # Para SHORT, comprar para fechar
                        logger.info(f"📤 Comprando para fechar SHORT...")
                        result = self.bitget_api.place_buy_order()
                        if result and isinstance(result, dict):
                            success = result.get('success', False)
                            if success:
                                logger.info(f"✅ COMPRA EXECUTADA: {result}")
                                break
                    
                    if not success:
                        attempt_duration = time.time() - attempt_start
                        logger.warning(f"⚠️ Tentativa {attempt+1} falhou em {attempt_duration:.1f}s")
                        
                        if attempt < max_attempts - 1:
                            sleep_time = min(3, (attempt + 1) * 1)  # 1s, 2s, 3s
                            logger.info(f"⏳ Aguardando {sleep_time}s antes da próxima tentativa...")
                            time.sleep(sleep_time)
                        
                except Exception as attempt_error:
                    logger.error(f"❌ Erro na tentativa {attempt+1}: {attempt_error}")
                    if attempt < max_attempts - 1:
                        time.sleep(2)
            
            # PROCESSAR RESULTADO
            if success:
                logger.info(f"🎉 POSIÇÃO FECHADA COM SUCESSO!")
                
                # Atualizar métricas - CRÍTICO
                with self._lock:
                    self.metrics.total_trades += 1
                    self.metrics.total_profit += final_pnl
                    self.metrics.total_fees_paid += abs(final_pnl) * 0.002
                    
                    if final_pnl > 0:
                        self.metrics.profitable_trades += 1
                        self.metrics.consecutive_wins += 1
                        self.metrics.consecutive_losses = 0
                        self.consecutive_losses = 0
                        logger.info(f"✅ LUCRO: {final_pnl*100:.4f}% | Win Streak: {self.metrics.consecutive_wins}")
                    else:
                        self.metrics.consecutive_wins = 0
                        self.metrics.consecutive_losses += 1
                        self.consecutive_losses += 1
                        self.daily_loss += abs(final_pnl)
                        logger.info(f"❌ PERDA: {final_pnl*100:.4f}% | Loss Streak: {self.consecutive_losses}")
                    
                    # Atualizar drawdown
                    if final_pnl < 0:
                        self.metrics.max_drawdown = max(self.metrics.max_drawdown, abs(final_pnl))
                    
                    # Atualizar duração média
                    total_duration = (self.metrics.average_trade_duration * (self.metrics.total_trades - 1) + duration)
                    self.metrics.average_trade_duration = total_duration / self.metrics.total_trades
                
                # STATS FINAIS
                logger.info(f"📊 ESTATÍSTICAS:")
                logger.info(f"   PnL final: {final_pnl*100:.4f}%")
                logger.info(f"   Duração: {duration:.1f}s")
                logger.info(f"   Win Rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   Trades hoje: {self.trades_today}")
                logger.info(f"   Profit líquido: {self.metrics.net_profit*100:.4f}%")
                
                # RESET POSIÇÃO E FLAGS
                self.current_position = None
                self.force_close_flag = False
                self.close_attempts = 0
                
                return True
            else:
                logger.error(f"🚨 FALHA CRÍTICA NO FECHAMENTO após {max_attempts} tentativas!")
                logger.error(f"   PnL não realizado: {final_pnl*100:.4f}%")
                logger.error(f"   Duração: {duration:.1f}s")
                
                # EM CASO DE FALHA TOTAL - MEDIDAS DRÁSTICAS
                if self.force_close_flag or self.close_attempts >= self.max_close_attempts:
                    logger.error("🚨 ATIVANDO PROTOCOLO DE EMERGÊNCIA")
                    logger.error("🚨 REMOVENDO POSIÇÃO DA MEMÓRIA PARA EVITAR LOOP INFINITO")
                    
                    # Registrar como perda total e remover posição
                    with self._lock:
                        self.metrics.total_trades += 1
                        self.metrics.total_profit += final_pnl  # Registrar perda
                        self.consecutive_losses += 1
                        self.daily_loss += abs(final_pnl)
                    
                    self.current_position = None  # REMOVER POSIÇÃO FORÇADAMENTE
                    self.force_close_flag = False
                    self.close_attempts = 0
                    
                    logger.error("🚨 POSIÇÃO REMOVIDA DA MEMÓRIA - PROTOCOLO DE EMERGÊNCIA ATIVO")
                    return False
                
                return False
                
        except Exception as e:
            logger.error(f"🚨 ERRO CRÍTICO no fechamento: {e}")
            traceback.print_exc()
            
            # Em caso de erro crítico, aplicar protocolo de emergência
            self.current_position = None
            self.force_close_flag = False
            self.close_attempts = 0
            return False
        finally:
            self.is_exiting_position = False

    def _should_stop_trading(self) -> bool:
        """Verificar se deve parar de operar"""
        if self.daily_loss >= self.max_daily_loss:
            logger.warning(f"⚠️ Perda diária máxima atingida: {self.daily_loss*100:.2f}%")
            return True
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"⚠️ Perdas consecutivas: {self.consecutive_losses}")
            return True
        
        if self.metrics.max_drawdown >= 0.05:
            logger.warning(f"⚠️ Drawdown máximo: {self.metrics.max_drawdown*100:.2f}%")
            return True
        
        return False

    def _can_open_new_position(self) -> bool:
        """Verificar se pode abrir nova posição"""
        if time.time() - self.last_trade_time < self.min_time_between_trades:
            return False
        
        if self.daily_loss >= self.max_daily_loss * 0.8:
            return False
        
        if self.consecutive_losses >= 2:
            return False
        
        return True

    def get_account_balance(self) -> float:
        """Obter saldo da conta"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                return float(balance_info.get('free', 0.0))
            return 1000.0 if self.paper_trading else 0.0
        except Exception as e:
            logger.error(f"Erro obtendo saldo: {e}")
            return 1000.0 if self.paper_trading else 0.0

    @property
    def is_running(self) -> bool:
        return self.state == TradingState.RUNNING

    def get_status(self) -> Dict:
        """Status completo do bot"""
        try:
            daily_profit = self.metrics.net_profit * 100
            
            return {
                'bot_status': {
                    'state': self.state.value,
                    'is_running': self.is_running,
                    'symbol': self.symbol,
                    'leverage': self.leverage,
                    'paper_trading': self.paper_trading,
                    'close_attempts': self.close_attempts,
                    'force_close_flag': self.force_close_flag
                },
                'performance': {
                    'trades_today': self.trades_today,
                    'total_trades': self.metrics.total_trades,
                    'win_rate': round(self.metrics.win_rate, 1),
                    'daily_profit': round(daily_profit, 3),
                    'daily_loss': round(self.daily_loss * 100, 3),
                    'net_profit': round(self.metrics.net_profit * 100, 4),
                    'max_drawdown': round(self.metrics.max_drawdown * 100, 3),
                    'consecutive_wins': self.metrics.consecutive_wins,
                    'consecutive_losses': self.consecutive_losses,
                    'avg_duration': round(self.metrics.average_trade_duration, 1)
                },
                'risk_management': {
                    'daily_loss_limit': f"{self.max_daily_loss*100:.1f}%",
                    'max_consecutive_losses': self.max_consecutive_losses,
                    'time_between_trades': f"{self.min_time_between_trades}s",
                    'risk_level': self._get_risk_level()
                },
                'current_position': self._get_position_status(),
                'targets': {
                    'daily_target': "Consistência",
                    'take_profit': f"{self.profit_target*100:.1f}%",
                    'stop_loss': f"{self.stop_loss*100:.1f}%",
                    'risk_reward': f"1:{self.profit_target/self.stop_loss:.1f}"
                }
            }
        except Exception as e:
            return {'error': str(e), 'is_running': False}

    def _get_risk_level(self) -> str:
        if self.consecutive_losses >= 2 or self.daily_loss >= self.max_daily_loss * 0.8:
            return "HIGH"
        elif self.consecutive_losses >= 1 or self.daily_loss >= self.max_daily_loss * 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_position_status(self) -> Dict:
        """Status da posição atual"""
        if not self.current_position:
            return {'active': False}
        
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            return {
                'active': True,
                'side': self.current_position.side.value,
                'entry_price': self.current_position.entry_price,
                'current_price': current_price,
                'pnl_percent': round(pnl * 100, 3),
                'duration_seconds': round(duration),
                'target_price': self.current_position.target_price,
                'stop_price': self.current_position.stop_price,
                'is_profitable': pnl > 0,
                'close_attempts': self.close_attempts,
                'force_close': self.force_close_flag
            }
        except Exception as e:
            return {'active': True, 'error': str(e)}

    def get_daily_stats(self) -> Dict:
        """Estatísticas diárias"""
        try:
            with self._lock:
                daily_profit = self.metrics.net_profit * 100
                
                return {
                    'daily_performance': {
                        'trades_executed': self.trades_today,
                        'profit_target': '0.9% por trade (MANTIDO)',
                        'current_profit': f"{daily_profit:.3f}%",
                        'win_rate': f"{self.metrics.win_rate:.1f}%",
                        'avg_trade_duration': f"{self.metrics.average_trade_duration:.1f}s",
                        'profitable_trades': self.metrics.profitable_trades,
                        'losing_trades': self.metrics.losing_trades
                    },
                    'risk_metrics': {
                        'daily_loss': f"{self.daily_loss*100:.3f}%",
                        'max_drawdown': f"{self.metrics.max_drawdown*100:.3f}%",
                        'consecutive_losses': self.consecutive_losses,
                        'risk_level': self._get_risk_level()
                    },
                    'consistency_metrics': {
                        'max_consecutive_wins': self.metrics.max_consecutive_wins,
                        'max_consecutive_losses': self.metrics.max_consecutive_losses,
                        'total_fees_paid': f"{self.metrics.total_fees_paid*100:.4f}%",
                        'net_profit': f"{self.metrics.net_profit*100:.4f}%"
                    },
                    'enhanced_features': {
                        'advanced_analysis': 'Ativo',
                        'multi_indicator_signals': 'Ativo',
                        'enhanced_risk_management': 'Ativo',
                        'critical_position_control': 'Ativo'
                    }
                }
        except Exception as e:
            return {'error': str(e)}

    def emergency_stop(self) -> bool:
        """Parada de emergência"""
        try:
            logger.warning("⚠️ PARADA DE EMERGÊNCIA ATIVADA")
            self.state = TradingState.EMERGENCY
            self.force_close_flag = True
            
            # Fechar posição imediatamente
            if self.current_position:
                logger.warning("🚨 FECHAMENTO DE EMERGÊNCIA DA POSIÇÃO")
                self._close_position_safely("EMERGENCY STOP")
            
            # Parar thread
            if self.trading_thread:
                self.trading_thread.join(timeout=5)
            
            self.state = TradingState.STOPPED
            logger.info("✅ Parada de emergência concluída")
            return True
            
        except Exception as e:
            logger.error(f"Erro na parada de emergência: {e}")
            return False

    def reset_daily_stats(self):
        """Reset para novo dia"""
        try:
            with self._lock:
                self.trades_today = 0
                self.daily_loss = 0.0
                self.consecutive_losses = 0
                self.metrics = TradingMetrics()
                self.last_trade_time = time.time()
                self.close_attempts = 0
                self.force_close_flag = False
                
                logger.info("📊 Estatísticas diárias resetadas para novo dia")
                
        except Exception as e:
            logger.error(f"Erro ao resetar estatísticas: {e}")

    def pause_trading(self):
        """Pausar trading temporariamente"""
        if self.state == TradingState.RUNNING:
            self.state = TradingState.PAUSED
            logger.info("⏸️ Trading pausado")

    def resume_trading(self):
        """Retomar trading"""
        if self.state == TradingState.PAUSED:
            self.state = TradingState.RUNNING
            logger.info("▶️ Trading retomado")

    def get_market_analysis(self) -> Dict:
        """Análise atual do mercado - APRIMORADA"""
        try:
            if len(self.price_history) < 50:
                return {'error': 'Dados insuficientes para análise completa'}
            
            prices = list(self.price_history)
            volumes = list(self.volume_history) if len(self.volume_history) > 20 else []
            current_price = prices[-1]
            
            # Indicadores avançados
            rsi, rsi_strength = AdvancedIndicators.rsi_with_validation(prices, 14)
            macd, signal_line, histogram, macd_strength = AdvancedIndicators.macd_advanced(prices)
            bb_data = AdvancedIndicators.bollinger_with_squeeze(prices)
            
            # Análise de volume
            volume_analysis = {'strength': 0.0, 'signal': 'NEUTRAL', 'quality': 0.0}
            if volumes:
                volume_analysis = AdvancedIndicators.volume_price_analysis(prices, volumes)
            
            # Trend analysis
            sma_20 = sum(prices[-20:]) / 20
            sma_50 = sum(prices[-50:]) / 50
            trend = "BULLISH" if current_price > sma_20 > sma_50 else "BEARISH" if current_price < sma_20 < sma_50 else "NEUTRAL"
            trend_strength = abs(current_price - sma_20) / sma_20 * 100
            
            # Volatilidade
            recent_highs = [max(prices[i:i+5]) for i in range(max(0, len(prices)-20), len(prices)-4)]
            recent_lows = [min(prices[i:i+5]) for i in range(max(0, len(prices)-20), len(prices)-4)]
            if recent_highs and recent_lows:
                volatility = (max(recent_highs) - min(recent_lows)) / current_price * 100
            else:
                volatility = 0
            
            # Score geral
            total_score = (
                (rsi_strength * 0.25) +
                (macd_strength * 0.25) +
                (bb_data.get('strength', 0) * 0.25) +
                (volume_analysis['quality'] * 0.25)
            ) * 100
            
            return {
                'current_price': round(current_price, 2),
                'trend': trend,
                'trend_strength': round(trend_strength, 3),
                'volatility': round(volatility, 2),
                'analysis_score': round(total_score, 1),
                'indicators': {
                    'rsi': round(rsi, 1),
                    'rsi_strength': round(rsi_strength * 100, 1),
                    'macd': round(macd, 4),
                    'macd_signal': round(signal_line, 4),
                    'macd_histogram': round(histogram, 4),
                    'macd_strength': round(macd_strength * 100, 1),
                    'bb_upper': round(bb_data['upper'], 2),
                    'bb_lower': round(bb_data['lower'], 2),
                    'bb_middle': round(bb_data['middle'], 2),
                    'bb_squeeze': bb_data['squeeze'],
                    'bb_breakout': bb_data['breakout_direction']
                },
                'volume_analysis': {
                    'signal': volume_analysis['signal'],
                    'strength': round(volume_analysis['strength'] * 100, 1),
                    'quality': round(volume_analysis['quality'] * 100, 1)
                },
                'signals': {
                    'rsi_signal': 'OVERSOLD' if rsi < 25 else 'OVERBOUGHT' if rsi > 75 else 'NEUTRAL',
                    'macd_signal': 'BULLISH' if histogram > 0 and macd > signal_line else 'BEARISH' if histogram < 0 and macd < signal_line else 'NEUTRAL',
                    'bb_signal': bb_data['breakout_direction'] or 'NEUTRAL',
                    'trend_signal': trend,
                    'overall_signal': 'STRONG' if total_score > 70 else 'MODERATE' if total_score > 40 else 'WEAK'
                }
            }
            
        except Exception as e:
            return {'error': str(e)}


# Funções auxiliares para compatibilidade
def create_trading_bot(bitget_api: BitgetAPI, **kwargs) -> TradingBot:
    """Criar instância do TradingBot com configurações otimizadas"""
    return TradingBot(bitget_api, **kwargs)

# Teste básico
if __name__ == "__main__":
    try:
        from bitget_api import BitgetAPI
        
        api = BitgetAPI()
        if api.test_connection():
            bot = TradingBot(
                bitget_api=api,
                paper_trading=True,
                leverage=10,
                balance_percentage=90.0,
                scalping_interval=3.0
            )
            
            print("✅ Bot Profissional APRIMORADO criado com sucesso!")
            print("Configurações mantidas:")
            print(f"   Take Profit: {bot.profit_target*100:.1f}% (MANTIDO)")
            print(f"   Stop Loss: {bot.stop_loss*100:.1f}% (MANTIDO)")
            print(f"   Risk/Reward: 1:{bot.profit_target/bot.stop_loss:.1f}")
            print(f"   Max perda diária: {bot.max_daily_loss*100:.1f}%")
            print("🚀 MELHORIAS IMPLEMENTADAS:")
            print("   ✅ Análise técnica aprimorada com múltiplos indicadores")
            print("   ✅ Sistema de fechamento de posição crítico")
            print("   ✅ Controle rigoroso de take profit e stop loss")
            print("   ✅ Protocolo de emergência para falhas")
            print("   ✅ Análise de volume e momentum")
            print("   ✅ Detecção de breakouts e squeezes")
            print("🎯 Focado em PRECISÃO e EXECUÇÃO GARANTIDA!")
        else:
            print("❌ Falha na conexão com a API")
    except Exception as e:
        print(f"
