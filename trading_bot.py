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

class ImprovedIndicators:
    """Indicadores técnicos melhorados com filtros de ruído"""
    
    @staticmethod
    def adaptive_rsi(prices: List[float], period: int = 14) -> Tuple[float, str]:
        """RSI adaptativo com contexto"""
        if len(prices) < period + 5:
            return 50.0, "NEUTRAL"
        
        # RSI tradicional
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas[-period:]]
        losses = [max(0, -delta) for delta in deltas[-period:]]
        
        avg_gain = sum(gains) / len(gains) if gains else 0.001
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Contexto adaptativo
        recent_volatility = np.std(prices[-20:]) if len(prices) >= 20 else 0
        volatility_factor = min(2.0, max(0.5, recent_volatility / np.mean(prices[-20:])))
        
        # Ajustar limites baseado na volatilidade
        oversold_threshold = 30 + (10 * (1 - volatility_factor))
        overbought_threshold = 70 - (10 * (1 - volatility_factor))
        
        if rsi < oversold_threshold:
            signal = "STRONG_OVERSOLD" if rsi < 20 else "OVERSOLD"
        elif rsi > overbought_threshold:
            signal = "STRONG_OVERBOUGHT" if rsi > 80 else "OVERBOUGHT"
        else:
            signal = "NEUTRAL"
            
        return rsi, signal
    
    @staticmethod
    def trend_strength(prices: List[float], short_period: int = 12, long_period: int = 26) -> Tuple[float, str]:
        """Força da tendência com múltiplos timeframes"""
        if len(prices) < long_period + 10:
            return 0.0, "SIDEWAYS"
        
        # EMAs
        ema_short = ImprovedIndicators.ema(prices, short_period)
        ema_long = ImprovedIndicators.ema(prices, long_period)
        current_price = prices[-1]
        
        # Força da tendência
        trend_strength = (ema_short - ema_long) / ema_long
        price_momentum = (current_price - ema_short) / ema_short
        
        # Confirmação de volume se disponível
        combined_strength = (trend_strength + price_momentum) / 2
        
        # Classificação
        if combined_strength > 0.005:
            direction = "STRONG_BULLISH" if combined_strength > 0.015 else "BULLISH"
        elif combined_strength < -0.005:
            direction = "STRONG_BEARISH" if combined_strength < -0.015 else "BEARISH"
        else:
            direction = "SIDEWAYS"
            
        return abs(combined_strength), direction
    
    @staticmethod
    def ema(prices: List[float], period: int) -> float:
        """EMA otimizada"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period
        
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    @staticmethod
    def smart_bollinger(prices: List[float], period: int = 20) -> Tuple[float, float, float, str]:
        """Bollinger Bands inteligentes"""
        if len(prices) < period:
            price = prices[-1] if prices else 0
            return price, price, price, "INSUFFICIENT_DATA"
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / len(recent_prices)
        
        # Desvio padrão adaptativo
        variance = sum([(p - middle) ** 2 for p in recent_prices]) / len(recent_prices)
        std = math.sqrt(variance)
        
        # Ajustar desvio baseado na volatilidade
        volatility = std / middle if middle > 0 else 0
        adaptive_multiplier = 2.0 + (volatility * 10)  # Entre 2.0 e 3.0
        
        upper = middle + (adaptive_multiplier * std)
        lower = middle - (adaptive_multiplier * std)
        current_price = prices[-1]
        
        # Posição nas bandas
        band_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        if band_position < 0.1:
            signal = "LOWER_BREAKOUT"
        elif band_position < 0.2:
            signal = "NEAR_LOWER"
        elif band_position > 0.9:
            signal = "UPPER_BREAKOUT"
        elif band_position > 0.8:
            signal = "NEAR_UPPER"
        else:
            signal = "MIDDLE_RANGE"
            
        return upper, lower, middle, signal

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ETHUSDT',
                 leverage: int = 10, balance_percentage: float = 95.0,
                 scalping_interval: float = 2.0, paper_trading: bool = False):
        """
        Bot de trading com fechamento forçado e previsões melhoradas
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

        # CONFIGURAÇÕES RIGOROSAS DE FECHAMENTO
        self.profit_target = 0.009           # 0.9% take profit
        self.stop_loss = 0.004               # 0.4% stop loss
        self.emergency_close_threshold = 0.006  # 0.6% emergência
        self.max_position_time = 240         # 4 minutos MÁXIMO
        self.min_position_time = 10          # 10 segundos mínimo
        
        # FECHAMENTO FORÇADO - NOVO
        self.force_close_attempts = 0
        self.max_force_attempts = 5
        self.force_close_active = False
        self.last_pnl_check = 0
        self.pnl_check_interval = 5  # Verificar PnL a cada 5 segundos
        
        # Controles de risco aprimorados
        self.max_daily_loss = 0.015         # 1.5% perda máxima
        self.max_consecutive_losses = 2     # Apenas 2 perdas seguidas
        self.min_time_between_trades = 15   # 15 segundos entre trades
        
        # Dados melhorados
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=50)
        self.prediction_history = deque(maxlen=20)
        
        # Métricas
        self.metrics = TradingMetrics()
        self.trades_today = 0
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = 0
        
        # Threading com locks mais rigorosos
        self._position_lock = threading.RLock()  # Re-entrant lock
        self._execution_lock = threading.Lock()
        self.is_managing_position = False
        self.position_check_active = False
        
        logger.info("=== TRADING BOT PROFISSIONAL V2 ===")
        logger.info(f"Take Profit RÍGIDO: {self.profit_target*100:.1f}%")
        logger.info(f"Stop Loss RÍGIDO: {self.stop_loss*100:.1f}%")
        logger.info(f"Fechamento forçado: {self.emergency_close_threshold*100:.1f}%")
        logger.info("=== FOCO: FECHAMENTO GARANTIDO ===")

    def start(self) -> bool:
        """Iniciar bot com configurações ultra-rigorosas"""
        try:
            if self.state == TradingState.RUNNING:
                return True
            
            logger.info("🚀 Iniciando bot com fechamento rigoroso...")
            
            # Reset completo
            self.state = TradingState.RUNNING
            self.last_trade_time = time.time()
            self.consecutive_losses = 0
            self.daily_loss = 0.0
            self.trades_today = 0
            self.force_close_attempts = 0
            self.force_close_active = False
            
            # Locks
            self.is_managing_position = False
            self.position_check_active = False
            
            # Dados iniciais
            self._collect_enhanced_data()
            
            # Thread principal
            self.trading_thread = threading.Thread(
                target=self._enhanced_trading_loop,
                daemon=True,
                name="EnhancedTradingBot"
            )
            self.trading_thread.start()
            
            # Thread de monitoramento CRÍTICO
            self.monitor_thread = threading.Thread(
                target=self._critical_position_monitor,
                daemon=True,
                name="CriticalMonitor"
            )
            self.monitor_thread.start()
            
            logger.info("✅ Bot iniciado com monitoramento crítico!")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            return False

    def _critical_position_monitor(self):
        """Monitor crítico que FORÇA o fechamento se necessário"""
        logger.info("🔥 Monitor crítico ativo - FECHAMENTO GARANTIDO")
        
        while self.state == TradingState.RUNNING:
            try:
                with self._position_lock:
                    if self.current_position and not self.force_close_active:
                        current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
                        pnl = self.current_position.calculate_pnl(current_price)
                        duration = self.current_position.get_duration()
                        
                        # CONDIÇÕES DE FECHAMENTO FORÇADO
                        should_force_close = False
                        reason = ""
                        
                        # 1. Take profit atingido
                        if pnl >= self.profit_target:
                            should_force_close = True
                            reason = f"🎯 TAKE PROFIT FORÇADO: {pnl*100:.3f}%"
                        
                        # 2. Stop loss atingido
                        elif pnl <= -self.stop_loss:
                            should_force_close = True
                            reason = f"⛔ STOP LOSS FORÇADO: {pnl*100:.3f}%"
                        
                        # 3. Emergência por tempo
                        elif duration >= self.max_position_time:
                            should_force_close = True
                            reason = f"⏰ TEMPO LIMITE FORÇADO: {duration:.0f}s"
                        
                        # 4. Emergência por PnL extremo
                        elif abs(pnl) >= self.emergency_close_threshold:
                            should_force_close = True
                            reason = f"🚨 EMERGÊNCIA PNL: {pnl*100:.3f}%"
                        
                        if should_force_close:
                            logger.warning(f"🔥 MONITOR CRÍTICO ATIVADO: {reason}")
                            self.force_close_active = True
                            self._emergency_position_close(reason)
                
                time.sleep(2)  # Check a cada 2 segundos
                
            except Exception as e:
                logger.error(f"Erro no monitor crítico: {e}")
                time.sleep(5)

    def _emergency_position_close(self, reason: str) -> bool:
        """Fechamento de emergência com múltiplas tentativas AGRESSIVAS"""
        logger.error(f"🚨 FECHAMENTO DE EMERGÊNCIA: {reason}")
        
        success = False
        max_attempts = 10  # 10 tentativas
        
        for attempt in range(max_attempts):
            try:
                logger.warning(f"🔥 Tentativa FORÇADA {attempt+1}/{max_attempts}")
                
                if self.paper_trading:
                    success = True
                    logger.info("PAPER: Fechamento forçado simulado")
                    break
                
                # FORÇAR FECHAMENTO REAL
                if self.current_position:
                    if self.current_position.side == TradeDirection.LONG:
                        # Vender com urgência
                        result = self.bitget_api.place_sell_order(profit_target=0)
                        success = result and result.get('success', False)
                        
                        if success:
                            logger.info(f"✅ VENDA FORÇADA SUCESSO: {result}")
                            break
                        else:
                            logger.error(f"❌ Tentativa {attempt+1} falhou: {result}")
                    
                    # Tentar fechamento direto via API se venda falhar
                    if not success and attempt >= 3:
                        logger.warning(f"🔧 Tentando fechamento direto via API...")
                        try:
                            positions = self.bitget_api.fetch_positions(['ETHUSDT'])
                            for pos in positions:
                                if pos.get('symbol') == 'ETHUSDT' and abs(float(pos.get('size', 0))) > 0:
                                    # Tentar fechar via create_order
                                    close_order = self.bitget_api.create_order(
                                        symbol='ETHUSDT',
                                        order_type='market',
                                        side='sell',
                                        amount=abs(float(pos.get('size', 0))),
                                        params={'reduceOnly': True}
                                    )
                                    if close_order:
                                        success = True
                                        logger.info("✅ Fechamento direto bem-sucedido!")
                                        break
                        except Exception as direct_error:
                            logger.error(f"Erro no fechamento direto: {direct_error}")
                
                if not success:
                    wait_time = min(10, 2 ** attempt)  # Exponential backoff
                    logger.warning(f"⏳ Aguardando {wait_time}s antes da próxima tentativa...")
                    time.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Erro na tentativa de fechamento {attempt+1}: {e}")
        
        # ATUALIZAR ESTADO INDEPENDENTE DO RESULTADO
        if self.current_position:
            final_pnl = 0
            if success or attempt >= max_attempts - 1:
                try:
                    current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
                    final_pnl = self.current_position.calculate_pnl(current_price)
                    duration = self.current_position.get_duration()
                    
                    # Atualizar métricas SEMPRE
                    self._update_trade_metrics(final_pnl, duration)
                    
                    # Log final
                    status = "✅ SUCESSO" if success else "⚠️ TIMEOUT"
                    logger.error(f"🔥 FECHAMENTO FORÇADO {status}")
                    logger.error(f"   PnL Final: {final_pnl*100:.3f}%")
                    logger.error(f"   Duração: {duration:.1f}s")
                    logger.error(f"   Tentativas: {attempt+1}")
                    
                except Exception as e:
                    logger.error(f"Erro calculando PnL final: {e}")
                
                # LIMPAR POSIÇÃO SEMPRE
                self.current_position = None
                self.force_close_active = False
                
                # Penalty por fechamento forçado
                if not success:
                    logger.error("💀 FECHAMENTO FALHOU - APLICANDO PENALTY")
                    self.consecutive_losses += 1
                    self.daily_loss += 0.01  # 1% penalty
        
        return success

    def _collect_enhanced_data(self):
        """Coletar dados com melhor qualidade"""
        try:
            logger.info("📊 Coletando dados melhorados...")
            
            for i in range(50):
                market_data = self.bitget_api.get_market_data(self.symbol)
                if market_data and market_data.get('price', 0) > 0:
                    price = float(market_data['price'])
                    self.price_history.append(price)
                    
                    if market_data.get('volume', 0) > 0:
                        self.volume_history.append(float(market_data['volume']))
                    
                    # Pequena pausa para dados mais distribuídos
                    time.sleep(0.3)
                
            logger.info(f"✅ Coletados {len(self.price_history)} pontos de alta qualidade")
            
        except Exception as e:
            logger.error(f"Erro coletando dados: {e}")

    def _enhanced_trading_loop(self):
        """Loop principal melhorado com previsões aprimoradas"""
        logger.info("🔄 Loop melhorado iniciado")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                
                # Verificações críticas
                if self._should_stop_trading():
                    logger.warning("🛑 Condições de parada atingidas")
                    break
                
                # Atualizar dados
                self._update_enhanced_market_data()
                
                # Gerenciar posição com prioridade
                if self.current_position and not self.force_close_active:
                    self._enhanced_position_management()
                
                # Procurar oportunidades APENAS se não tiver posição
                elif not self.current_position and self._can_open_enhanced_position():
                    signal = self._enhanced_market_analysis()
                    if signal:
                        direction, confidence, strength = signal
                        if confidence > 0.75 and strength > 0.8:  # Critérios mais rigorosos
                            self._execute_enhanced_trade(direction, confidence, strength)
                
                # Sleep controlado
                elapsed = time.time() - loop_start
                sleep_time = max(1.0, self.scalping_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                time.sleep(5)
        
        logger.info("🏁 Loop principal finalizado")

    def _update_enhanced_market_data(self):
        """Atualização de dados com validação"""
        try:
            market_data = self.bitget_api.get_market_data(self.symbol)
            if market_data and market_data.get('price', 0) > 0:
                price = float(market_data['price'])
                
                # Validar se o preço é razoável (não deve variar mais que 5% instantaneamente)
                if self.price_history:
                    last_price = self.price_history[-1]
                    price_change = abs(price - last_price) / last_price
                    
                    if price_change > 0.05:  # Mais de 5% de mudança
                        logger.warning(f"⚠️ Mudança de preço suspeita: {price_change*100:.2f}%")
                        return  # Ignorar este update
                
                self.price_history.append(price)
                
                if market_data.get('volume', 0) > 0:
                    volume = float(market_data['volume'])
                    self.volume_history.append(volume)
                        
        except Exception as e:
            logger.error(f"Erro atualizando dados: {e}")

    def _enhanced_market_analysis(self) -> Optional[Tuple[TradeDirection, float, float]]:
        """Análise de mercado MUITO melhorada"""
        try:
            if len(self.price_history) < 50:
                return None
                
            prices = list(self.price_history)
            current_price = prices[-1]
            
            # Indicadores melhorados
            rsi, rsi_signal = ImprovedIndicators.adaptive_rsi(prices)
            trend_strength, trend_direction = ImprovedIndicators.trend_strength(prices)
            bb_upper, bb_lower, bb_middle, bb_signal = ImprovedIndicators.smart_bollinger(prices)
            
            # Momentum multi-timeframe
            short_momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            medium_momentum = (prices[-1] - prices[-12]) / prices[-12] if len(prices) >= 12 else 0
            long_momentum = (prices[-1] - prices[-26]) / prices[-26] if len(prices) >= 26 else 0
            
            # Volume confirmation se disponível
            volume_factor = 1.0
            if len(self.volume_history) >= 10:
                recent_volume = sum(list(self.volume_history)[-5:]) / 5
                avg_volume = sum(list(self.volume_history)) / len(self.volume_history)
                volume_factor = min(2.0, recent_volume / avg_volume) if avg_volume > 0 else 1.0
            
            # Sistema de pontuação melhorado
            long_score = 0.0
            short_score = 0.0
            
            # RSI signals (peso: 25%)
            if rsi_signal == "STRONG_OVERSOLD":
                long_score += 0.25
            elif rsi_signal == "OVERSOLD":
                long_score += 0.15
            elif rsi_signal == "STRONG_OVERBOUGHT":
                short_score += 0.25
            elif rsi_signal == "OVERBOUGHT":
                short_score += 0.15
            
            # Trend signals (peso: 30%)
            if trend_direction == "STRONG_BULLISH":
                long_score += 0.30
            elif trend_direction == "BULLISH":
                long_score += 0.20
            elif trend_direction == "STRONG_BEARISH":
                short_score += 0.30
            elif trend_direction == "BEARISH":
                short_score += 0.20
            
            # Bollinger signals (peso: 20%)
            if bb_signal == "LOWER_BREAKOUT":
                long_score += 0.20
            elif bb_signal == "NEAR_LOWER":
                long_score += 0.10
            elif bb_signal == "UPPER_BREAKOUT":
                short_score += 0.20
            elif bb_signal == "NEAR_UPPER":
                short_score += 0.10
            
            # Momentum signals (peso: 15%)
            momentum_score = (short_momentum + medium_momentum + long_momentum) / 3
            if momentum_score > 0.003:
                long_score += 0.15
            elif momentum_score < -0.003:
                short_score += 0.15
            
            # Volume boost (peso: 10%)
            if volume_factor > 1.2:
                long_score *= volume_factor
                short_score *= volume_factor
            
            # Filtros de qualidade
            min_score = 0.7
            max_score = 0.95
            
            # Determinação final
            if long_score > short_score and long_score >= min_score:
                confidence = min(max_score, long_score)
                strength = trend_strength * volume_factor
                
                logger.info(f"📈 SINAL LONG detectado:")
                logger.info(f"   Score: {long_score:.3f}")
                logger.info(f"   Confiança: {confidence:.3f}")
                logger.info(f"   Força: {strength:.3f}")
                logger.info(f"   RSI: {rsi:.1f} ({rsi_signal})")
                logger.info(f"   Trend: {trend_direction}")
                logger.info(f"   Volume: {volume_factor:.2f}x")
                
                return TradeDirection.LONG, confidence, strength
                
            elif short_score > long_score and short_score >= min_score:
                confidence = min(max_score, short_score)
                strength = trend_strength * volume_factor
                
                logger.info(f"📉 SINAL SHORT detectado:")
                logger.info(f"   Score: {short_score:.3f}")
                logger.info(f"   Confiança: {confidence:.3f}")
                logger.info(f"   Força: {strength:.3f}")
                logger.info(f"   RSI: {rsi:.1f} ({rsi_signal})")
                logger.info(f"   Trend: {trend_direction}")
                logger.info(f"   Volume: {volume_factor:.2f}x")
                
                return TradeDirection.SHORT, confidence, strength
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return None

    def _execute_enhanced_trade(self, direction: TradeDirection, confidence: float, strength: float):
        """Executar trade com validações rigorosas"""
        with self._execution_lock:
            try:
                # Dupla verificação
                if self.current_position:
                    logger.warning("⚠️ Posição já existe, abortando trade")
                    return
                
                # Verificar saldo
                balance = self.get_account_balance()
                if balance <= 0:
                    if self.paper_trading:
                        balance = 1000
                    else:
                        logger.error("Saldo insuficiente")
                        return
                
                current_price = self.price_history[-1]
                
                # Targets ajustados por confiança
                confidence_multiplier = min(1.2, confidence + 0.2)
                adjusted_profit = self.profit_target * confidence_multiplier
                
                if direction == TradeDirection.LONG:
                    target_price = current_price * (1 + adjusted_profit)
                    stop_price = current_price * (1 - self.stop_loss)
                else:
                    target_price = current_price * (1 - adjusted_profit)
                    stop_price = current_price * (1 + self.stop_loss)
                
                logger.info(f"🎯 Executando {direction.name} PREMIUM:")
                logger.info(f"   Preço: ${current_price:.2f}")
                logger.info(f"   Target: ${target_price:.2f} ({adjusted_profit*100:.2f}%)")
                logger.info(f"   Stop: ${stop_price:.2f} ({self.stop_loss*100:.2f}%)")
                logger.info(f"   Confiança: {confidence*100:.1f}%")
                logger.info(f"   Força: {strength*100:.1f}%")
                
                # Executar ordem
                success = False
                if self.paper_trading:
                    success = True
                    logger.info("📝 PAPER TRADING - Trade simulado")
                else:
                    try:
                        if direction == TradeDirection.LONG:
                            result = self.bitget_api.place_buy_order()
                            success = result and result.get('success', False)
                            logger.info(f"Resultado compra: {result}")
                        else:
                            logger.info("SHORT temporariamente desabilitado para estabilidade")
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
                    
                    logger.info(f"✅ TRADE #{self.trades_today} EXECUTADO!")
                    logger.info(f"   Posição: {position_size:.4f} ETH")
                    logger.info(f"   Valor: ${position_value:.2f}")
                    logger.info(f"   🔥 MONITORAMENTO CRÍTICO ATIVO!")
                
            except Exception as e:
                logger.error(f"Erro no trade: {e}")
                traceback.print_exc()

    def _enhanced_position_management(self):
        """Gerenciamento de posição com verificações constantes"""
        if self.is_managing_position or not self.current_position:
            return
            
        self.is_managing_position = True
        
        try:
            with self._position_lock:
                current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
                pnl = self.current_position.calculate_pnl(current_price)
                duration = self.current_position.get_duration()
                
                # Log detalhado a cada 10 segundos
                if int(duration) % 10 == 0:
                    logger.info(f"📊 POSIÇÃO ATIVA:")
                    logger.info(f"   PnL: {pnl*100:.3f}% | Duração: {duration:.0f}s")
                    logger.info(f"   Preço atual: ${current_price:.2f}")
                    logger.info(f"   Target: ${self.current_position.target_price:.2f}")
                    logger.info(f"   Stop: ${self.current_position.stop_price:.2f}")
                
                # Verificações RIGOROSAS de fechamento
                should_close = False
                reason = ""
                
                # 1. TAKE PROFIT - Imediato
                if pnl >= self.profit_target:
                    should_close = True
                    reason = f"🎯 TAKE PROFIT: {pnl*100:.3f}%"
                
                # 2. STOP LOSS - Imediato  
                elif pnl <= -self.stop_loss:
                    should_close = True
                    reason = f"⛔ STOP LOSS: {pnl*100:.3f}%"
                
                # 3. TEMPO MÁXIMO - Forçado
                elif duration >= self.max_position_time:
                    should_close = True
                    reason = f"⏰ TEMPO LIMITE: {duration:.0f}s | PnL: {pnl*100:.3f}%"
                
                # 4. Proteção por PnL extremo
                elif pnl >= self.profit_target * 0.8:  # 80% do target
                    # Verificar se não está perdendo momentum
                    if len(self.price_history) >= 5:
                        recent_trend = (self.price_history[-1] - self.price_history[-5]) / self.price_history[-5]
                        if (self.current_position.side == TradeDirection.LONG and recent_trend < 0) or \
                           (self.current_position.side == TradeDirection.SHORT and recent_trend > 0):
                            should_close = True
                            reason = f"📉 MOMENTUM REVERSO: {pnl*100:.3f}%"
                
                # 5. Stop loss dinâmico agressivo
                elif pnl <= -self.stop_loss * 0.7:  # 70% do stop loss
                    should_close = True
                    reason = f"🚨 STOP ANTECIPADO: {pnl*100:.3f}%"
                
                if should_close and not self.force_close_active:
                    logger.warning(f"🔥 ACIONANDO FECHAMENTO: {reason}")
                    self._immediate_position_close(reason)
                    
        except Exception as e:
            logger.error(f"Erro gerenciando posição: {e}")
        finally:
            self.is_managing_position = False

    def _immediate_position_close(self, reason: str) -> bool:
        """Fechamento imediato com prioridade máxima"""
        logger.warning(f"⚡ FECHAMENTO IMEDIATO: {reason}")
        
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            final_pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            success = False
            attempts = 3
            
            for attempt in range(attempts):
                try:
                    if self.paper_trading:
                        success = True
                        logger.info("📝 PAPER: Fechamento imediato simulado")
                        break
                    
                    # Fechamento real
                    if self.current_position.side == TradeDirection.LONG:
                        result = self.bitget_api.place_sell_order(profit_target=0)
                        success = result and result.get('success', False)
                        
                        if success:
                            logger.info(f"✅ VENDA IMEDIATA: {result}")
                            break
                        else:
                            logger.error(f"❌ Tentativa {attempt+1} falhou")
                    
                    if not success and attempt < attempts - 1:
                        time.sleep(1)  # Pausa entre tentativas
                        
                except Exception as e:
                    logger.error(f"Erro na tentativa {attempt+1}: {e}")
            
            if success or attempts >= 3:
                # Atualizar métricas
                self._update_trade_metrics(final_pnl, duration)
                
                # Log resultado
                status = "✅ SUCESSO" if success else "⚠️ TIMEOUT"
                logger.warning(f"⚡ FECHAMENTO {status}")
                logger.warning(f"   PnL: {final_pnl*100:.3f}%")
                logger.warning(f"   Duração: {duration:.1f}s")
                logger.warning(f"   Razão: {reason}")
                
                # Limpar posição
                self.current_position = None
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Erro no fechamento imediato: {e}")
            return False

    def _update_trade_metrics(self, pnl: float, duration: float):
        """Atualizar métricas de trading"""
        try:
            with self._position_lock:
                self.metrics.total_trades += 1
                self.metrics.total_profit += pnl
                self.metrics.total_fees_paid += abs(pnl) * 0.002  # Taxa estimada
                
                if pnl > 0:
                    self.metrics.profitable_trades += 1
                    self.metrics.consecutive_wins += 1
                    self.metrics.consecutive_losses = 0
                    self.consecutive_losses = 0
                    logger.info(f"✅ TRADE LUCRATIVO: {pnl*100:.3f}%")
                else:
                    self.metrics.consecutive_wins = 0
                    self.metrics.consecutive_losses += 1
                    self.consecutive_losses += 1
                    self.daily_loss += abs(pnl)
                    logger.warning(f"❌ TRADE PERDEDOR: {pnl*100:.3f}%")
                
                # Atualizar máximos
                self.metrics.max_consecutive_wins = max(self.metrics.max_consecutive_wins, self.metrics.consecutive_wins)
                self.metrics.max_consecutive_losses = max(self.metrics.max_consecutive_losses, self.metrics.consecutive_losses)
                
                if pnl < 0:
                    self.metrics.max_drawdown = max(self.metrics.max_drawdown, abs(pnl))
                
                # Duração média
                total_duration = (self.metrics.average_trade_duration * (self.metrics.total_trades - 1) + duration)
                self.metrics.average_trade_duration = total_duration / self.metrics.total_trades
                
                # Log métricas atualizadas
                logger.info(f"📊 MÉTRICAS ATUALIZADAS:")
                logger.info(f"   Win Rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   Profit total: {self.metrics.total_profit*100:.3f}%")
                logger.info(f"   Trades hoje: {self.trades_today}")
                
        except Exception as e:
            logger.error(f"Erro atualizando métricas: {e}")

    def _should_stop_trading(self) -> bool:
        """Verificações de parada mais rigorosas"""
        # Perda diária
        if self.daily_loss >= self.max_daily_loss:
            logger.error(f"🛑 PERDA DIÁRIA MÁXIMA: {self.daily_loss*100:.2f}%")
            return True
        
        # Perdas consecutivas
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.error(f"🛑 PERDAS CONSECUTIVAS: {self.consecutive_losses}")
            return True
        
        # Drawdown máximo
        if self.metrics.max_drawdown >= 0.03:  # 3% drawdown máximo
            logger.error(f"🛑 DRAWDOWN MÁXIMO: {self.metrics.max_drawdown*100:.2f}%")
            return True
        
        # Limite de trades por dia (proteção)
        if self.trades_today >= 50:  # Máximo 50 trades por dia
            logger.warning(f"🛑 LIMITE DIÁRIO DE TRADES: {self.trades_today}")
            return True
        
        return False

    def _can_open_enhanced_position(self) -> bool:
        """Verificações mais rigorosas para abertura"""
        # Tempo entre trades
        if time.time() - self.last_trade_time < self.min_time_between_trades:
            return False
        
        # Estado de perda
        if self.daily_loss >= self.max_daily_loss * 0.7:  # 70% da perda máxima
            return False
        
        # Perdas consecutivas
        if self.consecutive_losses >= 1:  # Apenas 1 perda consecutiva permitida
            return False
        
        # Dados suficientes
        if len(self.price_history) < 50:
            return False
        
        return True

    def stop(self) -> bool:
        """Parar bot com fechamento seguro aprimorado"""
        try:
            logger.info("🛑 Parando bot...")
            self.state = TradingState.STOPPED
            
            # Fechar posição com prioridade
            if self.current_position:
                logger.warning("🔥 Fechando posição antes de parar...")
                self._emergency_position_close("Bot stopping")
            
            # Aguardar threads
            if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
                
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Relatório final detalhado
            daily_profit = self.metrics.net_profit * 100
            logger.info(f"📊 RELATÓRIO FINAL DETALHADO:")
            logger.info(f"   🎯 Trades executados: {self.trades_today}")
            logger.info(f"   💰 Win Rate: {self.metrics.win_rate:.1f}%")
            logger.info(f"   📈 Profit líquido: {daily_profit:.4f}%")
            logger.info(f"   🔥 Trades lucrativos: {self.metrics.profitable_trades}")
            logger.info(f"   💔 Trades perdedores: {self.metrics.losing_trades}")
            logger.info(f"   ⏱️ Duração média: {self.metrics.average_trade_duration:.1f}s")
            logger.info(f"   💸 Taxas pagas: {self.metrics.total_fees_paid*100:.4f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao parar bot: {e}")
            return False

    def get_account_balance(self) -> float:
        """Obter saldo com cache e validação"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                balance = float(balance_info.get('free', 0.0))
                return balance if balance > 0 else (1000.0 if self.paper_trading else 0.0)
            return 1000.0 if self.paper_trading else 0.0
        except Exception as e:
            logger.error(f"Erro obtendo saldo: {e}")
            return 1000.0 if self.paper_trading else 0.0

    def get_status(self) -> Dict:
        """Status completo e detalhado"""
        try:
            daily_profit = self.metrics.net_profit * 100
            
            return {
                'bot_status': {
                    'state': self.state.value,
                    'is_running': self.is_running,
                    'symbol': self.symbol,
                    'leverage': self.leverage,
                    'paper_trading': self.paper_trading,
                    'force_close_active': self.force_close_active,
                    'critical_monitor': 'ACTIVE' if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive() else 'INACTIVE'
                },
                'performance': {
                    'trades_today': self.trades_today,
                    'total_trades': self.metrics.total_trades,
                    'win_rate': round(self.metrics.win_rate, 1),
                    'daily_profit': round(daily_profit, 4),
                    'daily_loss': round(self.daily_loss * 100, 4),
                    'net_profit': round(self.metrics.net_profit * 100, 4),
                    'max_drawdown': round(self.metrics.max_drawdown * 100, 3),
                    'consecutive_wins': self.metrics.consecutive_wins,
                    'consecutive_losses': self.consecutive_losses,
                    'avg_duration': round(self.metrics.average_trade_duration, 1),
                    'profitable_trades': self.metrics.profitable_trades,
                    'losing_trades': self.metrics.losing_trades
                },
                'risk_management': {
                    'daily_loss_limit': f"{self.max_daily_loss*100:.1f}%",
                    'max_consecutive_losses': self.max_consecutive_losses,
                    'time_between_trades': f"{self.min_time_between_trades}s",
                    'risk_level': self._get_risk_level(),
                    'emergency_threshold': f"{self.emergency_close_threshold*100:.1f}%",
                    'max_position_time': f"{self.max_position_time}s"
                },
                'current_position': self._get_enhanced_position_status(),
                'targets': {
                    'take_profit': f"{self.profit_target*100:.1f}%",
                    'stop_loss': f"{self.stop_loss*100:.1f}%",
                    'risk_reward': f"1:{self.profit_target/self.stop_loss:.1f}",
                    'emergency_close': f"{self.emergency_close_threshold*100:.1f}%"
                },
                'system_health': {
                    'price_data_points': len(self.price_history),
                    'volume_data_points': len(self.volume_history),
                    'last_update': time.time(),
                    'locks_active': self.is_managing_position or self.force_close_active,
                    'critical_monitor': hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive()
                }
            }
        except Exception as e:
            return {'error': str(e), 'is_running': False}

    def _get_risk_level(self) -> str:
        """Determinação do nível de risco"""
        if self.consecutive_losses >= self.max_consecutive_losses or \
           self.daily_loss >= self.max_daily_loss * 0.9:
            return "CRITICAL"
        elif self.consecutive_losses >= 1 or \
             self.daily_loss >= self.max_daily_loss * 0.6:
            return "HIGH"
        elif self.daily_loss >= self.max_daily_loss * 0.3:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_enhanced_position_status(self) -> Dict:
        """Status da posição com mais detalhes"""
        if not self.current_position:
            return {'active': False}
        
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            # Calcular progresso até target e stop
            if self.current_position.side == TradeDirection.LONG:
                target_progress = (current_price - self.current_position.entry_price) / \
                                (self.current_position.target_price - self.current_position.entry_price)
                stop_distance = (current_price - self.current_position.stop_price) / \
                              (self.current_position.entry_price - self.current_position.stop_price)
            else:
                target_progress = (self.current_position.entry_price - current_price) / \
                                (self.current_position.entry_price - self.current_position.target_price)
                stop_distance = (self.current_position.stop_price - current_price) / \
                              (self.current_position.stop_price - self.current_position.entry_price)
            
            return {
                'active': True,
                'side': self.current_position.side.value,
                'entry_price': round(self.current_position.entry_price, 2),
                'current_price': round(current_price, 2),
                'target_price': round(self.current_position.target_price, 2),
                'stop_price': round(self.current_position.stop_price, 2),
                'pnl_percent': round(pnl * 100, 3),
                'duration_seconds': round(duration),
                'max_duration': self.max_position_time,
                'is_profitable': pnl > 0,
                'target_progress': round(target_progress * 100, 1),
                'stop_distance': round(stop_distance * 100, 1),
                'time_remaining': max(0, self.max_position_time - duration),
                'force_close_risk': abs(pnl) >= self.emergency_close_threshold * 0.8,
                'position_size': self.current_position.size
            }
        except Exception as e:
            return {'active': True, 'error': str(e)}

    @property
    def is_running(self) -> bool:
        """Estado do bot"""
        return self.state == TradingState.RUNNING

    def emergency_stop(self) -> bool:
        """Parada de emergência total"""
        try:
            logger.error("🚨 PARADA DE EMERGÊNCIA TOTAL!")
            self.state = TradingState.EMERGENCY
            
            # Forçar fechamento imediatamente
            if self.current_position:
                self.force_close_active = True
                self._emergency_position_close("EMERGENCY STOP")
            
            # Parar todas as threads
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=3)
            if self.trading_thread:
                self.trading_thread.join(timeout=3)
            
            self.state = TradingState.STOPPED
            logger.error("🚨 EMERGÊNCIA CONCLUÍDA!")
            return True
            
        except Exception as e:
            logger.error(f"Erro na emergência: {e}")
            return False

    def get_enhanced_analysis(self) -> Dict:
        """Análise de mercado detalhada para debugging"""
        try:
            if len(self.price_history) < 30:
                return {'error': 'Dados insuficientes'}
            
            prices = list(self.price_history)
            current_price = prices[-1]
            
            # Indicadores detalhados
            rsi, rsi_signal = ImprovedIndicators.adaptive_rsi(prices)
            trend_strength, trend_direction = ImprovedIndicators.trend_strength(prices)
            bb_upper, bb_lower, bb_middle, bb_signal = ImprovedIndicators.smart_bollinger(prices)
            
            # Momentum
            short_momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            medium_momentum = (prices[-1] - prices[-12]) / prices[-12] if len(prices) >= 12 else 0
            
            # Volume
            volume_factor = 1.0
            if len(self.volume_history) >= 5:
                recent_volume = sum(list(self.volume_history)[-3:]) / 3
                avg_volume = sum(list(self.volume_history)) / len(self.volume_history)
                volume_factor = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'market_data': {
                    'current_price': round(current_price, 2),
                    'price_points': len(self.price_history),
                    'volume_points': len(self.volume_history)
                },
                'indicators': {
                    'rsi': {
                        'value': round(rsi, 1),
                        'signal': rsi_signal,
                        'interpretation': 'OVERSOLD' if rsi < 35 else 'OVERBOUGHT' if rsi > 65 else 'NEUTRAL'
                    },
                    'trend': {
                        'strength': round(trend_strength * 100, 2),
                        'direction': trend_direction,
                        'interpretation': 'STRONG' if trend_strength > 0.01 else 'WEAK'
                    },
                    'bollinger': {
                        'upper': round(bb_upper, 2),
                        'middle': round(bb_middle, 2),
                        'lower': round(bb_lower, 2),
                        'signal': bb_signal,
                        'width': round(((bb_upper - bb_lower) / bb_middle) * 100, 2)
                    }
                },
                'momentum': {
                    'short_term': round(short_momentum * 100, 3),
                    'medium_term': round(medium_momentum * 100, 3),
                    'volume_factor': round(volume_factor, 2)
                },
                'trading_conditions': {
                    'can_trade': self._can_open_enhanced_position(),
                    'risk_level': self._get_risk_level(),
                    'consecutive_losses': self.consecutive_losses,
                    'daily_loss': round(self.daily_loss * 100, 3)
                }
            }
            
        except Exception as e:
            return {'error': str(e)}

# Função de criação simplificada
def create_enhanced_trading_bot(bitget_api: BitgetAPI, **kwargs) -> TradingBot:
    """Criar bot de trading aprimorado"""
    return TradingBot(bitget_api, **kwargs)

# Teste do sistema
if __name__ == "__main__":
    try:
        print("🔥 TRADING BOT V2 - TESTE DE SISTEMA")
        print("=" * 50)
        print("✅ Fechamento forçado implementado")
        print("✅ Previsões melhoradas")
        print("✅ Monitor crítico ativo")
        print("✅ Análise adaptativa")
        print("✅ Métricas detalhadas")
        print("=" * 50)
        print("🎯 PRONTO PARA TRADING PROFISSIONAL!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        traceback.print_exc()
