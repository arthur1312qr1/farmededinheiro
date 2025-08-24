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

class SimpleIndicators:
    """Indicadores técnicos simplificados e confiáveis"""
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        """RSI simples e confiável"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas[-period:]]
        losses = [max(0, -delta) for delta in deltas[-period:]]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0.001
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def sma(prices: List[float], period: int) -> float:
        """Média móvel simples"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        return sum(prices[-period:]) / period
    
    @staticmethod
    def ema(prices: List[float], period: int) -> float:
        """Média móvel exponencial"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bandas de Bollinger"""
        if len(prices) < period:
            price = prices[-1] if prices else 0
            return price, price, price
        
        recent_prices = prices[-period:]
        middle = sum(recent_prices) / len(recent_prices)
        variance = sum([(p - middle) ** 2 for p in recent_prices]) / len(recent_prices)
        std = math.sqrt(variance)
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, lower, middle

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

        # CONFIGURAÇÕES CORRIGIDAS - Take Profit 0.9%
        self.profit_target = 0.009           # 0.9% take profit (CORRIGIDO)
        self.stop_loss = 0.004               # 0.4% stop loss 
        self.min_profit_target = 0.005       # 0.5% mínimo para compensar fees
        self.max_position_time = 300         # 5 minutos máximo por trade
        self.min_position_time = 15          # 15 segundos mínimo
        
        # Controles de risco
        self.max_daily_loss = 0.02          # 2% perda máxima por dia
        self.max_consecutive_losses = 3     # Parar após 3 perdas seguidas
        self.min_time_between_trades = 10   # 10 segundos entre trades
        
        # Indicadores e dados
        self.price_history = deque(maxlen=200)
        self.volume_history = deque(maxlen=50)
        
        # Métricas
        self.metrics = TradingMetrics()
        self.trades_today = 0
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = 0
        
        # Threading - CORRIGIDO
        self._lock = threading.Lock()
        self.is_entering_position = False
        self.is_exiting_position = False
        
        # Inicialização
        logger.info("Trading Bot Profissional Inicializado")
        logger.info(f"Take Profit: {self.profit_target*100:.1f}%")
        logger.info(f"Stop Loss: {self.stop_loss*100:.1f}%")

    def start(self) -> bool:
        """Iniciar bot com configurações otimizadas"""
        try:
            if self.state == TradingState.RUNNING:
                return True
            
            logger.info("Iniciando bot profissional...")
            
            # Reset estado
            self.state = TradingState.RUNNING
            self.last_trade_time = time.time()
            self.consecutive_losses = 0
            self.daily_loss = 0.0
            self.trades_today = 0
            
            # Reset locks
            self.is_entering_position = False
            self.is_exiting_position = False
            
            # Coletar dados iniciais
            self._collect_initial_data()
            
            # Iniciar thread de trading
            self.trading_thread = threading.Thread(
                target=self._main_trading_loop,
                daemon=True,
                name="ProfessionalTradingBot"
            )
            self.trading_thread.start()
            
            logger.info("Bot profissional iniciado com sucesso!")
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
            
            # Fechar posição se existir
            if self.current_position:
                self._close_position_safely("Bot stopping")
            
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
        """Coletar dados históricos iniciais"""
        try:
            logger.info("Coletando dados iniciais...")
            for _ in range(30):
                market_data = self.bitget_api.get_market_data(self.symbol)
                if market_data and market_data.get('price', 0) > 0:
                    self.price_history.append(float(market_data['price']))
                    if market_data.get('volume', 0) > 0:
                        self.volume_history.append(float(market_data['volume']))
                time.sleep(0.5)
            logger.info(f"Coletados {len(self.price_history)} pontos de preço")
        except Exception as e:
            logger.error(f"Erro coletando dados: {e}")

    def _main_trading_loop(self):
        """Loop principal otimizado"""
        logger.info("Loop principal iniciado")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                
                # Verificar condições de parada
                if self._should_stop_trading():
                    logger.warning("Condições de parada atingidas")
                    break
                
                # Atualizar dados de mercado
                self._update_market_data()
                
                # Gerenciar posição existente
                if self.current_position:
                    self._manage_position()
                
                # Procurar nova oportunidade
                elif self._can_open_new_position():
                    signal = self._analyze_market()
                    if signal:
                        direction, confidence = signal
                        if confidence > 0.7:
                            self._execute_trade(direction, confidence)
                
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

    def _analyze_market(self) -> Optional[Tuple[TradeDirection, float]]:
        """Análise de mercado simplificada"""
        try:
            if len(self.price_history) < 30:
                return None
                
            prices = list(self.price_history)
            current_price = prices[-1]
            
            # Indicadores principais
            rsi = SimpleIndicators.rsi(prices)
            sma_20 = SimpleIndicators.sma(prices, 20)
            ema_12 = SimpleIndicators.ema(prices, 12)
            ema_26 = SimpleIndicators.ema(prices, 26)
            bb_upper, bb_lower, bb_middle = SimpleIndicators.bollinger_bands(prices)
            
            # Sinais de entrada
            signals = []
            
            # RSI oversold/overbought
            if rsi < 30:
                signals.append(("LONG", 0.3))
            elif rsi > 70:
                signals.append(("SHORT", 0.3))
            
            # EMA crossover
            if ema_12 > ema_26:
                signals.append(("LONG", 0.2))
            else:
                signals.append(("SHORT", 0.2))
            
            # Bollinger Bands
            if bb_upper != bb_lower:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                if bb_position < 0.2:
                    signals.append(("LONG", 0.25))
                elif bb_position > 0.8:
                    signals.append(("SHORT", 0.25))
            
            # Trend following
            if current_price > sma_20 * 1.002:
                signals.append(("LONG", 0.15))
            elif current_price < sma_20 * 0.998:
                signals.append(("SHORT", 0.15))
            
            # Calcular direção final
            long_strength = sum([strength for direction, strength in signals if direction == "LONG"])
            short_strength = sum([strength for direction, strength in signals if direction == "SHORT"])
            
            if long_strength > short_strength and long_strength > 0.6:
                return TradeDirection.LONG, min(long_strength, 0.95)
            elif short_strength > long_strength and short_strength > 0.6:
                return TradeDirection.SHORT, min(short_strength, 0.95)
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return None

    def _execute_trade(self, direction: TradeDirection, confidence: float):
        """Executar trade de forma segura - CORRIGIDO"""
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
            
            # Calcular targets
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)
                stop_price = current_price * (1 - self.stop_loss)
            else:
                target_price = current_price * (1 - self.profit_target)
                stop_price = current_price * (1 + self.stop_loss)
            
            logger.info(f"Executando {direction.name}:")
            logger.info(f"   Preço: ${current_price:.2f}")
            logger.info(f"   Target: ${target_price:.2f} ({self.profit_target*100:.1f}%)")
            logger.info(f"   Stop: ${stop_price:.2f}")
            logger.info(f"   Confiança: {confidence*100:.1f}%")
            
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
                logger.info(f"✅ Trade #{self.trades_today} executado com sucesso!")
                logger.info(f"   Posição: {position_size:.4f} ETH")
                logger.info(f"   Valor: ${position_value:.2f}")
            
        except Exception as e:
            logger.error(f"Erro no trade: {e}")
            traceback.print_exc()
        finally:
            self.is_entering_position = False

    def _manage_position(self):
        """Gerenciar posição - MÉTODO PRINCIPAL DE FECHAMENTO"""
        if not self.current_position or self.is_exiting_position:
            return
            
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            should_close = False
            reason = ""
            
            # CONDIÇÕES DE FECHAMENTO CORRIGIDAS
            if pnl >= self.profit_target:
                should_close = True
                reason = f"✅ TAKE PROFIT: {pnl*100:.3f}%"
                
            elif pnl <= -self.stop_loss:
                should_close = True
                reason = f"❌ STOP LOSS: {pnl*100:.3f}%"
                
            elif duration >= self.max_position_time:
                should_close = True
                reason = f"⏰ TEMPO LIMITE: {duration:.0f}s, PnL: {pnl*100:.3f}%"
            
            # Log periódico da posição
            if int(duration) % 15 == 0:  # A cada 15 segundos
                logger.info(f"📊 Posição ativa: {pnl*100:.3f}% | {duration:.0f}s | Target: {self.profit_target*100:.1f}%")
            
            if should_close:
                logger.info(f"🔄 Fechando posição: {reason}")
                self._close_position_safely(reason)
                
        except Exception as e:
            logger.error(f"Erro gerenciando posição: {e}")

    def _close_position_safely(self, reason: str) -> bool:
        """Fechar posição com múltiplas tentativas - CORRIGIDO"""
        if self.is_exiting_position or not self.current_position:
            return False
            
        self.is_exiting_position = True
        
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            final_pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            logger.info(f"🔄 Iniciando fechamento: {reason}")
            logger.info(f"   PnL atual: {final_pnl*100:.3f}%")
            logger.info(f"   Duração: {duration:.1f}s")
            
            success = False
            max_attempts = 3
            
            for attempt in range(max_attempts):
                try:
                    if self.paper_trading:
                        success = True
                        logger.info("PAPER TRADING - Fechamento simulado")
                        break
                    
                    # FECHAR POSIÇÃO REAL - CORRIGIDO
                    if self.current_position.side == TradeDirection.LONG:
                        # Para LONG, vender para fechar
                        logger.info(f"Tentativa {attempt+1}: Vendendo posição LONG...")
                        result = self.bitget_api.place_sell_order(profit_target=0)  # Vender imediatamente
                        if result:
                            success = result.get('success', False)
                            logger.info(f"Resultado venda: {result}")
                        else:
                            logger.warning("Resultado venda é None")
                    else:
                        # Para SHORT, comprar para fechar
                        logger.info(f"Tentativa {attempt+1}: Comprando para fechar SHORT...")
                        # Implementar quando necessário
                        success = False
                    
                    if success:
                        break
                    else:
                        logger.warning(f"Tentativa {attempt+1} falhou")
                        
                except Exception as e:
                    logger.error(f"Tentativa {attempt+1} com erro: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(2)
            
            if success:
                # Atualizar métricas
                with self._lock:
                    self.metrics.total_trades += 1
                    self.metrics.total_profit += final_pnl
                    self.metrics.total_fees_paid += abs(final_pnl) * 0.002
                    
                    if final_pnl > 0:
                        self.metrics.profitable_trades += 1
                        self.metrics.consecutive_wins += 1
                        self.metrics.consecutive_losses = 0
                        self.consecutive_losses = 0
                        logger.info(f"✅ LUCRO: {final_pnl*100:.3f}%")
                    else:
                        self.metrics.consecutive_wins = 0
                        self.metrics.consecutive_losses += 1
                        self.consecutive_losses += 1
                        self.daily_loss += abs(final_pnl)
                        logger.info(f"❌ PERDA: {final_pnl*100:.3f}%")
                    
                    # Atualizar drawdown
                    if final_pnl < 0:
                        self.metrics.max_drawdown = max(self.metrics.max_drawdown, abs(final_pnl))
                    
                    # Atualizar duração média
                    total_duration = (self.metrics.average_trade_duration * (self.metrics.total_trades - 1) + duration)
                    self.metrics.average_trade_duration = total_duration / self.metrics.total_trades
                
                logger.info(f"🎯 POSIÇÃO FECHADA!")
                logger.info(f"   PnL final: {final_pnl*100:.3f}%")
                logger.info(f"   Duração: {duration:.1f}s")
                logger.info(f"   Win Rate: {self.metrics.win_rate:.1f}%")
                
                self.current_position = None
                return True
            else:
                logger.error("❌ FALHA CRÍTICA ao fechar posição!")
                # Em caso de falha crítica, marcar posição como None para evitar loop
                self.current_position = None
                return False
                
        except Exception as e:
            logger.error(f"Erro crítico fechando posição: {e}")
            traceback.print_exc()
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
                    'paper_trading': self.paper_trading
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
                'is_profitable': pnl > 0
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
                        'profit_target': '0.9% por trade',
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
                    }
                }
        except Exception as e:
            return {'error': str(e)}

    def emergency_stop(self) -> bool:
        """Parada de emergência"""
        try:
            logger.warning("⚠️ PARADA DE EMERGÊNCIA ATIVADA")
            self.state = TradingState.EMERGENCY
            
            # Fechar posição imediatamente
            if self.current_position:
                self._close_position_safely("Emergency stop")
            
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
        """Análise atual do mercado"""
        try:
            if len(self.price_history) < 20:
                return {'error': 'Dados insuficientes'}
            
            prices = list(self.price_history)
            current_price = prices[-1]
            
            # Indicadores
            rsi = SimpleIndicators.rsi(prices)
            sma_20 = SimpleIndicators.sma(prices, 20)
            ema_12 = SimpleIndicators.ema(prices, 12)
            ema_26 = SimpleIndicators.ema(prices, 26)
            bb_upper, bb_lower, bb_middle = SimpleIndicators.bollinger_bands(prices)
            
            # Trend
            trend = "BULLISH" if current_price > sma_20 else "BEARISH"
            trend_strength = abs(current_price - sma_20) / sma_20 * 100
            
            # Volatilidade
            recent_highs = [max(prices[i:i+5]) for i in range(max(0, len(prices)-20), len(prices)-4)]
            recent_lows = [min(prices[i:i+5]) for i in range(max(0, len(prices)-20), len(prices)-4)]
            if recent_highs and recent_lows:
                volatility = (max(recent_highs) - min(recent_lows)) / current_price * 100
            else:
                volatility = 0
            
            return {
                'current_price': round(current_price, 2),
                'trend': trend,
                'trend_strength': round(trend_strength, 3),
                'volatility': round(volatility, 2),
                'indicators': {
                    'rsi': round(rsi, 1),
                    'sma_20': round(sma_20, 2),
                    'ema_12': round(ema_12, 2),
                    'ema_26': round(ema_26, 2),
                    'bb_upper': round(bb_upper, 2),
                    'bb_lower': round(bb_lower, 2),
                    'bb_middle': round(bb_middle, 2)
                },
                'signals': {
                    'rsi_signal': 'OVERSOLD' if rsi < 30 else 'OVERBOUGHT' if rsi > 70 else 'NEUTRAL',
                    'bb_signal': 'LOWER_BAND' if current_price <= bb_lower else 'UPPER_BAND' if current_price >= bb_upper else 'MIDDLE',
                    'ema_signal': 'BULLISH' if ema_12 > ema_26 else 'BEARISH'
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
            
            print("✅ Bot Profissional criado com sucesso!")
            print("Configurações otimizadas:")
            print(f"   Take Profit: {bot.profit_target*100:.1f}%")
            print(f"   Stop Loss: {bot.stop_loss*100:.1f}%")
            print(f"   Risk/Reward: 1:{bot.profit_target/bot.stop_loss:.1f}")
            print(f"   Max perda diária: {bot.max_daily_loss*100:.1f}%")
            print("🎯 Focado em consistência e preservação de capital!")
        else:
            print("❌ Falha na conexão com a API")
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        traceback.print_exc()
