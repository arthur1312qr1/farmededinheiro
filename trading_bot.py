import logging
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum

from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Estados do bot"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"

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
    
    @property
    def win_rate(self) -> float:
        return (self.profitable_trades / max(1, self.total_trades)) * 100
    
    @property
    def net_profit(self) -> float:
        return self.total_profit

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ETHUSDT',
                 leverage: int = 10, balance_percentage: float = 95.0,
                 scalping_interval: float = 1.5, paper_trading: bool = False):
        """
        Bot de trading ULTRA-AGRESSIVO para scalping de alta frequência
        """
        
        if not isinstance(bitget_api, BitgetAPI):
            raise TypeError(f"bitget_api deve ser BitgetAPI, recebido: {type(bitget_api)}")

        # Configurações básicas
        self.bitget_api = bitget_api
        self.symbol = symbol
        self.leverage = leverage
        self.balance_percentage = min(balance_percentage, 95.0)
        self.scalping_interval = max(scalping_interval, 1.0)  # Mínimo 1s
        self.paper_trading = paper_trading

        # Estado do bot
        self.state = TradingState.STOPPED
        self.current_position: Optional[TradePosition] = None
        self.trading_thread: Optional[threading.Thread] = None

        # CONFIGURAÇÕES CORRIGIDAS - MANTENDO TARGETS ORIGINAIS
        self.profit_target = 0.009           # 0.9% take profit (ORIGINAL)
        self.stop_loss = 0.004               # 0.4% stop loss (ORIGINAL)
        self.max_position_time = 60          # 1 minuto máximo
        
        # Controles de risco AGRESSIVOS
        self.max_daily_loss = 0.08          # 8% perda máxima por dia
        self.max_consecutive_losses = 6     # Muito tolerante
        self.min_time_between_trades = 1    # 1 segundo entre trades
        
        # Dados de mercado ULTRA-RÁPIDOS
        self.price_history = deque(maxlen=30)  # Histórico mínimo
        
        # Métricas
        self.metrics = TradingMetrics()
        self.trades_today = 0
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_trade_time = 0
        
        # CONTROLE ÚNICO DE FECHAMENTO
        self.is_closing = False
        self._lock = threading.Lock()
        
        logger.info("Trading Bot ULTRA-AGRESSIVO Inicializado")
        logger.info(f"Take Profit: {self.profit_target*100:.1f}%")
        logger.info(f"Stop Loss: {self.stop_loss*100:.1f}%")

    def start(self) -> bool:
        """Iniciar bot"""
        try:
            if self.state == TradingState.RUNNING:
                return True
            
            logger.info("Iniciando bot ultra-agressivo...")
            
            # Reset estado
            self.state = TradingState.RUNNING
            self.last_trade_time = time.time()
            self.consecutive_losses = 0
            self.daily_loss = 0.0
            self.trades_today = 0
            self.is_closing = False
            
            # Coletar dados iniciais RÁPIDO
            self._collect_initial_data()
            
            # Iniciar thread principal
            self.trading_thread = threading.Thread(
                target=self._main_loop,
                daemon=True,
                name="TradingBot"
            )
            self.trading_thread.start()
            
            logger.info("Bot iniciado - MODO SCALPING AGRESSIVO!")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            return False

    def stop(self) -> bool:
        """Parar bot"""
        try:
            logger.info("Parando bot...")
            self.state = TradingState.STOPPED
            
            # Fechar posição se existir
            if self.current_position:
                self._close_position_simple("Bot stopping")
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            logger.info(f"RELATÓRIO FINAL:")
            logger.info(f"   Trades: {self.trades_today}")
            logger.info(f"   Win Rate: {self.metrics.win_rate:.1f}%")
            logger.info(f"   Profit: {self.metrics.net_profit*100:.3f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao parar bot: {e}")
            return False

    def _collect_initial_data(self):
        """Coletar dados iniciais ULTRA-RÁPIDOS"""
        try:
            logger.info("Coletando dados iniciais...")
            
            # Tentar obter preço atual primeiro
            current_price = self.bitget_api.get_eth_price()
            if current_price > 0:
                # Preencher com preço atual para começar imediatamente
                for _ in range(5):
                    self.price_history.append(current_price)
                logger.info(f"Iniciado com preço: ${current_price:.2f}")
                return
            
            # Se falhar, coletar normalmente mas bem rápido
            for i in range(5):
                market_data = self.bitget_api.get_market_data(self.symbol)
                if market_data and market_data.get('price', 0) > 0:
                    self.price_history.append(float(market_data['price']))
                time.sleep(0.05)  # 50ms entre coletas
            
            logger.info(f"Coletados {len(self.price_history)} pontos")
            
        except Exception as e:
            logger.error(f"Erro coletando dados: {e}")
            # Fallback: usar um preço base
            self.price_history.extend([3500.0] * 5)

    def _main_loop(self):
        """Loop principal ULTRA-RÁPIDO"""
        logger.info("Loop principal iniciado - MODO AGRESSIVO")
        
        while self.state == TradingState.RUNNING:
            try:
                # Verificar condições de parada
                if self._should_stop_trading():
                    logger.warning("Condições de parada atingidas")
                    break
                
                # Atualizar preço
                self._update_price()
                
                # Gerenciar posição existente
                if self.current_position:
                    self._manage_position_simple()
                
                # Procurar nova oportunidade
                elif self._can_trade():
                    signal = self._analyze_market_aggressive()
                    if signal:
                        direction, confidence = signal
                        if confidence > 55:  # Threshold muito baixo
                            self._execute_trade(direction, confidence)
                
                time.sleep(self.scalping_interval)
                
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                time.sleep(2)
        
        logger.info("Loop finalizado")

    def _update_price(self):
        """Atualizar preço atual"""
        try:
            market_data = self.bitget_api.get_market_data(self.symbol)
            if market_data and market_data.get('price', 0) > 0:
                price = float(market_data['price'])
                self.price_history.append(price)
            except Exception as e:
                # Se falhar, usar último preço conhecido ou manter atual
                if self.price_history:
                    last_price = self.price_history[-1]
                    self.price_history.append(last_price)
                logger.debug(f"Erro atualizando preço: {e}")

    def _analyze_market_aggressive(self) -> Optional[Tuple[TradeDirection, float]]:
        """Análise ULTRA-AGRESSIVA para scalping"""
        try:
            if len(self.price_history) < 3:  # Mínimo absoluto
                return None
                
            prices = list(self.price_history)
            current_price = prices[-1]
            
            if current_price <= 0:
                logger.error("Preço atual inválido")
                return None
            
            # Análise ultra-simples baseada em movimento de preço
            signals = 0
            direction = None
            
            # 1. Momentum instantâneo (últimos 2 ticks)
            if len(prices) >= 2:
                prev_price = prices[-2]
                if prev_price > 0:
                    instant_change = (current_price - prev_price) / prev_price * 100
                    
                    if instant_change > 0.01:  # 0.01% movimento para cima
                        signals += 2
                        direction = TradeDirection.LONG
                    elif instant_change < -0.01:  # 0.01% movimento para baixo
                        signals += 2
                        direction = TradeDirection.SHORT
            
            # 2. Momentum de curto prazo (últimos 3-5 ticks)
            if len(prices) >= 5:
                old_price = prices[-5]
                if old_price > 0:
                    short_momentum = (current_price - old_price) / old_price * 100
                    
                    if short_momentum > 0.02 and direction == TradeDirection.LONG:
                        signals += 1
                    elif short_momentum < -0.02 and direction == TradeDirection.SHORT:
                        signals += 1
            
            # 3. Tendência de micro-prazo
            if len(prices) >= 3:
                p1, p2, p3 = prices[-3], prices[-2], prices[-1]
                if p1 > 0 and p2 > 0 and p3 > 0:
                    trend_up = p3 > p2 > p1
                    trend_down = p3 < p2 < p1
                    
                    if trend_up and direction == TradeDirection.LONG:
                        signals += 1
                    elif trend_down and direction == TradeDirection.SHORT:
                        signals += 1
            
            # 4. Volume implícito (volatilidade)
            if len(prices) >= 5:
                recent_prices = [p for p in prices[-5:] if p > 0]
                if len(recent_prices) >= 3:
                    recent_volatility = max(recent_prices) - min(recent_prices)
                    volatility_pct = (recent_volatility / current_price) * 100
                    
                    if volatility_pct > 0.05:  # Mais de 0.05% de volatilidade
                        signals += 1
            
            # Calcular confiança baseada em sinais
            if signals >= 2 and direction:
                confidence = min(85, 40 + signals * 15)
                
                logger.info(f"{'LONG' if direction == TradeDirection.LONG else 'SHORT'}: "
                          f"Sinais={signals}, Conf={confidence:.1f}%, "
                          f"Preço=${current_price:.2f}")
                
                return direction, confidence
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return None

    def _execute_trade(self, direction: TradeDirection, confidence: float):
        """Executar trade ULTRA-RÁPIDO"""
        try:
            # Verificar saldo
            balance = self._get_balance()
            if balance <= 0:
                if self.paper_trading:
                    balance = 1000
                else:
                    logger.error("Saldo insuficiente")
                    return
            
            current_price = self.price_history[-1]
            
            # Calcular targets AGRESSIVOS
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)
                stop_price = current_price * (1 - self.stop_loss)
            else:
                target_price = current_price * (1 - self.profit_target)
                stop_price = current_price * (1 + self.stop_loss)
            
            logger.info(f"Executando {direction.name}:")
            logger.info(f"   Preço: ${current_price:.2f}")
            logger.info(f"   Target: ${target_price:.2f} ({self.profit_target*100:.1f}%)")
            logger.info(f"   Stop: ${stop_price:.2f} ({self.stop_loss*100:.1f}%)")
            logger.info(f"   Confiança: {confidence:.1f}%")
            
            # Executar ordem
            success = False
            if self.paper_trading:
                success = True
                logger.info("PAPER TRADING - Ordem simulada")
            else:
                try:
                    if direction == TradeDirection.LONG:
                        result = self.bitget_api.place_buy_order()
                        success = result and result.get('success', False)
                        if not success:
                            logger.error(f"Falha na compra: {result}")
                    else:
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
                
                logger.info(f"Trade #{self.trades_today} executado!")
                logger.info(f"   Posição: {position_size:.4f} ETH")
                logger.info(f"   Valor: ${position_value:.2f}")
            
        except Exception as e:
            logger.error(f"Erro no trade: {e}")

    def _manage_position_simple(self):
        """Gerenciar posição ULTRA-AGRESSIVA"""
        if not self.current_position or self.is_closing:
            return
            
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            should_close = False
            reason = ""
            
            # 1. TAKE PROFIT AGRESSIVO
            if pnl >= self.profit_target:
                should_close = True
                reason = f"TAKE PROFIT: {pnl*100:.3f}%"
                
            # 2. STOP LOSS AGRESSIVO
            elif pnl <= -self.stop_loss:
                should_close = True
                reason = f"STOP LOSS: {pnl*100:.3f}%"
                
            # 3. TEMPO LIMITE CURTO
            elif duration >= self.max_position_time:
                should_close = True
                reason = f"TEMPO LIMITE: {duration:.0f}s"
                
            # 4. BREAKEVEN após 15 segundos se não está ganhando
            elif duration >= 15 and pnl < 0.002:  # 0.2%
                should_close = True
                reason = f"BREAKEVEN: {pnl*100:.3f}%"
                
            # 5. TRAILING STOP dinâmico
            elif pnl > 0.003 and duration >= 20:  # Se teve lucro mas pode estar caindo
                # Verificar se preço está caindo nos últimos ticks
                if len(self.price_history) >= 3:
                    recent_trend = self.price_history[-1] < self.price_history[-2] < self.price_history[-3]
                    if recent_trend and pnl < self.profit_target * 0.6:
                        should_close = True
                        reason = f"TRAILING: {pnl*100:.3f}%"
            
            # Log a cada 5 segundos
            if int(duration) % 5 == 0 and int(duration) > 0:
                logger.info(f"Posição: {pnl*100:.3f}% | {duration:.0f}s")
            
            if should_close:
                logger.info(f"FECHANDO: {reason}")
                self._close_position_simple(reason)
                
        except Exception as e:
            logger.error(f"Erro gerenciando posição: {e}")
            self._close_position_simple("ERRO CRÍTICO")

    def _close_position_simple(self, reason: str) -> bool:
        """MÉTODO ÚNICO DE FECHAMENTO"""
        
        # Prevenir execuções múltiplas
        with self._lock:
            if self.is_closing or not self.current_position:
                return False
            self.is_closing = True
        
        try:
            current_price = self.price_history[-1] if self.price_history else self.current_position.entry_price
            final_pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            logger.info(f"FECHANDO POSIÇÃO: {reason}")
            logger.info(f"   PnL: {final_pnl*100:.4f}%")
            logger.info(f"   Duração: {duration:.1f}s")
            
            success = False
            
            if self.paper_trading:
                success = True
                logger.info("PAPER TRADING - Fechamento simulado")
            else:
                try:
                    # FECHAR POSIÇÃO REAL
                    if self.current_position.side == TradeDirection.LONG:
                        result = self.bitget_api.place_sell_order(profit_target=0)
                        success = result and result.get('success', False)
                        
                        if not success:
                            error = result.get('error', 'Erro desconhecido') if result else 'Sem resposta'
                            logger.error(f"Falha na venda: {error}")
                    else:
                        # Para SHORT (quando implementado)
                        result = self.bitget_api.place_buy_order()
                        success = result and result.get('success', False)
                        
                except Exception as e:
                    logger.error(f"Erro executando fechamento: {e}")
                    success = False
            
            # PROCESSAR RESULTADO
            if success or self.paper_trading:
                logger.info(f"POSIÇÃO FECHADA COM SUCESSO!")
                
                # Atualizar métricas
                self.metrics.total_trades += 1
                self.metrics.total_profit += final_pnl
                
                if final_pnl > 0:
                    self.metrics.profitable_trades += 1
                    self.metrics.consecutive_wins += 1
                    self.consecutive_losses = 0
                    logger.info(f"LUCRO: {final_pnl*100:.4f}%")
                else:
                    self.metrics.consecutive_wins = 0
                    self.consecutive_losses += 1
                    self.daily_loss += abs(final_pnl)
                    logger.info(f"PERDA: {final_pnl*100:.4f}%")
                
                # Atualizar drawdown
                if final_pnl < 0:
                    self.metrics.max_drawdown = max(self.metrics.max_drawdown, abs(final_pnl))
                
                logger.info(f"Win Rate: {self.metrics.win_rate:.1f}% | Trades: {self.trades_today}")
                
                # Limpar posição
                self.current_position = None
                
                return True
            else:
                logger.error(f"FALHA NO FECHAMENTO!")
                self.current_position = None
                return False
                
        except Exception as e:
            logger.error(f"ERRO CRÍTICO no fechamento: {e}")
            self.current_position = None
            return False
        finally:
            self.is_closing = False

    def _should_stop_trading(self) -> bool:
        """Verificar se deve parar - MAIS TOLERANTE"""
        if self.daily_loss >= self.max_daily_loss:
            logger.warning(f"Perda diária máxima: {self.daily_loss*100:.2f}%")
            return True
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"Perdas consecutivas: {self.consecutive_losses}")
            return True
        
        return False

    def _can_trade(self) -> bool:
        """Verificar se pode operar - MUITO PERMISSIVO"""
        if time.time() - self.last_trade_time < self.min_time_between_trades:
            return False
        
        # Muito tolerante com perdas consecutivas
        if self.consecutive_losses >= 5:  # Só parar após 5 perdas
            return False
        
        return True

    def _get_balance(self) -> float:
        """Obter saldo"""
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
        """Status do bot"""
        try:
            return {
                'bot_status': {
                    'state': self.state.value,
                    'is_running': self.is_running,
                    'symbol': self.symbol,
                    'paper_trading': self.paper_trading,
                    'mode': 'ULTRA_AGGRESSIVE_SCALPING'
                },
                'performance': {
                    'trades_today': self.trades_today,
                    'total_trades': self.metrics.total_trades,
                    'win_rate': round(self.metrics.win_rate, 1),
                    'net_profit': round(self.metrics.net_profit * 100, 4),
                    'daily_loss': round(self.daily_loss * 100, 3),
                    'consecutive_losses': self.consecutive_losses,
                    'max_drawdown': round(self.metrics.max_drawdown * 100, 3)
                },
                'current_position': self._get_position_status(),
                'targets': {
                    'take_profit': f"{self.profit_target*100:.1f}%",
                    'stop_loss': f"{self.stop_loss*100:.1f}%",
                    'max_time': f"{self.max_position_time}s"
                },
                'trading_config': {
                    'scalping_interval': f"{self.scalping_interval}s",
                    'min_confidence': "55%",
                    'max_daily_loss': f"{self.max_daily_loss*100:.0f}%"
                }
            }
        except Exception as e:
            return {'error': str(e), 'is_running': False}

    def _get_position_status(self) -> Dict:
        """Status da posição"""
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
                'is_closing': self.is_closing,
                'target_price': self.current_position.target_price,
                'stop_price': self.current_position.stop_price
            }
        except Exception as e:
            return {'active': True, 'error': str(e)}

    def emergency_stop(self) -> bool:
        """Parada de emergência"""
        try:
            logger.warning("PARADA DE EMERGÊNCIA")
            self.state = TradingState.STOPPED
            
            if self.current_position:
                self._close_position_simple("EMERGENCY STOP")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na parada de emergência: {e}")
            return False

    def get_daily_stats(self) -> Dict:
        """Estatísticas diárias"""
        try:
            return {
                'daily_performance': {
                    'trades_executed': self.trades_today,
                    'current_profit': f"{self.metrics.net_profit*100:.3f}%",
                    'win_rate': f"{self.metrics.win_rate:.1f}%",
                    'profitable_trades': self.metrics.profitable_trades,
                    'losing_trades': self.metrics.total_trades - self.metrics.profitable_trades,
                    'avg_trade_duration': f"{self.max_position_time}s max"
                },
                'risk_metrics': {
                    'daily_loss': f"{self.daily_loss*100:.3f}%",
                    'max_drawdown': f"{self.metrics.max_drawdown*100:.3f}%",
                    'consecutive_losses': self.consecutive_losses,
                    'risk_level': 'ULTRA_HIGH'
                },
                'bot_config': {
                    'take_profit': f"{self.profit_target*100:.1f}%",
                    'stop_loss': f"{self.stop_loss*100:.1f}%",
                    'max_position_time': f"{self.max_position_time}s",
                    'scalping_interval': f"{self.scalping_interval}s",
                    'trading_mode': 'ULTRA_AGGRESSIVE'
                }
            }
        except Exception as e:
            return {'error': str(e)}


# Função para criar bot
def create_trading_bot(bitget_api: BitgetAPI, **kwargs) -> TradingBot:
    """Criar bot ultra-agressivo"""
    return TradingBot(bitget_api, **kwargs)


# Teste
if __name__ == "__main__":
    try:
        from bitget_api import BitgetAPI
        
        api = BitgetAPI()
        if api.test_connection():
            bot = TradingBot(
                bitget_api=api,
                paper_trading=True,
                leverage=10,
                scalping_interval=1.5
            )
            
            print("Bot ULTRA-AGRESSIVO criado com sucesso!")
            print("CONFIGURAÇÕES CORRIGIDAS:")
            print("   Take Profit: 0.9% (mantido original)")
            print("   Stop Loss: 0.4% (mantido original)")
            print("   Max Time: 60s")
            print("   Min Confidence: 55%")
            print("   Scalping Interval: 1.5s")
            print("   MODO: Scalping agressivo com análise rápida")
            
        else:
            print("Falha na conexão com a API")
    except Exception as e:
        print(f"Erro: {e}")
        traceback.print_exc()
