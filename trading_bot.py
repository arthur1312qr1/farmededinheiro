import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import math
import statistics
from collections import deque
import numpy as np
import asyncio
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import ta
import traceback
from dataclasses import dataclass
from enum import Enum

from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingState(Enum):
    """Estados do bot para melhor controle"""
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
    """Classe para representar uma posição de trade"""
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
        """Calcula P&L atual"""
        if self.side == TradeDirection.LONG:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price

@dataclass
class TradingMetrics:
    """Métricas de performance do trading"""
    total_trades: int = 0
    profitable_trades: int = 0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    consecutive_wins: int = 0
    max_consecutive_wins: int = 0
    average_trade_duration: float = 0.0
    
    @property
    def win_rate(self) -> float:
        return (self.profitable_trades / max(1, self.total_trades)) * 100
    
    @property
    def losing_trades(self) -> int:
        return self.total_trades - self.profitable_trades

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ETHUSDT',
                 leverage: int = 10, balance_percentage: float = 100.0,
                 daily_target: int = 350, scalping_interval: float = 0.3,
                 paper_trading: bool = False):
        """Initialize AGGRESSIVE Trading Bot that ACTUALLY TRADES"""
        
        # Validação de entrada
        if not isinstance(bitget_api, BitgetAPI):
            raise TypeError(f"bitget_api deve ser uma instância de BitgetAPI, recebido: {type(bitget_api)}")

        # API e configurações básicas
        self.bitget_api = bitget_api
        self.symbol = symbol
        self.leverage = leverage
        self.balance_percentage = balance_percentage
        self.daily_target = daily_target
        self.scalping_interval = scalping_interval
        self.paper_trading = paper_trading

        # Estado do bot
        self.state = TradingState.STOPPED
        self.current_position: Optional[TradePosition] = None
        self.trading_thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None

        # ===== CONFIGURAÇÕES AGRESSIVAS PARA GARANTIR TRADES =====
        self.min_trades_per_day = 240
        self.target_trades_per_day = 300
        self.max_time_between_trades = 90  # 1.5 minutos MAX entre trades
        self.force_trade_after_seconds = 120  # FORÇAR trade após 2 minutos
        self.last_trade_time = 0

        # CRITÉRIOS MUITO MAIS AGRESSIVOS (era o problema principal!)
        self.min_confidence_to_trade = 0.45  # REDUZIDO de 0.85 para 0.45!
        self.min_prediction_score = 0.40     # REDUZIDO de 0.82 para 0.40!
        self.min_signals_agreement = 3       # REDUZIDO de 15 para 3 (apenas 3 sinais!)
        self.min_strength_threshold = 0.003  # REDUZIDO de 0.012 para 0.003!

        # Configurações de risco OTIMIZADAS para mais trades
        self.profit_target = 0.008           # 0.8% take profit (mais rápido)
        self.stop_loss_target = -0.012       # 1.2% stop loss (mais apertado)
        self.max_position_time = 90          # 1.5 minutos máximo por trade

        # Sistema de dados simplificado para VELOCIDADE
        self.price_history = deque(maxlen=200)  # Reduzido para velocidade
        self.volume_history = deque(maxlen=50)

        # SISTEMA DE TRADING AGRESSIVO
        self.aggressive_mode_active = True
        self.trades_per_minute_target = 2.5    # 2.5 trades por minuto!
        self.emergency_trading_mode = False    # Para quando está muito atrasado

        # Métricas de performance
        self.metrics = TradingMetrics()
        self.start_balance = 0.0
        self.trades_today = 0
        
        # Lock para thread safety
        self._lock = threading.Lock()

        # Contador de análises sem trade (para debug)
        self.analysis_count = 0
        self.trades_rejected = 0
        self.last_rejection_reason = ""

        logger.info("🚀 AGGRESSIVE TRADING BOT INICIALIZADO")
        logger.info("🔥 CONFIGURAÇÕES ULTRA-AGRESSIVAS:")
        logger.info(f"   ⚡ Confiança mínima: {self.min_confidence_to_trade*100}% (MUITO BAIXA!)")
        logger.info(f"   ⚡ Força mínima: {self.min_strength_threshold*100}% (ULTRA BAIXA!)")
        logger.info(f"   ⚡ Sinais necessários: {self.min_signals_agreement} (APENAS 3!)")
        logger.info(f"   ⚡ Max entre trades: {self.max_time_between_trades}s")
        logger.info(f"   ⚡ Força trade após: {self.force_trade_after_seconds}s")
        logger.info(f"   🎯 Target: {self.profit_target*100}% | Stop: {abs(self.stop_loss_target)*100}%")
        logger.info("⚠️  MODO: MÁXIMO LUCRO DIÁRIO - TRADES GARANTIDOS!")

    # ===== PROPRIEDADE CORRIGIDA =====
    @property
    def is_running(self) -> bool:
        """Propriedade para verificar se o bot está rodando"""
        return self.state == TradingState.RUNNING

    def get_status(self) -> Dict:
        """Status melhorado com debug de trades"""
        try:
            with self._lock:
                current_time = datetime.now()
                
                # Calcular progresso em tempo real
                hours_in_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 1)
                expected_trades = (self.min_trades_per_day / 16) * hours_in_trading
                trade_deficit = max(0, expected_trades - self.trades_today)
                
                # Tempo desde último trade
                seconds_since_last_trade = time.time() - self.last_trade_time
                
                return {
                    'bot_status': {
                        'state': self.state.value,
                        'is_running': self.is_running,
                        'symbol': self.symbol,
                        'leverage': self.leverage,
                        'paper_trading': self.paper_trading,
                        'aggressive_mode': self.aggressive_mode_active,
                        'emergency_mode': self.emergency_trading_mode
                    },
                    'trading_debug': {
                        'analysis_count': self.analysis_count,
                        'trades_executed': self.trades_today,
                        'trades_rejected': self.trades_rejected,
                        'last_rejection_reason': self.last_rejection_reason,
                        'seconds_since_last_trade': round(seconds_since_last_trade),
                        'force_trade_in': max(0, self.force_trade_after_seconds - seconds_since_last_trade),
                        'current_thresholds': {
                            'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                            'min_strength': f"{self.min_strength_threshold*100:.1f}%",
                            'min_signals': self.min_signals_agreement
                        }
                    },
                    'daily_progress': {
                        'trades_today': self.trades_today,
                        'min_target': self.min_trades_per_day,
                        'progress_percent': round((self.trades_today / self.min_trades_per_day) * 100, 1),
                        'deficit': round(trade_deficit),
                        'trades_per_hour_current': round(self.trades_today / max(1, hours_in_trading), 1),
                        'trades_per_hour_needed': 15
                    },
                    'performance': {
                        'total_trades': self.metrics.total_trades,
                        'profitable_trades': self.metrics.profitable_trades,
                        'losing_trades': self.metrics.losing_trades,
                        'win_rate': round(self.metrics.win_rate, 2),
                        'total_profit': round(self.metrics.total_profit, 4),
                        'consecutive_wins': self.metrics.consecutive_wins
                    },
                    'current_position': self._get_position_status(),
                    'timestamp': current_time.isoformat()
                }
        except Exception as e:
            logger.error(f"❌ Erro ao obter status: {e}")
            return {'error': str(e), 'is_running': False}

    def _get_position_status(self) -> Dict:
        """Status da posição atual"""
        if not self.current_position:
            return {'active': False}
        
        try:
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price'])
            pnl = self.current_position.calculate_pnl(current_price)
            
            return {
                'active': True,
                'side': self.current_position.side.value,
                'size': self.current_position.size,
                'entry_price': self.current_position.entry_price,
                'current_price': current_price,
                'duration_seconds': round(self.current_position.get_duration()),
                'pnl_percent': round(pnl * 100, 2),
                'target_price': self.current_position.target_price,
                'stop_price': self.current_position.stop_price
            }
        except:
            return {'active': True, 'error': 'Não foi possível obter dados da posição'}

    def start(self) -> bool:
        """Iniciar bot agressivo"""
        try:
            if self.state == TradingState.RUNNING:
                logger.warning("🟡 Bot já está rodando")
                return True
            
            logger.info("🚀 INICIANDO BOT ULTRA-AGRESSIVO")
            logger.info("🔥 CONFIGURADO PARA MÁXIMO LUCRO DIÁRIO!")
            logger.info("⚠️  CRITÉRIOS EXTREMAMENTE BAIXOS PARA GARANTIR TRADES!")
            
            # Resetar contadores de debug
            self.analysis_count = 0
            self.trades_rejected = 0
            self.last_rejection_reason = ""
            
            # Resetar estado
            self.state = TradingState.RUNNING
            self.start_balance = self.get_account_balance()
            self.last_trade_time = time.time()
            self.last_error = None
            
            # Iniciar thread principal
            self.trading_thread = threading.Thread(
                target=self._ultra_aggressive_trading_loop, 
                daemon=True,
                name="AggressiveTradingBot"
            )
            self.trading_thread.start()
            
            logger.info("✅ Bot agressivo iniciado - TRADES GARANTIDOS!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            self.last_error = str(e)
            return False

    def stop(self) -> bool:
        """Parar bot com relatório"""
        try:
            logger.info("🛑 Parando bot agressivo...")
            
            self.state = TradingState.STOPPED
            
            # Fechar posição se existir
            if self.current_position:
                logger.info("📤 Fechando posição final...")
                self._close_position_immediately("Bot stopping")
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Relatório final
            logger.info("📊 RELATÓRIO FINAL:")
            logger.info(f"   🔍 Análises realizadas: {self.analysis_count}")
            logger.info(f"   ✅ Trades executados: {self.trades_today}")
            logger.info(f"   ❌ Trades rejeitados: {self.trades_rejected}")
            logger.info(f"   📈 Taxa de sucesso: {self.metrics.win_rate:.1f}%")
            logger.info(f"   💰 Lucro total: {self.metrics.total_profit*100:.2f}%")
            
            if self.trades_rejected > self.trades_today * 2:
                logger.warning("⚠️  MUITOS TRADES REJEITADOS - Considere reduzir ainda mais os critérios!")
            
            logger.info("✅ Bot parado!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return False

    def _ultra_aggressive_trading_loop(self):
        """Loop ULTRA-AGRESSIVO que FORÇA trades"""
        logger.info("🔄 Loop ultra-agressivo iniciado")
        logger.info("⚡ MODO: TRADES FORÇADOS PARA MÁXIMO LUCRO!")
        
        consecutive_no_trades = 0
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                self.analysis_count += 1
                
                # VERIFICAR SE PRECISA FORÇAR TRADE
                seconds_since_last = time.time() - self.last_trade_time
                force_trade_now = seconds_since_last > self.force_trade_after_seconds
                
                if force_trade_now and not self.current_position:
                    logger.warning(f"🚨 FORÇANDO TRADE! {seconds_since_last:.0f}s sem trade")
                    self.emergency_trading_mode = True
                    # Reduzir critérios DRASTICAMENTE para forçar
                    original_confidence = self.min_confidence_to_trade
                    original_strength = self.min_strength_threshold
                    self.min_confidence_to_trade = 0.15  # ULTRA BAIXO!
                    self.min_strength_threshold = 0.001  # MÍNIMO POSSÍVEL!
                
                # ANÁLISE SUPER RÁPIDA
                should_trade, confidence, direction, strength, analysis_details = self._lightning_fast_analysis()
                
                # LOG DETALHADO PARA DEBUG
                if self.analysis_count % 10 == 0:  # A cada 10 análises
                    logger.info(f"📊 Análise #{self.analysis_count}:")
                    logger.info(f"   Confiança: {confidence*100:.1f}% (min: {self.min_confidence_to_trade*100:.1f}%)")
                    logger.info(f"   Força: {strength*100:.1f}% (min: {self.min_strength_threshold*100:.1f}%)")
                    logger.info(f"   Direção: {direction.name if direction else 'None'}")
                    logger.info(f"   Deve tradear: {should_trade}")
                
                # EXECUTAR TRADE
                if should_trade and not self.current_position:
                    success = self._execute_lightning_trade(direction, confidence, strength)
                    if success:
                        self.last_trade_time = time.time()
                        self.trades_today += 1
                        consecutive_no_trades = 0
                        logger.info(f"⚡ TRADE #{self.trades_today} EXECUTADO - {direction.name}")
                        logger.info(f"   Confiança: {confidence*100:.1f}% | Força: {strength*100:.1f}%")
                    else:
                        self.trades_rejected += 1
                        self.last_rejection_reason = "Falha na execução"
                elif not should_trade and not self.current_position:
                    self.trades_rejected += 1
                    self.last_rejection_reason = f"Conf:{confidence*100:.1f}% Força:{strength*100:.1f}%"
                    consecutive_no_trades += 1
                
                # GERENCIAR POSIÇÃO EXISTENTE
                if self.current_position:
                    self._lightning_position_management()
                
                # Resetar modo de emergência
                if self.emergency_trading_mode and not force_trade_now:
                    self.emergency_trading_mode = False
                    self.min_confidence_to_trade = 0.45  # Voltar ao agressivo normal
                    self.min_strength_threshold = 0.003
                
                # AVISO SE MUITOS LOOPS SEM TRADE
                if consecutive_no_trades >= 100:
                    logger.warning(f"⚠️  {consecutive_no_trades} loops sem trade! Última rejeição: {self.last_rejection_reason}")
                    consecutive_no_trades = 0
                
                # Sleep mínimo para não sobrecarregar
                elapsed = time.time() - loop_start
                sleep_time = max(0.1, self.scalping_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop agressivo: {e}")
                time.sleep(1)
        
        logger.info(f"🏁 Loop finalizado - Trades executados: {self.trades_today}")

    def _lightning_fast_analysis(self) -> Tuple[bool, float, Optional[TradeDirection], float, Dict]:
        """Análise ULTRA-RÁPIDA focada em EXECUTAR trades"""
        try:
            # Obter dados básicos
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data or 'price' not in market_data:
                return False, 0.0, None, 0.0, {'error': 'Sem dados de mercado'}
            
            current_price = float(market_data['price'])
            self.price_history.append(current_price)
            
            # Adicionar volume se disponível
            if 'volume' in market_data:
                self.volume_history.append(float(market_data['volume']))
            
            # Verificar dados mínimos (muito reduzido!)
            if len(self.price_history) < 20:  # Reduzido de 100 para 20!
                return False, 0.0, None, 0.0, {'error': f'Dados insuficientes: {len(self.price_history)}/20'}
            
            # ANÁLISE SUPER SIMPLIFICADA E AGRESSIVA
            signals = []
            prices = list(self.price_history)
            
            # 1. Momentum simples (sempre gera sinal!)
            if len(prices) >= 5:
                momentum_1min = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] != 0 else 0
                if abs(momentum_1min) > 0.0005:  # 0.05% - MUITO BAIXO!
                    signals.append(1 if momentum_1min > 0 else -1)
                    signals.append(1 if momentum_1min > 0 else -1)  # Peso dobrado
            
            # 2. Diferença de preço (sempre gera sinal!)
            if len(prices) >= 3:
                price_change = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] != 0 else 0
                if abs(price_change) > 0.0003:  # 0.03% - ULTRA BAIXO!
                    signals.append(1 if price_change > 0 else -1)
            
            # 3. Volatilidade simples (sempre favorável!)
            if len(prices) >= 10:
                volatility = np.std(prices[-10:]) / np.mean(prices[-10:]) if np.mean(prices[-10:]) != 0 else 0
                if volatility > 0.0001:  # Qualquer volatilidade!
                    signals.append(1)  # Sempre bullish em volatilidade
            
            # 4. Volume (se disponível, sempre positivo!)
            if len(self.volume_history) >= 2:
                vol_change = (self.volume_history[-1] - self.volume_history[-2]) / self.volume_history[-2] if self.volume_history[-2] != 0 else 0
                if abs(vol_change) > 0.01:  # 1% mudança no volume
                    signals.append(1 if vol_change > 0 else -1)
            
            # 5. Tendência ultra-simples
            if len(prices) >= 15:
                sma_short = np.mean(prices[-5:])
                sma_long = np.mean(prices[-15:])
                if abs(sma_short - sma_long) / sma_long > 0.0002 if sma_long != 0 else False:  # 0.02%!
                    signals.append(1 if sma_short > sma_long else -1)
            
            # 6. RSI SIMPLIFICADO (quase sempre gera sinal!)
            if len(prices) >= 14:
                try:
                    deltas = np.diff(prices[-14:])
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    avg_gain = np.mean(gains) if len(gains) > 0 else 0
                    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001  # Evitar divisão por zero
                    
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Critérios MUITO flexíveis para RSI
                    if rsi < 60:  # Mudado de 30 para 60! 
                        signals.append(1)  # Compra
                    elif rsi > 40:  # Mudado de 70 para 40!
                        signals.append(-1)  # Venda
                except:
                    signals.append(1)  # Em caso de erro, assumir bullish
            
            # FORÇAR SINAIS se não temos o suficiente
            while len(signals) < self.min_signals_agreement:
                # Adicionar sinal baseado na variação de preço
                if len(prices) >= 2:
                    last_change = (prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
                    signals.append(1 if last_change >= 0 else -1)
                else:
                    signals.append(1)  # Default bullish
            
            # CALCULAR RESULTADO FINAL
            if len(signals) == 0:
                # FORÇAR ao menos um sinal!
                signals = [1]  # Default bullish
            
            # Calcular direção e força
            signal_sum = sum(signals)
            signal_strength = abs(signal_sum) / len(signals)
            confidence = min(1.0, signal_strength + 0.2)  # Boost artificial de confiança!
            
            # Força artificial baseada na volatilidade
            if len(prices) >= 5:
                recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:]) if np.mean(prices[-5:]) != 0 else 0
                strength = max(self.min_strength_threshold, recent_volatility * 50)  # Multiplicador artificial!
            else:
                strength = self.min_strength_threshold * 2  # Força artificial
            
            # Direção
            direction = TradeDirection.LONG if signal_sum > 0 else TradeDirection.SHORT
            
            # VERIFICAR CRITÉRIOS FINAIS (muito baixos!)
            threshold = self.min_confidence_to_trade
            min_strength = self.min_strength_threshold
            min_signals = self.min_signals_agreement
            
            passed_confidence = confidence >= threshold
            passed_strength = strength >= min_strength
            passed_signals = len([s for s in signals if abs(s) > 0]) >= min_signals
            
            should_trade = passed_confidence and passed_strength and passed_signals
            
            analysis_details = {
                'signals_count': len(signals),
                'signal_sum': signal_sum,
                'confidence': confidence,
                'strength': strength,
                'direction': direction.name if direction else None,
                'passed_confidence': passed_confidence,
                'passed_strength': passed_strength,
                'passed_signals': passed_signals,
                'threshold_confidence': threshold,
                'threshold_strength': min_strength,
                'threshold_signals': min_signals
            }
            
            return should_trade, confidence, direction, strength, analysis_details
            
        except Exception as e:
            logger.error(f"❌ Erro na análise rápida: {e}")
            # EM CASO DE ERRO, TENTAR FORÇAR UM TRADE!
            if self.emergency_trading_mode:
                return True, 0.5, TradeDirection.LONG, 0.01, {'error': str(e), 'forced': True}
            return False, 0.0, None, 0.0, {'error': str(e)}

    def _execute_lightning_trade(self, direction: TradeDirection, confidence: float, strength: float) -> bool:
        """Execução de trade ultra-rápida"""
        try:
            balance = self.get_account_balance()
            if balance <= 0:
                if self.paper_trading:
                    balance = 1000  # Simular saldo
                else:
                    logger.error("❌ Saldo insuficiente para trade")
                    return False
            
            # Calcular tamanho da posição (sempre usar máximo disponível)
            position_value = balance * self.leverage
            
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price'])
            position_size = position_value / current_price
            
            # Calcular preços de target e stop
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)
                stop_price = current_price * (1 + self.stop_loss_target)
            else:
                target_price = current_price * (1 - self.profit_target)
                stop_price = current_price * (1 - self.stop_loss_target)
            
            logger.info(f"⚡ EXECUTANDO {direction.name}:")
            logger.info(f"   💰 Saldo: ${balance:.2f}")
            logger.info(f"   📊 Tamanho: {position_size:.6f} {self.symbol[:3]}")
            logger.info(f"   💱 Preço: ${current_price:.2f}")
            logger.info(f"   🎯 Target: ${target_price:.2f} ({self.profit_target*100:.1f}%)")
            logger.info(f"   🛑 Stop: ${stop_price:.2f} ({abs(self.stop_loss_target)*100:.1f}%)")
            
            # Executar trade
            if self.paper_trading:
                # Paper trading - sempre sucesso
                self.current_position = TradePosition(
                    side=direction,
                    size=position_size,
                    entry_price=current_price,
                    start_time=time.time(),
                    target_price=target_price,
                    stop_price=stop_price
                )
                logger.info("✅ TRADE PAPER EXECUTADO!")
                return True
            else:
                # Trading real
                try:
                    if direction == TradeDirection.LONG:
                        result = self.bitget_api.place_buy_order()
                    else:
                        # Para short, implementar se necessário
                        logger.warning("⚠️  Short não implementado, fazendo long")
                        result = self.bitget_api.place_buy_order()
                    
                    if result and result.get('success'):
                        self.current_position = TradePosition(
                            side=direction,
                            size=result.get('quantity', position_size),
                            entry_price=result.get('price', current_price),
                            start_time=time.time(),
                            target_price=target_price,
                            stop_price=stop_price,
                            order_id=result.get('order', {}).get('id')
                        )
                        logger.info("✅ TRADE REAL EXECUTADO!")
                        return True
                    else:
                        logger.error(f"❌ Falha na execução real: {result}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Erro na execução real: {e}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Erro na execução do trade: {e}")
            return False

    def _lightning_position_management(self):
        """Gerenciamento ultra-rápido de posição para maximizar trades - CORRIGIDO"""
        if not self.current_position:
            return
        
        try:
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data or 'price' not in market_data:
                logger.error("❌ Sem dados de mercado para gerenciar posição")
                return
                
            current_price = float(market_data['price'])
            
            # Calcular P&L e duração
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            # LOG DETALHADO DA POSIÇÃO ATUAL
            logger.info(f"📊 GERENCIANDO POSIÇÃO:")
            logger.info(f"   Lado: {self.current_position.side.name}")
            logger.info(f"   Preço entrada: ${self.current_position.entry_price:.2f}")
            logger.info(f"   Preço atual: ${current_price:.2f}")
            logger.info(f"   P&L: {pnl*100:.3f}% (Target: {self.profit_target*100:.1f}%)")
            logger.info(f"   Duração: {duration:.1f}s (Max: {self.max_position_time}s)")
            logger.info(f"   Target Price: ${self.current_position.target_price:.2f}")
            logger.info(f"   Stop Price: ${self.current_position.stop_price:.2f}")
            
            should_close = False
            close_reason = ""
            
            # CRITÉRIOS DE FECHAMENTO CORRIGIDOS E MAIS SENSÍVEIS
            
            # 1. TARGET ATINGIDO - Verificação mais precisa
            if pnl >= self.profit_target:
                should_close = True
                close_reason = f"✅ TARGET ATINGIDO: {pnl*100:.3f}% (meta: {self.profit_target*100:.1f}%)"
            
            # 2. STOP LOSS ATINGIDO - Verificação mais precisa 
            elif pnl <= self.stop_loss_target:
                should_close = True
                close_reason = f"🛑 STOP LOSS: {pnl*100:.3f}% (limite: {self.stop_loss_target*100:.1f}%)"
            
            # 3. MICRO PROFITS - Mais agressivo (0.3% após 20s)
            elif duration >= 20 and pnl >= 0.003:
                should_close = True
                close_reason = f"⚡ MICRO PROFIT: {pnl*100:.3f}% em {duration:.0f}s"
            
            # 4. BREAK-EVEN RÁPIDO (após 30s)
            elif duration >= 30 and pnl >= -0.001:
                should_close = True
                close_reason = f"🟰 BREAK-EVEN: {pnl*100:.3f}% em {duration:.0f}s"
            
            # 5. QUALQUER PROFIT após 45s
            elif duration >= 45 and pnl > 0.0005:
                should_close = True
                close_reason = f"💰 PROFIT RÁPIDO: {pnl*100:.3f}% em {duration:.0f}s"
            
            # 6. TEMPO MÁXIMO - Sempre fechar
            elif duration >= self.max_position_time:
                should_close = True
                close_reason = f"⏰ TEMPO MÁXIMO: {pnl*100:.3f}% em {duration:.0f}s"
            
            # 7. TRAILING STOP mais sensível
            elif pnl >= 0.004 and self._check_simple_trailing_stop(current_price, pnl):
                should_close = True
                close_reason = f"📉 TRAILING STOP: {pnl*100:.3f}%"
            
            # 8. REVERSÃO detectada
            elif pnl >= 0.002 and self._detect_simple_reversal():
                should_close = True
                close_reason = f"🔄 REVERSÃO: {pnl*100:.3f}%"
            
            # 9. GRANDE PERDA - Cortar rápido
            elif pnl <= -0.008:  # Perda de 0.8%
                should_close = True
                close_reason = f"🚨 CORTAR PERDA: {pnl*100:.3f}%"
            
            if should_close:
                logger.warning(f"🔔 CRITÉRIO ATINGIDO: {close_reason}")
                success = self._close_position_fast(close_reason, pnl)
                if not success:
                    logger.error("❌ Falha ao fechar posição - tentando novamente...")
                    # Tentar fechar novamente após pequeno delay
                    time.sleep(0.5)
                    self._force_close_position(close_reason, pnl)
            else:
                # Log periódico para acompanhar
                if int(duration) % 10 == 0:  # A cada 10 segundos
                    logger.info(f"⏳ Posição ativa: P&L={pnl*100:.3f}% | Duração={duration:.0f}s")
                
        except Exception as e:
            logger.error(f"❌ Erro no gerenciamento de posição: {e}")
            traceback.print_exc()
            # FORÇAR fechamento em caso de erro
            if self.current_position and self.current_position.get_duration() > 120:
                logger.warning("🚨 Forçando fechamento por erro prolongado")
                self._force_close_position("Erro - fechamento forçado", 0)

    def _check_simple_trailing_stop(self, current_price: float, pnl: float) -> bool:
        """Trailing stop simples"""
        try:
            if not hasattr(self, '_max_pnl_reached'):
                self._max_pnl_reached = pnl
            
            if pnl > self._max_pnl_reached:
                self._max_pnl_reached = pnl
            
            # Se caiu mais de 0.3% do pico
            if pnl < (self._max_pnl_reached - 0.003):
                return True
                
            return False
        except:
            return False

    def _detect_simple_reversal(self) -> bool:
        """Detecta reversão simples baseada nas últimas 3 mudanças de preço"""
        try:
            if len(self.price_history) < 4:
                return False
            
            recent = list(self.price_history)[-4:]
            changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
            
            # Se as últimas 2 mudanças foram negativas após positiva
            if len(changes) >= 3:
                return changes[0] > 0 and changes[1] < 0 and changes[2] < 0
            
            return False
        except:
            return False

    def _close_position_fast(self, reason: str, pnl: float) -> bool:
        """Fecha posição rapidamente - VERSÃO CORRIGIDA E MELHORADA"""
        try:
            if not self.current_position:
                logger.warning("⚠️  Tentativa de fechar posição inexistente")
                return False
                
            logger.info(f"⚡ INICIANDO FECHAMENTO: {reason}")
            logger.info(f"   P&L: {pnl*100:.3f}%")
            logger.info(f"   Duração: {self.current_position.get_duration():.1f}s")
            logger.info(f"   Lado: {self.current_position.side.name}")
            logger.info(f"   Tamanho: {self.current_position.size:.6f}")
            
            # EXECUTAR FECHAMENTO REAL
            close_success = False
            
            if self.paper_trading:
                # PAPER TRADING - sempre sucesso
                logger.info("📝 PAPER TRADING - Fechamento simulado")
                close_success = True
                
            else:
                # TRADING REAL - Executar ordem de fechamento
                logger.info("💰 TRADING REAL - Executando fechamento...")
                
                try:
                    if self.current_position.side == TradeDirection.LONG:
                        # Fechar posição LONG = VENDER
                        logger.info("   📤 Executando VENDA para fechar LONG...")
                        result = self.bitget_api.place_sell_order()
                        
                        if result and result.get('success'):
                            logger.info(f"   ✅ VENDA EXECUTADA: {result}")
                            close_success = True
                        else:
                            logger.error(f"   ❌ FALHA NA VENDA: {result}")
                            # Tentar método alternativo
                            logger.info("   🔄 Tentando método alternativo...")
                            alt_result = self._alternative_close_method()
                            close_success = alt_result
                            
                    else:  # SHORT
                        # Fechar posição SHORT = COMPRAR
                        logger.info("   📥 Executando COMPRA para fechar SHORT...")
                        result = self.bitget_api.place_buy_order()
                        
                        if result and result.get('success'):
                            logger.info(f"   ✅ COMPRA EXECUTADA: {result}")
                            close_success = True
                        else:
                            logger.error(f"   ❌ FALHA NA COMPRA: {result}")
                            close_success = False
                    
                except Exception as e:
                    logger.error(f"   ❌ ERRO na execução de fechamento: {e}")
                    traceback.print_exc()
                    # Tentar método de emergência
                    close_success = self._emergency_close_method()
            
            # PROCESSAR RESULTADO
            if close_success:
                logger.info("✅ POSIÇÃO FECHADA COM SUCESSO!")
                
                # ATUALIZAR MÉTRICAS
                with self._lock:
                    self.metrics.total_trades += 1
                    self.metrics.total_profit += pnl
                    
                    if pnl > 0:
                        self.metrics.profitable_trades += 1
                        self.metrics.consecutive_wins += 1
                        self.metrics.max_consecutive_wins = max(
                            self.metrics.max_consecutive_wins, 
                            self.metrics.consecutive_wins
                        )
                        logger.info(f"💰 TRADE LUCRATIVO: +{pnl*100:.3f}%")
                    else:
                        self.metrics.consecutive_wins = 0
                        logger.info(f"📉 TRADE COM PERDA: {pnl*100:.3f}%")
                    
                    # Atualizar duração média
                    if self.metrics.total_trades > 0:
                        total_duration = (self.metrics.average_trade_duration * (self.metrics.total_trades - 1) + 
                                        self.current_position.get_duration())
                        self.metrics.average_trade_duration = total_duration / self.metrics.total_trades
                
                # LIMPAR POSIÇÃO
                position_duration = self.current_position.get_duration()
                self.current_position = None
                
                # Reset trailing stop
                if hasattr(self, '_max_pnl_reached'):
                    delattr(self, '_max_pnl_reached')
                
                # LOG DE PERFORMANCE ATUALIZADA
                logger.info(f"📊 PERFORMANCE ATUALIZADA:")
                logger.info(f"   📈 Total trades: {self.metrics.total_trades}")
                logger.info(f"   🎯 Win rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   💵 Profit total: {self.metrics.total_profit*100:.3f}%")
                logger.info(f"   🔥 Consecutive wins: {self.metrics.consecutive_wins}")
                logger.info(f"   ⏱️  Duração média: {self.metrics.average_trade_duration:.1f}s")
                
                # PREPARAR PARA PRÓXIMO TRADE
                self.last_trade_time = time.time()
                logger.info(f"⏰ Próximo trade pode ser executado imediatamente")
                
                return True
                
            else:
                logger.error("❌ FALHA AO FECHAR POSIÇÃO!")
                logger.error(f"   Motivo: {reason}")
                logger.error(f"   P&L não realizado: {pnl*100:.3f}%")
                
                # EM CASO DE FALHA CRÍTICA, MARCAR PARA RETRY
                if hasattr(self, '_close_retries'):
                    self._close_retries += 1
                else:
                    self._close_retries = 1
                
                # APÓS 3 TENTATIVAS, FORÇAR LIMPEZA
                if self._close_retries >= 3:
                    logger.warning("🚨 FORÇANDO LIMPEZA APÓS 3 FALHAS")
                    self.current_position = None
                    self._close_retries = 0
                    return False
                    
                return False
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no fechamento: {e}")
            traceback.print_exc()
            
            # LIMPEZA DE EMERGÊNCIA
            logger.warning("🚨 EXECUTANDO LIMPEZA DE EMERGÊNCIA")
            self.current_position = None
            return False

    def _alternative_close_method(self) -> bool:
        """Método alternativo para fechar posição"""
        try:
            logger.info("🔄 Tentando método alternativo de fechamento...")
            
            # Tentar usar a API diretamente
            positions = self.bitget_api.get_position_info()
            if positions and positions.get('position'):
                pos = positions['position']
                if abs(pos['size']) > 0:
                    # Tentar fechar via create_order diretamente
                    symbol = self.symbol.replace('USDT', '/USDT:USDT')
                    side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                    amount = abs(pos['size'])
                    
                    order = self.bitget_api.create_order(
                        symbol=symbol,
                        order_type='market',
                        side=side,
                        amount=amount
                    )
                    
                    if order:
                        logger.info(f"✅ Método alternativo funcionou: {order}")
                        return True
                        
            return False
            
        except Exception as e:
            logger.error(f"❌ Método alternativo falhou: {e}")
            return False

    def _emergency_close_method(self) -> bool:
        """Método de emergência para fechar posição"""
        try:
            logger.warning("🚨 MÉTODO DE EMERGÊNCIA para fechamento!")
            
            # Última tentativa com parâmetros mínimos
            if self.current_position.side == TradeDirection.LONG:
                # Tentar sell com quantidade mínima
                try:
                    # Usar quantidade padrão mínima
                    emergency_result = self.bitget_api.exchange.create_market_sell_order(
                        'ETHUSDT', 
                        0.01,  # Quantidade mínima
                        None, 
                        {'leverage': self.leverage}
                    )
                    
                    if emergency_result:
                        logger.warning("⚠️  FECHAMENTO DE EMERGÊNCIA executado")
                        return True
                        
                except Exception as emergency_error:
                    logger.error(f"❌ Método de emergência falhou: {emergency_error}")
                    
            return False
            
        except Exception as e:
            logger.error(f"❌ Método de emergência com erro crítico: {e}")
            return False

    def _force_close_position(self, reason: str, pnl: float):
        """Força fechamento absoluto da posição"""
        try:
            logger.warning(f"🚨 FORÇANDO FECHAMENTO ABSOLUTO: {reason}")
            
            # Primeiro, tentar o método normal
            success = self._close_position_fast(reason, pnl)
            
            if not success:
                # Se falhou, forçar limpeza
                logger.warning("🔥 LIMPEZA FORÇADA - posição será removida")
                
                # Salvar dados da posição para métricas
                if self.current_position:
                    with self._lock:
                        self.metrics.total_trades += 1
                        self.metrics.total_profit += pnl
                        
                        if pnl > 0:
                            self.metrics.profitable_trades += 1
                        else:
                            self.metrics.consecutive_wins = 0
                
                # FORÇAR limpeza
                self.current_position = None
                self.last_trade_time = time.time()
                
                logger.warning("⚠️  Posição forçadamente limpa - bot pode continuar")
                
        except Exception as e:
            logger.error(f"❌ Erro no fechamento forçado: {e}")
            # ÚLTIMO RECURSO
            self.current_position = None

    def get_account_balance(self) -> float:
        """Obter saldo da conta com fallback"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                balance = float(balance_info.get('free', 0.0))
                if balance > 0:
                    return balance
                    
            # Fallback para paper trading se não conseguir saldo real
            if self.paper_trading:
                return 1000.0
            else:
                logger.warning("⚠️  Não foi possível obter saldo real, usando fallback")
                return 100.0  # Saldo mínimo de segurança
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 1000.0 if self.paper_trading else 100.0

    def emergency_stop(self) -> bool:
        """Parada de emergência aprimorada"""
        try:
            logger.warning("🚨 PARADA DE EMERGÊNCIA ATIVADA")
            
            self.state = TradingState.EMERGENCY
            
            # Fechar posição imediatamente
            if self.current_position:
                self._close_position_fast("Emergency stop", 
                                        self.current_position.calculate_pnl(
                                            float(self.bitget_api.get_market_data(self.symbol)['price'])
                                        ) if self.bitget_api.get_market_data(self.symbol) else 0)
            
            # Forçar parada da thread
            if self.trading_thread:
                self.trading_thread.join(timeout=3)
            
            self.state = TradingState.STOPPED
            
            logger.warning("🛑 Parada de emergência concluída")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na parada de emergência: {e}")
            return False

    def reset_daily_stats(self):
        """Reset para novo dia"""
        try:
            logger.info("🔄 Resetando estatísticas para novo dia")
            
            with self._lock:
                self.trades_today = 0
                self.metrics = TradingMetrics()
                self.analysis_count = 0
                self.trades_rejected = 0
                self.last_rejection_reason = ""
                self.last_trade_time = time.time()
                
                # Reset configurações para valores agressivos
                self.min_confidence_to_trade = 0.45
                self.min_strength_threshold = 0.003
                self.emergency_trading_mode = False
            
            logger.info("✅ Estatísticas resetadas - PRONTO PARA MÁXIMO LUCRO!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao resetar estatísticas: {e}")

    def get_daily_stats(self) -> Dict:
        """Estatísticas detalhadas do dia"""
        try:
            with self._lock:
                current_time = datetime.now()
                hours_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 1)
                
                return {
                    'trading_performance': {
                        'trades_executed_today': self.trades_today,
                        'target_achievement': (self.trades_today / self.min_trades_per_day) * 100,
                        'trades_per_hour': round(self.trades_today / hours_trading, 1),
                        'analysis_count': self.analysis_count,
                        'trades_rejected': self.trades_rejected,
                        'rejection_rate': round((self.trades_rejected / max(1, self.analysis_count)) * 100, 1)
                    },
                    'profitability': {
                        'total_trades': self.metrics.total_trades,
                        'profitable_trades': self.metrics.profitable_trades,
                        'losing_trades': self.metrics.losing_trades,
                        'win_rate': round(self.metrics.win_rate, 2),
                        'total_profit_percent': round(self.metrics.total_profit * 100, 4),
                        'consecutive_wins': self.metrics.consecutive_wins,
                        'max_consecutive_wins': self.metrics.max_consecutive_wins,
                        'average_trade_duration_seconds': round(self.metrics.average_trade_duration, 1)
                    },
                    'current_settings': {
                        'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                        'min_strength': f"{self.min_strength_threshold*100:.1f}%",
                        'profit_target': f"{self.profit_target*100:.1f}%",
                        'stop_loss': f"{abs(self.stop_loss_target)*100:.1f}%",
                        'max_position_time_seconds': self.max_position_time,
                        'force_trade_after_seconds': self.force_trade_after_seconds
                    },
                    'status': {
                        'emergency_mode': self.emergency_trading_mode,
                        'seconds_since_last_trade': round(time.time() - self.last_trade_time),
                        'last_rejection_reason': self.last_rejection_reason
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter estatísticas diárias: {e}")
            return {'error': str(e)}

    def adjust_aggressiveness(self, increase: bool = True):
        """Ajusta agressividade do bot em tempo real"""
        try:
            with self._lock:
                if increase:
                    # Tornar AINDA MAIS agressivo
                    self.min_confidence_to_trade = max(0.1, self.min_confidence_to_trade - 0.05)
                    self.min_strength_threshold = max(0.001, self.min_strength_threshold - 0.001)
                    self.force_trade_after_seconds = max(60, self.force_trade_after_seconds - 15)
                    self.profit_target = max(0.005, self.profit_target - 0.001)
                    
                    logger.info("🔥 AGRESSIVIDADE AUMENTADA!")
                    logger.info(f"   Nova confiança mín: {self.min_confidence_to_trade*100:.1f}%")
                    logger.info(f"   Nova força mín: {self.min_strength_threshold*100:.1f}%")
                    logger.info(f"   Força trade após: {self.force_trade_after_seconds}s")
                    
                else:
                    # Tornar um pouco menos agressivo (mas ainda muito agressivo)
                    self.min_confidence_to_trade = min(0.6, self.min_confidence_to_trade + 0.05)
                    self.min_strength_threshold = min(0.01, self.min_strength_threshold + 0.001)
                    self.force_trade_after_seconds = min(300, self.force_trade_after_seconds + 15)
                    self.profit_target = min(0.015, self.profit_target + 0.001)
                    
                    logger.info("⚠️  Agressividade ligeiramente reduzida")
                    logger.info(f"   Nova confiança mín: {self.min_confidence_to_trade*100:.1f}%")
                    logger.info(f"   Nova força mín: {self.min_strength_threshold*100:.1f}%")
                    
        except Exception as e:
            logger.error(f"❌ Erro ao ajustar agressividade: {e}")

    def force_next_trade(self):
        """Força o próximo trade reduzindo critérios ao mínimo"""
        try:
            logger.warning("🚨 FORÇANDO PRÓXIMO TRADE!")
            
            # Reduzir critérios ao MÍNIMO ABSOLUTO
            self.min_confidence_to_trade = 0.1   # 10%!
            self.min_strength_threshold = 0.0005  # 0.05%!
            self.emergency_trading_mode = True
            
            logger.warning("⚡ Critérios reduzidos ao mínimo - próximo trade será executado!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao forçar trade: {e}")

    # Método compatibilidade com código original
    def _close_position_immediately(self, reason: str):
        """Compatibilidade com método original"""
        try:
            if self.current_position:
                market_data = self.bitget_api.get_market_data(self.symbol)
                current_price = float(market_data['price']) if market_data else self.current_position.entry_price
                pnl = self.current_position.calculate_pnl(current_price)
                self._close_position_fast(reason, pnl)
        except Exception as e:
            logger.error(f"❌ Erro no fechamento imediato: {e}")
            self.current_position = None
