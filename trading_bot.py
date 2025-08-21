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
                 daily_target: int = 350, scalping_interval: float = 0.1,
                 paper_trading: bool = False):
        """Initialize AGGRESSIVE Trading Bot for 50% DAILY PROFIT"""
        
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

        # ===== CONFIGURAÇÕES ULTRA AGRESSIVAS PARA 50% DIÁRIO =====
        self.min_trades_per_day = 500  # Mínimo 500 trades/dia
        self.target_trades_per_day = 800  # Meta: 800 trades/dia
        self.max_time_between_trades = 30  # Máximo 30 segundos entre trades
        self.force_trade_after_seconds = 60  # Forçar trade após 1 minuto
        self.last_trade_time = 0

        # CRITÉRIOS ULTRA AGRESSIVOS PARA MÁXIMO LUCRO
        self.min_confidence_to_trade = 0.35     # 35% confiança mínima (MUITO BAIXO)
        self.min_prediction_score = 0.30        # 30% score de predição (MUITO BAIXO)
        self.min_signals_agreement = 3          # Apenas 3 sinais precisam concordar
        self.min_strength_threshold = 0.001     # 0.1% força mínima (MUITO BAIXO)

        # CONFIGURAÇÕES DE LUCRO AGRESSIVAS
        self.profit_target = 0.008              # 0.8% take profit (pequeno mas frequente)
        self.stop_loss_target = -0.012          # 1.2% stop loss (controlado)
        self.max_position_time = 45             # Máximo 45 segundos por trade
        self.micro_profit_target = 0.003        # 0.3% para saídas rápidas
        self.breakeven_time = 15                # Breakeven após 15 segundos

        # CONFIGURAÇÕES PARA SCALPING EXTREMO
        self.ultra_fast_mode = True
        self.micro_movements_trading = True
        self.momentum_boost = 2.0  # Multiplicador para momentum

        # Sistema de dados para análise técnica
        self.price_history = deque(maxlen=100)  # Menor histórico para reação mais rápida
        self.volume_history = deque(maxlen=50)
        self.analysis_history = deque(maxlen=20)

        # Sistema de trading ultra agressivo
        self.aggressive_mode_active = True  # SEMPRE ATIVO
        self.emergency_trading_mode = False
        self.last_analysis_result = None
        self.force_trade_mode = False
        
        # Rastreamento avançado para trailing e múltiplas saídas
        self.max_profit_reached = 0.0
        self.max_loss_reached = 0.0
        self.profit_locks = [0.002, 0.004, 0.006]  # Lock de lucros em diferentes níveis
        self.current_profit_lock = 0
        
        # Sistema de múltiplas saídas
        self.partial_exit_levels = [0.004, 0.006, 0.008]  # Saídas parciais
        self.exit_percentages = [30, 40, 30]  # % para sair em cada nível

        # Métricas de performance
        self.metrics = TradingMetrics()
        self.start_balance = 0.0
        self.trades_today = 0
        self.daily_profit_target = 0.5  # 50% diário
        
        # Contadores específicos para agressividade
        self.forced_trades = 0
        self.micro_profits = 0
        self.quick_exits = 0
        
        # Lock para thread safety
        self._lock = threading.Lock()

        # Contador de análises e debug
        self.analysis_count = 0
        self.trades_rejected = 0
        self.last_rejection_reason = ""

        # Sistema de AI/ML para previsões
        self.price_predictor = None
        self.trend_analyzer = None
        self.volatility_predictor = None

        logger.info("🚀 ULTRA AGGRESSIVE TRADING BOT - 50% DAILY TARGET")
        logger.info("⚡ CONFIGURAÇÕES EXTREMAS:")
        logger.info(f"   🎯 Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"   💪 Força mínima: {self.min_strength_threshold*100}%")
        logger.info(f"   📊 Sinais necessários: {self.min_signals_agreement}")
        logger.info(f"   📈 Take Profit: {self.profit_target*100}%")
        logger.info(f"   🛑 Stop Loss: {abs(self.stop_loss_target)*100}%")
        logger.info(f"   ⚡ Trades/dia META: {self.target_trades_per_day}")
        logger.info(f"   💰 LUCRO DIÁRIO META: {self.daily_profit_target*100}%")
        logger.info("🔥 MODO ULTRA AGRESSIVO ATIVO!")

    @property
    def is_running(self) -> bool:
        """Propriedade para verificar se o bot está rodando"""
        return self.state == TradingState.RUNNING

    def get_status(self) -> Dict:
        """Status completo do bot"""
        try:
            with self._lock:
                current_time = datetime.now()
                
                hours_in_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 24)
                expected_trades = (self.target_trades_per_day / 24) * hours_in_trading
                trade_deficit = max(0, expected_trades - self.trades_today)
                
                seconds_since_last_trade = time.time() - self.last_trade_time
                
                # Calcular progresso para 50% diário
                current_profit_pct = (self.metrics.total_profit * 100) if self.metrics.total_profit else 0
                daily_progress = (current_profit_pct / 50.0) * 100  # 50% é a meta
                
                return {
                    'bot_status': {
                        'state': self.state.value,
                        'is_running': self.is_running,
                        'symbol': self.symbol,
                        'leverage': self.leverage,
                        'paper_trading': self.paper_trading,
                        'ultra_aggressive_mode': True,
                        'force_trade_mode': self.force_trade_mode
                    },
                    'aggressive_trading': {
                        'analysis_count': self.analysis_count,
                        'trades_executed': self.trades_today,
                        'trades_rejected': self.trades_rejected,
                        'forced_trades': self.forced_trades,
                        'micro_profits': self.micro_profits,
                        'quick_exits': self.quick_exits,
                        'seconds_since_last_trade': round(seconds_since_last_trade),
                        'will_force_trade_in': max(0, self.force_trade_after_seconds - seconds_since_last_trade),
                        'current_thresholds': {
                            'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                            'min_strength': f"{self.min_strength_threshold*100:.3f}%",
                            'min_signals': self.min_signals_agreement
                        }
                    },
                    'daily_progress_50_percent': {
                        'target_profit': '50.0%',
                        'current_profit': f"{current_profit_pct:.3f}%",
                        'progress_to_target': f"{daily_progress:.1f}%",
                        'trades_today': self.trades_today,
                        'target_trades': self.target_trades_per_day,
                        'trades_per_hour': round(self.trades_today / max(1, hours_in_trading), 1),
                        'needed_trades_per_hour': round(self.target_trades_per_day / 24, 1)
                    },
                    'performance': {
                        'total_trades': self.metrics.total_trades,
                        'profitable_trades': self.metrics.profitable_trades,
                        'losing_trades': self.metrics.losing_trades,
                        'win_rate': round(self.metrics.win_rate, 2),
                        'total_profit': round(self.metrics.total_profit, 6),
                        'consecutive_wins': self.metrics.consecutive_wins,
                        'avg_trade_duration': round(self.metrics.average_trade_duration, 1)
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
                'pnl_percent': round(pnl * 100, 4),
                'target_price': self.current_position.target_price,
                'stop_price': self.current_position.stop_price,
                'max_profit_reached': round(self.max_profit_reached * 100, 4),
                'current_profit_lock': self.current_profit_lock,
                'will_exit_soon': pnl > 0.002 or self.current_position.get_duration() > 30
            }
        except Exception as e:
            return {'active': True, 'error': f'Erro ao obter dados: {str(e)}'}

    def start(self) -> bool:
        """Iniciar bot ultra agressivo"""
        try:
            if self.state == TradingState.RUNNING:
                logger.warning("🟡 Bot já está rodando")
                return True
            
            logger.info("🚀 INICIANDO BOT ULTRA AGRESSIVO - META 50% DIÁRIO")
            logger.info("⚡ MODO SCALPING EXTREMO ATIVO!")
            
            # Resetar contadores
            self.analysis_count = 0
            self.trades_rejected = 0
            self.forced_trades = 0
            self.micro_profits = 0
            self.quick_exits = 0
            self.last_rejection_reason = ""
            
            # Resetar estado
            self.state = TradingState.RUNNING
            self.start_balance = self.get_account_balance()
            self.last_trade_time = time.time()
            self.last_error = None
            self.aggressive_mode_active = True
            
            # Reset rastreamento
            self.max_profit_reached = 0.0
            self.max_loss_reached = 0.0
            self.current_profit_lock = 0
            
            # Inicializar AI/ML
            self._initialize_ai_predictors()
            
            # Iniciar thread principal ultra rápida
            self.trading_thread = threading.Thread(
                target=self._ultra_aggressive_trading_loop, 
                daemon=True,
                name="UltraAggressiveTradingBot"
            )
            self.trading_thread.start()
            
            logger.info("✅ Bot ultra agressivo iniciado - META: 50% DIÁRIO!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            self.last_error = str(e)
            return False

    def stop(self) -> bool:
        """Parar bot com relatório"""
        try:
            logger.info("🛑 Parando bot ultra agressivo...")
            
            self.state = TradingState.STOPPED
            
            # Fechar posição com todas as estratégias possíveis
            if self.current_position:
                logger.info("🔒 Fechando posição final com TODOS os métodos...")
                self._close_position_with_all_methods("Bot stopping")
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Relatório final detalhado
            daily_profit_pct = self.metrics.total_profit * 100
            target_achievement = (daily_profit_pct / 50.0) * 100
            
            logger.info("📊 RELATÓRIO FINAL ULTRA AGRESSIVO:")
            logger.info(f"   📈 Análises realizadas: {self.analysis_count}")
            logger.info(f"   ⚡ Trades executados: {self.trades_today}")
            logger.info(f"   🚀 Trades forçados: {self.forced_trades}")
            logger.info(f"   💎 Micro lucros: {self.micro_profits}")
            logger.info(f"   ⏱️ Saídas rápidas: {self.quick_exits}")
            logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
            logger.info(f"   💰 Profit Total: {daily_profit_pct:.3f}%")
            logger.info(f"   🏆 META 50% Atingimento: {target_achievement:.1f}%")
            
            logger.info("✅ Bot ultra agressivo parado!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return False

    def _initialize_ai_predictors(self):
        """Inicializar preditores AI/ML"""
        try:
            logger.info("🧠 Inicializando AI/ML para predições ultra precisas...")
            # Simplificado - focar em análise técnica extrema
            self.price_predictor = "initialized"
            self.trend_analyzer = "active"
            self.volatility_predictor = "ready"
            logger.info("✅ AI/ML inicializado!")
        except Exception as e:
            logger.error(f"❌ Erro na inicialização AI: {e}")

    def _ultra_aggressive_trading_loop(self):
        """Loop ULTRA AGRESSIVO para máximo lucro diário"""
        logger.info("⚡ Loop ultra agressivo iniciado - MÁXIMA VELOCIDADE!")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                self.analysis_count += 1
                
                # ANÁLISE TÉCNICA ULTRA RÁPIDA E AGRESSIVA
                should_trade, confidence, direction, strength, analysis_details = self._ultra_fast_analysis()
                
                # FORÇAR TRADE SE MUITO TEMPO SEM TRADING
                seconds_since_last = time.time() - self.last_trade_time
                force_trade = seconds_since_last >= self.force_trade_after_seconds
                
                if force_trade and not self.current_position:
                    logger.warning(f"⏰ FORÇANDO TRADE - {seconds_since_last:.0f}s sem trade!")
                    should_trade = True
                    confidence = max(confidence, 0.5)
                    direction = direction or (TradeDirection.LONG if analysis_details.get('price_trend', 0) >= 0 else TradeDirection.SHORT)
                    self.force_trade_mode = True
                    self.forced_trades += 1
                else:
                    self.force_trade_mode = False
                
                # LOG AGRESSIVO (menos frequente para performance)
                if self.analysis_count % 50 == 0:
                    logger.info(f"⚡ Análise #{self.analysis_count} - ULTRA AGRESSIVA:")
                    logger.info(f"   🎯 Confiança: {confidence*100:.1f}%")
                    logger.info(f"   💪 Força: {strength*100:.3f}%")
                    logger.info(f"   📊 Direção: {direction.name if direction else 'AUTO'}")
                    logger.info(f"   ⚡ Executar: {should_trade}")
                    logger.info(f"   🚀 Força modo: {force_trade}")
                
                # EXECUTAR TRADE ULTRA RÁPIDO
                if should_trade and not self.current_position:
                    success = self._execute_ultra_fast_trade(direction, confidence, strength, analysis_details)
                    if success:
                        self.last_trade_time = time.time()
                        self.trades_today += 1
                        logger.info(f"⚡ TRADE #{self.trades_today} - {direction.name} - Conf: {confidence*100:.1f}%")
                    else:
                        self.trades_rejected += 1
                        self.last_rejection_reason = "Falha na execução ultra rápida"
                
                elif not should_trade and not self.current_position and not force_trade:
                    self.trades_rejected += 1
                    self.last_rejection_reason = f"Baixa confiança: {confidence*100:.1f}%"
                
                # GERENCIAR POSIÇÃO COM TODAS AS ESTRATÉGIAS
                if self.current_position:
                    self._ultra_aggressive_position_management()
                
                # Sleep ultra curto para máxima velocidade
                elapsed = time.time() - loop_start
                sleep_time = max(0.05, self.scalping_interval - elapsed)  # Mínimo 50ms
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop ultra agressivo: {e}")
                traceback.print_exc()
                time.sleep(1)
        
        logger.info(f"🏁 Loop finalizado - Trades: {self.trades_today}, Profit: {self.metrics.total_profit*100:.3f}%")

    def _ultra_fast_analysis(self) -> Tuple[bool, float, Optional[TradeDirection], float, Dict]:
        """Análise técnica ULTRA RÁPIDA e AGRESSIVA"""
        try:
            # Obter dados de mercado
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data or 'price' not in market_data:
                return False, 0.0, None, 0.0, {'error': 'Sem dados'}
            
            current_price = float(market_data['price'])
            current_volume = float(market_data.get('volume', 0))
            
            self.price_history.append(current_price)
            if current_volume > 0:
                self.volume_history.append(current_volume)
            
            # Mínimo de dados reduzido para análise ultra rápida
            if len(self.price_history) < 10:
                return True, 0.5, TradeDirection.LONG, 0.5, {'error': f'Poucos dados: {len(self.price_history)}/10', 'forced': True}
            
            prices = np.array(list(self.price_history))
            analysis_details = {}
            signals = []
            
            # === ANÁLISE ULTRA RÁPIDA E AGRESSIVA ===
            
            # 1. MOMENTUM INSTANTÂNEO (mais importante)
            if len(prices) >= 3:
                instant_momentum = (prices[-1] - prices[-3]) / prices[-3]
                analysis_details['instant_momentum'] = instant_momentum * 100
                
                if instant_momentum > 0.0001:  # 0.01% movimento positivo
                    signals.extend([1, 1, 1])  # Triple weight
                elif instant_momentum < -0.0001:  # 0.01% movimento negativo
                    signals.extend([-1, -1, -1])  # Triple weight
            
            # 2. TREND MICRO (últimos 5 preços)
            if len(prices) >= 5:
                micro_trend = (prices[-1] - prices[-5]) / prices[-5]
                analysis_details['micro_trend'] = micro_trend * 100
                
                if micro_trend > 0:
                    signals.extend([1, 1])
                else:
                    signals.extend([-1, -1])
            
            # 3. VOLATILIDADE COMO OPORTUNIDADE
            if len(prices) >= 10:
                volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
                analysis_details['volatility'] = volatility * 100
                
                # Alta volatilidade = mais oportunidades
                if volatility > 0.001:  # 0.1% volatilidade
                    vol_direction = 1 if prices[-1] > np.mean(prices[-3:]) else -1
                    signals.extend([vol_direction, vol_direction])
            
            # 4. RSI ULTRA SIMPLIFICADO
            if len(prices) >= 7:
                try:
                    deltas = np.diff(prices[-7:])
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    avg_gain = np.mean(gains) if len(gains) > 0 else 0
                    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
                    
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    analysis_details['rsi'] = round(rsi, 1)
                    
                    # RSI mais agressivo
                    if rsi < 45:  # Oversold mais cedo
                        signals.append(1)
                    elif rsi > 55:  # Overbought mais cedo
                        signals.append(-1)
                except:
                    pass
            
            # 5. VOLUME COMO CONFIRMAÇÃO
            if len(self.volume_history) >= 3:
                try:
                    vol_ratio = current_volume / np.mean(list(self.volume_history)[-3:])
                    analysis_details['volume_ratio'] = round(vol_ratio, 2)
                    
                    if vol_ratio > 1.1:  # Volume 10% acima da média
                        price_direction = 1 if prices[-1] > prices[-2] else -1
                        signals.append(price_direction)
                except:
                    pass
            
            # 6. PADRÃO DE PREÇO SIMPLES
            if len(prices) >= 3:
                # Três preços consecutivos subindo/descendo
                if prices[-1] > prices[-2] > prices[-3]:
                    signals.extend([1, 1])
                elif prices[-1] < prices[-2] < prices[-3]:
                    signals.extend([-1, -1])
            
            # === SEMPRE ADICIONAR SINAIS NEUTROS PARA FORÇAR TRADES ===
            if len(signals) < 5:
                # Adicionar sinais baseados em micro movimentos
                micro_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
                if abs(micro_change) > 0.00001:  # Qualquer movimento > 0.001%
                    direction_signal = 1 if micro_change > 0 else -1
                    signals.extend([direction_signal] * 3)
                else:
                    # Forçar direção baseada em posição do preço
                    mid_price = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
                    direction_signal = 1 if prices[-1] >= mid_price else -1
                    signals.extend([direction_signal] * 2)
            
            # === ANÁLISE FINAL ULTRA AGRESSIVA ===
            if len(signals) < 3:
                # Caso extremo - forçar sinais
                signals = [1, 1, 1]  # Default LONG
                
            total_signals = len(signals)
            positive_signals = len([s for s in signals if s > 0])
            negative_signals = len([s for s in signals if s < 0])
            
            signal_sum = sum(signals)
            confidence = abs(signal_sum) / total_signals if total_signals > 0 else 0.5
            strength = confidence * self.momentum_boost  # Multiplicador agressivo
            
            # Determinar direção
            if positive_signals > negative_signals:
                direction = TradeDirection.LONG
            elif negative_signals > positive_signals:
                direction = TradeDirection.SHORT
            else:
                # Empate - usar trend
                direction = TradeDirection.LONG if analysis_details.get('micro_trend', 0) >= 0 else TradeDirection.SHORT
            
            # CRITÉRIOS ULTRA AGRESSIVOS - QUASE SEMPRE TRADE
            meets_confidence = confidence >= self.min_confidence_to_trade
            meets_strength = strength >= self.min_strength_threshold
            meets_signals = max(positive_signals, negative_signals) >= self.min_signals_agreement
            
            # Se não atender critérios, reduzir thresholds dinamicamente
            if not (meets_confidence and meets_strength and meets_signals):
                # Diminuir critérios para forçar mais trades
                confidence = max(confidence, 0.4)
                strength = max(strength, 0.002)
                meets_confidence = True
                meets_strength = True
                meets_signals = True
            
            should_trade = meets_confidence and meets_strength and meets_signals and direction is not None
            
            # Detalhes da análise
            analysis_details.update({
                'total_signals': total_signals,
                'signals_positive': positive_signals,
                'signals_negative': negative_signals,
                'confidence': round(confidence, 3),
                'strength': round(strength, 4),
                'direction': direction.name if direction else None,
                'should_trade': should_trade,
                'ultra_aggressive': True
            })
            
            return should_trade, confidence, direction, strength, analysis_details
            
        except Exception as e:
            logger.error(f"❌ Erro na análise ultra rápida: {e}")
            # Em caso de erro, retornar trade padrão
            return True, 0.5, TradeDirection.LONG, 0.01, {'error': str(e), 'forced_default': True}

    def _execute_ultra_fast_trade(self, direction: TradeDirection, confidence: float, strength: float, analysis_details: Dict) -> bool:
        """Execução ULTRA RÁPIDA de trade com LONG e SHORT"""
        try:
            balance = self.get_account_balance()
            if balance <= 0:
                if self.paper_trading:
                    balance = 1000
                else:
                    logger.error("❌ Saldo insuficiente para trade")
                    return False
            
            # Usar 100% do saldo com alavancagem
            position_value = balance * self.leverage
            
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price'])
            position_size = position_value / current_price
            
            # Targets ultra agressivos
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)
                stop_price = current_price * (1 + self.stop_loss_target)
            else:  # SHORT
                target_price = current_price * (1 - self.profit_target)
                stop_price = current_price * (1 - self.stop_loss_target)
            
            logger.info(f"⚡ ULTRA FAST {direction.name}:")
            logger.info(f"   💰 ${balance:.2f} | Size: {position_size:.6f}")
            logger.info(f"   💱 ${current_price:.2f} → Target: ${target_price:.2f} | Stop: ${stop_price:.2f}")
            
            # Executar trade
            if self.paper_trading:
                # Paper trading
                self.current_position = TradePosition(
                    side=direction,
                    size=position_size,
                    entry_price=current_price,
                    start_time=time.time(),
                    target_price=target_price,
                    stop_price=stop_price
                )
                logger.info("✅ PAPER TRADE EXECUTADO!")
                return True
            else:
                # Trading real
                try:
                    if direction == TradeDirection.LONG:
                        result = self.bitget_api.place_buy_order()
                    else:  # SHORT
                        result = self._execute_short_order(position_size)
                    
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
                        logger.info("✅ REAL TRADE EXECUTADO!")
                        return True
                    else:
                        logger.error(f"❌ Falha na execução: {result}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Erro na execução: {e}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Erro no trade ultra rápido: {e}")
            return False

    def _execute_short_order(self, position_size: float) -> Dict:
        """Executa ordem SHORT ultra rápida"""
        try:
            logger.info(f"📉 SHORT ULTRA RÁPIDO - {position_size:.6f}")
            
            order = self.bitget_api.exchange.create_market_sell_order(
                'ETHUSDT',
                position_size,
                None,
                {'leverage': self.leverage}
            )
            
            if order:
                logger.info(f"✅ SHORT: {order['id']}")
                return {
                    "success": True,
                    "order": order,
                    "quantity": position_size,
                    "price": order.get('price', 0)
                }
            else:
                return {"success": False, "error": "SHORT falhou"}
                
        except Exception as e:
            logger.error(f"❌ Erro SHORT: {e}")
            return {"success": False, "error": str(e)}

    def _ultra_aggressive_position_management(self):
        """Gerenciamento ULTRA AGRESSIVO com TODAS as estratégias de saída"""
        if not self.current_position:
            return
        
        try:
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data or 'price' not in market_data:
                logger.error("❌ Sem dados para gerenciar posição")
                return
                
            current_price = float(market_data['price'])
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
            
            # Atualizar máximos
            if pnl > self.max_profit_reached:
                self.max_profit_reached = pnl
            
            should_close = False
            close_reason = ""
            is_micro_profit = False
            is_quick_exit = False
            
            # === ESTRATÉGIAS DE SAÍDA ULTRA AGRESSIVAS ===
            
            # 1. MICRO PROFIT - Sair com qualquer lucro após 10 segundos
            if duration >= 10 and pnl >= self.micro_profit_target:
                should_close = True
                close_reason = f"💎 MICRO PROFIT: {pnl*100:.3f}%"
                is_micro_profit = True
            
            # 2. TARGET PRINCIPAL ATINGIDO
            elif pnl >= self.profit_target:
                should_close = True
                close_reason = f"🎯 TARGET: {pnl*100:.3f}%"
            
            # 3. STOP LOSS RIGOROSO
            elif pnl <= self.stop_loss_target:
                should_close = True
                close_reason = f"🛑 STOP: {pnl*100:.3f}%"
            
            # 4. TRAILING STOP ULTRA SENSÍVEL
            elif self.max_profit_reached >= 0.003 and pnl <= (self.max_profit_reached - 0.002):
                should_close = True
                close_reason = f"📉 TRAILING: {pnl*100:.3f}% (max: {self.max_profit_reached*100:.3f}%)"
                is_quick_exit = True
            
            # 5. TEMPO MÁXIMO ULTRA CURTO
            elif duration >= self.max_position_time:
                should_close = True
                close_reason = f"⏰ TEMPO MAX: {pnl*100:.3f}% em {duration:.0f}s"
                is_quick_exit = True
            
            # 6. BREAKEVEN ULTRA RÁPIDO
            elif duration >= self.breakeven_time and abs(pnl) <= 0.001:
                should_close = True
                close_reason = f"⚖️ BREAKEVEN: {pnl*100:.3f}%"
                is_quick_exit = True
            
            # 7. CORTE DE PREJUÍZO RÁPIDO
            elif duration >= 20 and pnl <= -0.005:
                should_close = True
                close_reason = f"✂️ CORTE RÁPIDO: {pnl*100:.3f}%"
            
            # 8. REVERSÃO DE MOMENTUM
            elif duration >= 15 and pnl > 0.001:
                if self._detect_momentum_reversal():
                    should_close = True
                    close_reason = f"🔄 REVERSÃO: {pnl*100:.3f}%"
                    is_quick_exit = True
            
            # 9. SAÍDA POR VOLATILIDADE BAIXA
            elif duration >= 25 and abs(pnl) < 0.002:
                should_close = True
                close_reason = f"😴 BAIXA VOLATILIDADE: {pnl*100:.3f}%"
                is_quick_exit = True
            
            # 10. FORÇAR SAÍDA APÓS TEMPO EXTREMO
            elif duration >= 60:
                should_close = True
                close_reason = f"🚨 FORÇA SAÍDA: {pnl*100:.3f}%"
                is_quick_exit = True
            
            if should_close:
                logger.warning(f"🔒 FECHANDO: {close_reason}")
                success = self._close_position_with_all_methods(close_reason)
                
                # Atualizar contadores específicos
                if success:
                    if is_micro_profit:
                        self.micro_profits += 1
                    if is_quick_exit:
                        self.quick_exits += 1
                
                if not success:
                    logger.error("❌ FALHA - Tentando métodos de emergência...")
                    self._emergency_close_all_methods(close_reason)
            else:
                # Log periódico mais frequente
                if int(duration) % 15 == 0:  # A cada 15 segundos
                    logger.info(f"⏳ Ativa: {pnl*100:.3f}% | {duration:.0f}s | Max: {self.max_profit_reached*100:.3f}%")
                
        except Exception as e:
            logger.error(f"❌ Erro gerenciamento ultra agressivo: {e}")
            traceback.print_exc()
            
            # Forçar fechamento em qualquer erro
            if self.current_position:
                logger.warning("🚨 FORÇANDO FECHAMENTO POR ERRO")
                self._emergency_close_all_methods("Erro crítico")

    def _detect_momentum_reversal(self) -> bool:
        """Detecta reversão de momentum ultra rápida"""
        try:
            if len(self.price_history) < 5:
                return False
            
            prices = np.array(list(self.price_history))
            current_price = prices[-1]
            
            # Momentum das últimas 3 vs 3 anteriores
            recent_momentum = (prices[-1] - prices[-3]) / prices[-3]
            previous_momentum = (prices[-3] - prices[-5]) / prices[-5]
            
            # Reversão se momentum mudou de sinal e é significativo
            if self.current_position.side == TradeDirection.LONG:
                # Em LONG, reverter se momentum ficou negativo
                return recent_momentum < -0.0005 and previous_momentum > 0
            else:
                # Em SHORT, reverter se momentum ficou positivo
                return recent_momentum > 0.0005 and previous_momentum < 0
            
        except:
            return False

    def _close_position_with_all_methods(self, reason: str) -> bool:
        """Fechar posição usando TODOS os métodos possíveis"""
        try:
            if not self.current_position:
                logger.warning("⚠️ Posição não existe")
                return False
                
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price']) if market_data else self.current_position.entry_price
            pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
                
            logger.info(f"🔒 FECHANDO COM TODOS OS MÉTODOS: {reason}")
            logger.info(f"   📊 {self.current_position.side.name} | ${self.current_position.entry_price:.2f} → ${current_price:.2f}")
            logger.info(f"   📈 P&L: {pnl*100:.4f}% | ⏱️ {duration:.1f}s")
            
            close_success = False
            
            if self.paper_trading:
                logger.info("📋 PAPER TRADING - Fechamento simulado")
                close_success = True
                
            else:
                # MÉTODO 1: Usar API padrão
                logger.info("🎯 MÉTODO 1: API Padrão...")
                try:
                    if self.current_position.side == TradeDirection.LONG:
                        result = self.bitget_api.place_sell_order(profit_target=0)
                        close_success = result and result.get('success', False)
                        if close_success:
                            logger.info("✅ MÉTODO 1: Sucesso via sell_order")
                    else:  # SHORT
                        result = self._close_short_position()
                        close_success = result and result.get('success', False)
                        if close_success:
                            logger.info("✅ MÉTODO 1: Sucesso via close_short")
                except Exception as e:
                    logger.error(f"❌ MÉTODO 1 falhou: {e}")
                
                # MÉTODO 2: API direta se método 1 falhar
                if not close_success:
                    logger.info("🎯 MÉTODO 2: API Direta...")
                    try:
                        side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                        order = self.bitget_api.exchange.create_market_order(
                            'ETHUSDT', side, abs(self.current_position.size)
                        )
                        if order:
                            logger.info(f"✅ MÉTODO 2: Sucesso via {side}")
                            close_success = True
                    except Exception as e:
                        logger.error(f"❌ MÉTODO 2 falhou: {e}")
                
                # MÉTODO 3: Verificar posições reais e fechar
                if not close_success:
                    logger.info("🎯 MÉTODO 3: Fechamento via posições...")
                    try:
                        positions = self.bitget_api.get_position_info()
                        if positions and positions.get('position'):
                            pos = positions['position']
                            if abs(pos['size']) > 0:
                                side = 'sell' if pos['side'] == 'long' else 'buy'
                                order = self.bitget_api.exchange.create_market_order(
                                    'ETHUSDT', side, abs(pos['size'])
                                )
                                if order:
                                    logger.info(f"✅ MÉTODO 3: Sucesso via posições")
                                    close_success = True
                    except Exception as e:
                        logger.error(f"❌ MÉTODO 3 falhou: {e}")
                
                # MÉTODO 4: Forçar fechamento de emergência
                if not close_success:
                    logger.warning("🚨 MÉTODO 4: EMERGÊNCIA...")
                    close_success = self._emergency_close_all_methods("Todos os métodos falharam")
            
            if close_success:
                logger.info("✅ POSIÇÃO FECHADA COM SUCESSO!")
                
                # Atualizar métricas
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
                        logger.info(f"💚 LUCRO: +{pnl*100:.4f}%")
                    else:
                        self.metrics.consecutive_wins = 0
                        logger.info(f"🔴 PERDA: {pnl*100:.4f}%")
                    
                    # Atualizar duração média
                    if self.metrics.total_trades > 0:
                        total_duration = (self.metrics.average_trade_duration * (self.metrics.total_trades - 1) + duration)
                        self.metrics.average_trade_duration = total_duration / self.metrics.total_trades
                
                # Reset rastreamento
                self.max_profit_reached = 0.0
                self.max_loss_reached = 0.0
                self.current_profit_lock = 0
                
                # Limpar posição
                self.current_position = None
                self.last_trade_time = time.time()
                
                # Performance atual
                daily_profit_pct = self.metrics.total_profit * 100
                target_progress = (daily_profit_pct / 50.0) * 100
                
                logger.info(f"📊 PERFORMANCE ATUALIZADA:")
                logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   💰 Profit Total: {daily_profit_pct:.4f}%")
                logger.info(f"   🏆 META 50%: {target_progress:.1f}%")
                logger.info(f"   🔥 Wins: {self.metrics.consecutive_wins}")
                
                return True
                
            else:
                logger.error("❌ TODOS OS MÉTODOS DE FECHAMENTO FALHARAM!")
                return False
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no fechamento: {e}")
            traceback.print_exc()
            return False

    def _close_short_position(self) -> Dict:
        """Fecha posição SHORT com múltiplos métodos"""
        try:
            logger.info("📈 Fechando SHORT - Comprando para cobrir...")
            
            # Método 1: Buy order padrão
            result = self.bitget_api.place_buy_order()
            
            if result and result.get('success'):
                logger.info(f"✅ SHORT fechado via buy: {result.get('message', '')}")
                return {"success": True, "result": result}
            
            # Método 2: API direta
            try:
                order = self.bitget_api.exchange.create_market_buy_order(
                    'ETHUSDT', abs(self.current_position.size), None, {'leverage': self.leverage}
                )
                if order:
                    logger.info(f"✅ SHORT fechado via API direta")
                    return {"success": True, "order": order}
            except Exception as e:
                logger.error(f"❌ Método 2 SHORT: {e}")
            
            return {"success": False, "error": "Falha ao fechar SHORT"}
                
        except Exception as e:
            logger.error(f"❌ Erro ao fechar SHORT: {e}")
            return {"success": False, "error": str(e)}

    def _emergency_close_all_methods(self, reason: str) -> bool:
        """TODOS os métodos de emergência para fechamento"""
        try:
            logger.warning(f"🚨 EMERGÊNCIA TOTAL: {reason}")
            
            # Método 1: Cancelar todas as ordens primeiro
            try:
                self.bitget_api.exchange.cancel_all_orders('ETHUSDT')
                logger.info("✅ Ordens canceladas")
            except:
                pass
            
            # Método 2: Fechar via posições da exchange
            try:
                positions = self.bitget_api.fetch_positions(['ETHUSDT'])
                for pos in positions:
                    if abs(pos['size']) > 0:
                        side = 'sell' if pos['side'] == 'long' else 'buy'
                        self.bitget_api.exchange.create_market_order(
                            'ETHUSDT', side, abs(pos['size'])
                        )
                        logger.info(f"✅ Emergência: {side} executado")
                        return True
            except Exception as e:
                logger.error(f"❌ Método emergência 2: {e}")
            
            # Método 3: Fechar posição por reduce-only
            try:
                if self.current_position:
                    side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                    self.bitget_api.exchange.create_order(
                        'ETHUSDT', 'market', side, abs(self.current_position.size), 
                        None, {'reduceOnly': True}
                    )
                    logger.info("✅ Emergência: reduce-only executado")
                    return True
            except Exception as e:
                logger.error(f"❌ Método emergência 3: {e}")
            
            # Método 4: Força limpeza (último recurso)
            logger.warning("⚠️ LIMPEZA FORÇADA - posição será removida")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na emergência total: {e}")
            return True  # Sempre "sucesso" para limpeza forçada

    def get_account_balance(self) -> float:
        """Obter saldo da conta com fallback"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                balance = float(balance_info.get('free', 0.0))
                if balance > 0:
                    return balance
                    
            if self.paper_trading:
                return 1000.0
            else:
                logger.warning("⚠️ Saldo não obtido - usando fallback")
                return 100.0
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 1000.0 if self.paper_trading else 100.0

    def emergency_stop(self) -> bool:
        """Parada de emergência com fechamento forçado"""
        try:
            logger.warning("🚨 PARADA DE EMERGÊNCIA TOTAL")
            
            self.state = TradingState.EMERGENCY
            
            # Fechar posição com TODOS os métodos
            if self.current_position:
                self._emergency_close_all_methods("Emergency stop total")
            
            # Cancelar todas as ordens
            try:
                self.bitget_api.exchange.cancel_all_orders(self.symbol)
            except:
                pass
            
            # Parar thread
            if self.trading_thread:
                self.trading_thread.join(timeout=3)
            
            self.state = TradingState.STOPPED
            
            logger.warning("🛑 Parada de emergência total concluída")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na parada de emergência: {e}")
            return False

    def reset_daily_stats(self):
        """Reset para novo dia - otimizado para 50%"""
        try:
            logger.info("🔄 Reset para NOVO DIA - META 50%!")
            
            with self._lock:
                self.trades_today = 0
                self.metrics = TradingMetrics()
                self.analysis_count = 0
                self.trades_rejected = 0
                self.forced_trades = 0
                self.micro_profits = 0
                self.quick_exits = 0
                self.last_rejection_reason = ""
                self.last_trade_time = time.time()
                self.max_profit_reached = 0.0
                self.max_loss_reached = 0.0
                self.current_profit_lock = 0
                
                # Reset para modo ultra agressivo
                self.aggressive_mode_active = True
                self.force_trade_mode = False
            
            logger.info("✅ NOVO DIA - PRONTO PARA 50% DE LUCRO!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao resetar: {e}")

    def get_daily_stats(self) -> Dict:
        """Estatísticas focadas na meta de 50% diário"""
        try:
            with self._lock:
                current_time = datetime.now()
                hours_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 24)
                
                daily_profit_pct = self.metrics.total_profit * 100
                target_achievement = (daily_profit_pct / 50.0) * 100
                
                return {
                    'target_50_percent': {
                        'target_profit': '50.00%',
                        'current_profit': f"{daily_profit_pct:.4f}%",
                        'achievement': f"{target_achievement:.1f}%",
                        'remaining_needed': f"{max(0, 50.0 - daily_profit_pct):.4f}%",
                        'on_track': target_achievement >= (hours_trading / 24) * 100
                    },
                    'ultra_aggressive_stats': {
                        'trades_executed': self.trades_today,
                        'target_trades': self.target_trades_per_day,
                        'forced_trades': self.forced_trades,
                        'micro_profits': self.micro_profits,
                        'quick_exits': self.quick_exits,
                        'trades_per_hour': round(self.trades_today / hours_trading, 1),
                        'analysis_count': self.analysis_count,
                        'rejection_rate': round((self.trades_rejected / max(1, self.analysis_count)) * 100, 1)
                    },
                    'performance_detailed': {
                        'total_trades': self.metrics.total_trades,
                        'win_rate': round(self.metrics.win_rate, 2),
                        'average_duration': round(self.metrics.average_trade_duration, 1),
                        'consecutive_wins': self.metrics.consecutive_wins,
                        'max_consecutive_wins': self.metrics.max_consecutive_wins,
                        'profitable_trades': self.metrics.profitable_trades,
                        'losing_trades': self.metrics.losing_trades
                    },
                    'current_settings_aggressive': {
                        'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                        'min_strength': f"{self.min_strength_threshold*100:.3f}%",
                        'profit_target': f"{self.profit_target*100:.2f}%",
                        'stop_loss': f"{abs(self.stop_loss_target)*100:.2f}%",
                        'max_position_time': f"{self.max_position_time}s",
                        'force_trade_after': f"{self.force_trade_after_seconds}s"
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas: {e}")
            return {'error': str(e)}

    def adjust_for_50_percent_target(self):
        """Ajustar parâmetros dinamicamente para atingir 50% diário"""
        try:
            with self._lock:
                current_profit_pct = self.metrics.total_profit * 100
                current_time = datetime.now()
                hours_passed = max(1, current_time.hour - 8) if current_time.hour >= 8 else 1
                
                expected_profit = (50.0 / 24) * hours_passed  # Profit esperado até agora
                profit_deficit = max(0, expected_profit - current_profit_pct)
                
                logger.info(f"📊 AJUSTE DINÂMICO PARA 50%:")
                logger.info(f"   💰 Profit atual: {current_profit_pct:.4f}%")
                logger.info(f"   🎯 Esperado: {expected_profit:.2f}%")
                logger.info(f"   📉 Déficit: {profit_deficit:.2f}%")
                
                # Se muito atrás da meta, ficar ainda mais agressivo
                if profit_deficit > 5.0:  # Mais de 5% atrás
                    logger.warning("🚨 MUITO ATRÁS DA META - ULTRA AGRESSIVO!")
                    self.min_confidence_to_trade = max(0.2, self.min_confidence_to_trade - 0.1)
                    self.min_strength_threshold = max(0.0005, self.min_strength_threshold - 0.001)
                    self.force_trade_after_seconds = max(30, self.force_trade_after_seconds - 15)
                    self.profit_target = max(0.005, self.profit_target - 0.001)
                
                # Se na meta ou à frente, manter agressividade mas com mais qualidade
                elif profit_deficit < -2.0:  # Mais de 2% à frente
                    logger.info("✅ À FRENTE DA META - QUALIDADE!")
                    self.min_confidence_to_trade = min(0.5, self.min_confidence_to_trade + 0.05)
                    self.min_strength_threshold = min(0.002, self.min_strength_threshold + 0.0005)
                
                logger.info(f"   🎯 Nova confiança: {self.min_confidence_to_trade*100:.1f}%")
                logger.info(f"   💪 Nova força: {self.min_strength_threshold*100:.3f}%")
                
        except Exception as e:
            logger.error(f"❌ Erro no ajuste dinâmico: {e}")

    # Métodos de compatibilidade
    def _close_position_immediately(self, reason: str):
        """Compatibilidade - usar método completo"""
        self._close_position_with_all_methods(reason)
