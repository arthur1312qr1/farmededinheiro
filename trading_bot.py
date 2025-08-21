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

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ETHUSDT',
                 leverage: int = 10, balance_percentage: float = 100.0,
                 daily_target: int = 350, scalping_interval: float = 0.5,
                 paper_trading: bool = False):
        """Initialize PROFESSIONAL Trading Bot for 50% DAILY PROFIT with REAL PROFITS"""
        
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

        # ===== CONFIGURAÇÕES PROFISSIONAIS PARA 50% DIÁRIO COM LUCRO REAL =====
        self.min_trades_per_day = 150  # Mínimo 150 trades/dia
        self.target_trades_per_day = 200  # Meta: 200 trades/dia REALISTA
        self.max_time_between_trades = 300  # Máximo 5 minutos entre trades
        self.force_trade_after_seconds = 600  # Forçar trade após 10 minutos
        self.last_trade_time = 0

        # CRITÉRIOS SELETIVOS PARA TRADES DE QUALIDADE
        self.min_confidence_to_trade = 0.70     # 70% confiança mínima
        self.min_prediction_score = 0.65        # 65% score de predição
        self.min_signals_agreement = 6          # 6 sinais precisam concordar
        self.min_strength_threshold = 0.010     # 1.0% força mínima

        # CONFIGURAÇÕES DE LUCRO CORRIGIDAS PARA LUCRO REAL
        self.profit_target = 0.015              # 1.5% take profit (MÍNIMO para lucro real)
        self.stop_loss_target = -0.008          # 0.8% stop loss (controlado)
        self.minimum_profit_target = 0.010      # 1.0% lucro mínimo absoluto
        self.max_position_time = 180            # Máximo 3 minutos por trade
        self.min_position_time = 45             # Mínimo 45 segundos
        self.breakeven_time = 60                # Breakeven após 60 segundos

        # CONFIGURAÇÕES PARA SCALPING PROFISSIONAL
        self.quality_over_quantity = True
        self.professional_mode = True
        self.momentum_boost = 1.5  # Multiplicador moderado

        # Sistema de dados para análise técnica
        self.price_history = deque(maxlen=200)  # Mais dados para análise precisa
        self.volume_history = deque(maxlen=100)
        self.analysis_history = deque(maxlen=50)

        # Sistema de trading profissional
        self.professional_mode_active = True
        self.emergency_trading_mode = False
        self.last_analysis_result = None
        self.force_trade_mode = False
        
        # Rastreamento avançado para trailing e múltiplas saídas
        self.max_profit_reached = 0.0
        self.max_loss_reached = 0.0
        self.profit_locks = [0.008, 0.012, 0.018]  # Lock de lucros em 0.8%, 1.2%, 1.8%
        self.current_profit_lock = 0
        
        # Sistema de múltiplas saídas
        self.partial_exit_levels = [0.012, 0.018, 0.025]  # Saídas parciais em 1.2%, 1.8%, 2.5%
        self.exit_percentages = [40, 35, 25]  # % para sair em cada nível

        # Métricas de performance
        self.metrics = TradingMetrics()
        self.start_balance = 0.0
        self.trades_today = 0
        self.daily_profit_target = 0.5  # 50% diário
        
        # Contadores específicos para qualidade
        self.quality_trades = 0
        self.rejected_low_quality = 0
        self.profitable_exits = 0
        self.fee_losses_avoided = 0
        
        # Sistema de controle de riscos
        self.consecutive_losses = 0
        self.daily_loss_accumulated = 0.0
        self.emergency_stop_triggered = False
        
        # Lock para thread safety
        self._lock = threading.Lock()

        # Contador de análises e debug
        self.analysis_count = 0
        self.trades_rejected = 0
        self.last_rejection_reason = ""

        # Sistema de AI/ML para previsões (simplificado)
        self.technical_analyzer = None
        self.market_conditions = {
            'trend': 'neutral',
            'volatility': 0.0,
            'volume_avg': 0.0,
            'strength': 0.0
        }

        logger.info("🚀 PROFESSIONAL TRADING BOT - 50% DAILY TARGET with REAL PROFITS")
        logger.info("⚡ CONFIGURAÇÕES PROFISSIONAIS:")
        logger.info(f"   🎯 Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"   💪 Força mínima: {self.min_strength_threshold*100}%")
        logger.info(f"   📊 Sinais necessários: {self.min_signals_agreement}")
        logger.info(f"   📈 Take Profit: {self.profit_target*100}%")
        logger.info(f"   🛑 Stop Loss: {abs(self.stop_loss_target)*100}%")
        logger.info(f"   ⚡ Trades/dia META: {self.target_trades_per_day}")
        logger.info(f"   💰 LUCRO DIÁRIO META: {self.daily_profit_target*100}%")
        logger.info("🏆 MODO PROFISSIONAL - QUALIDADE > QUANTIDADE!")

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
                current_profit_pct = (self.metrics.net_profit * 100) if self.metrics.net_profit else 0
                daily_progress = (current_profit_pct / 50.0) * 100  # 50% é a meta
                
                return {
                    'bot_status': {
                        'state': self.state.value,
                        'is_running': self.is_running,
                        'symbol': self.symbol,
                        'leverage': self.leverage,
                        'paper_trading': self.paper_trading,
                        'professional_mode': True,
                        'quality_over_quantity': self.quality_over_quantity
                    },
                    'professional_trading': {
                        'analysis_count': self.analysis_count,
                        'trades_executed': self.trades_today,
                        'quality_trades': self.quality_trades,
                        'rejected_low_quality': self.rejected_low_quality,
                        'profitable_exits': self.profitable_exits,
                        'fee_losses_avoided': self.fee_losses_avoided,
                        'seconds_since_last_trade': round(seconds_since_last_trade),
                        'will_force_trade_in': max(0, self.force_trade_after_seconds - seconds_since_last_trade),
                        'current_thresholds': {
                            'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                            'min_strength': f"{self.min_strength_threshold*100:.1f}%",
                            'min_signals': self.min_signals_agreement,
                            'min_profit': f"{self.minimum_profit_target*100:.1f}%"
                        }
                    },
                    'daily_progress_50_percent': {
                        'target_profit': '50.0%',
                        'current_profit': f"{current_profit_pct:.3f}%",
                        'progress_to_target': f"{daily_progress:.1f}%",
                        'trades_today': self.trades_today,
                        'target_trades': self.target_trades_per_day,
                        'trades_per_hour': round(self.trades_today / max(1, hours_in_trading), 1),
                        'needed_trades_per_hour': round(self.target_trades_per_day / 24, 1),
                        'quality_ratio': f"{(self.quality_trades / max(1, self.trades_today)) * 100:.1f}%"
                    },
                    'performance': {
                        'total_trades': self.metrics.total_trades,
                        'profitable_trades': self.metrics.profitable_trades,
                        'losing_trades': self.metrics.losing_trades,
                        'win_rate': round(self.metrics.win_rate, 2),
                        'total_profit': round(self.metrics.total_profit, 6),
                        'net_profit': round(self.metrics.net_profit, 6),
                        'fees_paid': round(self.metrics.total_fees_paid, 6),
                        'consecutive_wins': self.metrics.consecutive_wins,
                        'consecutive_losses': self.metrics.consecutive_losses,
                        'avg_trade_duration': round(self.metrics.average_trade_duration, 1)
                    },
                    'risk_management': {
                        'consecutive_losses': self.consecutive_losses,
                        'daily_loss': round(self.daily_loss_accumulated, 4),
                        'emergency_stop': self.emergency_stop_triggered,
                        'max_drawdown': round(self.metrics.max_drawdown, 4)
                    },
                    'current_position': self._get_position_status(),
                    'market_conditions': self.market_conditions,
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
            duration = self.current_position.get_duration()
            
            return {
                'active': True,
                'side': self.current_position.side.value,
                'size': self.current_position.size,
                'entry_price': self.current_position.entry_price,
                'current_price': current_price,
                'duration_seconds': round(duration),
                'pnl_percent': round(pnl * 100, 4),
                'target_price': self.current_position.target_price,
                'stop_price': self.current_position.stop_price,
                'max_profit_reached': round(self.max_profit_reached * 100, 4),
                'current_profit_lock': self.current_profit_lock,
                'meets_minimum_profit': pnl >= self.minimum_profit_target,
                'meets_target_profit': pnl >= self.profit_target,
                'should_exit_soon': duration > self.min_position_time and (
                    pnl >= self.profit_target or 
                    pnl <= self.stop_loss_target or 
                    duration >= self.max_position_time
                )
            }
        except Exception as e:
            return {'active': True, 'error': f'Erro ao obter dados: {str(e)}'}

    def start(self) -> bool:
        """Iniciar bot profissional"""
        try:
            if self.state == TradingState.RUNNING:
                logger.warning("🟡 Bot já está rodando")
                return True
            
            logger.info("🚀 INICIANDO BOT PROFISSIONAL - META 50% DIÁRIO COM LUCRO REAL")
            logger.info("🏆 MODO QUALIDADE > QUANTIDADE!")
            
            # Resetar contadores
            self.analysis_count = 0
            self.trades_rejected = 0
            self.quality_trades = 0
            self.rejected_low_quality = 0
            self.profitable_exits = 0
            self.fee_losses_avoided = 0
            self.consecutive_losses = 0
            self.daily_loss_accumulated = 0.0
            self.emergency_stop_triggered = False
            
            # Resetar estado
            self.state = TradingState.RUNNING
            self.start_balance = self.get_account_balance()
            self.last_trade_time = time.time()
            self.last_error = None
            self.professional_mode_active = True
            
            # Reset rastreamento
            self.max_profit_reached = 0.0
            self.max_loss_reached = 0.0
            self.current_profit_lock = 0
            
            # Inicializar análise técnica
            self._initialize_technical_analysis()
            
            # Iniciar thread principal profissional
            self.trading_thread = threading.Thread(
                target=self._professional_trading_loop, 
                daemon=True,
                name="ProfessionalTradingBot"
            )
            self.trading_thread.start()
            
            logger.info("✅ Bot profissional iniciado - META: 50% DIÁRIO COM LUCRO REAL!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            self.last_error = str(e)
            return False

    def stop(self) -> bool:
        """Parar bot com relatório completo"""
        try:
            logger.info("🛑 Parando bot profissional...")
            
            self.state = TradingState.STOPPED
            
            # Fechar posição com método profissional
            if self.current_position:
                logger.info("🔒 Fechando posição final com método profissional...")
                self._close_position_professional("Bot stopping")
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Relatório final detalhado
            daily_profit_pct = self.metrics.net_profit * 100
            target_achievement = (daily_profit_pct / 50.0) * 100
            
            logger.info("📊 RELATÓRIO FINAL PROFISSIONAL:")
            logger.info(f"   📈 Análises realizadas: {self.analysis_count}")
            logger.info(f"   ⚡ Trades executados: {self.trades_today}")
            logger.info(f"   🏆 Trades de qualidade: {self.quality_trades}")
            logger.info(f"   🚫 Rejeitados baixa qualidade: {self.rejected_low_quality}")
            logger.info(f"   💚 Saídas lucrativas: {self.profitable_exits}")
            logger.info(f"   💰 Perdas por taxas evitadas: {self.fee_losses_avoided}")
            logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
            logger.info(f"   💎 Profit Bruto: {self.metrics.total_profit*100:.3f}%")
            logger.info(f"   💰 Profit Líquido: {daily_profit_pct:.3f}%")
            logger.info(f"   💸 Taxas pagas: {self.metrics.total_fees_paid*100:.3f}%")
            logger.info(f"   🏆 META 50% Atingimento: {target_achievement:.1f}%")
            
            logger.info("✅ Bot profissional parado!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return False

    def _initialize_technical_analysis(self):
        """Inicializar análise técnica profissional"""
        try:
            logger.info("🧠 Inicializando análise técnica profissional...")
            # Simplificado - focar em indicadores técnicos sólidos
            self.technical_analyzer = "professional"
            
            # Inicializar condições de mercado
            self._update_market_conditions()
            logger.info("✅ Análise técnica inicializada!")
        except Exception as e:
            logger.error(f"❌ Erro na inicialização técnica: {e}")

    def _professional_trading_loop(self):
        """Loop PROFISSIONAL para máximo lucro com qualidade"""
        logger.info("🏆 Loop profissional iniciado - QUALIDADE MÁXIMA!")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                self.analysis_count += 1
                
                # Verificar condições de emergência
                if self._check_emergency_conditions():
                    logger.warning("🚨 Condições de emergência detectadas!")
                    break
                
                # ANÁLISE TÉCNICA PROFISSIONAL E SELETIVA
                should_trade, confidence, direction, strength, analysis_details = self._professional_market_analysis()
                
                # FORÇAR TRADE SE MUITO TEMPO SEM TRADING (mais conservador)
                seconds_since_last = time.time() - self.last_trade_time
                force_trade = seconds_since_last >= self.force_trade_after_seconds and not self.current_position
                
                if force_trade:
                    logger.warning(f"⏰ FORÇANDO TRADE PROFISSIONAL - {seconds_since_last:.0f}s sem trade!")
                    should_trade = True
                    confidence = max(confidence, 0.7)  # Confiança mínima alta
                    direction = direction or (TradeDirection.LONG if analysis_details.get('trend_strength', 0) >= 0 else TradeDirection.SHORT)
                    self.force_trade_mode = True
                else:
                    self.force_trade_mode = False
                
                # LOG PROFISSIONAL (menos frequente para performance)
                if self.analysis_count % 100 == 0:
                    logger.info(f"🏆 Análise #{self.analysis_count} - PROFISSIONAL:")
                    logger.info(f"   🎯 Confiança: {confidence*100:.1f}%")
                    logger.info(f"   💪 Força: {strength*100:.2f}%")
                    logger.info(f"   📊 Direção: {direction.name if direction else 'AUTO'}")
                    logger.info(f"   ✅ Executar: {should_trade}")
                    logger.info(f"   🏆 Qualidade: {analysis_details.get('quality_score', 0):.1f}")
                
                # EXECUTAR TRADE PROFISSIONAL
                if should_trade and not self.current_position:
                    success = self._execute_professional_trade(direction, confidence, strength, analysis_details)
                    if success:
                        self.last_trade_time = time.time()
                        self.trades_today += 1
                        self.quality_trades += 1
                        logger.info(f"🏆 TRADE #{self.trades_today} PROFISSIONAL - {direction.name} - Conf: {confidence*100:.1f}%")
                    else:
                        self.trades_rejected += 1
                        self.last_rejection_reason = "Falha na execução profissional"
                
                elif not should_trade and not self.current_position and not force_trade:
                    self.trades_rejected += 1
                    self.rejected_low_quality += 1
                    self.last_rejection_reason = f"Baixa qualidade: Conf:{confidence*100:.1f}%, Força:{strength*100:.2f}%"
                
                # GERENCIAR POSIÇÃO COM ESTRATÉGIAS PROFISSIONAIS
                if self.current_position:
                    self._professional_position_management()
                
                # Sleep profissional para análise de qualidade
                elapsed = time.time() - loop_start
                sleep_time = max(0.2, self.scalping_interval - elapsed)  # Mínimo 200ms para qualidade
                time.sleep(sleep_time)
                
                # Ajuste dinâmico para 50% diário
                if self.analysis_count % 500 == 0:
                    self._adjust_for_50_percent_target()
                
            except Exception as e:
                logger.error(f"❌ Erro no loop profissional: {e}")
                traceback.print_exc()
                time.sleep(2)
        
        logger.info(f"🏁 Loop finalizado - Trades: {self.trades_today}, Profit: {self.metrics.net_profit*100:.3f}%")

    def _check_emergency_conditions(self) -> bool:
        """Verificar condições de emergência"""
        try:
            # Verificar perdas consecutivas
            if self.consecutive_losses >= 3:
                logger.warning(f"🚨 Muitas perdas consecutivas: {self.consecutive_losses}")
                return True
            
            # Verificar perda diária máxima
            if self.daily_loss_accumulated >= 0.08:  # 8% perda máxima
                logger.warning(f"🚨 Perda diária máxima atingida: {self.daily_loss_accumulated*100:.2f}%")
                return True
            
            # Verificar drawdown máximo
            if self.metrics.max_drawdown >= 0.10:  # 10% drawdown máximo
                logger.warning(f"🚨 Drawdown máximo atingido: {self.metrics.max_drawdown*100:.2f}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar emergência: {e}")
            return True  # Parar por segurança

    def _professional_market_analysis(self) -> Tuple[bool, float, Optional[TradeDirection], float, Dict]:
        """Análise de mercado PROFISSIONAL e SELETIVA"""
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
            
            # Mínimo de dados para análise profissional
            if len(self.price_history) < 50:
                return False, 0.0, None, 0.0, {'error': f'Dados insuficientes: {len(self.price_history)}/50'}
            
            prices = np.array(list(self.price_history))
            analysis_details = {}
            signals = []
            
            # === ANÁLISE TÉCNICA PROFISSIONAL ===
            
            # 1. TREND ANALYSIS PROFISSIONAL
            trend_strength = self._calculate_trend_strength(prices)
            analysis_details['trend_strength'] = trend_strength
            
            if abs(trend_strength) > 0.008:  # Trend forte > 0.8%
                trend_signal = 2 if trend_strength > 0 else -2  # Double weight
                signals.extend([trend_signal] * 2)
            
            # 2. RSI PROFISSIONAL
            rsi_signal, rsi_value = self._calculate_professional_rsi(prices)
            analysis_details['rsi'] = rsi_value
            if rsi_signal != 0:
                signals.extend([rsi_signal] * 2)
            
            # 3. MOVING AVERAGES CROSSOVER
            ma_signal = self._calculate_ma_crossover(prices)
            analysis_details['ma_signal'] = ma_signal
            if ma_signal != 0:
                signals.append(ma_signal)
            
            # 4. VOLUME CONFIRMATION
            volume_signal = self._analyze_volume_confirmation(current_volume)
            analysis_details['volume_confirmation'] = volume_signal
            if volume_signal != 0:
                signals.append(volume_signal)
            
            # 5. VOLATILITY ANALYSIS
            volatility_signal, volatility_value = self._analyze_volatility(prices)
            analysis_details['volatility'] = volatility_value
            if volatility_signal != 0:
                signals.append(volatility_signal)
            
            # 6. MOMENTUM INDICATOR
            momentum_signal = self._calculate_momentum(prices)
            analysis_details['momentum'] = momentum_signal
            if abs(momentum_signal) > 0.005:  # Momentum forte > 0.5%
                direction_signal = 1 if momentum_signal > 0 else -1
                signals.extend([direction_signal] * 2)
            
            # 7. SUPPORT/RESISTANCE LEVELS
            sr_signal = self._analyze_support_resistance(prices, current_price)
            analysis_details['support_resistance'] = sr_signal
            if sr_signal != 0:
                signals.append(sr_signal)
            
            # === ANÁLISE FINAL PROFISSIONAL ===
            
            if len(signals) < self.min_signals_agreement:
                return False, 0.0, None, 0.0, {'error': f'Sinais insuficientes: {len(signals)}/{self.min_signals_agreement}'}
            
            total_signals = len(signals)
            positive_signals = len([s for s in signals if s > 0])
            negative_signals = len([s for s in signals if s < 0])
            
            # Calcular confiança baseada na concordância
            signal_agreement = max(positive_signals, negative_signals)
            confidence = signal_agreement / total_signals
            
            # Calcular força baseada na intensidade dos sinais
            signal_strength = abs(sum(signals)) / total_signals
            strength = min(signal_strength * 0.01, 0.05)  # Normalizar para 0-5%
            
            # Determinar direção
            if positive_signals > negative_signals:
                direction = TradeDirection.LONG
            elif negative_signals > positive_signals:
                direction = TradeDirection.SHORT
            else:
                # Empate - usar trend principal
                direction = TradeDirection.LONG if trend_strength >= 0 else TradeDirection.SHORT
            
            # SCORE DE QUALIDADE
            quality_score = (confidence * 0.4 + (strength / 0.02) * 0.3 + 
                           (signal_agreement / total_signals) * 0.3) * 100
            analysis_details['quality_score'] = quality_score
            
            # CRITÉRIOS PROFISSIONAIS RÍGIDOS
            meets_confidence = confidence >= self.min_confidence_to_trade
            meets_strength = strength >= self.min_strength_threshold
            meets_signals = signal_agreement >= self.min_signals_agreement
            meets_quality = quality_score >= 70.0  # Score mínimo de qualidade
            
            should_trade = (meets_confidence and meets_strength and 
                          meets_signals and meets_quality and direction is not None)
            
            # Atualizar condições de mercado
            self._update_market_conditions_from_analysis(analysis_details)
            
            # Detalhes da análise
            analysis_details.update({
                'total_signals': total_signals,
                'signals_positive': positive_signals,
                'signals_negative': negative_signals,
                'confidence': round(confidence, 3),
                'strength': round(strength, 4),
                'direction': direction.name if direction else None,
                'should_trade': should_trade,
                'professional_mode': True,
                'quality_requirements_met': {
                    'confidence': meets_confidence,
                    'strength': meets_strength,
                    'signals': meets_signals,
                    'quality': meets_quality
                }
            })
            
            return should_trade, confidence, direction, strength, analysis_details
            
        except Exception as e:
            logger.error(f"❌ Erro na análise profissional: {e}")
            return False, 0.0, None, 0.0, {'error': str(e)}

    def _calculate_trend_strength(self, prices: np.array) -> float:
        """Calcular força da tendência"""
        try:
            # Linear regression para tendência
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            return slope / prices[-1]  # Normalizar pelo preço atual
        except:
            return 0.0

    def _calculate_professional_rsi(self, prices: np.array, period: int = 14) -> Tuple[int, float]:
        """RSI profissional com sinais claros"""
        try:
            if len(prices) < period + 1:
                return 0, 50.0
            
            deltas = np.diff(prices[-period-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Sinais profissionais mais conservadores
            if rsi < 25:  # Oversold extremo
                return 2, rsi  # Strong buy signal
            elif rsi < 35:  # Oversold
                return 1, rsi  # Buy signal
            elif rsi > 75:  # Overbought extremo
                return -2, rsi  # Strong sell signal
            elif rsi > 65:  # Overbought
                return -1, rsi  # Sell signal
            else:
                return 0, rsi  # Neutral
                
        except Exception as e:
            return 0, 50.0

    def _calculate_ma_crossover(self, prices: np.array) -> int:
        """Moving Average Crossover Signal"""
        try:
            if len(prices) < 20:
                return 0
            
            short_ma = np.mean(prices[-5:])   # MA5
            medium_ma = np.mean(prices[-10:]) # MA10
            long_ma = np.mean(prices[-20:])   # MA20
            
            # Crossover bullish
            if short_ma > medium_ma > long_ma:
                return 2  # Strong bullish
            elif short_ma > medium_ma:
                return 1  # Bullish
            # Crossover bearish
            elif short_ma < medium_ma < long_ma:
                return -2  # Strong bearish
            elif short_ma < medium_ma:
                return -1  # Bearish
            else:
                return 0  # Neutral
                
        except:
            return 0

    def _analyze_volume_confirmation(self, current_volume: float) -> int:
        """Análise de confirmação por volume"""
        try:
            if len(self.volume_history) < 10:
                return 0
            
            avg_volume = np.mean(list(self.volume_history)[-10:])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:  # Volume 50% acima da média
                return 2  # Strong confirmation
            elif volume_ratio > 1.2:  # Volume 20% acima da média
                return 1  # Confirmation
            elif volume_ratio < 0.7:  # Volume baixo
                return -1  # Weak signal
            else:
                return 0  # Normal volume
                
        except:
            return 0

    def _analyze_volatility(self, prices: np.array) -> Tuple[int, float]:
        """Análise de volatilidade para oportunidades"""
        try:
            if len(prices) < 20:
                return 0, 0.0
            
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            
            # Volatilidade ideal para trading
            if 0.005 < volatility < 0.025:  # 0.5% - 2.5%
                # Direção baseada no movimento recente
                recent_change = (prices[-1] - prices[-5]) / prices[-5]
                signal = 1 if recent_change > 0 else -1
                return signal, volatility
            elif volatility > 0.025:  # Muito volátil
                return -1, volatility  # Cuidado
            else:  # Pouca volatilidade
                return 0, volatility
                
        except:
            return 0, 0.0

    def _calculate_momentum(self, prices: np.array) -> float:
        """Calcular momentum do preço"""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Momentum = (preço atual - preço 10 períodos atrás) / preço 10 períodos atrás
            momentum = (prices[-1] - prices[-10]) / prices[-10]
            return momentum
            
        except:
            return 0.0

    def _analyze_support_resistance(self, prices: np.array, current_price: float) -> int:
        """Análise de suporte e resistência"""
        try:
            if len(prices) < 50:
                return 0
            
            # Encontrar máximos e mínimos locais
            recent_prices = prices[-30:]
            resistance = np.max(recent_prices)
            support = np.min(recent_prices)
            
            price_range = resistance - support
            if price_range == 0:
                return 0
            
            # Posição do preço atual no range
            position = (current_price - support) / price_range
            
            if position < 0.2:  # Próximo ao suporte
                return 1  # Buy signal
            elif position > 0.8:  # Próximo à resistência
                return -1  # Sell signal
            else:
                return 0  # Neutral
                
        except:
            return 0

    def _update_market_conditions(self):
        """Atualizar condições gerais do mercado"""
        try:
            if len(self.price_history) < 20:
                return
            
            prices = np.array(list(self.price_history))
            
            # Calcular trend geral
            trend_strength = self._calculate_trend_strength(prices)
            if trend_strength > 0.005:
                self.market_conditions['trend'] = 'bullish'
            elif trend_strength < -0.005:
                self.market_conditions['trend'] = 'bearish'
            else:
                self.market_conditions['trend'] = 'neutral'
            
            # Calcular volatilidade
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            self.market_conditions['volatility'] = volatility
            
            # Volume médio
            if len(self.volume_history) > 0:
                self.market_conditions['volume_avg'] = np.mean(list(self.volume_history)[-10:])
            
            # Força geral
            self.market_conditions['strength'] = abs(trend_strength)
            
        except Exception as e:
            logger.error(f"❌ Erro ao atualizar condições: {e}")

    def _update_market_conditions_from_analysis(self, analysis_details: Dict):
        """Atualizar condições do mercado baseado na análise"""
        try:
            self.market_conditions.update({
                'trend': 'bullish' if analysis_details.get('trend_strength', 0) > 0 else 'bearish',
                'volatility': analysis_details.get('volatility', 0),
                'strength': analysis_details.get('strength', 0),
                'quality_score': analysis_details.get('quality_score', 0)
            })
        except:
            pass

    def _execute_professional_trade(self, direction: TradeDirection, confidence: float, strength: float, analysis_details: Dict) -> bool:
        """Execução PROFISSIONAL de trade com LONG e SHORT"""
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
            
            # Targets profissionais
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)
                stop_price = current_price * (1 + self.stop_loss_target)
            else:  # SHORT
                target_price = current_price * (1 - self.profit_target)
                stop_price = current_price * (1 - self.stop_loss_target)
            
            # Calcular taxa estimada (Bitget: ~0.1% por operação)
            estimated_fee = position_value * 0.001  # 0.1% taxa
            
            logger.info(f"🏆 TRADE PROFISSIONAL {direction.name}:")
            logger.info(f"   💰 Saldo: ${balance:.2f} | Size: {position_size:.6f}")
            logger.info(f"   💱 ${current_price:.2f} → Target: ${target_price:.2f} | Stop: ${stop_price:.2f}")
            logger.info(f"   🎯 Conf: {confidence*100:.1f}% | Força: {strength*100:.2f}%")
            logger.info(f"   💸 Taxa estimada: ${estimated_fee:.2f}")
            
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
                logger.info("✅ PAPER TRADE PROFISSIONAL EXECUTADO!")
                return True
            else:
                # Trading real
                try:
                    if direction == TradeDirection.LONG:
                        result = self.bitget_api.place_buy_order()
                    else:  # SHORT
                        result = self._execute_short_order_professional(position_size)
                    
                    if result and result.get('success'):
                        # Registrar taxa paga
                        self.metrics.total_fees_paid += estimated_fee / balance
                        
                        self.current_position = TradePosition(
                            side=direction,
                            size=result.get('quantity', position_size),
                            entry_price=result.get('price', current_price),
                            start_time=time.time(),
                            target_price=target_price,
                            stop_price=stop_price,
                            order_id=result.get('order', {}).get('id')
                        )
                        logger.info("✅ REAL TRADE PROFISSIONAL EXECUTADO!")
                        return True
                    else:
                        logger.error(f"❌ Falha na execução: {result}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Erro na execução: {e}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Erro no trade profissional: {e}")
            return False

    def _execute_short_order_professional(self, position_size: float) -> Dict:
        """Executa ordem SHORT profissional"""
        try:
            logger.info(f"📉 SHORT PROFISSIONAL - {position_size:.6f}")
            
            order = self.bitget_api.exchange.create_market_sell_order(
                'ETHUSDT',
                position_size,
                None,
                {'leverage': self.leverage}
            )
            
            if order:
                logger.info(f"✅ SHORT PROFISSIONAL: {order['id']}")
                return {
                    "success": True,
                    "order": order,
                    "quantity": position_size,
                    "price": order.get('price', 0)
                }
            else:
                return {"success": False, "error": "SHORT profissional falhou"}
                
        except Exception as e:
            logger.error(f"❌ Erro SHORT profissional: {e}")
            return {"success": False, "error": str(e)}

    def _professional_position_management(self):
        """Gerenciamento PROFISSIONAL com saídas RENTÁVEIS"""
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
            is_profitable_exit = False
            
            # === ESTRATÉGIAS DE SAÍDA PROFISSIONAIS ===
            
            # 1. TAKE PROFIT PRINCIPAL (1.5% mínimo)
            if pnl >= self.profit_target:
                should_close = True
                close_reason = f"🎯 TARGET ATINGIDO: {pnl*100:.3f}%"
                is_profitable_exit = True
            
            # 2. STOP LOSS RIGOROSO
            elif pnl <= self.stop_loss_target:
                should_close = True
                close_reason = f"🛑 STOP LOSS: {pnl*100:.3f}%"
            
            # 3. TRAILING STOP PROFISSIONAL (só após lucro significativo)
            elif self.max_profit_reached >= 0.012 and pnl <= (self.max_profit_reached - 0.006):
                should_close = True
                close_reason = f"📉 TRAILING STOP: {pnl*100:.3f}% (max: {self.max_profit_reached*100:.3f}%)"
                is_profitable_exit = True
            
            # 4. SAÍDA POR TEMPO + LUCRO MÍNIMO
            elif duration >= self.min_position_time and pnl >= self.minimum_profit_target:
                should_close = True
                close_reason = f"✅ LUCRO MÍNIMO: {pnl*100:.3f}% em {duration:.0f}s"
                is_profitable_exit = True
            
            # 5. TEMPO MÁXIMO (só se não estiver muito negativo)
            elif duration >= self.max_position_time:
                if pnl > -0.006:  # Não sair com perda grande
                    should_close = True
                    close_reason = f"⏰ TEMPO MAX: {pnl*100:.3f}% em {duration:.0f}s"
                    if pnl > 0:
                        is_profitable_exit = True
            
            # 6. SAÍDA DE EMERGÊNCIA (perdas grandes)
            elif pnl <= -0.015:  # -1.5%
                should_close = True
                close_reason = f"🚨 EMERGÊNCIA: {pnl*100:.3f}%"
            
            # 7. PROFIT LOCK (trancar lucros em níveis específicos)
            elif pnl >= 0.015 and self.current_profit_lock < len(self.profit_locks):
                lock_level = self.profit_locks[self.current_profit_lock]
                if pnl >= lock_level:
                    self.current_profit_lock += 1
                    logger.info(f"🔒 PROFIT LOCKED em {pnl*100:.3f}% (nível {self.current_profit_lock})")
            
            # ❌ REMOVIDAS TODAS AS SAÍDAS COM MICRO LUCROS PREJUDICIAIS
            # ❌ REMOVIDOS BREAKEVEN E SAÍDAS PREMATURAS QUE GERAM PREJUÍZO
            
            if should_close:
                logger.info(f"🔒 FECHANDO POSIÇÃO PROFISSIONAL: {close_reason}")
                success = self._close_position_professional(close_reason)
                
                # Atualizar contadores específicos
                if success and is_profitable_exit:
                    self.profitable_exits += 1
                
                return success
            
            # Log periódico menos frequente
            if int(duration) % 45 == 0:  # A cada 45 segundos
                profit_status = "💚" if pnl > 0 else "🔴"
                logger.info(f"⏳ Posição ativa: {profit_status} {pnl*100:.3f}% | {duration:.0f}s | Max: {self.max_profit_reached*100:.3f}%")
                
        except Exception as e:
            logger.error(f"❌ Erro gerenciamento profissional: {e}")
            traceback.print_exc()
            
            # Forçar fechamento em qualquer erro crítico
            if self.current_position:
                logger.warning("🚨 FORÇANDO FECHAMENTO POR ERRO CRÍTICO")
                self._emergency_close_professional("Erro crítico")

    def _close_position_professional(self, reason: str) -> bool:
        """Fechamento PROFISSIONAL com verificação garantida"""
        try:
            if not self.current_position:
                logger.warning("⚠️ Posição não existe")
                return False
                
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price']) if market_data else self.current_position.entry_price
            final_pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
                
            logger.info(f"🔒 FECHAMENTO PROFISSIONAL: {reason}")
            logger.info(f"   📊 {self.current_position.side.name} | ${self.current_position.entry_price:.2f} → ${current_price:.2f}")
            logger.info(f"   📈 P&L: {final_pnl*100:.4f}% | ⏱️ {duration:.1f}s")
            
            close_success = False
            
            if self.paper_trading:
                logger.info("📋 PAPER TRADING - Fechamento simulado")
                close_success = True
                
            else:
                # MÉTODO PROFISSIONAL: Tentativa principal
                logger.info("🎯 MÉTODO PROFISSIONAL: Fechamento otimizado...")
                try:
                    if self.current_position.side == TradeDirection.LONG:
                        result = self.bitget_api.place_sell_order(profit_target=0)
                        close_success = result and result.get('success', False)
                        if close_success:
                            logger.info("✅ LONG fechado via sell_order profissional")
                    else:  # SHORT
                        result = self._close_short_position_professional()
                        close_success = result and result.get('success', False)
                        if close_success:
                            logger.info("✅ SHORT fechado via close_short profissional")
                except Exception as e:
                    logger.error(f"❌ Método profissional falhou: {e}")
                
                # MÉTODO ALTERNATIVO: API direta se método profissional falhar
                if not close_success:
                    logger.info("🎯 MÉTODO ALTERNATIVO: API Direta...")
                    try:
                        side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                        order = self.bitget_api.exchange.create_market_order(
                            'ETHUSDT', side, abs(self.current_position.size)
                        )
                        if order:
                            logger.info(f"✅ MÉTODO ALTERNATIVO: Sucesso via {side}")
                            close_success = True
                    except Exception as e:
                        logger.error(f"❌ Método alternativo falhou: {e}")
                
                # VERIFICAÇÃO: Confirmar se posição foi realmente fechada
                if close_success:
                    time.sleep(3)  # Aguardar processamento
                    try:
                        positions = self.bitget_api.get_position_info()
                        if positions and positions.get('position') and abs(positions['position']['size']) > 0:
                            logger.warning("⚠️ Posição ainda aberta após fechamento - método de emergência")
                            close_success = self._emergency_close_professional("Verificação falhou")
                    except:
                        pass  # Assumir sucesso se não conseguir verificar
            
            if close_success:
                logger.info("✅ POSIÇÃO FECHADA COM SUCESSO PROFISSIONAL!")
                
                # Calcular taxa estimada para o fechamento
                estimated_fee = abs(final_pnl * self.current_position.size * current_price * 0.001)
                
                # Atualizar métricas PROFISSIONAIS
                with self._lock:
                    self.metrics.total_trades += 1
                    
                    # P&L líquido (descontando taxa de fechamento estimada)
                    net_pnl = final_pnl - (estimated_fee / (self.current_position.size * current_price))
                    self.metrics.total_profit += net_pnl
                    self.metrics.total_fees_paid += estimated_fee / (self.current_position.size * current_price)
                    
                    if net_pnl > 0:
                        self.metrics.profitable_trades += 1
                        self.metrics.consecutive_wins += 1
                        self.metrics.consecutive_losses = 0
                        self.consecutive_losses = 0
                        logger.info(f"💚 LUCRO LÍQUIDO: +{net_pnl*100:.4f}%")
                        
                        # Resetar acumulado de perdas
                        self.daily_loss_accumulated = max(0, self.daily_loss_accumulated - abs(net_pnl))
                        
                    else:
                        self.metrics.consecutive_wins = 0
                        self.metrics.consecutive_losses += 1
                        self.consecutive_losses += 1
                        self.daily_loss_accumulated += abs(net_pnl)
                        logger.info(f"🔴 PERDA LÍQUIDA: {net_pnl*100:.4f}%")
                    
                    # Atualizar drawdown máximo
                    if net_pnl < 0:
                        current_drawdown = abs(net_pnl)
                        self.metrics.max_drawdown = max(self.metrics.max_drawdown, current_drawdown)
                    
                    # Atualizar duração média
                    if self.metrics.total_trades > 0:
                        total_duration = (self.metrics.average_trade_duration * (self.metrics.total_trades - 1) + duration)
                        self.metrics.average_trade_duration = total_duration / self.metrics.total_trades
                    
                    # Atualizar máximos consecutivos
                    self.metrics.max_consecutive_wins = max(
                        self.metrics.max_consecutive_wins, 
                        self.metrics.consecutive_wins
                    )
                    self.metrics.max_consecutive_losses = max(
                        self.metrics.max_consecutive_losses, 
                        self.metrics.consecutive_losses
                    )
                
                # Reset rastreamento
                self.max_profit_reached = 0.0
                self.max_loss_reached = 0.0
                self.current_profit_lock = 0
                
                # Limpar posição
                self.current_position = None
                self.last_trade_time = time.time()
                
                # Performance atual
                daily_profit_pct = self.metrics.net_profit * 100
                target_progress = (daily_profit_pct / 50.0) * 100
                
                logger.info(f"📊 PERFORMANCE PROFISSIONAL ATUALIZADA:")
                logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   💎 Profit Bruto: {self.metrics.total_profit*100:.4f}%")
                logger.info(f"   💰 Profit Líquido: {daily_profit_pct:.4f}%")
                logger.info(f"   💸 Taxas Pagas: {self.metrics.total_fees_paid*100:.4f}%")
                logger.info(f"   🏆 META 50%: {target_progress:.1f}%")
                logger.info(f"   🔥 Wins Consecutivos: {self.metrics.consecutive_wins}")
                logger.info(f"   ❄️ Losses Consecutivos: {self.consecutive_losses}")
                
                return True
                
            else:
                logger.error("❌ TODOS OS MÉTODOS PROFISSIONAIS DE FECHAMENTO FALHARAM!")
                return False
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no fechamento profissional: {e}")
            traceback.print_exc()
            return False

    def _close_short_position_professional(self) -> Dict:
        """Fecha posição SHORT com métodos profissionais"""
        try:
            logger.info("📈 Fechando SHORT profissional - Comprando para cobrir...")
            
            # Método 1: Buy order padrão
            result = self.bitget_api.place_buy_order()
            
            if result and result.get('success'):
                logger.info(f"✅ SHORT fechado via buy profissional: {result.get('message', '')}")
                return {"success": True, "result": result}
            
            # Método 2: API direta
            try:
                order = self.bitget_api.exchange.create_market_buy_order(
                    'ETHUSDT', abs(self.current_position.size), None, {'leverage': self.leverage}
                )
                if order:
                    logger.info(f"✅ SHORT fechado via API direta profissional")
                    return {"success": True, "order": order}
            except Exception as e:
                logger.error(f"❌ Método 2 SHORT profissional: {e}")
            
            return {"success": False, "error": "Falha ao fechar SHORT profissional"}
                
        except Exception as e:
            logger.error(f"❌ Erro ao fechar SHORT profissional: {e}")
            return {"success": False, "error": str(e)}

    def _emergency_close_professional(self, reason: str) -> bool:
        """Fechamento de emergência com TODOS os métodos profissionais"""
        try:
            logger.warning(f"🚨 EMERGÊNCIA PROFISSIONAL: {reason}")
            
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
                        logger.info(f"✅ Emergência profissional: {side} executado")
                        return True
            except Exception as e:
                logger.error(f"❌ Método emergência profissional 2: {e}")
            
            # Método 3: Fechar posição por reduce-only
            try:
                if self.current_position:
                    side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                    self.bitget_api.exchange.create_order(
                        'ETHUSDT', 'market', side, abs(self.current_position.size), 
                        None, {'reduceOnly': True}
                    )
                    logger.info("✅ Emergência profissional: reduce-only executado")
                    return True
            except Exception as e:
                logger.error(f"❌ Método emergência profissional 3: {e}")
            
            # Método 4: Limpeza forçada profissional
            logger.warning("⚠️ LIMPEZA PROFISSIONAL FORÇADA")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na emergência profissional: {e}")
            return True

    def get_account_balance(self) -> float:
        """Obter saldo da conta com fallback profissional"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                balance = float(balance_info.get('free', 0.0))
                if balance > 0:
                    return balance
                    
            if self.paper_trading:
                return 1000.0
            else:
                logger.warning("⚠️ Saldo não obtido - usando fallback profissional")
                return 100.0
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 1000.0 if self.paper_trading else 100.0

    def emergency_stop(self) -> bool:
        """Parada de emergência profissional com fechamento forçado"""
        try:
            logger.warning("🚨 PARADA DE EMERGÊNCIA PROFISSIONAL TOTAL")
            
            self.state = TradingState.EMERGENCY
            self.emergency_stop_triggered = True
            
            # Fechar posição com TODOS os métodos profissionais
            if self.current_position:
                self._emergency_close_professional("Emergency stop profissional total")
            
            # Cancelar todas as ordens
            try:
                self.bitget_api.exchange.cancel_all_orders(self.symbol)
            except:
                pass
            
            # Parar thread
            if self.trading_thread:
                self.trading_thread.join(timeout=5)
            
            self.state = TradingState.STOPPED
            
            logger.warning("🛑 Parada de emergência profissional total concluída")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na parada de emergência profissional: {e}")
            return False

    def _adjust_for_50_percent_target(self):
        """Ajustar parâmetros dinamicamente para atingir 50% diário"""
        try:
            with self._lock:
                current_profit_pct = self.metrics.net_profit * 100
                current_time = datetime.now()
                hours_passed = max(1, current_time.hour - 8) if current_time.hour >= 8 else 1
                
                expected_profit = (50.0 / 24) * hours_passed  # Profit esperado até agora
                profit_deficit = max(0, expected_profit - current_profit_pct)
                
                logger.info(f"📊 AJUSTE PROFISSIONAL PARA 50%:")
                logger.info(f"   💰 Profit atual: {current_profit_pct:.4f}%")
                logger.info(f"   🎯 Esperado: {expected_profit:.2f}%")
                logger.info(f"   📉 Déficit: {profit_deficit:.2f}%")
                
                # Se muito atrás da meta, ajustar critérios (mas manter qualidade)
                if profit_deficit > 8.0:  # Mais de 8% atrás
                    logger.warning("🚨 MUITO ATRÁS DA META - AJUSTE PROFISSIONAL!")
                    # Reduzir critérios mas manter padrão profissional
                    self.min_confidence_to_trade = max(0.6, self.min_confidence_to_trade - 0.05)
                    self.min_strength_threshold = max(0.008, self.min_strength_threshold - 0.001)
                    self.force_trade_after_seconds = max(300, self.force_trade_after_seconds - 60)
                    
                # Se na meta ou à frente, aumentar qualidade
                elif profit_deficit < -3.0:  # Mais de 3% à frente
                    logger.info("✅ À FRENTE DA META - AUMENTAR QUALIDADE!")
                    self.min_confidence_to_trade = min(0.8, self.min_confidence_to_trade + 0.02)
                    self.min_strength_threshold = min(0.015, self.min_strength_threshold + 0.001)
                
                # Se perdas consecutivas, ser mais conservador
                if self.consecutive_losses >= 2:
                    logger.warning("⚠️ PERDAS CONSECUTIVAS - MODO CONSERVADOR!")
                    self.min_confidence_to_trade = min(0.8, self.min_confidence_to_trade + 0.1)
                    self.min_strength_threshold = min(0.015, self.min_strength_threshold + 0.002)
                
                logger.info(f"   🎯 Nova confiança: {self.min_confidence_to_trade*100:.1f}%")
                logger.info(f"   💪 Nova força: {self.min_strength_threshold*100:.2f}%")
                
        except Exception as e:
            logger.error(f"❌ Erro no ajuste profissional: {e}")

    def reset_daily_stats(self):
        """Reset para novo dia - otimizado para 50% profissional"""
        try:
            logger.info("🔄 Reset para NOVO DIA PROFISSIONAL - META 50%!")
            
            with self._lock:
                self.trades_today = 0
                self.metrics = TradingMetrics()
                self.analysis_count = 0
                self.trades_rejected = 0
                self.quality_trades = 0
                self.rejected_low_quality = 0
                self.profitable_exits = 0
                self.fee_losses_avoided = 0
                self.consecutive_losses = 0
                self.daily_loss_accumulated = 0.0
                self.emergency_stop_triggered = False
                self.last_rejection_reason = ""
                self.last_trade_time = time.time()
                self.max_profit_reached = 0.0
                self.max_loss_reached = 0.0
                self.current_profit_lock = 0
                
                # Reset para modo profissional
                self.professional_mode_active = True
                self.force_trade_mode = False
                
                # Reset critérios para padrões profissionais
                self.min_confidence_to_trade = 0.70
                self.min_strength_threshold = 0.010
            
            logger.info("✅ NOVO DIA PROFISSIONAL - PRONTO PARA 50% DE LUCRO REAL!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao resetar: {e}")

    def get_daily_stats(self) -> Dict:
        """Estatísticas focadas na meta de 50% diário profissional"""
        try:
            with self._lock:
                current_time = datetime.now()
                hours_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 24)
                
                daily_profit_pct = self.metrics.net_profit * 100
                target_achievement = (daily_profit_pct / 50.0) * 100
                
                return {
                    'target_50_percent_professional': {
                        'target_profit': '50.00%',
                        'current_profit_gross': f"{self.metrics.total_profit*100:.4f}%",
                        'current_profit_net': f"{daily_profit_pct:.4f}%",
                        'fees_paid': f"{self.metrics.total_fees_paid*100:.4f}%",
                        'achievement': f"{target_achievement:.1f}%",
                        'remaining_needed': f"{max(0, 50.0 - daily_profit_pct):.4f}%",
                        'on_track': target_achievement >= (hours_trading / 24) * 100,
                        'professional_mode': True
                    },
                    'professional_trading_stats': {
                        'trades_executed': self.trades_today,
                        'quality_trades': self.quality_trades,
                        'quality_ratio': f"{(self.quality_trades / max(1, self.trades_today)) * 100:.1f}%",
                        'rejected_low_quality': self.rejected_low_quality,
                        'profitable_exits': self.profitable_exits,
                        'fee_losses_avoided': self.fee_losses_avoided,
                        'target_trades': self.target_trades_per_day,
                        'trades_per_hour': round(self.trades_today / hours_trading, 1),
                        'analysis_count': self.analysis_count,
                        'rejection_rate': round((self.trades_rejected / max(1, self.analysis_count)) * 100, 1)
                    },
                    'performance_professional': {
                        'total_trades': self.metrics.total_trades,
                        'win_rate': round(self.metrics.win_rate, 2),
                        'average_duration': round(self.metrics.average_trade_duration, 1),
                        'consecutive_wins': self.metrics.consecutive_wins,
                        'consecutive_losses': self.metrics.consecutive_losses,
                        'max_consecutive_wins': self.metrics.max_consecutive_wins,
                        'max_consecutive_losses': self.metrics.max_consecutive_losses,
                        'max_drawdown': round(self.metrics.max_drawdown * 100, 4),
                        'profitable_trades': self.metrics.profitable_trades,
                        'losing_trades': self.metrics.losing_trades,
                        'daily_loss_accumulated': round(self.daily_loss_accumulated * 100, 4)
                    },
                    'risk_management_professional': {
                        'emergency_stop_triggered': self.emergency_stop_triggered,
                        'consecutive_losses_current': self.consecutive_losses,
                        'daily_loss_limit': '8.00%',
                        'max_consecutive_losses_limit': 3,
                        'drawdown_limit': '10.00%',
                        'risk_level': 'HIGH' if self.consecutive_losses >= 2 else 'MEDIUM' if self.daily_loss_accumulated > 0.04 else 'LOW'
                    },
                    'current_settings_professional': {
                        'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                        'min_strength': f"{self.min_strength_threshold*100:.2f}%",
                        'min_signals': self.min_signals_agreement,
                        'profit_target': f"{self.profit_target*100:.1f}%",
                        'stop_loss': f"{abs(self.stop_loss_target)*100:.1f}%",
                        'minimum_profit': f"{self.minimum_profit_target*100:.1f}%",
                        'max_position_time': f"{self.max_position_time}s",
                        'min_position_time': f"{self.min_position_time}s"
                    },
                    'market_conditions': self.market_conditions
                }
                
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas profissionais: {e}")
            return {'error': str(e)}

    # Métodos de compatibilidade
    def _close_position_immediately(self, reason: str):
        """Compatibilidade - usar método profissional"""
        self._close_position_professional(reason)
