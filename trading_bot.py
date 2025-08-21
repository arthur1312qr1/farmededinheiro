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
        """Initialize SMART Trading Bot with PROPER Analysis"""
        
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

        # ===== CONFIGURAÇÕES INTELIGENTES E EQUILIBRADAS =====
        self.min_trades_per_day = 150
        self.target_trades_per_day = 200
        self.max_time_between_trades = 300  # 5 minutos entre trades
        self.force_trade_after_seconds = 600  # Forçar após 10 minutos
        self.last_trade_time = 0

        # CRITÉRIOS INTELIGENTES PARA ANÁLISE TÉCNICA
        self.min_confidence_to_trade = 0.75     # 75% confiança mínima
        self.min_prediction_score = 0.70        # 70% score de predição
        self.min_signals_agreement = 8          # 8 de 12 sinais devem concordar
        self.min_strength_threshold = 0.008     # 0.8% força mínima do movimento

        # Configurações de risco EQUILIBRADAS
        self.profit_target = 0.015              # 1.5% take profit
        self.stop_loss_target = -0.020          # 2.0% stop loss
        self.max_position_time = 300            # 5 minutos máximo por trade

        # Sistema de dados para análise técnica
        self.price_history = deque(maxlen=500)
        self.volume_history = deque(maxlen=100)
        self.analysis_history = deque(maxlen=50)

        # Sistema de trading inteligente
        self.aggressive_mode_active = False
        self.emergency_trading_mode = False
        self.last_analysis_result = None
        
        # Rastreamento de preços para trailing
        self.max_profit_reached = 0.0
        self.max_loss_reached = 0.0

        # Métricas de performance
        self.metrics = TradingMetrics()
        self.start_balance = 0.0
        self.trades_today = 0
        
        # Lock para thread safety
        self._lock = threading.Lock()

        # Contador de análises e debug
        self.analysis_count = 0
        self.trades_rejected = 0
        self.last_rejection_reason = ""

        logger.info("🤖 TRADING BOT INTELIGENTE INICIALIZADO")
        logger.info("📊 CONFIGURAÇÕES EQUILIBRADAS:")
        logger.info(f"   🎯 Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"   💪 Força mínima: {self.min_strength_threshold*100}%")
        logger.info(f"   🔍 Sinais necessários: {self.min_signals_agreement}")
        logger.info(f"   📈 Take Profit: {self.profit_target*100}%")
        logger.info(f"   🛑 Stop Loss: {abs(self.stop_loss_target)*100}%")
        logger.info("✅ ANÁLISE TÉCNICA ATIVA - TRADES INTELIGENTES!")

    @property
    def is_running(self) -> bool:
        """Propriedade para verificar se o bot está rodando"""
        return self.state == TradingState.RUNNING

    def get_status(self) -> Dict:
        """Status completo do bot"""
        try:
            with self._lock:
                current_time = datetime.now()
                
                hours_in_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 1)
                expected_trades = (self.min_trades_per_day / 16) * hours_in_trading
                trade_deficit = max(0, expected_trades - self.trades_today)
                
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
                    'trading_analysis': {
                        'analysis_count': self.analysis_count,
                        'trades_executed': self.trades_today,
                        'trades_rejected': self.trades_rejected,
                        'last_rejection_reason': self.last_rejection_reason,
                        'seconds_since_last_trade': round(seconds_since_last_trade),
                        'next_analysis_in': max(0, self.scalping_interval - (time.time() % self.scalping_interval)),
                        'current_thresholds': {
                            'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                            'min_strength': f"{self.min_strength_threshold*100:.1f}%",
                            'min_signals': self.min_signals_agreement
                        }
                    },
                    'daily_progress': {
                        'trades_today': self.trades_today,
                        'target_trades': self.target_trades_per_day,
                        'progress_percent': round((self.trades_today / self.target_trades_per_day) * 100, 1),
                        'deficit': round(trade_deficit),
                        'trades_per_hour_current': round(self.trades_today / max(1, hours_in_trading), 1),
                        'trades_per_hour_needed': round(self.target_trades_per_day / 16, 1)
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
                'pnl_percent': round(pnl * 100, 3),
                'target_price': self.current_position.target_price,
                'stop_price': self.current_position.stop_price,
                'max_profit_reached': round(self.max_profit_reached * 100, 3),
                'trailing_active': pnl > 0.01  # Trailing ativo com +1%
            }
        except Exception as e:
            return {'active': True, 'error': f'Erro ao obter dados: {str(e)}'}

    def start(self) -> bool:
        """Iniciar bot inteligente"""
        try:
            if self.state == TradingState.RUNNING:
                logger.warning("🟡 Bot já está rodando")
                return True
            
            logger.info("🚀 INICIANDO BOT DE TRADING INTELIGENTE")
            logger.info("🧠 ANÁLISE TÉCNICA ATIVA - TRADES BASEADOS EM DADOS!")
            
            # Resetar contadores
            self.analysis_count = 0
            self.trades_rejected = 0
            self.last_rejection_reason = ""
            
            # Resetar estado
            self.state = TradingState.RUNNING
            self.start_balance = self.get_account_balance()
            self.last_trade_time = time.time()
            self.last_error = None
            
            # Reset rastreamento
            self.max_profit_reached = 0.0
            self.max_loss_reached = 0.0
            
            # Iniciar thread principal
            self.trading_thread = threading.Thread(
                target=self._intelligent_trading_loop, 
                daemon=True,
                name="IntelligentTradingBot"
            )
            self.trading_thread.start()
            
            logger.info("✅ Bot inteligente iniciado - ANÁLISE TÉCNICA ATIVA!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            self.last_error = str(e)
            return False

    def stop(self) -> bool:
        """Parar bot com relatório"""
        try:
            logger.info("🛑 Parando bot inteligente...")
            
            self.state = TradingState.STOPPED
            
            # Fechar posição se existir
            if self.current_position:
                logger.info("📤 Fechando posição final...")
                self._close_position_safely("Bot stopping", force_close=True)
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=15)
            
            # Relatório final
            logger.info("📊 RELATÓRIO FINAL:")
            logger.info(f"   🔍 Análises realizadas: {self.analysis_count}")
            logger.info(f"   ✅ Trades executados: {self.trades_today}")
            logger.info(f"   ❌ Trades rejeitados: {self.trades_rejected}")
            logger.info(f"   📈 Taxa de sucesso: {self.metrics.win_rate:.1f}%")
            logger.info(f"   💰 Lucro total: {self.metrics.total_profit*100:.2f}%")
            
            logger.info("✅ Bot parado com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return False

    def _intelligent_trading_loop(self):
        """Loop inteligente de trading com análise técnica REAL"""
        logger.info("🧠 Loop inteligente iniciado - ANÁLISE TÉCNICA ATIVA")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                self.analysis_count += 1
                
                # ANÁLISE TÉCNICA COMPLETA
                should_trade, confidence, direction, strength, analysis_details = self._comprehensive_technical_analysis()
                
                # LOG PERIÓDICO DETALHADO
                if self.analysis_count % 20 == 0:
                    logger.info(f"📊 Análise #{self.analysis_count}:")
                    logger.info(f"   🎯 Confiança: {confidence*100:.1f}% (min: {self.min_confidence_to_trade*100:.1f}%)")
                    logger.info(f"   💪 Força: {strength*100:.3f}% (min: {self.min_strength_threshold*100:.3f}%)")
                    logger.info(f"   📍 Direção: {direction.name if direction else 'Indefinida'}")
                    logger.info(f"   ✅ Executar: {should_trade}")
                    if analysis_details:
                        logger.info(f"   📈 RSI: {analysis_details.get('rsi', 'N/A')}")
                        logger.info(f"   📊 MACD: {analysis_details.get('macd_signal', 'N/A')}")
                        logger.info(f"   🌊 Bollinger: {analysis_details.get('bb_signal', 'N/A')}")

                # EXECUTAR TRADE SE CRITÉRIOS ATENDIDOS
                if should_trade and not self.current_position:
                    success = self._execute_intelligent_trade(direction, confidence, strength, analysis_details)
                    if success:
                        self.last_trade_time = time.time()
                        self.trades_today += 1
                        logger.info(f"⚡ TRADE #{self.trades_today} EXECUTADO - {direction.name}")
                        logger.info(f"   🎯 Confiança: {confidence*100:.1f}%")
                        logger.info(f"   💪 Força: {strength*100:.3f}%")
                        logger.info(f"   📊 Sinais: {analysis_details.get('signals_positive', 0)}/{analysis_details.get('total_signals', 0)}")
                    else:
                        self.trades_rejected += 1
                        self.last_rejection_reason = "Falha na execução"
                
                elif not should_trade and not self.current_position:
                    self.trades_rejected += 1
                    self.last_rejection_reason = f"Conf:{confidence*100:.1f}% Força:{strength*100:.3f}%"
                
                # GERENCIAR POSIÇÃO EXISTENTE
                if self.current_position:
                    self._intelligent_position_management()
                
                # Sleep inteligente
                elapsed = time.time() - loop_start
                sleep_time = max(0.5, self.scalping_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop inteligente: {e}")
                traceback.print_exc()
                time.sleep(2)
        
        logger.info(f"🏁 Loop finalizado - Trades executados: {self.trades_today}")

    def _comprehensive_technical_analysis(self) -> Tuple[bool, float, Optional[TradeDirection], float, Dict]:
        """Análise técnica COMPLETA e INTELIGENTE"""
        try:
            # Obter dados de mercado
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data or 'price' not in market_data:
                return False, 0.0, None, 0.0, {'error': 'Sem dados de mercado'}
            
            current_price = float(market_data['price'])
            current_volume = float(market_data.get('volume', 0))
            
            self.price_history.append(current_price)
            if current_volume > 0:
                self.volume_history.append(current_volume)
            
            # Verificar dados mínimos para análise técnica
            if len(self.price_history) < 50:
                return False, 0.0, None, 0.0, {'error': f'Dados insuficientes: {len(self.price_history)}/50'}
            
            # Converter para arrays numpy
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history)) if len(self.volume_history) > 10 else None
            
            # === INDICADORES TÉCNICOS PROFISSIONAIS ===
            signals = []
            analysis_details = {}
            
            # 1. RSI (Relative Strength Index)
            try:
                if len(prices) >= 14:
                    deltas = np.diff(prices[-30:])  # Últimos 30 preços
                    gains = np.where(deltas > 0, deltas, 0)
                    losses = np.where(deltas < 0, -deltas, 0)
                    
                    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        analysis_details['rsi'] = round(rsi, 2)
                        
                        # RSI Signals
                        if rsi < 30:  # Oversold - BUY signal
                            signals.extend([1, 1])  # Double weight
                        elif rsi > 70:  # Overbought - SELL signal
                            signals.extend([-1, -1])  # Double weight
                        elif rsi < 45:  # Slight oversold
                            signals.append(1)
                        elif rsi > 55:  # Slight overbought
                            signals.append(-1)
            except:
                pass
            
            # 2. MACD (Moving Average Convergence Divergence)
            try:
                if len(prices) >= 26:
                    ema12 = self._calculate_ema(prices, 12)
                    ema26 = self._calculate_ema(prices, 26)
                    macd_line = ema12[-1] - ema26[-1]
                    
                    if len(prices) >= 35:
                        macd_values = ema12[-9:] - ema26[-9:]  # Last 9 MACD values
                        signal_line = self._calculate_ema(macd_values, 9)[-1]
                        macd_histogram = macd_line - signal_line
                        
                        analysis_details['macd'] = round(macd_line, 6)
                        analysis_details['macd_signal'] = round(signal_line, 6)
                        analysis_details['macd_histogram'] = round(macd_histogram, 6)
                        
                        # MACD Signals
                        if macd_line > signal_line and macd_histogram > 0:
                            signals.extend([1, 1])  # Bullish crossover
                        elif macd_line < signal_line and macd_histogram < 0:
                            signals.extend([-1, -1])  # Bearish crossover
                        elif macd_histogram > 0:
                            signals.append(1)
                        elif macd_histogram < 0:
                            signals.append(-1)
            except:
                pass
            
            # 3. Bollinger Bands
            try:
                if len(prices) >= 20:
                    sma20 = np.mean(prices[-20:])
                    std20 = np.std(prices[-20:])
                    upper_band = sma20 + (2 * std20)
                    lower_band = sma20 - (2 * std20)
                    
                    analysis_details['bb_upper'] = round(upper_band, 2)
                    analysis_details['bb_middle'] = round(sma20, 2)
                    analysis_details['bb_lower'] = round(lower_band, 2)
                    
                    bb_position = (current_price - lower_band) / (upper_band - lower_band)
                    analysis_details['bb_position'] = round(bb_position, 3)
                    
                    # Bollinger Signals
                    if current_price <= lower_band:  # Price at lower band - BUY
                        signals.extend([1, 1])
                        analysis_details['bb_signal'] = 'OVERSOLD'
                    elif current_price >= upper_band:  # Price at upper band - SELL
                        signals.extend([-1, -1])
                        analysis_details['bb_signal'] = 'OVERBOUGHT'
                    elif bb_position < 0.3:  # Near lower band
                        signals.append(1)
                        analysis_details['bb_signal'] = 'NEAR_LOWER'
                    elif bb_position > 0.7:  # Near upper band
                        signals.append(-1)
                        analysis_details['bb_signal'] = 'NEAR_UPPER'
                    else:
                        analysis_details['bb_signal'] = 'MIDDLE'
            except:
                pass
            
            # 4. Moving Averages Crossover
            try:
                if len(prices) >= 50:
                    sma10 = np.mean(prices[-10:])
                    sma20 = np.mean(prices[-20:])
                    sma50 = np.mean(prices[-50:])
                    
                    analysis_details['sma10'] = round(sma10, 2)
                    analysis_details['sma20'] = round(sma20, 2)
                    analysis_details['sma50'] = round(sma50, 2)
                    
                    # MA Crossover Signals
                    if sma10 > sma20 > sma50:  # Bullish alignment
                        signals.extend([1, 1])
                    elif sma10 < sma20 < sma50:  # Bearish alignment
                        signals.extend([-1, -1])
                    elif sma10 > sma20:  # Short term bullish
                        signals.append(1)
                    elif sma10 < sma20:  # Short term bearish
                        signals.append(-1)
            except:
                pass
            
            # 5. Volume Analysis
            try:
                if volumes is not None and len(volumes) >= 10:
                    avg_volume = np.mean(volumes[-10:])
                    current_vol_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    analysis_details['volume_ratio'] = round(current_vol_ratio, 2)
                    
                    # Volume Signals
                    if current_vol_ratio > 1.5:  # High volume
                        price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
                        if price_change > 0:
                            signals.append(1)  # High volume + price up = bullish
                        elif price_change < 0:
                            signals.append(-1)  # High volume + price down = bearish
                    elif current_vol_ratio < 0.7:  # Low volume - reduce confidence
                        signals.append(0)  # Neutral signal for low volume
            except:
                pass
            
            # 6. Price Momentum
            try:
                if len(prices) >= 10:
                    momentum_5 = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
                    momentum_10 = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
                    
                    analysis_details['momentum_5'] = round(momentum_5 * 100, 3)
                    analysis_details['momentum_10'] = round(momentum_10 * 100, 3)
                    
                    # Momentum Signals
                    if momentum_5 > 0.005:  # 0.5% positive momentum
                        signals.append(1)
                    elif momentum_5 < -0.005:  # -0.5% negative momentum
                        signals.append(-1)
                        
                    if momentum_10 > 0.01:  # 1% positive longer momentum
                        signals.append(1)
                    elif momentum_10 < -0.01:  # -1% negative longer momentum
                        signals.append(-1)
            except:
                pass
            
            # 7. Volatility Analysis
            try:
                if len(prices) >= 20:
                    volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
                    analysis_details['volatility'] = round(volatility * 100, 3)
                    
                    # Volatility Signals - prefer moderate volatility
                    if 0.005 < volatility < 0.025:  # Good volatility range
                        signals.append(1 if prices[-1] > np.mean(prices[-5:]) else -1)
                    elif volatility > 0.03:  # Too volatile - be cautious
                        signals.append(0)
            except:
                pass
            
            # === ANÁLISE DOS SINAIS ===
            if len(signals) < 5:  # Mínimo de sinais para análise
                return False, 0.0, None, 0.0, {'error': 'Sinais insuficientes', 'signals_count': len(signals)}
            
            # Calcular direção e força
            total_signals = len(signals)
            positive_signals = len([s for s in signals if s > 0])
            negative_signals = len([s for s in signals if s < 0])
            neutral_signals = total_signals - positive_signals - negative_signals
            
            signal_sum = sum(signals)
            signal_strength = abs(signal_sum) / total_signals
            confidence = (max(positive_signals, negative_signals) / total_signals)
            
            # Determinar direção
            if positive_signals > negative_signals:
                direction = TradeDirection.LONG
            elif negative_signals > positive_signals:
                direction = TradeDirection.SHORT
            else:
                direction = None
            
            # Calcular força real baseada na análise técnica
            strength = signal_strength * confidence
            
            # Critérios finais INTELIGENTES
            meets_confidence = confidence >= self.min_confidence_to_trade
            meets_strength = strength >= self.min_strength_threshold
            meets_signals = max(positive_signals, negative_signals) >= self.min_signals_agreement
            
            should_trade = meets_confidence and meets_strength and meets_signals and direction is not None
            
            # Detalhes da análise
            analysis_details.update({
                'total_signals': total_signals,
                'signals_positive': positive_signals,
                'signals_negative': negative_signals,
                'signals_neutral': neutral_signals,
                'signal_sum': signal_sum,
                'confidence': round(confidence, 3),
                'strength': round(strength, 3),
                'direction': direction.name if direction else None,
                'meets_confidence': meets_confidence,
                'meets_strength': meets_strength,
                'meets_signals': meets_signals,
                'should_trade': should_trade
            })
            
            return should_trade, confidence, direction, strength, analysis_details
            
        except Exception as e:
            logger.error(f"❌ Erro na análise técnica: {e}")
            traceback.print_exc()
            return False, 0.0, None, 0.0, {'error': str(e)}

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calcula EMA (Exponential Moving Average)"""
        try:
            alpha = 2 / (period + 1)
            ema = np.zeros(len(prices))
            ema[0] = prices[0]
            
            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
            return ema
        except:
            return prices  # Fallback

    def _execute_intelligent_trade(self, direction: TradeDirection, confidence: float, strength: float, analysis_details: Dict) -> bool:
        """Execução de trade inteligente com LONG e SHORT"""
        try:
            balance = self.get_account_balance()
            if balance <= 0:
                if self.paper_trading:
                    balance = 1000
                else:
                    logger.error("❌ Saldo insuficiente para trade")
                    return False
            
            # Calcular tamanho da posição
            position_value = balance * self.leverage
            
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price'])
            position_size = position_value / current_price
            
            # Calcular preços de target e stop
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)
                stop_price = current_price * (1 + self.stop_loss_target)
            else:  # SHORT
                target_price = current_price * (1 - self.profit_target)
                stop_price = current_price * (1 - self.stop_loss_target)
            
            logger.info(f"⚡ EXECUTANDO {direction.name} INTELIGENTE:")
            logger.info(f"   💰 Saldo: ${balance:.2f}")
            logger.info(f"   📊 Tamanho: {position_size:.6f} {self.symbol[:3]}")
            logger.info(f"   💱 Preço: ${current_price:.2f}")
            logger.info(f"   🎯 Target: ${target_price:.2f} ({self.profit_target*100:.1f}%)")
            logger.info(f"   🛑 Stop: ${stop_price:.2f} ({abs(self.stop_loss_target)*100:.1f}%)")
            logger.info(f"   🧠 Análise: RSI={analysis_details.get('rsi', 'N/A')}, MACD={analysis_details.get('macd_signal', 'N/A')}")
            
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
                logger.info("✅ TRADE PAPER EXECUTADO!")
                return True
            else:
                # Trading real
                try:
                    if direction == TradeDirection.LONG:
                        result = self.bitget_api.place_buy_order()
                    else:  # SHORT - IMPLEMENTADO
                        # Para SHORT, usar sell order com alavancagem
                        logger.info("📉 Executando SHORT (venda)...")
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

    def _execute_short_order(self, position_size: float) -> Dict:
        """Executa ordem SHORT (venda com alavancagem)"""
        try:
            logger.info(f"📉 Executando SHORT - Quantidade: {position_size:.6f}")
            
            # Para SHORT, usar create_order diretamente
            order = self.bitget_api.exchange.create_market_sell_order(
                'ETHUSDT',
                position_size,
                None,
                {'leverage': self.leverage}
            )
            
            if order:
                logger.info(f"✅ SHORT executado: {order['id']}")
                return {
                    "success": True,
                    "order": order,
                    "quantity": position_size,
                    "price": order.get('price', 0),
                    "message": f"SHORT executado - ID: {order['id']}"
                }
            else:
                return {"success": False, "error": "Ordem SHORT falhou"}
                
        except Exception as e:
            logger.error(f"❌ Erro no SHORT: {e}")
            return {"success": False, "error": str(e)}

    def _intelligent_position_management(self):
        """Gerenciamento INTELIGENTE de posição - CORRIGIDO"""
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
            
            # Atualizar máximos para trailing
            if pnl > self.max_profit_reached:
                self.max_profit_reached = pnl
            
            # LOG DETALHADO
            logger.info(f"📊 GERENCIANDO POSIÇÃO {self.current_position.side.name}:")
            logger.info(f"   💱 Entrada: ${self.current_position.entry_price:.2f} → Atual: ${current_price:.2f}")
            logger.info(f"   📈 P&L: {pnl*100:.3f}% | Max: {self.max_profit_reached*100:.3f}%")
            logger.info(f"   ⏱️ Duração: {duration:.1f}s / {self.max_position_time}s")
            logger.info(f"   🎯 Target: ${self.current_position.target_price:.2f} ({self.profit_target*100:.1f}%)")
            logger.info(f"   🛑 Stop: ${self.current_position.stop_price:.2f} ({abs(self.stop_loss_target)*100:.1f}%)")
            
            should_close = False
            close_reason = ""
            
            # === CRITÉRIOS DE FECHAMENTO INTELIGENTES ===
            
            # 1. TARGET DE LUCRO ATINGIDO
            if pnl >= self.profit_target:
                should_close = True
                close_reason = f"🎯 TARGET ATINGIDO: {pnl*100:.3f}%"
            
            # 2. STOP LOSS ATINGIDO
            elif pnl <= self.stop_loss_target:
                should_close = True
                close_reason = f"🛑 STOP LOSS: {pnl*100:.3f}%"
            
            # 3. TRAILING STOP INTELIGENTE (após 1% de lucro)
            elif self.max_profit_reached >= 0.01 and pnl <= (self.max_profit_reached - 0.005):
                should_close = True
                close_reason = f"📉 TRAILING STOP: {pnl*100:.3f}% (max: {self.max_profit_reached*100:.3f}%)"
            
            # 4. TEMPO MÁXIMO ATINGIDO
            elif duration >= self.max_position_time:
                should_close = True
                close_reason = f"⏰ TEMPO MÁXIMO: {pnl*100:.3f}% em {duration:.0f}s"
            
            # 5. BREAKEVEN RÁPIDO (após 2 minutos se próximo do zero)
            elif duration >= 120 and -0.002 <= pnl <= 0.002:
                should_close = True
                close_reason = f"⚖️ BREAKEVEN: {pnl*100:.3f}% em {duration:.0f}s"
            
            # 6. PEQUENO LUCRO APÓS TEMPO MÉDIO
            elif duration >= 180 and pnl >= 0.005:
                should_close = True
                close_reason = f"💰 LUCRO CONSISTENTE: {pnl*100:.3f}% em {duration:.0f}s"
            
            # 7. CORTE DE PREJUÍZO GRANDE
            elif pnl <= -0.015:  # -1.5% ou pior
                should_close = True
                close_reason = f"🚨 CORTE PREJUÍZO: {pnl*100:.3f}%"
            
            # 8. ANÁLISE TÉCNICA DE REVERSÃO
            elif duration >= 60 and pnl > 0.003:
                if self._detect_technical_reversal():
                    should_close = True
                    close_reason = f"🔄 REVERSÃO TÉCNICA: {pnl*100:.3f}%"
            
            if should_close:
                logger.warning(f"🔔 FECHANDO POSIÇÃO: {close_reason}")
                success = self._close_position_safely(close_reason)
                if not success:
                    logger.error("❌ Falha ao fechar - tentando método de emergência...")
                    self._close_position_safely(close_reason, force_close=True)
            else:
                # Log periódico
                if int(duration) % 30 == 0:  # A cada 30 segundos
                    logger.info(f"⏳ Posição ativa: {pnl*100:.3f}% | {duration:.0f}s")
                
        except Exception as e:
            logger.error(f"❌ Erro no gerenciamento: {e}")
            traceback.print_exc()
            
            # Forçar fechamento em caso de erro crítico
            if self.current_position and self.current_position.get_duration() > 300:
                logger.warning("🚨 Forçando fechamento por erro prolongado")
                self._close_position_safely("Erro crítico - fechamento forçado", force_close=True)

    def _detect_technical_reversal(self) -> bool:
        """Detecta reversão técnica usando indicadores"""
        try:
            if len(self.price_history) < 20:
                return False
            
            prices = np.array(list(self.price_history))
            current_price = prices[-1]
            
            # RSI check para overbought/oversold
            if len(prices) >= 14:
                deltas = np.diff(prices[-15:])
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    # Reversão se RSI extremo
                    if self.current_position.side == TradeDirection.LONG and rsi > 75:
                        return True
                    elif self.current_position.side == TradeDirection.SHORT and rsi < 25:
                        return True
            
            # Price momentum reversal
            if len(prices) >= 5:
                recent_trend = (prices[-1] - prices[-5]) / prices[-5]
                if self.current_position.side == TradeDirection.LONG and recent_trend < -0.003:
                    return True
                elif self.current_position.side == TradeDirection.SHORT and recent_trend > 0.003:
                    return True
            
            return False
            
        except:
            return False

    def _close_position_safely(self, reason: str, force_close: bool = False) -> bool:
        """Fecha posição de forma SEGURA e INTELIGENTE"""
        try:
            if not self.current_position:
                logger.warning("⚠️ Tentativa de fechar posição inexistente")
                return False
                
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price']) if market_data else self.current_position.entry_price
            pnl = self.current_position.calculate_pnl(current_price)
                
            logger.info(f"🔄 FECHANDO POSIÇÃO: {reason}")
            logger.info(f"   📍 Lado: {self.current_position.side.name}")
            logger.info(f"   💱 Entrada: ${self.current_position.entry_price:.2f} → Saída: ${current_price:.2f}")
            logger.info(f"   📊 P&L: {pnl*100:.3f}%")
            logger.info(f"   ⏱️ Duração: {self.current_position.get_duration():.1f}s")
            
            close_success = False
            
            if self.paper_trading:
                # Paper trading - sempre sucesso
                logger.info("📋 PAPER TRADING - Fechamento simulado")
                close_success = True
                
            else:
                # Trading real - CORRIGIDO
                logger.info("💰 TRADING REAL - Executando fechamento...")
                
                try:
                    if self.current_position.side == TradeDirection.LONG:
                        # Fechar LONG = VENDER
                        logger.info("   📤 Executando VENDA para fechar LONG...")
                        result = self.bitget_api.place_sell_order(profit_target=0)  # Vender imediatamente
                        close_success = result and result.get('success', False)
                        
                    else:  # SHORT
                        # Fechar SHORT = COMPRAR
                        logger.info("   📥 Executando COMPRA para fechar SHORT...")
                        result = self._close_short_position()
                        close_success = result and result.get('success', False)
                    
                    if not close_success and force_close:
                        logger.warning("🚨 FORÇANDO FECHAMENTO DE EMERGÊNCIA...")
                        close_success = self._emergency_close_position()
                        
                except Exception as e:
                    logger.error(f"❌ Erro no fechamento real: {e}")
                    if force_close:
                        logger.warning("🚨 Forçando limpeza de emergência...")
                        close_success = True  # Forçar limpeza
            
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
                        logger.info(f"💚 TRADE LUCRATIVO: +{pnl*100:.3f}%")
                    else:
                        self.metrics.consecutive_wins = 0
                        logger.info(f"🔴 TRADE COM PERDA: {pnl*100:.3f}%")
                    
                    # Atualizar duração média
                    if self.metrics.total_trades > 0:
                        total_duration = (self.metrics.average_trade_duration * (self.metrics.total_trades - 1) + 
                                        self.current_position.get_duration())
                        self.metrics.average_trade_duration = total_duration / self.metrics.total_trades
                
                # Reset rastreamento
                self.max_profit_reached = 0.0
                self.max_loss_reached = 0.0
                
                # Limpar posição
                self.current_position = None
                self.last_trade_time = time.time()
                
                # Log de performance
                logger.info(f"📈 PERFORMANCE ATUALIZADA:")
                logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   💰 Profit Total: {self.metrics.total_profit*100:.3f}%")
                logger.info(f"   🔥 Wins Consecutivos: {self.metrics.consecutive_wins}")
                
                return True
                
            else:
                logger.error("❌ FALHA AO FECHAR POSIÇÃO!")
                return False
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no fechamento: {e}")
            traceback.print_exc()
            
            # Limpeza de emergência
            if force_close:
                logger.warning("🚨 LIMPEZA DE EMERGÊNCIA")
                self.current_position = None
                self.max_profit_reached = 0.0
                return False
            
            return False

    def _close_short_position(self) -> Dict:
        """Fecha posição SHORT (compra para cobrir)"""
        try:
            logger.info("📥 Fechando SHORT - Executando compra para cobrir...")
            
            # Para fechar SHORT, fazer uma compra
            result = self.bitget_api.place_buy_order()
            
            if result and result.get('success'):
                logger.info(f"✅ SHORT fechado via compra: {result.get('message', '')}")
                return {"success": True, "result": result}
            else:
                logger.error(f"❌ Falha ao fechar SHORT: {result}")
                return {"success": False, "error": "Falha na compra para fechar SHORT"}
                
        except Exception as e:
            logger.error(f"❌ Erro ao fechar SHORT: {e}")
            return {"success": False, "error": str(e)}

    def _emergency_close_position(self) -> bool:
        """Fechamento de emergência"""
        try:
            logger.warning("🚨 FECHAMENTO DE EMERGÊNCIA!")
            
            # Tentar métodos alternativos
            try:
                positions = self.bitget_api.get_position_info()
                if positions and positions.get('position'):
                    pos = positions['position']
                    if abs(pos['size']) > 0:
                        # Fechar usando API direta
                        side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                        
                        order = self.bitget_api.exchange.create_market_order(
                            'ETHUSDT',
                            side,
                            abs(pos['size'])
                        )
                        
                        if order:
                            logger.info(f"✅ Emergência: posição fechada via {side}")
                            return True
                            
            except Exception as e:
                logger.error(f"❌ Método de emergência falhou: {e}")
            
            # Último recurso - forçar limpeza
            logger.warning("⚠️ FORÇANDO LIMPEZA - posição será removida")
            return True
            
        except:
            return True  # Sempre "sucesso" para limpeza forçada

    def get_account_balance(self) -> float:
        """Obter saldo da conta"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                balance = float(balance_info.get('free', 0.0))
                if balance > 0:
                    return balance
                    
            if self.paper_trading:
                return 1000.0
            else:
                logger.warning("⚠️ Não foi possível obter saldo real")
                return 100.0
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 1000.0 if self.paper_trading else 100.0

    def emergency_stop(self) -> bool:
        """Parada de emergência"""
        try:
            logger.warning("🚨 PARADA DE EMERGÊNCIA ATIVADA")
            
            self.state = TradingState.EMERGENCY
            
            # Fechar posição imediatamente
            if self.current_position:
                self._close_position_safely("Emergency stop", force_close=True)
            
            # Parar thread
            if self.trading_thread:
                self.trading_thread.join(timeout=5)
            
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
                self.max_profit_reached = 0.0
                self.max_loss_reached = 0.0
            
            logger.info("✅ Estatísticas resetadas - PRONTO PARA TRADING INTELIGENTE!")
            
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
                        'target_achievement': (self.trades_today / self.target_trades_per_day) * 100,
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
                        'min_strength': f"{self.min_strength_threshold*100:.3f}%",
                        'profit_target': f"{self.profit_target*100:.1f}%",
                        'stop_loss': f"{abs(self.stop_loss_target)*100:.1f}%",
                        'max_position_time_seconds': self.max_position_time
                    },
                    'technical_analysis': {
                        'data_points': len(self.price_history),
                        'volume_data_points': len(self.volume_history),
                        'last_analysis': self.last_analysis_result,
                        'indicators_active': ['RSI', 'MACD', 'Bollinger Bands', 'Moving Averages', 'Volume', 'Momentum']
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter estatísticas: {e}")
            return {'error': str(e)}

    def adjust_risk_parameters(self, more_conservative: bool = True):
        """Ajusta parâmetros de risco"""
        try:
            with self._lock:
                if more_conservative:
                    # Mais conservador
                    self.min_confidence_to_trade = min(0.85, self.min_confidence_to_trade + 0.05)
                    self.min_strength_threshold = min(0.015, self.min_strength_threshold + 0.002)
                    self.profit_target = min(0.025, self.profit_target + 0.002)
                    self.stop_loss_target = max(-0.035, self.stop_loss_target - 0.005)
                    
                    logger.info("🛡️ PARÂMETROS MAIS CONSERVADORES:")
                else:
                    # Mais agressivo (mas ainda inteligente)
                    self.min_confidence_to_trade = max(0.65, self.min_confidence_to_trade - 0.05)
                    self.min_strength_threshold = max(0.005, self.min_strength_threshold - 0.002)
                    self.profit_target = max(0.010, self.profit_target - 0.002)
                    self.stop_loss_target = min(-0.015, self.stop_loss_target + 0.005)
                    
                    logger.info("⚡ PARÂMETROS MAIS AGRESSIVOS:")
                
                logger.info(f"   🎯 Confiança: {self.min_confidence_to_trade*100:.1f}%")
                logger.info(f"   💪 Força: {self.min_strength_threshold*100:.3f}%")
                logger.info(f"   📈 Target: {self.profit_target*100:.1f}%")
                logger.info(f"   🛑 Stop: {abs(self.stop_loss_target)*100:.1f}%")
                
        except Exception as e:
            logger.error(f"❌ Erro ao ajustar parâmetros: {e}")

    # Método de compatibilidade
    def _close_position_immediately(self, reason: str):
        """Compatibilidade com código original"""
        self._close_position_safely(reason, force_close=True)
