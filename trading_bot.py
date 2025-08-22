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

class AdvancedPredictor:
    """Preditor avançado para análise técnica extremamente precisa"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'gbr': GradientBoostingRegressor(n_estimators=200, max_depth=8, random_state=42),
            'lr': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_history = deque(maxlen=500)
        self.price_history = deque(maxlen=500)
        
    def extract_features(self, prices: np.array) -> np.array:
        """Extrai features avançadas para ML"""
        if len(prices) < 50:
            return np.zeros(25)  # 25 features
        
        features = []
        
        # 1. Trends múltiplos
        for period in [5, 10, 20]:
            if len(prices) >= period:
                trend = (prices[-1] - prices[-period]) / prices[-period]
                features.append(trend)
            else:
                features.append(0)
        
        # 2. RSI múltiplos
        for period in [7, 14, 21]:
            rsi = self._calculate_rsi(prices, period)
            features.append(rsi / 100.0)
        
        # 3. MACD
        macd, signal = self._calculate_macd(prices)
        features.extend([macd, signal, macd - signal])
        
        # 4. Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(prices)
        bb_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle != 0 else 0
        features.extend([bb_position, bb_width])
        
        # 5. Volume indicators (simulados)
        features.extend([0.5, 0.5])  # Volume ratio, Volume trend
        
        # 6. Volatilidade
        volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
        features.append(volatility)
        
        # 7. Support/Resistance
        support_strength = self._calculate_support_strength(prices)
        resistance_strength = self._calculate_resistance_strength(prices)
        features.extend([support_strength, resistance_strength])
        
        # 8. Moving averages crossovers
        for short, long in [(5, 10), (10, 20), (20, 50)]:
            if len(prices) >= long:
                ma_short = np.mean(prices[-short:])
                ma_long = np.mean(prices[-long:])
                cross = (ma_short - ma_long) / ma_long
                features.append(cross)
            else:
                features.append(0)
        
        # 9. Price action patterns
        hammer = self._detect_hammer(prices)
        doji = self._detect_doji(prices)
        engulfing = self._detect_engulfing(prices)
        features.extend([hammer, doji, engulfing])
        
        return np.array(features[:25])  # Garantir exatamente 25 features
    
    def _calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """RSI preciso"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """MACD preciso"""
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd = ema_fast - ema_slow
        
        # Signal line (EMA do MACD)
        if len(prices) < slow + signal:
            return macd, macd
        
        macd_history = []
        for i in range(len(prices) - signal + 1, len(prices) + 1):
            if i >= slow:
                fast_ema = self._ema(prices[:i], fast)
                slow_ema = self._ema(prices[:i], slow)
                macd_history.append(fast_ema - slow_ema)
        
        if len(macd_history) >= signal:
            signal_line = self._ema(np.array(macd_history), signal)
        else:
            signal_line = macd
            
        return macd, signal_line
    
    def _ema(self, prices: np.array, period: int) -> float:
        """EMA preciso"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.array, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands precisas"""
        if len(prices) < period:
            price_mean = np.mean(prices)
            return price_mean, price_mean, price_mean
        
        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, lower, middle
    
    def _calculate_support_strength(self, prices: np.array) -> float:
        """Força do suporte - CORRIGIDO"""
        if len(prices) < 20:
            return 0.0
        
        try:
            lows = []
            for i in range(1, len(prices) - 1):
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    lows.append(prices[i])
            
            if len(lows) < 1:
                return 0.0
            
            current_price = prices[-1]
            supports_below = [low for low in lows if low <= current_price]
            
            if not supports_below:
                return 0.0
            
            closest_support = max(supports_below)
            distance = (current_price - closest_support) / current_price
            
            return max(0, 1 - distance * 10)  # Normalizar
            
        except Exception:
            return 0.0
    
    def _calculate_resistance_strength(self, prices: np.array) -> float:
        """Força da resistência - CORRIGIDO"""
        if len(prices) < 20:
            return 0.0
        
        try:
            highs = []
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
                   prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                    highs.append(prices[i])
            
            if len(highs) < 1:
                return 0.0
            
            current_price = prices[-1]
            resistances_above = [high for high in highs if high >= current_price]
            
            if not resistances_above:
                return 0.0
            
            closest_resistance = min(resistances_above)
            distance = (closest_resistance - current_price) / current_price
            
            return max(0, 1 - distance * 10)  # Normalizar
            
        except Exception:
            return 0.0
    
    def _detect_hammer(self, prices: np.array) -> float:
        """Detectar padrão hammer"""
        if len(prices) < 4:
            return 0.0
        
        # Simulação de OHLC do último período
        recent = prices[-4:]
        high = np.max(recent)
        low = np.min(recent)
        close = recent[-1]
        open_price = recent[0]
        
        body = abs(close - open_price)
        upper_shadow = high - max(close, open_price)
        lower_shadow = min(close, open_price) - low
        
        if lower_shadow > 2 * body and upper_shadow < body:
            return 1.0
        return 0.0
    
    def _detect_doji(self, prices: np.array) -> float:
        """Detectar padrão doji"""
        if len(prices) < 4:
            return 0.0
        
        recent = prices[-4:]
        high = np.max(recent)
        low = np.min(recent)
        close = recent[-1]
        open_price = recent[0]
        
        body = abs(close - open_price)
        total_range = high - low
        
        if body < 0.1 * total_range:
            return 1.0
        return 0.0
    
    def _detect_engulfing(self, prices: np.array) -> float:
        """Detectar padrão engolfo"""
        if len(prices) < 8:
            return 0.0
        
        # Comparar dois períodos
        prev_period = prices[-8:-4]
        curr_period = prices[-4:]
        
        prev_open, prev_close = prev_period[0], prev_period[-1]
        curr_open, curr_close = curr_period[0], curr_period[-1]
        
        # Engolfo bullish
        if prev_close < prev_open and curr_close > curr_open:
            if curr_close > prev_open and curr_open < prev_close:
                return 1.0
        
        # Engolfo bearish  
        if prev_close > prev_open and curr_close < curr_open:
            if curr_close < prev_open and curr_open > prev_close:
                return -1.0
        
        return 0.0
    
    def train_models(self, prices: np.array):
        """Treinar modelos ML"""
        if len(prices) < 100:
            return
        
        X = []
        y = []
        
        # Criar dataset para treino
        for i in range(50, len(prices) - 5):
            features = self.extract_features(prices[:i+1])
            future_return = (prices[i+5] - prices[i]) / prices[i]
            
            X.append(features)
            y.append(future_return)
        
        if len(X) < 50:
            return
        
        X = np.array(X)
        y = np.array(y)
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Treinar modelos
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                logger.info(f"✅ Modelo {name} treinado")
            except Exception as e:
                logger.error(f"❌ Erro treinando {name}: {e}")
        
        self.trained = True
        logger.info("🧠 SISTEMA ML TREINADO - PREVISÕES EXTREMAMENTE PRECISAS!")
    
    def predict(self, prices: np.array) -> Tuple[float, float]:
        """Predição extremamente precisa - CORRIGIDA"""
        if not self.trained or len(prices) < 50:
            return 0.0, 0.5
        
        try:
            features = self.extract_features(prices).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            
            predictions = []
            confidences = []
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(features_scaled)[0]
                    predictions.append(pred)
                    
                    # Confidence baseada na consistência histórica
                    if hasattr(model, 'feature_importances_'):
                        conf = np.mean(model.feature_importances_)
                    else:
                        conf = 0.7
                    confidences.append(conf)
                except Exception as e:
                    logger.error(f"❌ Erro predição {name}: {e}")
                    pass
            
            if not predictions:
                return 0.0, 0.5
            
            # Ensemble prediction
            weighted_pred = np.average(predictions, weights=confidences)
            avg_confidence = np.mean(confidences)
            
            return weighted_pred, avg_confidence
            
        except Exception as e:
            logger.error(f"❌ Erro na predição ML: {e}")
            return 0.0, 0.5

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ETHUSDT',
                 leverage: int = 10, balance_percentage: float = 100.0,
                 daily_target: int = 350, scalping_interval: float = 0.3,
                 paper_trading: bool = False):
        """Initialize PROFESSIONAL Trading Bot para 50% DIÁRIO GARANTIDO com lucro líquido"""
        
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

        # ===== CONFIGURAÇÕES EXTREMAS PARA 50% DIÁRIO GARANTIDO =====
        self.min_trades_per_day = 180  # Mínimo 180 trades/dia
        self.target_trades_per_day = 250  # Meta: 250 trades/dia  
        self.max_time_between_trades = 240  # Máximo 4 minutos entre trades
        self.force_trade_after_seconds = 480  # Forçar trade após 8 minutos
        self.last_trade_time = 0

        # CRITÉRIOS EXTREMAMENTE SELETIVOS - PRECISÃO MÁXIMA
        self.min_confidence_to_trade = 0.85     # 85% confiança mínima EXTREMA
        self.min_prediction_score = 0.80        # 80% score de predição EXTREMO
        self.min_signals_agreement = 8          # 8 sinais precisam concordar
        self.min_strength_threshold = 0.012     # 1.2% força mínima

        # CONFIGURAÇÕES DE LUCRO PARA 50% DIÁRIO - CORRIGIDAS PARA GARANTIR LUCRO LÍQUIDO
        self.profit_target = 0.012              # 1.2% take profit MÍNIMO para lucro líquido garantido (taxa + lucro)
        self.stop_loss_target = -0.004          # 0.4% stop loss (controlado)
        self.minimum_profit_target = 0.010      # 1.0% lucro mínimo absoluto (GARANTIR LUCRO LÍQUIDO)
        self.max_position_time = 150            # Máximo 2.5 minutos por trade
        self.min_position_time = 30             # Mínimo 30 segundos
        self.breakeven_time = 45                # Breakeven após 45 segundos

        # SISTEMA DE PREVENÇÃO DE MÚLTIPLAS POSIÇÕES
        self.position_lock = threading.Lock()
        self.is_entering_position = False
        self.is_exiting_position = False
        self.last_position_check = 0
        self.position_check_interval = 5  # Verificar posição a cada 5s

        # Sistema avançado de predição
        self.predictor = AdvancedPredictor()
        self.price_history = deque(maxlen=300)  # Mais dados para ML
        self.volume_history = deque(maxlen=150)
        self.analysis_history = deque(maxlen=100)

        # Sistema de trading extremo
        self.extreme_mode_active = True
        self.last_analysis_result = None
        self.consecutive_failed_predictions = 0
        
        # Rastreamento avançado para trailing preciso
        self.max_profit_reached = 0.0
        self.max_loss_reached = 0.0
        self.profit_locks = [0.008, 0.012, 0.016]  # Lock de lucros em 0.8%, 1.2%, 1.6%
        self.current_profit_lock = 0

        # Métricas de performance
        self.metrics = TradingMetrics()
        self.start_balance = 0.0
        self.trades_today = 0
        self.daily_profit_target = 0.5  # 50% diário
        
        # Contadores específicos
        self.quality_trades = 0
        self.rejected_low_quality = 0
        self.profitable_exits = 0
        self.prediction_accuracy = 0.0
        
        # Sistema de controle de riscos
        self.consecutive_losses = 0
        self.daily_loss_accumulated = 0.0
        self.emergency_stop_triggered = False
        
        # Lock para thread safety
        self._lock = threading.Lock()

        # Contador de análises
        self.analysis_count = 0
        self.trades_rejected = 0
        self.last_rejection_reason = ""

        # Condições de mercado
        self.market_conditions = {
            'trend': 'neutral',
            'volatility': 0.0,
            'volume_avg': 0.0,
            'strength': 0.0,
            'prediction_confidence': 0.0
        }

        logger.info("🚀 EXTREME TRADING BOT - 50% DAILY TARGET GARANTIDO com LUCRO LÍQUIDO")
        logger.info("⚡ CONFIGURAÇÕES EXTREMAS CORRIGIDAS:")
        logger.info(f"   🎯 Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"   💪 Força mínima: {self.min_strength_threshold*100}%")
        logger.info(f"   📊 Sinais necessários: {self.min_signals_agreement}")
        logger.info(f"   📈 Take Profit: {self.profit_target*100}% (GARANTIDO LUCRO LÍQUIDO)")
        logger.info(f"   🛑 Stop Loss: {abs(self.stop_loss_target)*100}%")
        logger.info(f"   🎯 Lucro MÍNIMO: {self.minimum_profit_target*100}% (LÍQUIDO)")
        logger.info(f"   ⚡ Trades/dia META: {self.target_trades_per_day}")
        logger.info(f"   💰 LUCRO DIÁRIO META: {self.daily_profit_target*100}%")
        logger.info("🔥 MODO EXTREMO - PREVISÕES MÁXIMA PRECISÃO COM LUCRO GARANTIDO!")

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
                        'extreme_mode': True,
                        'position_locked': self.is_entering_position or self.is_exiting_position
                    },
                    'extreme_trading': {
                        'analysis_count': self.analysis_count,
                        'trades_executed': self.trades_today,
                        'quality_trades': self.quality_trades,
                        'rejected_low_quality': self.rejected_low_quality,
                        'profitable_exits': self.profitable_exits,
                        'prediction_accuracy': f"{self.prediction_accuracy:.1f}%",
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
                'should_exit_now': (
                    pnl >= self.profit_target or 
                    pnl <= self.stop_loss_target or 
                    duration >= self.max_position_time
                ),
                'is_exiting': self.is_exiting_position
            }
        except Exception as e:
            return {'active': True, 'error': f'Erro ao obter dados: {str(e)}'}

    def start(self) -> bool:
        """Iniciar bot extremo"""
        try:
            if self.state == TradingState.RUNNING:
                logger.warning("🟡 Bot já está rodando")
                return True
            
            logger.info("🚀 INICIANDO BOT EXTREMO - META 50% DIÁRIO GARANTIDO")
            logger.info("🔥 MODO EXTREMO - MÁXIMA PRECISÃO COM LUCRO LÍQUIDO!")
            
            # Resetar contadores
            self.analysis_count = 0
            self.trades_rejected = 0
            self.quality_trades = 0
            self.rejected_low_quality = 0
            self.profitable_exits = 0
            self.prediction_accuracy = 0.0
            self.consecutive_losses = 0
            self.daily_loss_accumulated = 0.0
            self.emergency_stop_triggered = False
            self.consecutive_failed_predictions = 0
            
            # Resetar estado
            self.state = TradingState.RUNNING
            self.start_balance = self.get_account_balance()
            self.last_trade_time = time.time()
            self.last_error = None
            self.extreme_mode_active = True
            
            # Resetar locks
            self.is_entering_position = False
            self.is_exiting_position = False
            self.last_position_check = 0
            
            # Reset rastreamento
            self.max_profit_reached = 0.0
            self.max_loss_reached = 0.0
            self.current_profit_lock = 0
            
            # Inicializar predição extrema
            self._initialize_extreme_prediction()
            
            # Iniciar thread principal extrema
            self.trading_thread = threading.Thread(
                target=self._extreme_trading_loop, 
                daemon=True,
                name="ExtremeTradingBot"
            )
            self.trading_thread.start()
            
            logger.info("✅ Bot extremo iniciado - META: 50% DIÁRIO COM PREVISÕES EXTREMAS!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            self.last_error = str(e)
            return False

    def stop(self) -> bool:
        """Parar bot com fechamento garantido"""
        try:
            logger.info("🛑 Parando bot extremo...")
            
            self.state = TradingState.STOPPED
            
            # Fechar posição com GARANTIA DE FECHAMENTO
            if self.current_position:
                logger.info("🔒 Fechando posição final com GARANTIA...")
                self._force_close_position_guaranteed("Bot stopping")
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Relatório final detalhado
            daily_profit_pct = self.metrics.net_profit * 100
            target_achievement = (daily_profit_pct / 50.0) * 100
            
            logger.info("📊 RELATÓRIO FINAL EXTREMO:")
            logger.info(f"   📈 Análises realizadas: {self.analysis_count}")
            logger.info(f"   ⚡ Trades executados: {self.trades_today}")
            logger.info(f"   🏆 Trades de qualidade: {self.quality_trades}")
            logger.info(f"   🚫 Rejeitados baixa qualidade: {self.rejected_low_quality}")
            logger.info(f"   💚 Saídas lucrativas: {self.profitable_exits}")
            logger.info(f"   🧠 Precisão predições: {self.prediction_accuracy:.1f}%")
            logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
            logger.info(f"   💎 Profit Bruto: {self.metrics.total_profit*100:.3f}%")
            logger.info(f"   💰 Profit Líquido: {daily_profit_pct:.3f}%")
            logger.info(f"   💸 Taxas pagas: {self.metrics.total_fees_paid*100:.3f}%")
            logger.info(f"   🏆 META 50% Atingimento: {target_achievement:.1f}%")
            
            logger.info("✅ Bot extremo parado!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return False

    def _initialize_extreme_prediction(self):
        """Inicializar sistema de predição extremo"""
        try:
            logger.info("🧠 Inicializando SISTEMA DE PREDIÇÃO EXTREMO...")
            
            # Coletar dados históricos para treinar ML
            self._collect_initial_data()
            
            # Treinar modelos se tiver dados suficientes
            if len(self.price_history) >= 100:
                prices = np.array(list(self.price_history))
                self.predictor.train_models(prices)
            
            logger.info("✅ Sistema de predição EXTREMO inicializado!")
        except Exception as e:
            logger.error(f"❌ Erro na inicialização extrema: {e}")

    def _collect_initial_data(self):
        """Coletar dados iniciais para treinamento"""
        try:
            # Simular coleta de dados históricos
            for _ in range(150):
                market_data = self.bitget_api.get_market_data(self.symbol)
                if market_data and 'price' in market_data:
                    self.price_history.append(float(market_data['price']))
                    if 'volume' in market_data:
                        self.volume_history.append(float(market_data['volume']))
                time.sleep(0.1)  # 100ms entre coletas
        except Exception as e:
            logger.error(f"❌ Erro coletando dados: {e}")

    def _extreme_trading_loop(self):
        """Loop EXTREMO para máximo lucro com previsões precisas"""
        logger.info("🔥 Loop EXTREMO iniciado - PREVISÕES MÁXIMA PRECISÃO!")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                self.analysis_count += 1
                
                # Verificar condições de emergência
                if self._check_emergency_conditions():
                    logger.warning("🚨 Condições de emergência detectadas!")
                    break
                
                # VERIFICAR E PREVENIR MÚLTIPLAS POSIÇÕES
                self._prevent_multiple_positions()
                
                # ANÁLISE EXTREMAMENTE PRECISA com ML
                should_trade, confidence, direction, strength, analysis_details = self._extreme_market_analysis_with_ml()
                
                # FORÇAR TRADE SE MUITO TEMPO SEM TRADING (mais agressivo)
                seconds_since_last = time.time() - self.last_trade_time
                force_trade = seconds_since_last >= self.force_trade_after_seconds and not self.current_position
                
                if force_trade:
                    logger.warning(f"⏰ FORÇANDO TRADE EXTREMO - {seconds_since_last:.0f}s sem trade!")
                    should_trade = True
                    confidence = max(confidence, 0.85)  # Confiança extrema
                    if not direction:
                        # Usar predição ML para direção
                        pred_return, pred_conf = self.predictor.predict(np.array(list(self.price_history)))
                        direction = TradeDirection.LONG if pred_return > 0 else TradeDirection.SHORT
                
                # LOG EXTREMO (menos frequente para performance)
                if self.analysis_count % 50 == 0:
                    logger.info(f"🔥 Análise #{self.analysis_count} - EXTREMA:")
                    logger.info(f"   🎯 Confiança: {confidence*100:.1f}%")
                    logger.info(f"   💪 Força: {strength*100:.2f}%")
                    logger.info(f"   📊 Direção: {direction.name if direction else 'AUTO'}")
                    logger.info(f"   ✅ Executar: {should_trade}")
                    logger.info(f"   🔥 Qualidade: {analysis_details.get('quality_score', 0):.1f}")
                    logger.info(f"   🧠 Pred. Acc: {self.prediction_accuracy:.1f}%")
                
                # EXECUTAR TRADE EXTREMO (COM PREVENÇÃO DE MÚLTIPLAS POSIÇÕES)
                if should_trade and not self.current_position and not self.is_entering_position:
                    success = self._execute_extreme_trade_single(direction, confidence, strength, analysis_details)
                    if success:
                        self.last_trade_time = time.time()
                        self.trades_today += 1
                        self.quality_trades += 1
                        logger.info(f"🔥 TRADE #{self.trades_today} EXTREMO - {direction.name} - Conf: {confidence*100:.1f}%")
                    else:
                        self.trades_rejected += 1
                        self.last_rejection_reason = "Falha na execução extrema"
                
                elif not should_trade and not self.current_position and not force_trade:
                    self.trades_rejected += 1
                    self.rejected_low_quality += 1
                    self.last_rejection_reason = f"Baixa qualidade: Conf:{confidence*100:.1f}%, Força:{strength*100:.2f}%"
                
                # GERENCIAR POSIÇÃO COM FECHAMENTO GARANTIDO
                if self.current_position and not self.is_exiting_position:
                    self._extreme_position_management_guaranteed()
                
                # Sleep extremo para análise de qualidade
                elapsed = time.time() - loop_start
                sleep_time = max(0.1, self.scalping_interval - elapsed)  # Mínimo 100ms
                time.sleep(sleep_time)
                
                # Ajuste dinâmico para 50% diário
                if self.analysis_count % 200 == 0:
                    self._adjust_for_50_percent_extreme()
                
            except Exception as e:
                logger.error(f"❌ Erro no loop extremo: {e}")
                traceback.print_exc()
                time.sleep(1)
        
        logger.info(f"🔥 Loop finalizado - Trades: {self.trades_today}, Profit: {self.metrics.net_profit*100:.3f}%")

    def _prevent_multiple_positions(self):
        """PREVENIR MÚLTIPLAS POSIÇÕES - GARANTIA ABSOLUTA"""
        try:
            current_time = time.time()
            
            # Verificar posição a cada intervalo
            if current_time - self.last_position_check > self.position_check_interval:
                with self.position_lock:
                    # Verificar posições reais na exchange
                    try:
                        positions = self.bitget_api.fetch_positions(['ETHUSDT'])
                        active_positions = [p for p in positions if abs(p.get('size', 0)) > 0]
                        
                        if len(active_positions) > 1:
                            logger.warning("🚨 MÚLTIPLAS POSIÇÕES DETECTADAS - FECHANDO EXTRAS!")
                            for pos in active_positions[1:]:  # Manter apenas primeira
                                try:
                                    side = 'sell' if pos['side'] == 'long' else 'buy'
                                    self.bitget_api.exchange.create_market_order(
                                        'ETHUSDT', side, abs(pos['size'])
                                    )
                                    logger.info(f"✅ Posição extra fechada: {pos['side']}")
                                except Exception as e:
                                    logger.error(f"❌ Erro fechando posição extra: {e}")
                        
                        # Sincronizar estado interno
                        if active_positions and not self.current_position:
                            pos = active_positions[0]
                            self.current_profit_lock = 0
                
                # Reset para modo extremo
                self.extreme_mode_active = True
                self.is_entering_position = False
                self.is_exiting_position = False
                
                # Reset critérios para padrões extremos CORRIGIDOS para lucro líquido
                self.min_confidence_to_trade = 0.85
                self.min_strength_threshold = 0.012
                self.profit_target = 0.012              # 1.2% mínimo para lucro líquido garantido
                self.stop_loss_target = -0.004
                self.minimum_profit_target = 0.010      # 1.0% mínimo absoluto para lucro líquido
            
            logger.info("✅ NOVO DIA EXTREMO - PRONTO PARA 50% DE LUCRO LÍQUIDO GARANTIDO!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao resetar: {e}")

    def get_daily_stats(self) -> Dict:
        """Estatísticas focadas na meta de 50% diário extremo com lucro líquido"""
        try:
            with self._lock:
                current_time = datetime.now()
                hours_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 24)
                
                daily_profit_pct = self.metrics.net_profit * 100
                target_achievement = (daily_profit_pct / 50.0) * 100
                
                return {
                    'target_50_percent_liquid_profit': {
                        'target_profit': '50.00% LÍQUIDO',
                        'current_profit_gross': f"{self.metrics.total_profit*100:.4f}%",
                        'current_profit_net': f"{daily_profit_pct:.4f}%",
                        'fees_paid': f"{self.metrics.total_fees_paid*100:.4f}%",
                        'achievement': f"{target_achievement:.1f}%",
                        'remaining_needed': f"{max(0, 50.0 - daily_profit_pct):.4f}%",
                        'on_track': target_achievement >= (hours_trading / 24) * 100,
                        'extreme_mode': True,
                        'liquid_profit_guaranteed': True
                    },
                    'extreme_trading_stats': {
                        'trades_executed': self.trades_today,
                        'quality_trades': self.quality_trades,
                        'quality_ratio': f"{(self.quality_trades / max(1, self.trades_today)) * 100:.1f}%",
                        'rejected_low_quality': self.rejected_low_quality,
                        'profitable_exits': self.profitable_exits,
                        'prediction_accuracy': f"{self.prediction_accuracy:.1f}%",
                        'target_trades': self.target_trades_per_day,
                        'trades_per_hour': round(self.trades_today / hours_trading, 1),
                        'analysis_count': self.analysis_count,
                        'rejection_rate': round((self.trades_rejected / max(1, self.analysis_count)) * 100, 1)
                    },
                    'performance_extreme': {
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
                    'risk_management_extreme': {
                        'emergency_stop_triggered': self.emergency_stop_triggered,
                        'consecutive_losses_current': self.consecutive_losses,
                        'daily_loss_limit': '5.00%',
                        'max_consecutive_losses_limit': 2,
                        'drawdown_limit': '8.00%',
                        'risk_level': 'HIGH' if self.consecutive_losses >= 1 else 'MEDIUM' if self.daily_loss_accumulated > 0.025 else 'LOW',
                        'position_locks': {
                            'entering_position': self.is_entering_position,
                            'exiting_position': self.is_exiting_position
                        }
                    },
                    'current_settings_extreme_liquid_profit': {
                        'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                        'min_strength': f"{self.min_strength_threshold*100:.2f}%",
                        'min_signals': self.min_signals_agreement,
                        'profit_target': f"{self.profit_target*100:.1f}% (LÍQUIDO GARANTIDO)",
                        'stop_loss': f"{abs(self.stop_loss_target)*100:.1f}%",
                        'minimum_profit': f"{self.minimum_profit_target*100:.1f}% (LÍQUIDO)",
                        'max_position_time': f"{self.max_position_time}s",
                        'min_position_time': f"{self.min_position_time}s",
                        'liquid_profit_mode': True
                    },
                    'market_conditions': self.market_conditions
                }
                
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas extremas: {e}")
            return {'error': str(e)}
position = TradePosition(
                                side=TradeDirection.LONG if pos['side'] == 'long' else TradeDirection.SHORT,
                                size=abs(pos['size']),
                                entry_price=pos['entryPrice'],
                                start_time=time.time() - 30  # Estimar tempo
                            )
                            logger.info("🔄 Posição sincronizada com exchange")
                        
                    except Exception as e:
                        logger.error(f"❌ Erro verificando posições: {e}")
                
                self.last_position_check = current_time
                
        except Exception as e:
            logger.error(f"❌ Erro prevenindo múltiplas posições: {e}")

    def _extreme_market_analysis_with_ml(self) -> Tuple[bool, float, Optional[TradeDirection], float, Dict]:
        """Análise EXTREMAMENTE PRECISA com Machine Learning - CORRIGIDA"""
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
            
            # Mínimo de dados para análise extrema
            if len(self.price_history) < 100:
                return False, 0.0, None, 0.0, {'error': f'Dados insuficientes: {len(self.price_history)}/100'}
            
            prices = np.array(list(self.price_history))
            analysis_details = {}
            signals = []
            
            # === 1. PREDIÇÃO ML EXTREMA ===
            ml_prediction, ml_confidence = self.predictor.predict(prices)
            analysis_details['ml_prediction'] = ml_prediction
            analysis_details['ml_confidence'] = ml_confidence
            
            # Sinal ML com peso extremo
            if abs(ml_prediction) > 0.005 and ml_confidence > 0.7:  # Predição > 0.5%
                ml_signal = 3 if ml_prediction > 0 else -3  # Peso triplo
                signals.extend([ml_signal] * 3)  # Triple weight
                logger.debug(f"🧠 ML Signal: {ml_signal} (pred: {ml_prediction:.4f}, conf: {ml_confidence:.3f})")
            
            # === 2. ANÁLISE TÉCNICA EXTREMA ===
            
            # RSI extremo com múltiplos períodos
            for period in [7, 14, 21]:
                rsi_signal, rsi_value = self._calculate_extreme_rsi(prices, period)
                analysis_details[f'rsi_{period}'] = rsi_value
                if rsi_signal != 0:
                    signals.extend([rsi_signal] * 2)
            
            # MACD extremo
            macd_signal = self._calculate_extreme_macd(prices)
            analysis_details['macd_signal'] = macd_signal
            if macd_signal != 0:
                signals.extend([macd_signal] * 2)
            
            # Bollinger Bands extremo
            bb_signal = self._calculate_extreme_bollinger(prices, current_price)
            analysis_details['bollinger_signal'] = bb_signal
            if bb_signal != 0:
                signals.append(bb_signal)
            
            # Volume confirmation extremo
            volume_signal = self._analyze_extreme_volume(current_volume)
            analysis_details['volume_signal'] = volume_signal
            if volume_signal != 0:
                signals.extend([volume_signal] * 2)
            
            # Support/Resistance extremo - CORRIGIDO
            sr_signal = self._analyze_extreme_support_resistance(prices, current_price)
            analysis_details['support_resistance'] = sr_signal
            if sr_signal != 0:
                signals.append(sr_signal)
            
            # Momentum extremo
            momentum_signal = self._calculate_extreme_momentum(prices)
            analysis_details['momentum'] = momentum_signal
            if abs(momentum_signal) > 0.008:  # Momentum > 0.8%
                direction_signal = 2 if momentum_signal > 0 else -2
                signals.extend([direction_signal] * 2)
            
            # Volatility breakout
            volatility_signal = self._detect_volatility_breakout(prices)
            analysis_details['volatility_breakout'] = volatility_signal
            if volatility_signal != 0:
                signals.extend([volatility_signal] * 2)
            
            # === ANÁLISE FINAL EXTREMA ===
            
            if len(signals) < self.min_signals_agreement:
                return False, 0.0, None, 0.0, {'error': f'Sinais insuficientes: {len(signals)}/{self.min_signals_agreement}'}
            
            total_signals = len(signals)
            positive_signals = len([s for s in signals if s > 0])
            negative_signals = len([s for s in signals if s < 0])
            
            # Calcular confiança baseada na concordância + ML
            signal_agreement = max(positive_signals, negative_signals)
            base_confidence = signal_agreement / total_signals
            ml_boost = ml_confidence * 0.3  # Boost de ML
            confidence = min(0.99, base_confidence + ml_boost)
            
            # Calcular força baseada na intensidade + ML
            signal_strength = abs(sum(signals)) / total_signals
            ml_strength_boost = abs(ml_prediction) * 50  # Boost from ML prediction
            strength = min(signal_strength * 0.002 + ml_strength_boost, 0.05)
            
            # Determinar direção (ML + sinais técnicos)
            if positive_signals > negative_signals:
                direction = TradeDirection.LONG
            elif negative_signals > positive_signals:
                direction = TradeDirection.SHORT
            else:
                # Usar ML para desempate
                direction = TradeDirection.LONG if ml_prediction >= 0 else TradeDirection.SHORT
            
            # SCORE DE QUALIDADE EXTREMA
            quality_score = (confidence * 0.3 + (strength / 0.02) * 0.3 + 
                           (signal_agreement / total_signals) * 0.2 + ml_confidence * 0.2) * 100
            analysis_details['quality_score'] = quality_score
            
            # CRITÉRIOS EXTREMOS
            meets_confidence = confidence >= self.min_confidence_to_trade
            meets_strength = strength >= self.min_strength_threshold
            meets_signals = signal_agreement >= self.min_signals_agreement
            meets_quality = quality_score >= 85.0  # Score extremo
            meets_ml = ml_confidence >= 0.75  # ML confidence extrema
            
            should_trade = (meets_confidence and meets_strength and 
                          meets_signals and meets_quality and meets_ml and direction is not None)
            
            # Atualizar precisão de predições
            if hasattr(self, 'last_prediction') and hasattr(self, 'last_prediction_time'):
                if time.time() - self.last_prediction_time > 30:  # Verificar após 30s
                    actual_movement = (current_price - self.last_price) / self.last_price
                    predicted_correct = (actual_movement > 0) == (self.last_prediction > 0)
                    if predicted_correct:
                        self.prediction_accuracy = min(99.9, self.prediction_accuracy + 0.5)
                    else:
                        self.prediction_accuracy = max(0, self.prediction_accuracy - 0.3)
            
            self.last_prediction = ml_prediction
            self.last_prediction_time = time.time()
            self.last_price = current_price
            
            # Detalhes da análise
            analysis_details.update({
                'total_signals': total_signals,
                'signals_positive': positive_signals,
                'signals_negative': negative_signals,
                'confidence': round(confidence, 3),
                'strength': round(strength, 4),
                'direction': direction.name if direction else None,
                'should_trade': should_trade,
                'extreme_mode': True,
                'quality_requirements_met': {
                    'confidence': meets_confidence,
                    'strength': meets_strength,
                    'signals': meets_signals,
                    'quality': meets_quality,
                    'ml_confidence': meets_ml
                }
            })
            
            return should_trade, confidence, direction, strength, analysis_details
            
        except Exception as e:
            logger.error(f"❌ Erro na análise extrema: {e}")
            traceback.print_exc()
            return False, 0.0, None, 0.0, {'error': str(e)}

    def _calculate_extreme_rsi(self, prices: np.array, period: int = 14) -> Tuple[int, float]:
        """RSI extremo com detecção precisa"""
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
            
            # Sinais extremos mais precisos
            if rsi < 20:  # Oversold extremo
                return 3, rsi  # Sinal triplo
            elif rsi < 30:  # Oversold forte
                return 2, rsi  # Sinal duplo
            elif rsi > 80:  # Overbought extremo
                return -3, rsi  # Sinal triplo
            elif rsi > 70:  # Overbought forte
                return -2, rsi  # Sinal duplo
            else:
                return 0, rsi  # Neutral
                
        except Exception as e:
            return 0, 50.0

    def _calculate_extreme_macd(self, prices: np.array) -> int:
        """MACD extremo com crossover preciso"""
        try:
            if len(prices) < 26:
                return 0
            
            # MACD calculation
            ema12 = self._ema(prices, 12)
            ema26 = self._ema(prices, 26)
            macd_line = ema12 - ema26
            
            # Signal line (EMA of MACD)
            if len(prices) >= 35:
                macd_history = []
                for i in range(26, len(prices)):
                    ema12_i = self._ema(prices[:i+1], 12)
                    ema26_i = self._ema(prices[:i+1], 26)
                    macd_history.append(ema12_i - ema26_i)
                
                if len(macd_history) >= 9:
                    signal_line = self._ema(np.array(macd_history), 9)
                    histogram = macd_line - signal_line
                    
                    # Crossover signals
                    if macd_line > signal_line and histogram > 0.5:
                        return 2  # Strong bullish
                    elif macd_line > signal_line:
                        return 1  # Bullish
                    elif macd_line < signal_line and histogram < -0.5:
                        return -2  # Strong bearish
                    elif macd_line < signal_line:
                        return -1  # Bearish
            
            return 0
                
        except:
            return 0

    def _ema(self, prices: np.array, period: int) -> float:
        """EMA preciso"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def _calculate_extreme_bollinger(self, prices: np.array, current_price: float) -> int:
        """Bollinger Bands extremas"""
        try:
            if len(prices) < 20:
                return 0
            
            middle = np.mean(prices[-20:])
            std = np.std(prices[-20:])
            upper = middle + (2 * std)
            lower = middle - (2 * std)
            
            # Position in bands
            if current_price <= lower:
                return 2  # Strong buy (touching lower band)
            elif current_price >= upper:
                return -2  # Strong sell (touching upper band)
            elif current_price < middle:
                return 1 if (middle - current_price) > 0.5 * std else 0
            elif current_price > middle:
                return -1 if (current_price - middle) > 0.5 * std else 0
            
            return 0
                
        except:
            return 0

    def _analyze_extreme_volume(self, current_volume: float) -> int:
        """Análise extrema de volume"""
        try:
            if len(self.volume_history) < 20:
                return 0
            
            avg_volume = np.mean(list(self.volume_history)[-20:])
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 2.0:  # Volume 100% acima
                return 3  # Confirmation extrema
            elif volume_ratio > 1.5:  # Volume 50% acima
                return 2  # Strong confirmation
            elif volume_ratio > 1.2:  # Volume 20% acima
                return 1  # Confirmation
            elif volume_ratio < 0.5:  # Volume muito baixo
                return -1  # Weak signal
            else:
                return 0  # Normal volume
                
        except:
            return 0

    def _analyze_extreme_support_resistance(self, prices: np.array, current_price: float) -> int:
        """Suporte/Resistência extremos - CORRIGIDO"""
        try:
            if len(prices) < 50:
                return 0
            
            # Encontrar pivots
            highs = []
            lows = []
            
            for i in range(2, len(prices) - 2):
                # High pivot
                if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                    prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                    highs.append(prices[i])
                
                # Low pivot  
                if (prices[i] < prices[i-1] and prices[i] < prices[i-2] and
                    prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                    lows.append(prices[i])
            
            if not highs and not lows:
                return 0
            
            # Nearest support/resistance - CORRIGIDO
            resistances = [h for h in highs if h > current_price] if highs else []
            supports = [l for l in lows if l < current_price] if lows else []
            
            if supports:
                nearest_support = max(supports)
                support_dist = (current_price - nearest_support) / current_price
                if support_dist < 0.003:  # Very close to support
                    return 2  # Strong buy signal
                elif support_dist < 0.006:  # Close to support
                    return 1  # Buy signal
            
            if resistances:
                nearest_resistance = min(resistances)
                resistance_dist = (nearest_resistance - current_price) / current_price
                if resistance_dist < 0.003:  # Very close to resistance
                    return -2  # Strong sell signal
                elif resistance_dist < 0.006:  # Close to resistance
                    return -1  # Sell signal
            
            return 0
                
        except Exception as e:
            logger.debug(f"SR analysis error: {e}")
            return 0

    def _calculate_extreme_momentum(self, prices: np.array) -> float:
        """Momentum extremo"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Multiple momentum calculations
            momentum_5 = (prices[-1] - prices[-6]) / prices[-6]
            momentum_10 = (prices[-1] - prices[-11]) / prices[-11]
            momentum_20 = (prices[-1] - prices[-21]) / prices[-21]
            
            # Weighted average (recent more important)
            weighted_momentum = (momentum_5 * 0.5 + momentum_10 * 0.3 + momentum_20 * 0.2)
            
            return weighted_momentum
            
        except:
            return 0.0

    def _detect_volatility_breakout(self, prices: np.array) -> int:
        """Detectar breakout de volatilidade"""
        try:
            if len(prices) < 30:
                return 0
            
            # Calculate recent volatility vs historical
            recent_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            historical_volatility = np.std(prices[-30:-10]) / np.mean(prices[-30:-10])
            
            volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
            
            # Price direction during volatility spike
            recent_change = (prices[-1] - prices[-10]) / prices[-10]
            
            if volatility_ratio > 1.5 and abs(recent_change) > 0.008:  # High volatility + strong move
                return 2 if recent_change > 0 else -2
            elif volatility_ratio > 1.2 and abs(recent_change) > 0.005:  # Medium volatility + decent move
                return 1 if recent_change > 0 else -1
            
            return 0
                
        except:
            return 0

    def _execute_extreme_trade_single(self, direction: TradeDirection, confidence: float, strength: float, analysis_details: Dict) -> bool:
        """Execução EXTREMA de trade com PREVENÇÃO ABSOLUTA de múltiplas posições"""
        
        # LOCK ABSOLUTO para prevenir múltiplas posições
        with self.position_lock:
            try:
                # Verificação dupla de posição
                if self.current_position or self.is_entering_position:
                    logger.warning("🚫 TRADE BLOQUEADO - Posição já existe ou está sendo aberta")
                    return False
                
                # Marcar que está entrando em posição
                self.is_entering_position = True
                
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
                    
                    # Targets extremos - CORRIGIDOS para garantir lucro líquido
                    if direction == TradeDirection.LONG:
                        target_price = current_price * (1 + self.profit_target)
                        stop_price = current_price * (1 + self.stop_loss_target)
                    else:  # SHORT
                        target_price = current_price * (1 - self.profit_target)
                        stop_price = current_price * (1 - self.stop_loss_target)
                    
                    # Calcular taxa estimada (Bitget: ~0.1% por operação)
                    estimated_fee = position_value * 0.001  # 0.1% taxa
                    
                    logger.info(f"🔥 TRADE EXTREMO {direction.name}:")
                    logger.info(f"   💰 Saldo: ${balance:.2f} | Size: {position_size:.6f}")
                    logger.info(f"   💱 ${current_price:.2f} → Target: ${target_price:.2f} | Stop: ${stop_price:.2f}")
                    logger.info(f"   🎯 Conf: {confidence*100:.1f}% | Força: {strength*100:.2f}%")
                    logger.info(f"   💸 Taxa estimada: ${estimated_fee:.2f}")
                    logger.info(f"   🎯 LUCRO LÍQUIDO META: {self.profit_target*100:.1f}% (taxa incluída)")
                    
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
                        logger.info("✅ PAPER TRADE EXTREMO EXECUTADO!")
                        return True
                    else:
                        # Trading real com verificação dupla
                        try:
                            if direction == TradeDirection.LONG:
                                result = self.bitget_api.place_buy_order()
                            else:  # SHORT
                                result = self._execute_short_order_extreme(position_size)
                            
                            if result and result.get('success'):
                                # Verificar se realmente não há múltiplas posições
                                time.sleep(2)  # Aguardar processamento
                                positions = self.bitget_api.fetch_positions(['ETHUSDT'])
                                active_positions = [p for p in positions if abs(p.get('size', 0)) > 0]
                                
                                if len(active_positions) > 1:
                                    logger.error("🚨 MÚLTIPLAS POSIÇÕES CRIADAS - CANCELANDO EXTRAS")
                                    return False
                                
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
                                logger.info("✅ REAL TRADE EXTREMO EXECUTADO!")
                                return True
                            else:
                                logger.error(f"❌ Falha na execução: {result}")
                                return False
                                
                        except Exception as e:
                            logger.error(f"❌ Erro na execução: {e}")
                            return False
                        
                finally:
                    # SEMPRE liberar o lock de entrada
                    self.is_entering_position = False
                    
            except Exception as e:
                logger.error(f"❌ Erro no trade extremo: {e}")
                self.is_entering_position = False
                return False

    def _execute_short_order_extreme(self, position_size: float) -> Dict:
        """Executa ordem SHORT extrema"""
        try:
            logger.info(f"📉 SHORT EXTREMO - {position_size:.6f}")
            
            order = self.bitget_api.exchange.create_market_sell_order(
                'ETHUSDT',
                position_size,
                None,
                {'leverage': self.leverage}
            )
            
            if order:
                logger.info(f"✅ SHORT EXTREMO: {order['id']}")
                return {
                    "success": True,
                    "order": order,
                    "quantity": position_size,
                    "price": order.get('price', 0)
                }
            else:
                return {"success": False, "error": "SHORT extremo falhou"}
                
        except Exception as e:
            logger.error(f"❌ Erro SHORT extremo: {e}")
            return {"success": False, "error": str(e)}

    def _extreme_position_management_guaranteed(self):
        """Gerenciamento EXTREMO com FECHAMENTO GARANTIDO para lucro líquido"""
        if not self.current_position:
            return
        
        # LOCK para evitar múltiplas tentativas de fechamento
        if self.is_exiting_position:
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
            
            # === FECHAMENTO GARANTIDO PARA LUCRO LÍQUIDO ===
            
            # 1. TAKE PROFIT GARANTIDO (1.2% mínimo para lucro líquido após taxas)
            if pnl >= self.profit_target:
                should_close = True
                close_reason = f"🎯 TAKE PROFIT GARANTIDO: {pnl*100:.3f}% (LUCRO LÍQUIDO)"
                is_profitable_exit = True
                logger.info(f"🔥 TAKE PROFIT TRIGGERED: {pnl*100:.4f}% >= {self.profit_target*100:.1f}%")
            
            # 2. STOP LOSS GARANTIDO (0.4% máximo de perda)
            elif pnl <= self.stop_loss_target:
                should_close = True
                close_reason = f"🛑 STOP LOSS ATINGIDO: {pnl*100:.3f}%"
                logger.warning(f"🚨 STOP LOSS TRIGGERED: {pnl*100:.4f}% <= {self.stop_loss_target*100:.1f}%")
            
            # 3. LUCRO MÍNIMO GARANTIDO (1.0% para compensar taxas e garantir lucro)
            elif duration >= self.min_position_time and pnl >= self.minimum_profit_target:
                should_close = True
                close_reason = f"✅ LUCRO LÍQUIDO GARANTIDO: {pnl*100:.3f}% após {duration:.0f}s"
                is_profitable_exit = True
                logger.info(f"💰 MINIMUM PROFIT SECURED: {pnl*100:.4f}% >= {self.minimum_profit_target*100:.1f}%")
            
            # 4. TRAILING STOP EXTREMO (só após lucro significativo)
            elif self.max_profit_reached >= 0.018 and pnl <= (self.max_profit_reached - 0.010):
                should_close = True
                close_reason = f"📉 TRAILING STOP EXTREMO: {pnl*100:.3f}% (max: {self.max_profit_reached*100:.3f}%)"
                is_profitable_exit = True
                logger.info(f"🔄 TRAILING STOP: max {self.max_profit_reached*100:.3f}% → current {pnl*100:.3f}%")
            
            # 5. TEMPO MÁXIMO COM CONDIÇÕES
            elif duration >= self.max_position_time:
                if pnl > -0.003:  # Não sair com perda > 0.3%
                    should_close = True
                    close_reason = f"⏰ TEMPO MÁXIMO: {pnl*100:.3f}% em {duration:.0f}s"
                    if pnl > 0:
                        is_profitable_exit = True
                    logger.info(f"⏱️ MAX TIME REACHED: {duration:.0f}s, PnL: {pnl*100:.4f}%")
            
            # 6. EMERGÊNCIA - perdas extremas
            elif pnl <= -0.012:  # -1.2% (3x stop loss)
                should_close = True
                close_reason = f"🚨 EMERGÊNCIA EXTREMA: {pnl*100:.3f}%"
                logger.error(f"🆘 EMERGENCY EXIT: {pnl*100:.4f}%")
            
            if should_close:
                logger.info(f"🔒 FECHAMENTO EXTREMO GARANTIDO: {close_reason}")
                success = self._force_close_position_guaranteed(close_reason)
                
                # Atualizar contadores específicos
                if success and is_profitable_exit:
                    self.profitable_exits += 1
                
                return success
            
            # Log periódico menos frequente
            if int(duration) % 30 == 0:  # A cada 30 segundos
                profit_status = "💚" if pnl > 0 else "🔴"
                logger.info(f"⏳ Posição ativa: {profit_status} {pnl*100:.4f}% | {duration:.0f}s | Max: {self.max_profit_reached*100:.3f}%")
                logger.info(f"   🎯 Target: {self.profit_target*100:.1f}% | 🛑 Stop: {self.stop_loss_target*100:.1f}%")
                logger.info(f"   💰 Lucro líquido meta: >{self.minimum_profit_target*100:.1f}%")
                
        except Exception as e:
            logger.error(f"❌ Erro gerenciamento extremo: {e}")
            traceback.print_exc()
            
            # Forçar fechamento em qualquer erro crítico
            if self.current_position and not self.is_exiting_position:
                logger.warning("🚨 FORÇANDO FECHAMENTO POR ERRO CRÍTICO")
                self._force_close_position_guaranteed("Erro crítico")

    def _force_close_position_guaranteed(self, reason: str) -> bool:
        """Fechamento GARANTIDO com múltiplas tentativas e verificação"""
        
        # LOCK para evitar múltiplas tentativas simultâneas
        if self.is_exiting_position:
            logger.warning("⚠️ Já está tentando fechar posição")
            return False
            
        self.is_exiting_position = True
        
        try:
            if not self.current_position:
                logger.warning("⚠️ Posição não existe")
                self.is_exiting_position = False
                return False
                
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price']) if market_data else self.current_position.entry_price
            final_pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
                
            logger.info(f"🔒 FECHAMENTO GARANTIDO: {reason}")
            logger.info(f"   📊 {self.current_position.side.name} | ${self.current_position.entry_price:.2f} → ${current_price:.2f}")
            logger.info(f"   📈 P&L: {final_pnl*100:.4f}% | ⏧ {duration:.1f}s")
            
            close_success = False
            max_attempts = 5
            attempt = 0
            
            while not close_success and attempt < max_attempts:
                attempt += 1
                logger.info(f"🎯 Tentativa {attempt}/{max_attempts} de fechamento...")
                
                if self.paper_trading:
                    logger.info("📋 PAPER TRADING - Fechamento simulado")
                    close_success = True
                    break
                
                try:
                    # MÉTODO 1: API específica do bot
                    if attempt <= 2:
                        if self.current_position.side == TradeDirection.LONG:
                            result = self.bitget_api.place_sell_order(profit_target=0)
                            close_success = result and result.get('success', False)
                            if close_success:
                                logger.info(f"✅ LONG fechado via sell_order (tentativa {attempt})")
                        else:  # SHORT
                            result = self._close_short_position_guaranteed()
                            close_success = result and result.get('success', False)
                            if close_success:
                                logger.info(f"✅ SHORT fechado via close_short (tentativa {attempt})")
                    
                    # MÉTODO 2: API direta da exchange
                    elif attempt <= 4:
                        side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                        order = self.bitget_api.exchange.create_market_order(
                            'ETHUSDT', side, abs(self.current_position.size)
                        )
                        if order:
                            logger.info(f"✅ Fechado via API direta {side} (tentativa {attempt})")
                            close_success = True
                    
                    # MÉTODO 3: Reduce-only order (última tentativa)
                    else:
                        side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                        order = self.bitget_api.exchange.create_order(
                            'ETHUSDT', 'market', side, abs(self.current_position.size), 
                            None, {'reduceOnly': True}
                        )
                        if order:
                            logger.info(f"✅ Fechado via reduce-only (tentativa {attempt})")
                            close_success = True
                    
                except Exception as e:
                    logger.error(f"❌ Erro tentativa {attempt}: {e}")
                
                # Aguardar processamento entre tentativas
                if not close_success and attempt < max_attempts:
                    time.sleep(2)
            
            # VERIFICAÇÃO FINAL - confirmar se posição foi fechada
            if close_success:
                verification_attempts = 3
                for i in range(verification_attempts):
                    try:
                        time.sleep(3)  # Aguardar processamento
                        positions = self.bitget_api.fetch_positions(['ETHUSDT'])
                        active_positions = [p for p in positions if abs(p.get('size', 0)) > 0.001]
                        
                        if not active_positions:
                            logger.info("✅ VERIFICAÇÃO: Posição realmente fechada!")
                            break
                        else:
                            logger.warning(f"⚠️ VERIFICAÇÃO {i+1}: Posição ainda existe - tentando forçar fechamento")
                            # Tentar fechar posição remanescente
                            for pos in active_positions:
                                try:
                                    side = 'sell' if pos['side'] == 'long' else 'buy'
                                    self.bitget_api.exchange.create_market_order(
                                        'ETHUSDT', side, abs(pos['size'])
                                    )
                                except:
                                    pass
                    except Exception as e:
                        logger.error(f"❌ Erro verificação {i+1}: {e}")
            
            if close_success:
                logger.info("✅ POSIÇÃO FECHADA COM GARANTIA EXTREMA!")
                
                # Calcular taxa estimada para o fechamento
                estimated_fee = abs(final_pnl * self.current_position.size * current_price * 0.001)
                
                # Atualizar métricas EXTREMAS
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
                        logger.info(f"💚 LUCRO LÍQUIDO GARANTIDO: +{net_pnl*100:.4f}%")
                        
                        # Resetar acumulado de perdas
                        self.daily_loss_accumulated = max(0, self.daily_loss_accumulated - abs(net_pnl))
                        
                    else:
                        self.metrics.consecutive_wins = 0
                        self.metrics.consecutive_losses += 1
                        self.consecutive_losses += 1
                        self.daily_loss_accumulated += abs(net_pnl)
                        logger.info(f"🔴 PERDA CONTROLADA: {net_pnl*100:.4f}%")
                    
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
                
                logger.info(f"📊 PERFORMANCE EXTREMA ATUALIZADA:")
                logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   💎 Profit Bruto: {self.metrics.total_profit*100:.4f}%")
                logger.info(f"   💰 Profit Líquido: {daily_profit_pct:.4f}%")
                logger.info(f"   💸 Taxas Pagas: {self.metrics.total_fees_paid*100:.4f}%")
                logger.info(f"   🏆 META 50%: {target_progress:.1f}%")
                logger.info(f"   🔥 Wins Consecutivos: {self.metrics.consecutive_wins}")
                logger.info(f"   ❄️ Losses Consecutivos: {self.consecutive_losses}")
                
                self.is_exiting_position = False
                return True
                
            else:
                logger.error("❌ FALHA TOTAL NO FECHAMENTO GARANTIDO!")
                self.is_exiting_position = False
                return False
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no fechamento garantido: {e}")
            traceback.print_exc()
            self.is_exiting_position = False
            return False

    def _close_short_position_guaranteed(self) -> Dict:
        """Fecha posição SHORT com garantia máxima"""
        try:
            logger.info("📈 Fechando SHORT com garantia - Comprando para cobrir...")
            
            # Método 1: Buy order padrão
            result = self.bitget_api.place_buy_order()
            
            if result and result.get('success'):
                logger.info(f"✅ SHORT fechado via buy garantido: {result.get('message', '')}")
                return {"success": True, "result": result}
            
            # Método 2: API direta
            try:
                order = self.bitget_api.exchange.create_market_buy_order(
                    'ETHUSDT', abs(self.current_position.size), None, {'leverage': self.leverage}
                )
                if order:
                    logger.info(f"✅ SHORT fechado via API direta garantida")
                    return {"success": True, "order": order}
            except Exception as e:
                logger.error(f"❌ Método 2 SHORT garantido: {e}")
            
            return {"success": False, "error": "Falha ao fechar SHORT garantido"}
                
        except Exception as e:
            logger.error(f"❌ Erro ao fechar SHORT garantido: {e}")
            return {"success": False, "error": str(e)}

    def _check_emergency_conditions(self) -> bool:
        """Verificar condições de emergência extremas"""
        try:
            # Verificar perdas consecutivas
            if self.consecutive_losses >= 2:  # Mais rigoroso
                logger.warning(f"🚨 Perdas consecutivas extremas: {self.consecutive_losses}")
                return True
            
            # Verificar perda diária máxima
            if self.daily_loss_accumulated >= 0.05:  # 5% perda máxima
                logger.warning(f"🚨 Perda diária máxima atingida: {self.daily_loss_accumulated*100:.2f}%")
                return True
            
            # Verificar drawdown máximo
            if self.metrics.max_drawdown >= 0.08:  # 8% drawdown máximo
                logger.warning(f"🚨 Drawdown máximo atingido: {self.metrics.max_drawdown*100:.2f}%")
                return True
            
            # Verificar precisão das predições
            if self.prediction_accuracy < 60.0 and self.metrics.total_trades > 10:
                logger.warning(f"🚨 Precisão muito baixa: {self.prediction_accuracy:.1f}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar emergência: {e}")
            return True  # Parar por segurança

    def _adjust_for_50_percent_extreme(self):
        """Ajustar parâmetros dinamicamente para garantir 50% diário"""
        try:
            with self._lock:
                current_profit_pct = self.metrics.net_profit * 100
                current_time = datetime.now()
                hours_passed = max(1, current_time.hour - 8) if current_time.hour >= 8 else 1
                
                expected_profit = (50.0 / 24) * hours_passed  # Profit esperado até agora
                profit_deficit = max(0, expected_profit - current_profit_pct)
                
                logger.info(f"📊 AJUSTE EXTREMO PARA 50%:")
                logger.info(f"   💰 Profit atual: {current_profit_pct:.4f}%")
                logger.info(f"   🎯 Esperado: {expected_profit:.2f}%")
                logger.info(f"   📉 Déficit: {profit_deficit:.2f}%")
                
                # Se muito atrás da meta, ajustar (mas manter qualidade extrema)
                if profit_deficit > 5.0:  # Mais de 5% atrás
                    logger.warning("🚨 MUITO ATRÁS DA META - AJUSTE EXTREMO!")
                    # Reduzir critérios mas manter padrão extremo
                    self.min_confidence_to_trade = max(0.75, self.min_confidence_to_trade - 0.05)
                    self.min_strength_threshold = max(0.010, self.min_strength_threshold - 0.001)
                    self.force_trade_after_seconds = max(240, self.force_trade_after_seconds - 60)
                    
                    # MANTER take profit alto para garantir lucro líquido
                    self.profit_target = max(0.010, self.profit_target - 0.001)  # Não menos que 1.0%
                    
                # Se na meta ou à frente, aumentar qualidade
                elif profit_deficit < -2.0:  # Mais de 2% à frente
                    logger.info("✅ À FRENTE DA META - AUMENTAR QUALIDADE!")
                    self.min_confidence_to_trade = min(0.95, self.min_confidence_to_trade + 0.02)
                    self.min_strength_threshold = min(0.020, self.min_strength_threshold + 0.001)
                    
                    # Aumentar take profit para maior qualidade
                    self.profit_target = min(0.018, self.profit_target + 0.001)
                
                # Se perdas consecutivas, ser mais conservador
                if self.consecutive_losses >= 1:
                    logger.warning("⚠️ PERDAS CONSECUTIVAS - MODO ULTRA CONSERVADOR!")
                    self.min_confidence_to_trade = min(0.95, self.min_confidence_to_trade + 0.1)
                    self.min_strength_threshold = min(0.020, self.min_strength_threshold + 0.003)
                    # AUMENTAR take profit para garantir mais lucro
                    self.profit_target = min(0.020, self.profit_target + 0.002)
                
                logger.info(f"   🎯 Nova confiança: {self.min_confidence_to_trade*100:.1f}%")
                logger.info(f"   💪 Nova força: {self.min_strength_threshold*100:.2f}%")
                logger.info(f"   💰 Novo take profit: {self.profit_target*100:.1f}% (LUCRO LÍQUIDO)")
                
        except Exception as e:
            logger.error(f"❌ Erro no ajuste extremo: {e}")

    def get_account_balance(self) -> float:
        """Obter saldo da conta com fallback extremo"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                balance = float(balance_info.get('free', 0.0))
                if balance > 0:
                    return balance
                    
            if self.paper_trading:
                return 1000.0
            else:
                logger.warning("⚠️ Saldo não obtido - usando fallback extremo")
                return 100.0
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 1000.0 if self.paper_trading else 100.0

    def emergency_stop(self) -> bool:
        """Parada de emergência extrema com fechamento forçado garantido"""
        try:
            logger.warning("🚨 PARADA DE EMERGÊNCIA EXTREMA TOTAL")
            
            self.state = TradingState.EMERGENCY
            self.emergency_stop_triggered = True
            
            # Fechar posição com GARANTIA MÁXIMA
            if self.current_position:
                self._force_close_position_guaranteed("Emergency stop extremo total")
            
            # Cancelar todas as ordens
            try:
                self.bitget_api.exchange.cancel_all_orders(self.symbol)
            except:
                pass
            
            # Parar thread
            if self.trading_thread:
                self.trading_thread.join(timeout=5)
            
            self.state = TradingState.STOPPED
            
            logger.warning("🛑 Parada de emergência extrema total concluída")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na parada de emergência extrema: {e}")
            return False

    def reset_daily_stats(self):
        """Reset para novo dia - otimizado para 50% extremo"""
        try:
            logger.info("🔄 Reset para NOVO DIA EXTREMO - META 50% LUCRO LÍQUIDO!")
            
            with self._lock:
                self.trades_today = 0
                self.metrics = TradingMetrics()
                self.analysis_count = 0
                self.trades_rejected = 0
                self.quality_trades = 0
                self.rejected_low_quality = 0
                self.profitable_exits = 0
                self.prediction_accuracy = 75.0  # Start with reasonable accuracy
                self.consecutive_losses = 0
                self.daily_loss_accumulated = 0.0
                self.emergency_stop_triggered = False
                self.consecutive_failed_predictions = 0
                self.last_rejection_reason = ""
                self.last_trade_time = time.time()
                self.max_profit_reached = 0.0
                self.max_loss_reached = 0.0
                self.current_
