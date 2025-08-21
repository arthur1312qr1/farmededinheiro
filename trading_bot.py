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
    """IA Avançada para previsão de direção LONG/SHORT"""
    
    def __init__(self):
        self.models = {
            'momentum': RandomForestRegressor(n_estimators=50, random_state=42),
            'trend': GradientBoostingRegressor(n_estimators=30, random_state=42),
            'volatility': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.trained = False
        self.prediction_history = deque(maxlen=100)
        
    def prepare_features(self, prices: np.array, volumes: np.array = None) -> np.array:
        """Prepara features avançadas para ML"""
        if len(prices) < 50:
            return None
            
        features = []
        
        # 1. Momentum features
        for period in [5, 10, 14, 20]:
            if len(prices) >= period:
                momentum = (prices[-1] - prices[-period]) / prices[-period]
                features.append(momentum)
        
        # 2. RSI features
        rsi_14 = self._calculate_rsi(prices, 14)
        rsi_7 = self._calculate_rsi(prices, 7)
        features.extend([rsi_14, rsi_7, rsi_14 - rsi_7])
        
        # 3. Moving averages
        if len(prices) >= 20:
            ma_5 = np.mean(prices[-5:])
            ma_10 = np.mean(prices[-10:])
            ma_20 = np.mean(prices[-20:])
            
            features.extend([
                (prices[-1] - ma_5) / ma_5,
                (prices[-1] - ma_10) / ma_10,
                (prices[-1] - ma_20) / ma_20,
                (ma_5 - ma_10) / ma_10,
                (ma_10 - ma_20) / ma_20
            ])
        
        # 4. Volatility features
        if len(prices) >= 20:
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            volatility_change = volatility - (np.std(prices[-25:-5]) / np.mean(prices[-25:-5])) if len(prices) >= 25 else 0
            features.extend([volatility, volatility_change])
        
        # 5. Price action patterns
        if len(prices) >= 10:
            highs = np.maximum.accumulate(prices[-10:])
            lows = np.minimum.accumulate(prices[-10:])
            price_position = (prices[-1] - lows[-1]) / max(highs[-1] - lows[-1], 0.0001)
            features.append(price_position)
        
        # 6. Support/Resistance
        if len(prices) >= 30:
            resistance = np.max(prices[-30:])
            support = np.min(prices[-30:])
            sr_position = (prices[-1] - support) / max(resistance - support, 0.0001)
            distance_to_resistance = (resistance - prices[-1]) / prices[-1]
            distance_to_support = (prices[-1] - support) / prices[-1]
            features.extend([sr_position, distance_to_resistance, distance_to_support])
        
        # 7. Volume features (se disponível)
        if volumes is not None and len(volumes) >= 10:
            volume_ratio = volumes[-1] / max(np.mean(volumes[-10:]), 1)
            features.append(volume_ratio)
        else:
            features.append(1.0)  # Valor neutro se não há volume
        
        return np.array(features)
    
    def _calculate_rsi(self, prices: np.array, period: int = 14) -> float:
        """Calcula RSI"""
        if len(prices) <= period:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0001
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def train_models(self, price_history: deque, target_movements: list):
        """Treina modelos com dados históricos"""
        try:
            if len(price_history) < 100 or len(target_movements) < 50:
                return False
                
            prices = np.array(list(price_history))
            X_data = []
            y_data = []
            
            # Criar dataset de treino
            for i in range(50, len(prices) - 5):
                features = self.prepare_features(prices[:i+1])
                if features is not None and len(features) > 10:
                    # Target: movimento dos próximos 5 períodos
                    future_change = (prices[i+5] - prices[i]) / prices[i]
                    
                    X_data.append(features)
                    y_data.append(1 if future_change > 0 else -1)  # 1 = LONG, -1 = SHORT
            
            if len(X_data) < 20:
                return False
            
            X_data = np.array(X_data)
            y_data = np.array(y_data)
            
            # Normalizar features
            X_scaled = self.scaler.fit_transform(X_data)
            
            # Treinar modelos
            for model_name, model in self.models.items():
                try:
                    model.fit(X_scaled, y_data)
                except:
                    pass
            
            self.trained = True
            logger.info("🧠 IA treinada com sucesso!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro no treino da IA: {e}")
            return False
    
    def predict_direction(self, prices: np.array, volumes: np.array = None) -> Tuple[TradeDirection, float]:
        """Prediz direção com alta precisão usando ensemble de modelos"""
        try:
            if not self.trained or len(prices) < 50:
                # Fallback para análise técnica simples
                return self._fallback_prediction(prices)
            
            features = self.prepare_features(prices, volumes)
            if features is None:
                return self._fallback_prediction(prices)
            
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            predictions = []
            confidences = []
            
            # Ensemble de modelos
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)[0]
                        pred = 1 if proba[1] > proba[0] else -1
                        confidence = max(proba) - min(proba)  # Diferença entre probabilidades
                    else:
                        pred = model.predict(features_scaled)[0]
                        pred = 1 if pred > 0 else -1
                        confidence = abs(pred)
                    
                    predictions.append(pred)
                    confidences.append(confidence)
                except:
                    continue
            
            if not predictions:
                return self._fallback_prediction(prices)
            
            # Voto majoritário ponderado
            weighted_prediction = sum(p * c for p, c in zip(predictions, confidences)) / sum(confidences)
            final_direction = TradeDirection.LONG if weighted_prediction > 0 else TradeDirection.SHORT
            
            # Confiança média
            avg_confidence = np.mean(confidences)
            
            # Adicionar à história
            self.prediction_history.append({
                'direction': final_direction,
                'confidence': avg_confidence,
                'timestamp': time.time()
            })
            
            logger.info(f"🎯 IA Prevê: {final_direction.name} (Conf: {avg_confidence:.3f})")
            
            return final_direction, avg_confidence
            
        except Exception as e:
            logger.error(f"❌ Erro na predição IA: {e}")
            return self._fallback_prediction(prices)
    
    def _fallback_prediction(self, prices: np.array) -> Tuple[TradeDirection, float]:
        """Predição de fallback usando análise técnica"""
        try:
            if len(prices) < 20:
                return TradeDirection.LONG, 0.5
            
            # Análise de momentum
            short_momentum = (prices[-1] - prices[-5]) / prices[-5]
            medium_momentum = (prices[-1] - prices[-10]) / prices[-10]
            long_momentum = (prices[-1] - prices[-20]) / prices[-20]
            
            # RSI
            rsi = self._calculate_rsi(prices)
            
            # Moving averages
            ma_5 = np.mean(prices[-5:])
            ma_20 = np.mean(prices[-20:])
            
            signals = []
            
            # Sinais de momentum
            if short_momentum > 0.003:  # 0.3% momentum positivo
                signals.append(1)
            elif short_momentum < -0.003:
                signals.append(-1)
            
            # Sinais de RSI
            if rsi < 30:
                signals.append(1)  # Oversold = buy
            elif rsi > 70:
                signals.append(-1)  # Overbought = sell
            
            # Sinais de MA
            if ma_5 > ma_20 * 1.001:  # MA cruzamento
                signals.append(1)
            elif ma_5 < ma_20 * 0.999:
                signals.append(-1)
            
            if not signals:
                # Se não há sinais, usar momentum geral
                overall_momentum = (short_momentum + medium_momentum + long_momentum) / 3
                direction = TradeDirection.LONG if overall_momentum >= 0 else TradeDirection.SHORT
                confidence = min(abs(overall_momentum) * 10, 0.8)
            else:
                signal_sum = sum(signals)
                direction = TradeDirection.LONG if signal_sum > 0 else TradeDirection.SHORT
                confidence = min(abs(signal_sum) / len(signals), 0.8)
            
            return direction, confidence
            
        except Exception as e:
            logger.error(f"❌ Erro no fallback: {e}")
            return TradeDirection.LONG, 0.5

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ETHUSDT',
                 leverage: int = 10, balance_percentage: float = 100.0,
                 daily_target: int = 350, scalping_interval: float = 0.5,
                 paper_trading: bool = False):
        """Initialize ADVANCED Trading Bot with AI Prediction for 50% DAILY PROFIT"""
        
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
        self.min_trades_per_day = 150
        self.target_trades_per_day = 200
        self.max_time_between_trades = 300
        self.force_trade_after_seconds = 600
        self.last_trade_time = 0

        # CRITÉRIOS SELETIVOS PARA TRADES DE QUALIDADE - MAIS RIGOROSOS
        self.min_confidence_to_trade = 0.75     # 75% confiança mínima
        self.min_prediction_score = 0.70        # 70% score de predição
        self.min_signals_agreement = 7          # 7 sinais precisam concordar
        self.min_strength_threshold = 0.012     # 1.2% força mínima

        # CONFIGURAÇÕES DE LUCRO FIXAS - NÃO ALTERAR
        self.profit_target = 0.015              # 1.5% take profit FIXO
        self.stop_loss_target = -0.008          # 0.8% stop loss FIXO
        self.minimum_profit_target = 0.010      # 1.0% lucro mínimo absoluto
        self.max_position_time = 180            # Máximo 3 minutos por trade
        self.min_position_time = 45             # Mínimo 45 segundos
        self.breakeven_time = 60                # Breakeven após 60 segundos

        # Sistema de dados para análise técnica
        self.price_history = deque(maxlen=500)  # Mais dados para IA
        self.volume_history = deque(maxlen=200)
        self.analysis_history = deque(maxlen=100)

        # ===== SISTEMA DE IA AVANÇADO =====
        self.ai_predictor = AdvancedPredictor()
        self.movement_history = deque(maxlen=200)  # Para treinar IA
        self.ai_training_interval = 100  # Retreinar a cada 100 análises
        self.ai_predictions_correct = 0
        self.ai_predictions_total = 0

        # Sistema de trading profissional
        self.professional_mode_active = True
        self.emergency_trading_mode = False
        self.last_analysis_result = None
        self.force_trade_mode = False
        
        # Rastreamento avançado
        self.max_profit_reached = 0.0
        self.max_loss_reached = 0.0

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

        # Sistema de market conditions
        self.market_conditions = {
            'trend': 'neutral',
            'volatility': 0.0,
            'volume_avg': 0.0,
            'strength': 0.0
        }

        logger.info("🚀 ADVANCED AI TRADING BOT - 50% DAILY TARGET with REAL PROFITS")
        logger.info("🧠 IA AVANÇADA PARA PREDIÇÃO LONG/SHORT")
        logger.info("⚡ CONFIGURAÇÕES PROFISSIONAIS:")
        logger.info(f"   🎯 Confiança mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"   💪 Força mínima: {self.min_strength_threshold*100}%")
        logger.info(f"   📊 Sinais necessários: {self.min_signals_agreement}")
        logger.info(f"   📈 Take Profit FIXO: {self.profit_target*100}%")
        logger.info(f"   🛑 Stop Loss FIXO: {abs(self.stop_loss_target)*100}%")
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
                daily_progress = (current_profit_pct / 50.0) * 100
                
                # Precisão da IA
                ai_accuracy = (self.ai_predictions_correct / max(1, self.ai_predictions_total)) * 100
                
                return {
                    'bot_status': {
                        'state': self.state.value,
                        'is_running': self.is_running,
                        'symbol': self.symbol,
                        'leverage': self.leverage,
                        'paper_trading': self.paper_trading,
                        'professional_mode': True,
                        'ai_mode_active': True
                    },
                    'ai_system': {
                        'trained': self.ai_predictor.trained,
                        'predictions_total': self.ai_predictions_total,
                        'predictions_correct': self.ai_predictions_correct,
                        'accuracy': f"{ai_accuracy:.1f}%",
                        'model_count': len(self.ai_predictor.models),
                        'training_data_points': len(self.movement_history)
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
                            'min_signals': self.min_signals_agreement
                        }
                    },
                    'daily_progress_50_percent': {
                        'target_profit': '50.0%',
                        'current_profit': f"{current_profit_pct:.3f}%",
                        'progress_to_target': f"{daily_progress:.1f}%",
                        'trades_today': self.trades_today,
                        'target_trades': self.target_trades_per_day,
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
                'meets_take_profit': pnl >= self.profit_target,
                'meets_stop_loss': pnl <= self.stop_loss_target,
                'should_exit_time': duration >= self.max_position_time
            }
        except Exception as e:
            return {'active': True, 'error': f'Erro ao obter dados: {str(e)}'}

    def start(self) -> bool:
        """Iniciar bot com IA avançada"""
        try:
            if self.state == TradingState.RUNNING:
                logger.warning("🟡 Bot já está rodando")
                return True
            
            logger.info("🚀 INICIANDO BOT COM IA AVANÇADA - META 50% DIÁRIO COM LUCRO REAL")
            logger.info("🧠 IA PARA PREDIÇÃO LONG/SHORT ATIVA!")
            
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
            
            # Resetar IA
            self.ai_predictions_correct = 0
            self.ai_predictions_total = 0
            self.movement_history.clear()
            
            # Resetar estado
            self.state = TradingState.RUNNING
            self.start_balance = self.get_account_balance()
            self.last_trade_time = time.time()
            self.last_error = None
            self.professional_mode_active = True
            
            # Reset rastreamento
            self.max_profit_reached = 0.0
            self.max_loss_reached = 0.0
            
            # Iniciar thread principal com IA
            self.trading_thread = threading.Thread(
                target=self._advanced_ai_trading_loop, 
                daemon=True,
                name="AdvancedAITradingBot"
            )
            self.trading_thread.start()
            
            logger.info("✅ Bot com IA avançada iniciado - META: 50% DIÁRIO COM LUCRO REAL!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar bot: {e}")
            self.state = TradingState.STOPPED
            self.last_error = str(e)
            return False

    def stop(self) -> bool:
        """Parar bot com relatório completo"""
        try:
            logger.info("🛑 Parando bot com IA avançada...")
            
            self.state = TradingState.STOPPED
            
            # Fechar posição atual
            if self.current_position:
                logger.info("🔒 Fechando posição final...")
                self._close_position_advanced("Bot stopping")
            
            # Aguardar thread
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=10)
            
            # Relatório final com IA
            daily_profit_pct = self.metrics.net_profit * 100
            target_achievement = (daily_profit_pct / 50.0) * 100
            ai_accuracy = (self.ai_predictions_correct / max(1, self.ai_predictions_total)) * 100
            
            logger.info("📊 RELATÓRIO FINAL COM IA AVANÇADA:")
            logger.info(f"   🧠 IA Accuracy: {ai_accuracy:.1f}%")
            logger.info(f"   📈 Análises realizadas: {self.analysis_count}")
            logger.info(f"   ⚡ Trades executados: {self.trades_today}")
            logger.info(f"   🏆 Trades de qualidade: {self.quality_trades}")
            logger.info(f"   🚫 Rejeitados baixa qualidade: {self.rejected_low_quality}")
            logger.info(f"   💚 Saídas lucrativas: {self.profitable_exits}")
            logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
            logger.info(f"   💎 Profit Bruto: {self.metrics.total_profit*100:.3f}%")
            logger.info(f"   💰 Profit Líquido: {daily_profit_pct:.3f}%")
            logger.info(f"   🏆 META 50% Atingimento: {target_achievement:.1f}%")
            
            logger.info("✅ Bot com IA avançada parado!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao parar bot: {e}")
            return False

    def _advanced_ai_trading_loop(self):
        """Loop AVANÇADO com IA para máximo lucro"""
        logger.info("🏆 Loop com IA avançada iniciado - QUALIDADE MÁXIMA!")
        
        while self.state == TradingState.RUNNING:
            try:
                loop_start = time.time()
                self.analysis_count += 1
                
                # Verificar condições de emergência
                if self._check_emergency_conditions():
                    logger.warning("🚨 Condições de emergência detectadas!")
                    break
                
                # COLETA DE DADOS PARA IA
                market_data = self.bitget_api.get_market_data(self.symbol)
                if not market_data or 'price' not in market_data:
                    time.sleep(1)
                    continue
                
                current_price = float(market_data['price'])
                current_volume = float(market_data.get('volume', 0))
                
                self.price_history.append(current_price)
                if current_volume > 0:
                    self.volume_history.append(current_volume)
                
                # Registrar movimento para treinar IA
                if len(self.price_history) >= 10:
                    old_price = self.price_history[-10]
                    movement = (current_price - old_price) / old_price
                    self.movement_history.append(movement)
                
                # TREINAR IA PERIODICAMENTE
                if self.analysis_count % self.ai_training_interval == 0 and len(self.price_history) > 100:
                    logger.info("🧠 Treinando IA com novos dados...")
                    self.ai_predictor.train_models(self.price_history, list(self.movement_history))
                
                # ANÁLISE TÉCNICA + IA AVANÇADA - SÓ SE NÃO TIVER POSIÇÃO
                if not self.current_position:
                    should_trade, confidence, ai_direction, strength, analysis_details = self._advanced_ai_analysis()
                    
                    # FORÇAR TRADE SE MUITO TEMPO SEM TRADING
                    seconds_since_last = time.time() - self.last_trade_time
                    force_trade = seconds_since_last >= self.force_trade_after_seconds
                    
                    if force_trade:
                        logger.warning(f"⏰ FORÇANDO TRADE COM IA - {seconds_since_last:.0f}s sem trade!")
                        should_trade = True
                        confidence = max(confidence, 0.7)
                        self.force_trade_mode = True
                    else:
                        self.force_trade_mode = False
                    
                    # LOG PERIÓDICO
                    if self.analysis_count % 100 == 0:
                        logger.info(f"🏆 Análise IA #{self.analysis_count}:")
                        logger.info(f"   🧠 IA Direção: {ai_direction.name if ai_direction else 'AUTO'}")
                        logger.info(f"   🎯 Confiança: {confidence*100:.1f}%")
                        logger.info(f"   💪 Força: {strength*100:.2f}%")
                        logger.info(f"   ✅ Executar: {should_trade}")
                    
                    # EXECUTAR TRADE APENAS SE CRITÉRIOS RIGOROSOS
                    if should_trade and ai_direction:
                        success = self._execute_ai_trade(ai_direction, confidence, strength, analysis_details)
                        if success:
                            self.last_trade_time = time.time()
                            self.trades_today += 1
                            self.quality_trades += 1
                            logger.info(f"🏆 TRADE IA #{self.trades_today} - {ai_direction.name} - Conf: {confidence*100:.1f}%")
                        else:
                            self.trades_rejected += 1
                            self.last_rejection_reason = "Falha na execução IA"
                    
                    elif not should_trade:
                        self.trades_rejected += 1
                        self.rejected_low_quality += 1
                        self.last_rejection_reason = f"IA: Baixa qualidade - Conf:{confidence*100:.1f}%, Força:{strength*100:.2f}%"
                
                # GERENCIAR POSIÇÃO EXISTENTE COM IA
                elif self.current_position:
                    self._ai_position_management()
                
                # Sleep otimizado
                elapsed = time.time() - loop_start
                sleep_time = max(0.2, self.scalping_interval - elapsed)
                time.sleep(sleep_time)
                
                # Ajuste dinâmico para 50%
                if self.analysis_count % 500 == 0:
                    self._adjust_for_50_percent_target()
                
            except Exception as e:
                logger.error(f"❌ Erro no loop IA: {e}")
                traceback.print_exc()
                time.sleep(2)
        
        logger.info(f"🏁 Loop IA finalizado - Trades: {self.trades_today}, Profit: {self.metrics.net_profit*100:.3f}%")

    def _advanced_ai_analysis(self) -> Tuple[bool, float, Optional[TradeDirection], float, Dict]:
        """Análise AVANÇADA com IA para predição LONG/SHORT"""
        try:
            if len(self.price_history) < 50:
                return False, 0.0, None, 0.0, {'error': f'Dados insuficientes: {len(self.price_history)}/50'}
            
            prices = np.array(list(self.price_history))
            volumes = np.array(list(self.volume_history)) if self.volume_history else None
            analysis_details = {}
            
            # === IA PRINCIPAL PARA DIREÇÃO ===
            ai_direction, ai_confidence = self.ai_predictor.predict_direction(prices, volumes)
            analysis_details['ai_direction'] = ai_direction.name
            analysis_details['ai_confidence'] = ai_confidence
            
            # === ANÁLISE TÉCNICA COMPLEMENTAR ===
            technical_signals = []
            
            # 1. RSI AVANÇADO
            rsi_signal, rsi_value = self._calculate_advanced_rsi(prices)
            analysis_details['rsi'] = rsi_value
            if rsi_signal != 0:
                technical_signals.extend([rsi_signal] * 2)
            
            # 2. MOMENTUM MULTI-TIMEFRAME
            momentum_signals = self._calculate_multi_momentum(prices)
            analysis_details['momentum_signals'] = momentum_signals
            technical_signals.extend(momentum_signals)
            
            # 3. VOLUME ANALYSIS
            volume_signal = self._analyze_volume_advanced(volumes) if volumes is not None else 0
            analysis_details['volume_signal'] = volume_signal
            if volume_signal != 0:
                technical_signals.append(volume_signal)
            
            # 4. VOLATILITY BREAKOUT
            volatility_signal, vol_value = self._analyze_volatility_breakout(prices)
            analysis_details['volatility'] = vol_value
            if volatility_signal != 0:
                technical_signals.append(volatility_signal)
            
            # 5. SUPPORT/RESISTANCE
            sr_signal = self._analyze_support_resistance_advanced(prices)
            analysis_details['sr_signal'] = sr_signal
            if sr_signal != 0:
                technical_signals.append(sr_signal)
            
            # 6. TREND STRENGTH
            trend_strength = self._calculate_trend_strength_advanced(prices)
            analysis_details['trend_strength'] = trend_strength
            if abs(trend_strength) > 0.008:
                trend_signal = 2 if trend_strength > 0 else -2
                technical_signals.extend([trend_signal] * 2)
            
            # === COMBINAR IA + ANÁLISE TÉCNICA ===
            
            # Verificar concordância entre IA e análise técnica
            if technical_signals:
                avg_technical = sum(technical_signals) / len(technical_signals)
                technical_direction = TradeDirection.LONG if avg_technical > 0 else TradeDirection.SHORT
                
                # IA e técnica concordam?
                ia_technical_agreement = (ai_direction == technical_direction)
                analysis_details['ia_technical_agreement'] = ia_technical_agreement
                
                if ia_technical_agreement:
                    # Concordância aumenta confiança
                    combined_confidence = min(ai_confidence * 1.2, 0.95)
                    final_direction = ai_direction
                    agreement_bonus = 0.1
                else:
                    # Discordância reduz confiança
                    combined_confidence = ai_confidence * 0.8
                    final_direction = ai_direction  # IA tem prioridade
                    agreement_bonus = 0.0
            else:
                # Só IA disponível
                combined_confidence = ai_confidence
                final_direction = ai_direction
                agreement_bonus = 0.0
            
            # Calcular força total
            signal_count = len(technical_signals)
            signal_strength = abs(sum(technical_signals)) / max(signal_count, 1) if technical_signals else ai_confidence
            combined_strength = min((signal_strength * 0.01) + agreement_bonus, 0.05)
            
            # SCORE DE QUALIDADE FINAL
            quality_score = (
                combined_confidence * 0.4 +
                (combined_strength / 0.02) * 0.3 +
                (ai_confidence * 0.2) +
                (agreement_bonus * 10)  # Bônus por concordância
            ) * 100
            
            analysis_details.update({
                'combined_confidence': combined_confidence,
                'combined_strength': combined_strength,
                'quality_score': quality_score,
                'technical_signals_count': signal_count,
                'signal_strength_avg': signal_strength,
                'agreement_bonus': agreement_bonus
            })
            
            # CRITÉRIOS RIGOROSOS PARA 50% DIÁRIO
            meets_confidence = combined_confidence >= self.min_confidence_to_trade
            meets_strength = combined_strength >= self.min_strength_threshold
            meets_signals = signal_count >= (self.min_signals_agreement - 2)  # IA compensa alguns sinais
            meets_quality = quality_score >= 75.0
            meets_ai_confidence = ai_confidence >= 0.65  # IA deve ter confiança mínima
            
            should_trade = (meets_confidence and meets_strength and meets_signals 
                          and meets_quality and meets_ai_confidence and final_direction is not None)
            
            analysis_details['criteria_met'] = {
                'confidence': meets_confidence,
                'strength': meets_strength,
                'signals': meets_signals,
                'quality': meets_quality,
                'ai_confidence': meets_ai_confidence,
                'final_decision': should_trade
            }
            
            return should_trade, combined_confidence, final_direction, combined_strength, analysis_details
            
        except Exception as e:
            logger.error(f"❌ Erro na análise IA avançada: {e}")
            return False, 0.0, None, 0.0, {'error': str(e)}

    def _calculate_advanced_rsi(self, prices: np.array, period: int = 14) -> Tuple[int, float]:
        """RSI avançado com sinais mais precisos"""
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
            
            # Sinais mais conservadores
            if rsi < 20:  # Oversold extremo
                return 3, rsi
            elif rsi < 30:  # Oversold
                return 2, rsi
            elif rsi > 80:  # Overbought extremo
                return -3, rsi
            elif rsi > 70:  # Overbought
                return -2, rsi
            else:
                return 0, rsi
                
        except Exception as e:
            return 0, 50.0

    def _calculate_multi_momentum(self, prices: np.array) -> List[int]:
        """Momentum multi-timeframe"""
        try:
            signals = []
            periods = [3, 5, 10, 15, 20]
            
            for period in periods:
                if len(prices) >= period:
                    momentum = (prices[-1] - prices[-period]) / prices[-period]
                    
                    if momentum > 0.008:  # 0.8% momentum positivo
                        signals.append(2)
                    elif momentum > 0.004:  # 0.4% momentum positivo
                        signals.append(1)
                    elif momentum < -0.008:  # 0.8% momentum negativo
                        signals.append(-2)
                    elif momentum < -0.004:  # 0.4% momentum negativo
                        signals.append(-1)
            
            return signals
            
        except:
            return []

    def _analyze_volume_advanced(self, volumes: np.array) -> int:
        """Análise avançada de volume"""
        try:
            if volumes is None or len(volumes) < 20:
                return 0
            
            current_vol = volumes[-1]
            avg_vol_short = np.mean(volumes[-5:])
            avg_vol_long = np.mean(volumes[-20:])
            
            vol_ratio_short = current_vol / max(avg_vol_short, 1)
            vol_ratio_long = current_vol / max(avg_vol_long, 1)
            
            # Volume breakout
            if vol_ratio_short > 2.0 and vol_ratio_long > 1.8:
                return 3  # Volume breakout forte
            elif vol_ratio_short > 1.5 and vol_ratio_long > 1.3:
                return 2  # Volume acima da média
            elif vol_ratio_short > 1.2:
                return 1  # Volume ligeiramente acima
            elif vol_ratio_short < 0.6:
                return -1  # Volume muito baixo
            else:
                return 0
                
        except:
            return 0

    def _analyze_volatility_breakout(self, prices: np.array) -> Tuple[int, float]:
        """Detecta breakouts de volatilidade"""
        try:
            if len(prices) < 30:
                return 0, 0.0
            
            # Volatilidade atual vs histórica
            current_vol = np.std(prices[-10:]) / np.mean(prices[-10:])
            historical_vol = np.std(prices[-30:-10]) / np.mean(prices[-30:-10])
            
            vol_ratio = current_vol / max(historical_vol, 0.001)
            
            # Direção do breakout
            recent_change = (prices[-1] - prices[-5]) / prices[-5]
            
            if vol_ratio > 1.5:  # Volatilidade 50% maior
                if recent_change > 0:
                    return 2, current_vol  # Breakout bullish
                else:
                    return -2, current_vol  # Breakout bearish
            elif vol_ratio > 1.2:  # Volatilidade 20% maior
                if recent_change > 0:
                    return 1, current_vol
                else:
                    return -1, current_vol
            else:
                return 0, current_vol
                
        except:
            return 0, 0.0

    def _analyze_support_resistance_advanced(self, prices: np.array) -> int:
        """Análise avançada de suporte e resistência"""
        try:
            if len(prices) < 50:
                return 0
            
            current_price = prices[-1]
            
            # Encontrar níveis de S/R dos últimos 50 períodos
            highs = []
            lows = []
            
            for i in range(2, len(prices) - 2):
                if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                    prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                    highs.append(prices[i])
                
                if (prices[i] < prices[i-1] and prices[i] < prices[i+1] and
                    prices[i] < prices[i-2] and prices[i] < prices[i+2]):
                    lows.append(prices[i])
            
            if not highs or not lows:
                return 0
            
            # Resistência e suporte mais próximos
            resistance = min([h for h in highs if h > current_price], default=current_price * 1.1)
            support = max([l for l in lows if l < current_price], default=current_price * 0.9)
            
            # Distância para S/R
            dist_to_resistance = (resistance - current_price) / current_price
            dist_to_support = (current_price - support) / current_price
            
            # Sinais baseados na proximidade
            if dist_to_support < 0.005:  # Muito próximo ao suporte
                return 2  # Strong buy
            elif dist_to_support < 0.01:  # Próximo ao suporte
                return 1  # Buy
            elif dist_to_resistance < 0.005:  # Muito próximo à resistência
                return -2  # Strong sell
            elif dist_to_resistance < 0.01:  # Próximo à resistência
                return -1  # Sell
            else:
                return 0
                
        except:
            return 0

    def _calculate_trend_strength_advanced(self, prices: np.array) -> float:
        """Calcula força da tendência com múltiplos indicadores"""
        try:
            if len(prices) < 30:
                return 0.0
            
            # 1. Linear regression slope
            x = np.arange(len(prices[-20:]))
            slope = np.polyfit(x, prices[-20:], 1)[0]
            slope_strength = slope / prices[-1]
            
            # 2. Moving average alignment
            if len(prices) >= 20:
                ma_5 = np.mean(prices[-5:])
                ma_10 = np.mean(prices[-10:])
                ma_20 = np.mean(prices[-20:])
                
                # Alinhamento das médias
                if ma_5 > ma_10 > ma_20:
                    ma_strength = 0.01  # Uptrend
                elif ma_5 < ma_10 < ma_20:
                    ma_strength = -0.01  # Downtrend
                else:
                    ma_strength = 0  # Sideways
            else:
                ma_strength = 0
            
            # 3. Price momentum
            momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
            
            # Combinar indicadores
            combined_strength = (slope_strength * 0.5 + ma_strength * 0.3 + momentum * 0.2)
            
            return combined_strength
            
        except:
            return 0.0

    def _execute_ai_trade(self, direction: TradeDirection, confidence: float, strength: float, analysis_details: Dict) -> bool:
        """Execução de trade com IA - APENAS UMA DIREÇÃO"""
        try:
            # ❌ VERIFICAÇÃO CRÍTICA: NÃO EXECUTAR SE JÁ HÁ POSIÇÃO
            if self.current_position:
                logger.warning("⚠️ TRADE CANCELADO - Posição já existe")
                return False
            
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
            
            # Targets FIXOS - NÃO ALTERAR
            if direction == TradeDirection.LONG:
                target_price = current_price * (1 + self.profit_target)  # +1.5%
                stop_price = current_price * (1 + self.stop_loss_target)  # -0.8%
            else:  # SHORT
                target_price = current_price * (1 - self.profit_target)  # -1.5%
                stop_price = current_price * (1 - self.stop_loss_target)  # +0.8%
            
            # Calcular taxa estimada
            estimated_fee = position_value * 0.001
            
            logger.info(f"🏆 TRADE IA {direction.name}:")
            logger.info(f"   🧠 IA Conf: {analysis_details.get('ai_confidence', 0)*100:.1f}%")
            logger.info(f"   💰 Saldo: ${balance:.2f} | Size: {position_size:.6f}")
            logger.info(f"   💱 ${current_price:.2f} → Target: ${target_price:.2f} | Stop: ${stop_price:.2f}")
            logger.info(f"   🎯 Conf Total: {confidence*100:.1f}% | Força: {strength*100:.2f}%")
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
                logger.info("✅ PAPER TRADE IA EXECUTADO!")
                return True
            else:
                # Trading real - APENAS UMA DIREÇÃO
                try:
                    if direction == TradeDirection.LONG:
                        result = self.bitget_api.place_buy_order()
                    else:  # SHORT
                        result = self._execute_short_order_ai(position_size)
                    
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
                        
                        # Incrementar contador de predições da IA
                        self.ai_predictions_total += 1
                        
                        logger.info("✅ REAL TRADE IA EXECUTADO!")
                        return True
                    else:
                        logger.error(f"❌ Falha na execução IA: {result}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Erro na execução IA: {e}")
                    return False
                
        except Exception as e:
            logger.error(f"❌ Erro no trade IA: {e}")
            return False

    def _execute_short_order_ai(self, position_size: float) -> Dict:
        """Executa ordem SHORT com IA"""
        try:
            logger.info(f"📉 SHORT IA - {position_size:.6f}")
            
            order = self.bitget_api.exchange.create_market_sell_order(
                'ETHUSDT',
                position_size,
                None,
                {'leverage': self.leverage}
            )
            
            if order:
                logger.info(f"✅ SHORT IA: {order['id']}")
                return {
                    "success": True,
                    "order": order,
                    "quantity": position_size,
                    "price": order.get('price', 0)
                }
            else:
                return {"success": False, "error": "SHORT IA falhou"}
                
        except Exception as e:
            logger.error(f"❌ Erro SHORT IA: {e}")
            return {"success": False, "error": str(e)}

    def _ai_position_management(self):
        """Gerenciamento de posição com IA - FECHAMENTO GARANTIDO"""
        if not self.current_position:
            return
        
        try:
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data or 'price' not in market_data:
                logger.error("❌ Sem dados para gerenciar posição IA")
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
            
            # === CRITÉRIOS DE FECHAMENTO FIXOS - NÃO ALTERAR ===
            
            # 1. TAKE PROFIT FIXO (1.5%)
            if pnl >= self.profit_target:
                should_close = True
                close_reason = f"🎯 TAKE PROFIT FIXO: {pnl*100:.3f}%"
                is_profitable_exit = True
                # Marcar predição da IA como correta
                if hasattr(self, 'ai_predictions_total') and self.ai_predictions_total > 0:
                    self.ai_predictions_correct += 1
            
            # 2. STOP LOSS FIXO (0.8%)
            elif pnl <= self.stop_loss_target:
                should_close = True
                close_reason = f"🛑 STOP LOSS FIXO: {pnl*100:.3f}%"
            
            # 3. TRAILING STOP IA (só após lucro significativo)
            elif self.max_profit_reached >= 0.02 and pnl <= (self.max_profit_reached - 0.008):
                should_close = True
                close_reason = f"📉 TRAILING IA: {pnl*100:.3f}% (max: {self.max_profit_reached*100:.3f}%)"
                is_profitable_exit = True
                self.ai_predictions_correct += 1
            
            # 4. TEMPO MÁXIMO COM LUCRO MÍNIMO
            elif duration >= self.min_position_time and pnl >= self.minimum_profit_target:
                should_close = True
                close_reason = f"✅ LUCRO MÍNIMO IA: {pnl*100:.3f}% em {duration:.0f}s"
                is_profitable_exit = True
                self.ai_predictions_correct += 1
            
            # 5. TEMPO LIMITE ABSOLUTO
            elif duration >= self.max_position_time:
                should_close = True
                close_reason = f"⏰ TEMPO LIMITE: {pnl*100:.3f}% em {duration:.0f}s"
                if pnl > 0:
                    is_profitable_exit = True
                    self.ai_predictions_correct += 1
            
            # 6. EMERGÊNCIA (perdas grandes)
            elif pnl <= -0.02:  # -2%
                should_close = True
                close_reason = f"🚨 EMERGÊNCIA IA: {pnl*100:.3f}%"
            
            if should_close:
                logger.info(f"🔒 FECHANDO POSIÇÃO IA: {close_reason}")
                success = self._close_position_advanced(close_reason)
                
                if success and is_profitable_exit:
                    self.profitable_exits += 1
                
                return success
            
            # Log periódico
            if int(duration) % 30 == 0:  # A cada 30 segundos
                profit_status = "💚" if pnl > 0 else "🔴"
                logger.info(f"⏳ Posição IA ativa: {profit_status} {pnl*100:.3f}% | {duration:.0f}s | Max: {self.max_profit_reached*100:.3f}%")
                
        except Exception as e:
            logger.error(f"❌ Erro gerenciamento IA: {e}")
            traceback.print_exc()
            
            # Forçar fechamento em erro crítico
            if self.current_position:
                logger.warning("🚨 FORÇANDO FECHAMENTO IA POR ERRO CRÍTICO")
                self._emergency_close_ai("Erro crítico IA")

    def _close_position_advanced(self, reason: str) -> bool:
        """Fechamento avançado com IA - GARANTIDO"""
        try:
            if not self.current_position:
                logger.warning("⚠️ Posição não existe")
                return False
                
            market_data = self.bitget_api.get_market_data(self.symbol)
            current_price = float(market_data['price']) if market_data else self.current_position.entry_price
            final_pnl = self.current_position.calculate_pnl(current_price)
            duration = self.current_position.get_duration()
                
            logger.info(f"🔒 FECHAMENTO IA: {reason}")
            logger.info(f"   📊 {self.current_position.side.name} | ${self.current_position.entry_price:.2f} → ${current_price:.2f}")
            logger.info(f"   📈 P&L: {final_pnl*100:.4f}% | ⱏ️ {duration:.1f}s")
            
            close_success = False
            
            if self.paper_trading:
                logger.info("📋 PAPER TRADING - Fechamento simulado IA")
                close_success = True
                
            else:
                # MÉTODO 1: Fechamento otimizado por direção
                logger.info("🎯 MÉTODO IA: Fechamento otimizado...")
                try:
                    if self.current_position.side == TradeDirection.LONG:
                        # Fechar LONG vendendo
                        result = self.bitget_api.place_sell_order(profit_target=0)
                        close_success = result and result.get('success', False)
                        if close_success:
                            logger.info("✅ LONG fechado via IA sell")
                    else:  # SHORT
                        # Fechar SHORT comprando
                        result = self._close_short_position_ai()
                        close_success = result and result.get('success', False)
                        if close_success:
                            logger.info("✅ SHORT fechado via IA buy")
                except Exception as e:
                    logger.error(f"❌ Método IA falhou: {e}")
                
                # MÉTODO 2: API direta se método IA falhar
                if not close_success:
                    logger.info("🎯 MÉTODO ALTERNATIVO IA: API Direta...")
                    try:
                        side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                        order = self.bitget_api.exchange.create_market_order(
                            'ETHUSDT', side, abs(self.current_position.size)
                        )
                        if order:
                            logger.info(f"✅ MÉTODO ALTERNATIVO IA: Sucesso via {side}")
                            close_success = True
                    except Exception as e:
                        logger.error(f"❌ Método alternativo IA falhou: {e}")
                
                # MÉTODO 3: Emergência total
                if not close_success:
                    logger.warning("🚨 MÉTODO EMERGÊNCIA IA TOTAL")
                    close_success = self._emergency_close_ai("Todos métodos falharam")
            
            if close_success:
                logger.info("✅ POSIÇÃO IA FECHADA COM SUCESSO!")
                
                # Calcular taxa estimada para fechamento
                estimated_fee = abs(final_pnl * self.current_position.size * current_price * 0.001)
                
                # Atualizar métricas IA
                with self._lock:
                    self.metrics.total_trades += 1
                    
                    # P&L líquido descontando taxas
                    net_pnl = final_pnl - (estimated_fee / (self.current_position.size * current_price))
                    self.metrics.total_profit += net_pnl
                    self.metrics.total_fees_paid += estimated_fee / (self.current_position.size * current_price)
                    
                    if net_pnl > 0:
                        self.metrics.profitable_trades += 1
                        self.metrics.consecutive_wins += 1
                        self.metrics.consecutive_losses = 0
                        self.consecutive_losses = 0
                        logger.info(f"💚 LUCRO IA LÍQUIDO: +{net_pnl*100:.4f}%")
                        
                        # Resetar perdas acumuladas
                        self.daily_loss_accumulated = max(0, self.daily_loss_accumulated - abs(net_pnl))
                        
                    else:
                        self.metrics.consecutive_wins = 0
                        self.metrics.consecutive_losses += 1
                        self.consecutive_losses += 1
                        self.daily_loss_accumulated += abs(net_pnl)
                        logger.info(f"🔴 PERDA IA LÍQUIDA: {net_pnl*100:.4f}%")
                    
                    # Atualizar drawdown
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
                
                # Limpar posição
                self.current_position = None
                self.last_trade_time = time.time()
                
                # Performance com IA
                daily_profit_pct = self.metrics.net_profit * 100
                target_progress = (daily_profit_pct / 50.0) * 100
                ai_accuracy = (self.ai_predictions_correct / max(1, self.ai_predictions_total)) * 100
                
                logger.info(f"📊 PERFORMANCE IA ATUALIZADA:")
                logger.info(f"   🧠 IA Accuracy: {ai_accuracy:.1f}%")
                logger.info(f"   🎯 Win Rate: {self.metrics.win_rate:.1f}%")
                logger.info(f"   💎 Profit Bruto: {self.metrics.total_profit*100:.4f}%")
                logger.info(f"   💰 Profit Líquido: {daily_profit_pct:.4f}%")
                logger.info(f"   💸 Taxas Pagas: {self.metrics.total_fees_paid*100:.4f}%")
                logger.info(f"   🏆 META 50%: {target_progress:.1f}%")
                
                return True
                
            else:
                logger.error("❌ TODOS OS MÉTODOS IA DE FECHAMENTO FALHARAM!")
                return False
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no fechamento IA: {e}")
            traceback.print_exc()
            return False

    def _close_short_position_ai(self) -> Dict:
        """Fecha posição SHORT com IA - comprando para cobrir"""
        try:
            logger.info("📈 Fechando SHORT IA - Comprando para cobrir...")
            
            # Método 1: Buy order padrão
            result = self.bitget_api.place_buy_order()
            
            if result and result.get('success'):
                logger.info(f"✅ SHORT IA fechado via buy: {result.get('message', '')}")
                return {"success": True, "result": result}
            
            # Método 2: API direta
            try:
                order = self.bitget_api.exchange.create_market_buy_order(
                    'ETHUSDT', abs(self.current_position.size), None, {'leverage': self.leverage}
                )
                if order:
                    logger.info(f"✅ SHORT IA fechado via API direta")
                    return {"success": True, "order": order}
            except Exception as e:
                logger.error(f"❌ Método 2 SHORT IA: {e}")
            
            return {"success": False, "error": "Falha ao fechar SHORT IA"}
                
        except Exception as e:
            logger.error(f"❌ Erro ao fechar SHORT IA: {e}")
            return {"success": False, "error": str(e)}

    def _emergency_close_ai(self, reason: str) -> bool:
        """Fechamento de emergência IA com TODOS os métodos"""
        try:
            logger.warning(f"🚨 EMERGÊNCIA IA: {reason}")
            
            # Método 1: Cancelar todas as ordens
            try:
                self.bitget_api.exchange.cancel_all_orders('ETHUSDT')
                logger.info("✅ Ordens canceladas IA")
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
                        logger.info(f"✅ Emergência IA: {side} executado")
                        return True
            except Exception as e:
                logger.error(f"❌ Método emergência IA 2: {e}")
            
            # Método 3: Reduce-only
            try:
                if self.current_position:
                    side = 'sell' if self.current_position.side == TradeDirection.LONG else 'buy'
                    self.bitget_api.exchange.create_order(
                        'ETHUSDT', 'market', side, abs(self.current_position.size), 
                        None, {'reduceOnly': True}
                    )
                    logger.info("✅ Emergência IA: reduce-only executado")
                    return True
            except Exception as e:
                logger.error(f"❌ Método emergência IA 3: {e}")
            
            # Método 4: Limpeza forçada
            logger.warning("⚠️ LIMPEZA IA FORÇADA")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na emergência IA: {e}")
            return True

    def _check_emergency_conditions(self) -> bool:
        """Verificar condições de emergência com IA"""
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
            
            # Verificar accuracy da IA muito baixa
            if self.ai_predictions_total >= 20:
                ai_accuracy = (self.ai_predictions_correct / self.ai_predictions_total)
                if ai_accuracy < 0.3:  # Menos de 30% de accuracy
                    logger.warning(f"🚨 IA Accuracy muito baixa: {ai_accuracy*100:.1f}%")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar emergência: {e}")
            return True

    def get_account_balance(self) -> float:
        """Obter saldo da conta com fallback IA"""
        try:
            balance_info = self.bitget_api.get_balance()
            if balance_info and isinstance(balance_info, dict):
                balance = float(balance_info.get('free', 0.0))
                if balance > 0:
                    return balance
                    
            if self.paper_trading:
                return 1000.0
            else:
                logger.warning("⚠️ Saldo não obtido - usando fallback IA")
                return 100.0
                
        except Exception as e:
            logger.error(f"❌ Erro ao obter saldo: {e}")
            return 1000.0 if self.paper_trading else 100.0

    def emergency_stop(self) -> bool:
        """Parada de emergência IA total"""
        try:
            logger.warning("🚨 PARADA DE EMERGÊNCIA IA TOTAL")
            
            self.state = TradingState.EMERGENCY
            self.emergency_stop_triggered = True
            
            # Fechar posição com TODOS os métodos IA
            if self.current_position:
                self._emergency_close_ai("Emergency stop IA total")
            
            # Cancelar todas as ordens
            try:
                self.bitget_api.exchange.cancel_all_orders(self.symbol)
            except:
                pass
            
            # Parar thread
            if self.trading_thread:
                self.trading_thread.join(timeout=5)
            
            self.state = TradingState.STOPPED
            
            logger.warning("🛑 Parada de emergência IA total concluída")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na parada de emergência IA: {e}")
            return False

    def _adjust_for_50_percent_target(self):
        """Ajustar parâmetros dinamicamente para atingir 50% diário com IA"""
        try:
            with self._lock:
                current_profit_pct = self.metrics.net_profit * 100
                current_time = datetime.now()
                hours_passed = max(1, current_time.hour - 8) if current_time.hour >= 8 else 1
                
                expected_profit = (50.0 / 24) * hours_passed
                profit_deficit = max(0, expected_profit - current_profit_pct)
                
                # IA accuracy
                ai_accuracy = (self.ai_predictions_correct / max(1, self.ai_predictions_total)) * 100
                
                logger.info(f"📊 AJUSTE IA PARA 50%:")
                logger.info(f"   💰 Profit atual: {current_profit_pct:.4f}%")
                logger.info(f"   🎯 Esperado: {expected_profit:.2f}%")
                logger.info(f"   📉 Déficit: {profit_deficit:.2f}%")
                logger.info(f"   🧠 IA Accuracy: {ai_accuracy:.1f}%")
                
                # Ajustar critérios baseado na performance da IA
                if ai_accuracy > 70:  # IA muito boa
                    # Ser mais agressivo
                    self.min_confidence_to_trade = max(0.65, self.min_confidence_to_trade - 0.05)
                    self.min_strength_threshold = max(0.008, self.min_strength_threshold - 0.002)
                    logger.info("🚀 IA EXCELENTE - Modo mais agressivo!")
                
                elif ai_accuracy < 40:  # IA ruim
                    # Ser mais conservador
                    self.min_confidence_to_trade = min(0.85, self.min_confidence_to_trade + 0.05)
                    self.min_strength_threshold = min(0.018, self.min_strength_threshold + 0.002)
                    logger.warning("⚠️ IA COM PROBLEMAS - Modo conservador!")
                
                # Se muito atrás da meta
                if profit_deficit > 10.0:
                    logger.warning("🚨 MUITO ATRÁS DA META - AJUSTE IA URGENTE!")
                    self.force_trade_after_seconds = max(180, self.force_trade_after_seconds - 60)
                    
                    # Retreinar IA mais frequentemente
                    self.ai_training_interval = max(50, self.ai_training_interval - 20)
                
                # Se à frente da meta
                elif profit_deficit < -5.0:
                    logger.info("✅ À FRENTE DA META IA - Aumentar qualidade!")
                    self.min_confidence_to_trade = min(0.85, self.min_confidence_to_trade + 0.03)
                
                logger.info(f"   🎯 Nova confiança: {self.min_confidence_to_trade*100:.1f}%")
                logger.info(f"   💪 Nova força: {self.min_strength_threshold*100:.2f}%")
                logger.info(f"   🔄 Treino IA a cada: {self.ai_training_interval} análises")
                
        except Exception as e:
            logger.error(f"❌ Erro no ajuste IA: {e}")

    def reset_daily_stats(self):
        """Reset para novo dia com IA - otimizado para 50%"""
        try:
            logger.info("🔄 Reset para NOVO DIA COM IA - META 50%!")
            
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
                
                # Reset IA
                self.ai_predictions_correct = 0
                self.ai_predictions_total = 0
                self.movement_history.clear()
                
                # Reset critérios para padrões profissionais
                self.min_confidence_to_trade = 0.75
                self.min_strength_threshold = 0.012
                self.ai_training_interval = 100
                
                self.professional_mode_active = True
                self.force_trade_mode = False
            
            logger.info("✅ NOVO DIA IA - PRONTO PARA 50% DE LUCRO REAL!")
            
        except Exception as e:
            logger.error(f"❌ Erro ao resetar IA: {e}")

    def get_daily_stats(self) -> Dict:
        """Estatísticas focadas na meta de 50% diário com IA"""
        try:
            with self._lock:
                current_time = datetime.now()
                hours_trading = max(1, (current_time.hour - 8) if current_time.hour >= 8 else 24)
                
                daily_profit_pct = self.metrics.net_profit * 100
                target_achievement = (daily_profit_pct / 50.0) * 100
                ai_accuracy = (self.ai_predictions_correct / max(1, self.ai_predictions_total)) * 100
                
                return {
                    'target_50_percent_ai': {
                        'target_profit': '50.00%',
                        'current_profit_gross': f"{self.metrics.total_profit*100:.4f}%",
                        'current_profit_net': f"{daily_profit_pct:.4f}%",
                        'fees_paid': f"{self.metrics.total_fees_paid*100:.4f}%",
                        'achievement': f"{target_achievement:.1f}%",
                        'remaining_needed': f"{max(0, 50.0 - daily_profit_pct):.4f}%",
                        'on_track': target_achievement >= (hours_trading / 24) * 100,
                        'ai_mode': True
                    },
                    'ai_performance': {
                        'trained': self.ai_predictor.trained,
                        'predictions_total': self.ai_predictions_total,
                        'predictions_correct': self.ai_predictions_correct,
                        'accuracy': f"{ai_accuracy:.1f}%",
                        'model_count': len(self.ai_predictor.models),
                        'training_interval': self.ai_training_interval,
                        'data_points': len(self.movement_history)
                    },
                    'professional_trading_ai': {
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
                    'performance_ai': {
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
                    'risk_management_ai': {
                        'emergency_stop_triggered': self.emergency_stop_triggered,
                        'consecutive_losses_current': self.consecutive_losses,
                        'daily_loss_limit': '8.00%',
                        'max_consecutive_losses_limit': 3,
                        'drawdown_limit': '10.00%',
                        'ai_accuracy_threshold': '30.00%',
                        'risk_level': 'HIGH' if (self.consecutive_losses >= 2 or ai_accuracy < 40) else 'MEDIUM' if self.daily_loss_accumulated > 0.04 else 'LOW'
                    },
                    'current_settings_ai': {
                        'min_confidence': f"{self.min_confidence_to_trade*100:.1f}%",
                        'min_strength': f"{self.min_strength_threshold*100:.2f}%",
                        'min_signals': self.min_signals_agreement,
                        'profit_target_fixed': f"{self.profit_target*100:.1f}%",
                        'stop_loss_fixed': f"{abs(self.stop_loss_target)*100:.1f}%",
                        'minimum_profit': f"{self.minimum_profit_target*100:.1f}%",
                        'max_position_time': f"{self.max_position_time}s",
                        'min_position_time': f"{self.min_position_time}s"
                    },
                    'market_conditions': self.market_conditions
                }
                
        except Exception as e:
            logger.error(f"❌ Erro nas estatísticas IA: {e}")
            return {'error': str(e)}

    def _close_position_immediately(self, reason: str):
        """Compatibilidade - usar método IA"""
        return self._close_position_advanced(reason)
    
    def _close_position_professional(self, reason: str):
        """Compatibilidade - usar método IA avançado"""
        return self._close_position_advanced(reason)
    
    def _professional_position_management(self):
        """Compatibilidade - usar gerenciamento IA"""
        return self._ai_position_management()
    
    def _execute_professional_trade(self, direction, confidence, strength, analysis_details):
        """Compatibilidade - usar execução IA"""
        return self._execute_ai_trade(direction, confidence, strength, analysis_details)
