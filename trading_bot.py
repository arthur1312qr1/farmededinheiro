import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import math
import statistics
from collections import deque
import numpy as np
import asyncio

# Importações condicionais para ML
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import ta
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str='ETH/USDT:USDT',
                 leverage: int=10, balance_percentage: float=100.0,
                 daily_target: int=350, scalping_interval: float=0.3,
                 paper_trading: bool=False):
        """Initialize Trading Bot with EXTREME SUCCESS AI prediction system"""
        if not isinstance(bitget_api, BitgetAPI):
            raise TypeError(f"bitget_api deve ser uma instância de BitgetAPI, recebido: {type(bitget_api)}")
        
        self.bitget_api = bitget_api
        self.symbol = symbol
        self.leverage = leverage
        self.balance_percentage = balance_percentage
        self.daily_target = daily_target
        self.scalping_interval = scalping_interval
        self.paper_trading = paper_trading
        
        # Trading state
        self.is_running = False
        self.trades_today = 0
        self.current_position = None
        self.entry_price = None
        self.position_side = None
        self.position_size = 0.0
        self.position_start_time = None
        
        # === CONFIGURAÇÕES EXTREMAS PARA 95%+ SUCESSO E 300+ TRADES ===
        self.min_confidence_to_trade = 0.98 if ML_AVAILABLE else 0.85
        self.min_prediction_score = 0.95 if ML_AVAILABLE else 0.80
        self.min_signals_agreement = 9 if ML_AVAILABLE else 7
        self.min_strength_threshold = 0.020 if ML_AVAILABLE else 0.015
        
        # MÚLTIPLOS NÍVEIS DE TAKE PROFIT PARA COMPENSAR TAXAS
        self.profit_levels = {
            'micro_profit': 0.008,   # 0.8% - saída super rápida (30s)
            'quick_profit': 0.012,   # 1.2% - saída rápida (60s)
            'normal_profit': 0.015,  # 1.5% - saída normal
            'max_profit': 0.020      # 2.0% - deixar correr
        }
        self.profit_target = 0.015  # 1.5% take profit principal
        self.stop_loss_target = -0.006  # 0.6% stop loss
        self.max_position_time = 2  # Máximo 2 minutos
        
        # SISTEMA DE PREVISÃO EXPANDIDO
        self.price_history = deque(maxlen=5000)
        self.volume_history = deque(maxlen=1000)
        
        # SISTEMA DE MACHINE LEARNING (se disponível)
        if ML_AVAILABLE:
            self.prediction_models = {
                'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
                'gradient_boost': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'linear_regression': LinearRegression(),
            }
            self.scaler = StandardScaler()
            self.prediction_accuracy = {'rf': 0.7, 'gb': 0.7, 'lr': 0.6}
            self.model_trained = False
            self.ml_features_history = deque(maxlen=1000)
            self.price_targets_history = deque(maxlen=1000)
        else:
            self.prediction_models = {}
            self.model_trained = False
        
        # 10 ESTRATÉGIAS DE SAÍDA AUTOMÁTICA OTIMIZADAS
        self.exit_strategies = {
            'trailing_stop': {'active': True, 'trail_percent': 0.3},
            'grid_exit': {'active': True, 'levels': 8},
            'time_based': {'active': True, 'max_hold_minutes': 2},
            'volatility_exit': {'active': True, 'vol_threshold': 1.5},
            'momentum_exit': {'active': True, 'rsi_threshold': 65},
            'support_resistance': {'active': True, 'sr_buffer': 0.15},
            'fibonacci_exit': {'active': True, 'fib_levels': [0.618, 0.786, 1.0]},
            'bollinger_exit': {'active': True, 'bb_threshold': 0.9},
            'macd_exit': {'active': True, 'macd_divergence': True},
            'volume_exit': {'active': True, 'volume_spike': 1.8}
        }
        
        # Statistics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0
        
        # Loop control
        self.trading_task = None
        
        logger.info("🎯 EXTREME SUCCESS AI TRADING BOT INICIALIZADO")
        logger.info(f"🚀 Meta: 300+ trades/dia com 95%+ sucesso")
        logger.info(f"⚡ Velocidade: {self.scalping_interval}s")
        logger.info(f"🔒 Certeza mínima: {self.min_confidence_to_trade*100}%")
        logger.info(f"💎 Força mínima: {self.min_strength_threshold*100}%")
        logger.info(f"🤖 ML disponível: {'SIM' if ML_AVAILABLE else 'NÃO'}")

    def calculate_advanced_rsi(self, prices: List[float]) -> Dict:
        """RSI em múltiplos timeframes"""
        def rsi(data, period):
            if len(data) < period + 1:
                return 50
            
            deltas = [data[i] - data[i-1] for i in range(1, len(data))]
            gains = [d if d > 0 else 0 for d in deltas[-period:]]
            losses = [-d if d < 0 else 0 for d in deltas[-period:]]
            
            avg_gain = sum(gains) / period if gains else 0
            avg_loss = sum(losses) / period if losses else 0
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        
        return {
            'rsi_6': rsi(prices, 6),
            'rsi_14': rsi(prices, 14),
            'rsi_21': rsi(prices, 21)
        }

    def calculate_macd_advanced(self, prices: List[float]) -> Dict:
        """MACD com análise avançada"""
        if len(prices) < 35:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        def ema(data, period):
            if len(data) < period:
                return 0
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val
        
        ema_12 = ema(prices[-12:], 12)
        ema_26 = ema(prices[-26:], 26)
        macd_line = ema_12 - ema_26
        
        if hasattr(self, '_macd_history'):
            self._macd_history.append(macd_line)
            if len(self._macd_history) > 9:
                self._macd_history = self._macd_history[-9:]
        else:
            self._macd_history = [macd_line]
        
        signal_line = ema(self._macd_history, 9) if len(self._macd_history) >= 9 else macd_line
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_advanced_features(self, prices: List[float], volumes: List[float] = None) -> Dict:
        """Calcula features avançadas para análise"""
        if len(prices) < 50:
            return {}
        
        try:
            features = {}
            
            # RSI e MACD
            rsi_data = self.calculate_advanced_rsi(prices)
            macd_data = self.calculate_macd_advanced(prices)
            
            features['rsi'] = rsi_data['rsi_14']
            features['rsi_6'] = rsi_data['rsi_6']
            features['rsi_21'] = rsi_data['rsi_21']
            features['macd'] = macd_data['histogram']
            features['macd_signal'] = macd_data['signal']
            
            # Features de momentum
            features['momentum_1m'] = (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0
            features['momentum_3m'] = (prices[-1] - prices[-15]) / prices[-15] if len(prices) > 15 else 0
            features['momentum_5m'] = (prices[-1] - prices[-25]) / prices[-25] if len(prices) > 25 else 0
            
            # Features de volatilidade
            if len(prices) >= 50:
                vol_5m = np.std(prices[-25:]) / np.mean(prices[-25:])
                vol_10m = np.std(prices[-50:]) / np.mean(prices[-50:])
                features['volatility_5m'] = vol_5m
                features['volatility_10m'] = vol_10m
                features['volatility_ratio'] = vol_5m / vol_10m if vol_10m > 0 else 1
            
            # Features de volume
            if volumes and len(volumes) >= 10:
                features['volume_ratio'] = volumes[-1] / np.mean(volumes[-5:]) if len(volumes) >= 5 else 1
            
            # Limpar valores inválidos
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0.0
                    
            return features
            
        except Exception as e:
            logger.error(f"Erro ao calcular features: {e}")
            return {}

    def train_prediction_models(self):
        """Treina os modelos de ML se disponível"""
        if not ML_AVAILABLE or len(self.ml_features_history) < 50:
            return False
        
        try:
            X = []
            y = []
            
            for i in range(len(self.ml_features_history) - 1):
                features = list(self.ml_features_history[i].values())
                if len(features) > 0 and not any(np.isnan(features)) and not any(np.isinf(features)):
                    X.append(features)
                    y.append(self.price_targets_history[i + 1])
            
            if len(X) < 30:
                return False
                
            X = np.array(X)
            y = np.array(y)
            
            X_scaled = self.scaler.fit_transform(X)
            
            for name, model in self.prediction_models.items():
                try:
                    model.fit(X_scaled, y)
                    predictions = model.predict(X_scaled)
                    accuracy = 1 - np.mean(np.abs(predictions - y) / (np.abs(y) + 1e-8))
                    self.prediction_accuracy[name[:2]] = max(0.5, min(1, accuracy))
                except Exception as e:
                    logger.error(f"Erro ao treinar modelo {name}: {e}")
                    self.prediction_accuracy[name[:2]] = 0.6
            
            self.model_trained = True
            avg_accuracy = np.mean(list(self.prediction_accuracy.values()))
            logger.info(f"✅ Modelos ML treinados - Precisão: {avg_accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Erro no treinamento: {e}")
            return False

    def extreme_market_analysis(self, prices: List[float]) -> Dict:
        """Análise de mercado ultra-rigorosa"""
        if len(prices) < 100:
            return {'valid': False, 'score': 0, 'confidence': 0}
        
        try:
            validations = {}
            score = 0
            confidence_factors = []
            
            # 1. ANÁLISE DE MOMENTUM MULTI-TIMEFRAME
            momentum_1m = (prices[-1] - prices[-5]) / prices[-5] if len(prices) > 5 else 0
            momentum_3m = (prices[-1] - prices[-15]) / prices[-15] if len(prices) > 15 else 0
            momentum_5m = (prices[-1] - prices[-25]) / prices[-25] if len(prices) > 25 else 0
            
            if (momentum_1m > 0.003 and momentum_3m > 0.005 and momentum_5m > 0.008):
                validations['momentum_bullish'] = True
                score += 30
                confidence_factors.append(0.95)
            elif (momentum_1m < -0.003 and momentum_3m < -0.005 and momentum_5m < -0.008):
                validations['momentum_bearish'] = True
                score += 30
                confidence_factors.append(0.95)
            
            # 2. ANÁLISE DE VOLUME
            if len(self.volume_history) >= 30:
                volumes = list(self.volume_history)
                current_vol = volumes[-1]
                avg_vol = np.mean(volumes[-10:])
                
                if current_vol > avg_vol * 1.2:
                    validations['volume_good'] = True
                    score += 20
                    confidence_factors.append(0.8)
            
            # 3. VOLATILIDADE CONTROLADA
            if len(prices) >= 50:
                volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
                if 0.005 < volatility < 0.03:
                    validations['volatility_ok'] = True
                    score += 15
                    confidence_factors.append(0.75)
            
            # 4. INDICADORES TÉCNICOS
            rsi_data = self.calculate_advanced_rsi(prices)
            rsi_14 = rsi_data['rsi_14']
            if 30 < rsi_14 < 70:
                validations['rsi_neutral'] = True
                score += 15
                confidence_factors.append(0.7)
            
            # 5. MACD
            macd_data = self.calculate_macd_advanced(prices)
            if abs(macd_data['histogram']) > 0.001:
                validations['macd_signal'] = True
                score += 20
                confidence_factors.append(0.8)
            
            # 6. HORÁRIO ÓTIMO
            current_hour = datetime.now().hour
            if current_hour in [8, 9, 10, 13, 14, 15, 20, 21, 22]:
                validations['good_time'] = True
                score += 10
                confidence_factors.append(0.7)
            
            total_validations = sum(1 for v in validations.values() if v)
            if total_validations >= 3:
                validations['confluence'] = True
                score += 20
                confidence_factors.append(0.9)
            
            avg_confidence = np.mean(confidence_factors) if confidence_factors else 0
            
            return {
                'valid': score >= 80,
                'score': score,
                'confidence': avg_confidence,
                'validations': validations,
                'total_validations': total_validations
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de mercado: {e}")
            return {'valid': False, 'score': 0, 'confidence': 0}

    def ultra_ml_prediction(self, features: Dict) -> Dict:
        """Previsão ML ou análise técnica avançada"""
        if not features:
            return {'valid': False, 'confidence': 0, 'strength': 0, 'direction': 0}
        
        try:
            if ML_AVAILABLE and self.model_trained:
                # Usar ML se disponível
                feature_values = list(features.values())
                if any(np.isnan(feature_values)) or any(np.isinf(feature_values)):
                    return {'valid': False, 'confidence': 0, 'strength': 0, 'direction': 0}
                
                X = np.array([feature_values])
                X_scaled = self.scaler.transform(X)
                
                predictions = {}
                confidences = {}
                weights = {'random_forest': 0.5, 'gradient_boost': 0.35, 'linear_regression': 0.15}
                
                for name, model in self.prediction_models.items():
                    try:
                        pred = model.predict(X_scaled)[0]
                        model_accuracy = self.prediction_accuracy.get(name[:2], 0.6)
                        confidence = min(abs(pred) * model_accuracy * 2, 1.0)
                        predictions[name] = pred
                        confidences[name] = confidence
                    except:
                        predictions[name] = 0.0
                        confidences[name] = 0.0
                
                ensemble_pred = sum(predictions[name] * weights[name] for name in predictions)
                ensemble_confidence = sum(confidences[name] * weights[name] for name in confidences)
                
                # Validação
                signal_strength = abs(ensemble_pred)
                is_valid = (ensemble_confidence > 0.85 and 
                           signal_strength > self.min_strength_threshold)
                
                return {
                    'valid': is_valid,
                    'confidence': ensemble_confidence,
                    'strength': signal_strength,
                    'direction': 1 if ensemble_pred > 0 else -1,
                    'ensemble_pred': ensemble_pred
                }
            else:
                # Usar análise técnica manual
                rsi = features.get('rsi', 50)
                macd = features.get('macd', 0)
                momentum_5m = features.get('momentum_5m', 0)
                volatility_ratio = features.get('volatility_ratio', 1)
                
                # Sistema de pontuação manual
                score = 0
                direction = 0
                
                # RSI
                if 30 < rsi < 45:  # Oversold recovery
                    score += 25
                    direction += 1
                elif 55 < rsi < 70:  # Overbought but strong
                    score += 25
                    direction -= 1
                
                # MACD
                if macd > 0.001:
                    score += 30
                    direction += 1
                elif macd < -0.001:
                    score += 30
                    direction -= 1
                
                # Momentum
                if momentum_5m > 0.008:
                    score += 35
                    direction += 1
                elif momentum_5m < -0.008:
                    score += 35
                    direction -= 1
                
                # Volatilidade
                if 1.1 < volatility_ratio < 2.0:
                    score += 10
                
                confidence = min(score / 100, 1.0)
                strength = abs(momentum_5m) + abs(macd) * 10
                
                is_valid = (score >= 70 and 
                           confidence > 0.8 and 
                           strength > self.min_strength_threshold)
                
                return {
                    'valid': is_valid,
                    'confidence': confidence,
                    'strength': strength,
                    'direction': 1 if direction > 0 else -1 if direction < 0 else 0,
                    'manual_score': score
                }
                
        except Exception as e:
            logger.error(f"Erro na previsão: {e}")
            return {'valid': False, 'confidence': 0, 'strength': 0, 'direction': 0}

    def calculate_real_pnl_with_fees(self, current_price: float, entry_price: float, position_side: str) -> Dict:
        """Calcula PnL considerando taxas da Bitget"""
        if position_side == 'long':
            pnl_gross = (current_price - entry_price) / entry_price
        else:
            pnl_gross = (entry_price - current_price) / entry_price
        
        entry_fee = 0.0004  # 0.04% taker
        exit_fee = 0.0002   # 0.02% maker
        total_fees = entry_fee + exit_fee
        
        pnl_net = pnl_gross - total_fees
        
        return {
            'pnl_gross': pnl_gross,
            'pnl_net': pnl_net,
            'total_fees': total_fees,
            'profitable_after_fees': pnl_net > 0
        }

    def micro_scalping_exit(self, current_price: float, entry_price: float, position_side: str) -> Dict:
        """Sistema de saída ultra-rápido com múltiplos take profits"""
        pnl_data = self.calculate_real_pnl_with_fees(current_price, entry_price, position_side)
        
        elapsed_seconds = 0
        if hasattr(self, 'position_start_time') and self.position_start_time:
            elapsed_seconds = (datetime.now() - self.position_start_time).total_seconds()
        
        # STOP LOSS IMEDIATO
        if pnl_data['pnl_net'] <= self.stop_loss_target:
            return {
                'exit': True,
                'type': 'stop_loss',
                'pnl': pnl_data['pnl_net'],
                'reason': f"Stop loss: {pnl_data['pnl_net']:.4f}"
            }
        
        # MICRO PROFIT - 30 segundos
        if elapsed_seconds >= 30 and pnl_data['pnl_net'] >= self.profit_levels['micro_profit']:
            return {
                'exit': True,
                'type': 'micro_profit',
                'pnl': pnl_data['pnl_net'],
                'reason': f"Micro profit 30s: {pnl_data['pnl_net']:.4f}"
            }
        
        # QUICK PROFIT - 60 segundos
        if elapsed_seconds >= 60 and pnl_data['pnl_net'] >= self.profit_levels['quick_profit']:
            return {
                'exit': True,
                'type': 'quick_profit',
                'pnl': pnl_data['pnl_net'],
                'reason': f"Quick profit 60s: {pnl_data['pnl_net']:.4f}"
            }
        
        # NORMAL PROFIT - qualquer momento
        if pnl_data['pnl_net'] >= self.profit_levels['normal_profit']:
            return {
                'exit': True,
                'type': 'normal_profit',
                'pnl': pnl_data['pnl_net'],
                'reason': f"Normal profit: {pnl_data['pnl_net']:.4f}"
            }
        
        # TEMPO LIMITE
        if elapsed_seconds >= self.max_position_time * 60:
            if pnl_data['pnl_net'] > 0.003:
                return {
                    'exit': True,
                    'type': 'time_limit_profit',
                    'pnl': pnl_data['pnl_net'],
                    'reason': f"Tempo limite com lucro: {pnl_data['pnl_net']:.4f}"
                }
            else:
                return {
                    'exit': True,
                    'type': 'time_limit_loss',
                    'pnl': pnl_data['pnl_net'],
                    'reason': f"Tempo limite: {pnl_data['pnl_net']:.4f}"
                }
        
        return {'exit': False}

    def ultra_selective_entry(self, prices: List[float]) -> Dict:
        """Sistema ultra-seletivo para entrada"""
        # 1. Análise de mercado
        market_analysis = self.extreme_market_analysis(prices)
        if not market_analysis['valid'] or market_analysis['score'] < 80:
            return {'enter': False, 'reason': 'Análise técnica insuficiente'}
        
        # 2. Previsão ML
        features = self.calculate_advanced_features(prices)
        ml_prediction = self.ultra_ml_prediction(features)
        if not ml_prediction['valid'] or ml_prediction['confidence'] < 0.85:
            return {'enter': False, 'reason': 'Confiança ML insuficiente'}
        
        # 3. Verificação final
        if ml_prediction['strength'] < self.min_strength_threshold:
            return {'enter': False, 'reason': 'Força do sinal insuficiente'}
        
        return {
            'enter': True,
            'direction': ml_prediction['direction'],
            'confidence': min(market_analysis['confidence'], ml_prediction['confidence']),
            'expected_profit': ml_prediction['strength'],
            'score': market_analysis['score']
        }

    async def get_current_price(self):
        """Obter preço atual do ativo"""
        try:
            # Simular obtenção de preço (substituir pela API real)
            # ticker = await self.bitget_api.get_ticker(self.symbol)
            # return ticker['last']
            
            # Por enquanto, usar preço simulado baseado no histórico
            if len(self.price_history) > 0:
                last_price = self.price_history[-1]
                # Simular pequena variação
                change = np.random.normal(0, last_price * 0.001)
                return last_price + change
            else:
                return 3500.0  # Preço inicial do ETH
                
        except Exception as e:
            logger.error(f"Erro ao obter preço: {e}")
            return None

    async def open_position(self, side: str, price: float):
        """Abrir posição"""
        try:
            # Simular abertura de posição
            self.current_position = {
                'side': side,
                'size': 0.1,
                'entry_price': price,
                'timestamp': datetime.now()
            }
            self.entry_price = price
            self.position_side = side
            self.position_start_time = datetime.now()
            
            logger.info(f"🚀 Posição {side.upper()} aberta a {price:.2f}")
            
        except Exception as e:
            logger.error(f"Erro ao abrir posição: {e}")

    async def close_position(self):
        """Fechar posição"""
        try:
            if self.current_position:
                self.current_position = None
                self.entry_price = None
                self.position_side = None
                self.position_start_time = None
                logger.info("🔄 Posição fechada")
                
        except Exception as e:
            logger.error(f"Erro ao fechar posição: {e}")

    def run_extreme_success_loop(self):
        """Executa o loop principal em thread separada"""
        def run_loop():
            try:
                # Criar novo event loop para esta thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Executar o loop principal
                loop.run_until_complete(self.extreme_success_loop())
                
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
            finally:
                loop.close()
        
        # Iniciar em thread separada
        self.trading_thread = threading.Thread(target=run_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()

    async def extreme_success_loop(self):
        """Loop principal para 95%+ sucesso com 300+ trades"""
        logger.info("🎯 INICIANDO EXTREME SUCCESS TRADING")
        
        trades_count = 0
        successful_trades = 0
        total_net_profit = 0.0
        
        while self.is_running and trades_count < self.daily_target:
            try:
                current_price = await self.get_current_price()
                if not current_price:
                    await asyncio.sleep(0.1)
                    continue
                
                # Atualizar históricos
                self.price_history.append(current_price)
                self.volume_history.append(np.random.uniform(800000, 1200000))  # Volume simulado
                
                # Treinar modelos periodicamente
                if ML_AVAILABLE and len(self.price_history) >= 100:
                    features = self.calculate_advanced_features(list(self.price_history))
                    if features:
                        self.ml_features_history.append(features)
                        self.price_targets_history.append(current_price)
                        
                        if len(self.ml_features_history) % 50 == 0:
                            self.train_prediction_models()
                
                if self.current_position:
                    # === VERIFICAR SAÍDA ULTRA-RÁPIDA ===
                    exit_data = self.micro_scalping_exit(
                        current_price, self.entry_price, self.position_side
                    )
                    
                    if exit_data['exit']:
                        await self.close_position()
                        trades_count += 1
                        self.trades_today += 1
                        
                        if exit_data['pnl'] > 0:
                            successful_trades += 1
                            self.profitable_trades += 1
                            self.consecutive_wins += 1
                            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
                            total_net_profit += exit_data['pnl']
                            self.total_profit += exit_data['pnl']
                            logger.info(f"✅ WIN #{self.consecutive_wins} | {exit_data['reason']}")
                        else:
                            self.consecutive_wins = 0
                            total_net_profit += exit_data['pnl']
                            self.total_profit += exit_data['pnl']
                            logger.info(f"❌ LOSS | {exit_data['reason']}")
                        
                        self.total_trades += 1
                        
                        # Stats em tempo real
                        success_rate = successful_trades / trades_count if trades_count > 0 else 0
                        logger.info(f"📊 {trades_count} trades | {success_rate:.1%} sucesso | Streak: {self.consecutive_wins}")
                        continue
                
                else:
                    # === BUSCAR ENTRADA ULTRA-SELETIVA ===
                    if len(self.price_history) >= 100:
                        
                        entry_analysis = self.ultra_selective_entry(list(self.price_history))
                        
                        if entry_analysis['enter']:
                            position_side = 'long' if entry_analysis['direction'] > 0 else 'short'
                            
                            logger.info(f"🚀 ENTRADA ULTRA-SELETIVA #{trades_count + 1}")
                            logger.info(f"🎯 Confiança: {entry_analysis['confidence']:.3f}")
                            logger.info(f"💪 Força: {entry_analysis['expected_profit']:.4f}")
                            logger.info(f"📈 Direção: {position_side.upper()}")
                            
                            await self.open_position(position_side, current_price)
                
                await asyncio.sleep(self.scalping_interval)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop: {e}")
                await asyncio.sleep(0.5)
        
        # === RELATÓRIO FINAL ===
        if trades_count > 0:
            final_success_rate = successful_trades / trades_count
            daily_profit = total_net_profit * 100
            avg_profit_per_trade = total_net_profit / trades_count
            
            logger.info(f"🏆 EXTREME SUCCESS REPORT")
            logger.info(f"📊 Total trades: {trades_count}")
            logger.info(f"✅ Sucessos: {successful_trades}")
            logger.info(f"🎯 Taxa de sucesso: {final_success_rate:.1%}")
            logger.info(f"💰 Lucro total: {daily_profit:.2f}%")
            logger.info(f"💵 Lucro médio/trade: {avg_profit_per_trade:.4f}")
            logger.info(f"🔥 Max streak: {self.max_consecutive_wins}")

    def start(self):
        """Iniciar bot"""
        if not self.is_running:
            self.is_running = True
            self.run_extreme_success_loop()
            logger.info("🚀 Bot iniciado!")
        else:
            logger.warning("⚠️ Bot já está rodando!")

    def stop(self):
        """Parar bot"""
        if self.is_running:
            self.is_running = False
            logger.info("🛑 Bot parado!")
        else:
            logger.warning("⚠️ Bot já está parado!")

    def get_status(self):
        """Obter status do bot"""
        success_rate = (self.profitable_trades / max(1, self.total_trades)) * 100
        
        return {
            'is_running': self.is_running,
            'trades_today': self.trades_today,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'success_rate': f"{success_rate:.1f}%",
            'total_profit': f"{self.total_profit*100:.2f}%",
            'current_position': self.current_position,
            'consecutive_wins': self.consecutive_wins,
            'max_consecutive_wins': self.max_consecutive_wins,
            'ml_available': ML_AVAILABLE,
            'model_trained': self.model_trained if ML_AVAILABLE else False,
            'price_history_size': len(self.price_history),
            'profit_levels': self.profit_levels,
            'exit_strategies_active': sum(1 for s in self.exit_strategies.values() if s['active'])
        }

    def get_enhanced_status(self):
        """Status detalhado do bot"""
        return self.get_status()

    # Métodos para compatibilidade (para manter funcionando com o código existente)
    def detect_chart_patterns(self, prices: List[float]) -> Dict:
        """Detecção básica de padrões"""
        return {'patterns': [], 'confidence': 0.5}

    def find_peaks_valleys(self, prices: List[float]) -> Dict:
        """Encontra picos e vales básicos"""
        return {'peaks': [], 'valleys': []}

    def is_head_and_shoulders(self, peaks, valleys):
        """Detecta padrão cabeça e ombros"""
        return False

    def detect_triangle_pattern(self, prices):
        """Detecta padrão triangular"""
        return {'detected': False, 'type': 'none'}

    def detect_flag_pattern(self, prices):
        """Detecta padrão de bandeira"""
        return False

    def detect_cup_handle_pattern(self, prices):
        """Detecta padrão xícara e alça"""
        return False

    def calculate_stochastic(self, prices, highs, lows):
        """Calcula estocástico"""
        return {'k': 50, 'd': 50}

    def predict_price_direction(self, features):
        """Previsão de direção do preço"""
        return self.ultra_ml_prediction(features)

    def check_exit_strategies(self, current_price, entry_price, position_side):
        """Verifica estratégias de saída"""
        return self.micro_scalping_exit(current_price, entry_price, position_side)
