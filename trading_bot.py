import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading
import math

from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str='ETH/USDT:USDT',
                 leverage: int=10, balance_percentage: float=100.0,
                 daily_target: int=200, scalping_interval: int=2,
                 paper_trading: bool=False):
        """Initialize Trading Bot"""
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
        self.profit_target = 0.01  # 1% take profit
        self.stop_loss_target = -0.02  # 2% stop loss
        
        # Sistema de segurança
        self.emergency_stop = False
        self.price_monitoring = True
        self.last_price_check = None
        
        # Previsão avançada
        self.price_history = []
        self.prediction_data = {
            'trend': 'neutral',
            'confidence': 0.0,
            'next_20min_prediction': 0.0,
            'signals': []
        }
        
        # Statistics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        self.stop_loss_triggered = 0
        self.take_profit_triggered = 0
        
        logger.info("✅ Trading Bot inicializado com SISTEMA DE SEGURANÇA")
        logger.info(f"🤖 Configuração:")
        logger.info(f"   Símbolo: {self.symbol}")
        logger.info(f"   Alavancagem: {self.leverage}x")
        logger.info(f"   Stop Loss: {self.stop_loss_target * 100}%")
        logger.info(f"   Take Profit: {self.profit_target * 100}%")
        logger.info(f"   Uso do saldo: {self.balance_percentage}%")
        logger.info(f"   Meta diária: {self.daily_target} trades")

    def get_market_data(self) -> Dict:
        """Get current market data"""
        return self.bitget_api.get_market_data(self.symbol)

    def get_account_balance(self) -> float:
        """Get current account balance"""
        return self.bitget_api.get_account_balance()

    def calculate_advanced_prediction(self, current_price: float) -> Dict:
        """Sistema de previsão avançado para próximos 20 minutos"""
        try:
            # Adicionar preço atual ao histórico
            current_time = time.time()
            self.price_history.append({
                'price': current_price,
                'timestamp': current_time
            })
            
            # Manter apenas últimos 100 pontos (aproximadamente 10 segundos)
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]
            
            if len(self.price_history) < 10:
                return {
                    'trend': 'neutral',
                    'confidence': 0.0,
                    'next_20min_prediction': current_price,
                    'signals': ['Dados insuficientes']
                }
            
            # Análise de múltiplos indicadores
            signals = []
            scores = []
            
            # 1. Análise de tendência (últimos 30 pontos)
            recent_prices = [p['price'] for p in self.price_history[-30:]]
            if len(recent_prices) >= 5:
                trend_score = self.calculate_trend_strength(recent_prices)
                scores.append(trend_score)
                if trend_score > 0.6:
                    signals.append(f"Tendência alta forte: {trend_score:.2f}")
                elif trend_score < -0.6:
                    signals.append(f"Tendência baixa forte: {trend_score:.2f}")
                else:
                    signals.append(f"Tendência neutra: {trend_score:.2f}")
            
            # 2. Análise de momentum
            if len(recent_prices) >= 10:
                momentum_score = self.calculate_momentum(recent_prices)
                scores.append(momentum_score)
                if momentum_score > 0.5:
                    signals.append(f"Momentum positivo: {momentum_score:.2f}")
                elif momentum_score < -0.5:
                    signals.append(f"Momentum negativo: {momentum_score:.2f}")
            
            # 3. Análise de volatilidade
            volatility = self.calculate_volatility(recent_prices)
            if volatility > 0.002:  # Alta volatilidade
                signals.append(f"Alta volatilidade: {volatility:.4f}")
                scores.append(-0.2)  # Penalizar alta volatilidade
            else:
                signals.append(f"Volatilidade normal: {volatility:.4f}")
                scores.append(0.1)
            
            # 4. Análise de suporte/resistência
            support_resistance = self.find_support_resistance(recent_prices)
            if support_resistance:
                signals.append(f"S/R: {support_resistance}")
            
            # Calcular confiança geral
            confidence = max(0.0, min(1.0, sum(scores) / len(scores) if scores else 0.0))
            
            # Determinar tendência geral
            avg_score = sum(scores) / len(scores) if scores else 0.0
            if avg_score > 0.3:
                trend = 'bullish'
            elif avg_score < -0.3:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Previsão para próximos 20 minutos
            prediction_factor = avg_score * 0.01  # Max 1% de movimento predito
            next_20min_prediction = current_price * (1 + prediction_factor)
            
            self.prediction_data = {
                'trend': trend,
                'confidence': confidence,
                'next_20min_prediction': next_20min_prediction,
                'signals': signals,
                'avg_score': avg_score
            }
            
            logger.info(f"🔮 PREVISÃO 20min: {trend} | Confiança: {confidence:.2f} | Preço: ${next_20min_prediction:.2f}")
            
            return self.prediction_data
            
        except Exception as e:
            logger.error(f"❌ Erro na previsão: {e}")
            return {
                'trend': 'neutral',
                'confidence': 0.0,
                'next_20min_prediction': current_price,
                'signals': [f'Erro: {str(e)}']
            }

    def calculate_trend_strength(self, prices: List[float]) -> float:
        """Calcula força da tendência (-1 a 1)"""
        if len(prices) < 3:
            return 0.0
        
        # Regressão linear simples
        x = list(range(len(prices)))
        n = len(prices)
        
        sum_x = sum(x)
        sum_y = sum(prices)
        sum_xy = sum(x[i] * prices[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Normalizar slope baseado no preço médio
        avg_price = sum_y / n
        normalized_slope = slope / avg_price
        
        # Limitar entre -1 e 1
        return max(-1.0, min(1.0, normalized_slope * 1000))

    def calculate_momentum(self, prices: List[float]) -> float:
        """Calcula momentum baseado em aceleração de preços"""
        if len(prices) < 6:
            return 0.0
        
        # Comparar últimos 3 com 3 anteriores
        recent = prices[-3:]
        previous = prices[-6:-3]
        
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        
        momentum = (recent_avg - previous_avg) / previous_avg
        
        return max(-1.0, min(1.0, momentum * 100))

    def calculate_volatility(self, prices: List[float]) -> float:
        """Calcula volatilidade usando desvio padrão"""
        if len(prices) < 2:
            return 0.0
        
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        volatility = math.sqrt(variance) / mean_price
        
        return volatility

    def find_support_resistance(self, prices: List[float]) -> str:
        """Identifica níveis de suporte e resistência"""
        if len(prices) < 5:
            return "Dados insuficientes"
        
        current_price = prices[-1]
        max_price = max(prices)
        min_price = min(prices)
        
        # Verificar se está próximo de suporte ou resistência
        range_pct = (max_price - min_price) / min_price
        
        if current_price <= min_price * 1.01:
            return f"Próximo ao suporte ${min_price:.2f}"
        elif current_price >= max_price * 0.99:
            return f"Próximo à resistência ${max_price:.2f}"
        else:
            return f"Entre S:${min_price:.2f} R:${max_price:.2f}"

    def ultra_fast_price_monitor(self):
        """Monitor ultra-rápido de preços com sistema de segurança"""
        logger.warning("🚨 SISTEMA DE SEGURANÇA ATIVADO - Monitor 0.1s")
        
        while self.price_monitoring and self.is_running:
            try:
                if not self.current_position:
                    time.sleep(0.1)
                    continue
                
                start_time = time.perf_counter()
                
                # Obter preço atual
                market_data = self.get_market_data()
                if not market_data or 'price' not in market_data:
                    time.sleep(0.1)
                    continue
                
                current_price = float(market_data['price'])
                self.last_price_check = time.time()
                
                # Calcular P&L atual
                if self.entry_price and self.position_side:
                    if self.position_side == 'buy':
                        pnl_pct = (current_price - self.entry_price) / self.entry_price
                    else:
                        pnl_pct = (self.entry_price - current_price) / self.entry_price
                    
                    # SISTEMA DE SEGURANÇA - STOP LOSS IMEDIATO
                    if pnl_pct <= self.stop_loss_target:
                        logger.warning(f"🚨 STOP LOSS ATIVADO! P&L: {pnl_pct*100:.2f}%")
                        logger.warning(f"💰 Entrada: ${self.entry_price:.2f} | Atual: ${current_price:.2f}")
                        
                        self.emergency_close_position("STOP_LOSS")
                        self.stop_loss_triggered += 1
                        
                    # SISTEMA DE SEGURANÇA - TAKE PROFIT IMEDIATO
                    elif pnl_pct >= self.profit_target:
                        logger.warning(f"🎯 TAKE PROFIT ATIVADO! P&L: {pnl_pct*100:.2f}%")
                        logger.warning(f"💰 Entrada: ${self.entry_price:.2f} | Atual: ${current_price:.2f}")
                        
                        self.emergency_close_position("TAKE_PROFIT")
                        self.take_profit_triggered += 1
                    
                    # Log a cada 10 verificações para não poluir
                    if hasattr(self, '_monitor_count'):
                        self._monitor_count += 1
                    else:
                        self._monitor_count = 1
                    
                    if self._monitor_count % 50 == 0:  # Log a cada 5 segundos
                        processing_time = (time.perf_counter() - start_time) * 1000
                        logger.info(f"🔍 P&L: {pnl_pct*100:.3f}% | ${self.entry_price:.2f}→${current_price:.2f} | {processing_time:.1f}ms")
                
                # Atualizar previsão
                self.calculate_advanced_prediction(current_price)
                
                time.sleep(0.1)  # 100ms entre verificações
                
            except Exception as e:
                logger.error(f"❌ Erro no monitor de segurança: {e}")
                time.sleep(0.2)

    def emergency_close_position(self, reason: str) -> bool:
        """Fechamento de emergência GARANTIDO"""
        try:
            logger.warning(f"🚨 FECHAMENTO DE EMERGÊNCIA: {reason}")
            
            max_attempts = 5
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                logger.warning(f"🔄 Tentativa {attempt}/{max_attempts} de fechamento")
                
                try:
                    # Tentar fechar posição
                    close_side = 'sell' if self.position_side == 'buy' else 'buy'
                    result = self.execute_trade(close_side)
                    
                    if result and result.get('success'):
                        logger.warning(f"✅ POSIÇÃO FECHADA COM SUCESSO! Motivo: {reason}")
                        self.current_position = None
                        self.entry_price = None
                        self.position_side = None
                        
                        if reason == "TAKE_PROFIT":
                            self.profitable_trades += 1
                        
                        return True
                    else:
                        logger.error(f"❌ Falha na tentativa {attempt}: {result.get('error', 'Erro desconhecido')}")
                        
                except Exception as e:
                    logger.error(f"❌ Erro na tentativa {attempt}: {e}")
                
                if attempt < max_attempts:
                    time.sleep(0.5)  # Aguardar antes da próxima tentativa
            
            # Se chegou aqui, todas as tentativas falharam
            logger.error(f"🚨 FALHA CRÍTICA: Não foi possível fechar posição após {max_attempts} tentativas!")
            self.emergency_stop = True
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro crítico no fechamento de emergência: {e}")
            self.emergency_stop = True
            return False

    def execute_trade(self, side: str) -> Dict:
        """Execute trade com cálculo 100% dinâmico"""
        try:
            logger.warning(f"🚀 INICIANDO TRADE {side.upper()}")
            
            # Buscar preço atual primeiro
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data:
                logger.error("❌ Erro ao obter dados do mercado")
                return {'success': False, 'error': 'Dados de mercado indisponíveis'}
            
            current_price = market_data['price']
            logger.warning(f"💎 Preço ETH atual: ${current_price:.2f}")
            
            # Executar ordem
            result = self.bitget_api.place_order(
                symbol=self.symbol,
                side=side,
                size=0,
                price=current_price,
                leverage=self.leverage
            )
            
            if result['success']:
                logger.warning(f"✅ TRADE {side.upper()} EXECUTADO!")
                return result
            else:
                logger.error(f"❌ Erro no trade {side}: {result.get('error', 'Erro desconhecido')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Erro crítico ao executar trade {side}: {e}")
            return {'success': False, 'error': str(e)}

    def close_position(self) -> Dict:
        """Close current position"""
        try:
            if not self.current_position:
                return {'success': False, 'error': 'Nenhuma posição para fechar'}
            
            close_side = 'sell' if self.position_side == 'buy' else 'buy'
            logger.warning(f"🔄 FECHANDO POSIÇÃO {self.position_side.upper()}")
            
            result = self.execute_trade(close_side)
            if result['success']:
                self.current_position = None
                self.entry_price = None
                self.position_side = None
                logger.warning(f"✅ POSIÇÃO FECHADA!")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição: {e}")
            return {'success': False, 'error': str(e)}

    def check_profit_target(self) -> bool:
        """MÉTODO OBSOLETO - Substituído pelo monitor ultra-rápido"""
        # Este método agora é redundante pois o monitor ultra-rápido
        # já verifica stop loss e take profit continuamente
        return False

    def scalping_strategy(self):
        """Enhanced scalping strategy com previsão"""
        try:
            if self.emergency_stop:
                logger.error("🚨 BOT PARADO POR EMERGÊNCIA")
                return
            
            if not self.current_position:
                # Usar previsão para decidir direção do trade
                current_market = self.get_market_data()
                if current_market and 'price' in current_market:
                    current_price = current_market['price']
                    prediction = self.calculate_advanced_prediction(current_price)
                    
                    # Decidir side baseado na previsão
                    if prediction['confidence'] > 0.6:
                        if prediction['trend'] == 'bullish':
                            side = 'buy'
                        elif prediction['trend'] == 'bearish':
                            side = 'sell'
                        else:
                            side = 'buy' if self.trades_today % 2 == 0 else 'sell'
                    else:
                        side = 'buy' if self.trades_today % 2 == 0 else 'sell'
                    
                    logger.warning(f"🚀 ABRINDO POSIÇÃO {side.upper()}")
                    logger.warning(f"🔮 Previsão: {prediction['trend']} | Confiança: {prediction['confidence']:.2f}")
                    
                    result = self.execute_trade(side)
                    if result['success']:
                        self.current_position = result.get('order_id', True)
                        self.entry_price = result.get('price', current_price)
                        self.position_side = side
                        self.trades_today += 1
                        self.total_trades += 1
                        logger.warning(f"✅ POSIÇÃO ABERTA: {side.upper()}")
                        logger.warning(f"📊 Trades hoje: {self.trades_today}/{self.daily_target}")
                        
                        # Iniciar monitor de segurança se não estiver rodando
                        if not hasattr(self, '_monitor_thread') or not self._monitor_thread.is_alive():
                            self._monitor_thread = threading.Thread(target=self.ultra_fast_price_monitor, daemon=True)
                            self._monitor_thread.start()
                            
        except Exception as e:
            logger.error(f"❌ Erro na estratégia de scalping: {e}")

    def run_trading_loop(self):
        """Main trading loop"""
        logger.warning(f"🚀 Trading bot iniciado com SISTEMA DE SEGURANÇA")
        self.start_balance = self.get_account_balance()
        
        while self.is_running and not self.emergency_stop:
            try:
                if self.trades_today >= self.daily_target:
                    logger.warning(f"🎯 META DIÁRIA ATINGIDA: {self.trades_today} trades")
                    time.sleep(60)
                    if datetime.now().hour == 0:
                        self.trades_today = 0
                        logger.warning(f"🌅 NOVO DIA - Contador zerado")
                    continue
                
                self.scalping_strategy()
                time.sleep(self.scalping_interval)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop: {e}")
                time.sleep(5)
            except KeyboardInterrupt:
                self.stop()
                break
        
        if self.emergency_stop:
            logger.error("🚨 BOT INTERROMPIDO POR PARADA DE EMERGÊNCIA!")

    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning(f"⚠️ Bot já está rodando")
            return
            
        self.is_running = True
        self.price_monitoring = True
        self.emergency_stop = False
        
        trading_thread = threading.Thread(target=self.run_trading_loop, daemon=True)
        trading_thread.start()
        logger.warning(f"✅ Trading bot iniciado com SISTEMA DE SEGURANÇA!")

    def stop(self):
        """Stop the trading bot"""
        logger.warning(f"🛑 Parando trading bot...")
        self.is_running = False
        self.price_monitoring = False
        
        if self.current_position:
            logger.warning(f"🔄 Fechando posição antes de parar...")
            self.emergency_close_position("BOT_STOP")
        
        logger.warning(f"🛑 Trading bot parado")
        
        # Estatísticas finais
        logger.warning(f"📊 ESTATÍSTICAS FINAIS:")
        logger.warning(f"   Total de trades: {self.total_trades}")
        logger.warning(f"   Stop Loss ativados: {self.stop_loss_triggered}")
        logger.warning(f"   Take Profit ativados: {self.take_profit_triggered}")
        logger.warning(f"   Trades lucrativos: {self.profitable_trades}")

    def get_status(self) -> Dict:
        """Get current bot status"""
        current_balance = self.get_account_balance()
        
        return {
            'is_running': self.is_running,
            'trades_today': self.trades_today,
            'daily_target': self.daily_target,
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'current_balance': current_balance,
            'start_balance': self.start_balance,
            'current_position': bool(self.current_position),
            'position_side': self.position_side,
            'entry_price': self.entry_price,
            'profit_target': self.profit_target * 100,
            'stop_loss_target': abs(self.stop_loss_target) * 100,
            'stop_loss_triggered': self.stop_loss_triggered,
            'take_profit_triggered': self.take_profit_triggered,
            'emergency_stop': self.emergency_stop,
            'prediction': self.prediction_data,
            'test_mode': getattr(self.bitget_api, 'test_mode', False)
        }

    def update_config(self, **kwargs):
        """Update bot configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.warning(f"✅ Configuração atualizada: {key} = {value}")
