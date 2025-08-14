import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from config import Config
from bitget_api import BitgetAPI
from technical_analysis import ScalpingAnalysis
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

class AggressiveScalpingBot:
    def __init__(self):
        self.config = Config()
        self.config.validate_config()
        
        # Componentes
        self.bitget = BitgetAPI()
        self.scalping_analysis = ScalpingAnalysis()
        self.risk_manager = RiskManager()
        
        # Estados do bot
        self.is_running = False
        self.current_balance = 0.0
        self.daily_trades_count = 0
        self.successful_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = 0
        self.consecutive_losses = 0
        
        # Cache para análises rápidas
        self.price_history = []
        self.volume_history = []
        
        logger.info("🚀 AggressiveScalpingBot - MODO SCALPING ATIVO")
    
    async def start_scalping(self):
        """Loop principal de scalping"""
        self.is_running = True
        logger.info("⚡ SCALPING INICIADO - Buscando trades constantes...")
        
        while self.is_running:
            try:
                # Update rápido de saldo (a cada 10 iterações)
                if self.daily_trades_count % 10 == 0:
                    await self.update_account_balance()
                
                # Verificar se pode fazer trade
                if not self.can_scalp():
                    await asyncio.sleep(5)  # Espera menor
                    continue
                
                # Análise rápida para scalping
                scalp_signal = await self.fast_scalping_analysis()
                
                if scalp_signal and scalp_signal['confidence'] >= self.config.MIN_CONFIDENCE_SCORE:
                    # Executar trade de scalping
                    await self.execute_scalp_trade(scalp_signal)
                
                # Gerenciar posições rapidamente
                await self.quick_position_management()
                
                # Intervalo mínimo entre análises
                await asyncio.sleep(5)  # 5 segundos entre análises
                
            except Exception as e:
                logger.error(f"❌ Erro no scalping: {e}")
                await asyncio.sleep(10)
    
    async def update_account_balance(self):
        """Update de saldo otimizado"""
        try:
            account_info = await self.bitget.get_account_info()
            if account_info:
                old_balance = self.current_balance
                self.current_balance = float(account_info.get('availableBalance', 0))
                
                if old_balance > 0:
                    pnl_change = self.current_balance - old_balance
                    self.daily_pnl += pnl_change
                    
                    if pnl_change > 0:
                        logger.info(f"💚 +${pnl_change:.2f} | Saldo: ${self.current_balance:.2f}")
                    elif pnl_change < 0:
                        logger.info(f"💔 ${pnl_change:.2f} | Saldo: ${self.current_balance:.2f}")
        except Exception as e:
            logger.error(f"❌ Erro ao atualizar saldo: {e}")
    
    def can_scalp(self) -> bool:
        """Verificação rápida se pode fazer scalping"""
        current_time = time.time()
        
        # Verificar saldo mínimo
        if self.current_balance < self.config.MIN_BALANCE_USDT:
            return False
        
        # Verificar máximo de trades diários
        if self.daily_trades_count >= self.config.MAX_DAILY_TRADES:
            logger.info(f"📊 Máximo de trades diários atingido: {self.daily_trades_count}")
            return False
        
        # Verificar perdas consecutivas
        if self.consecutive_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            logger.info(f"⚠️ Muitas perdas consecutivas: {self.consecutive_losses}")
            return False
        
        # Verificar intervalo mínimo
        if current_time - self.last_trade_time < self.config.MIN_TRADE_INTERVAL:
            return False
        
        return True
    
    async def fast_scalping_analysis(self) -> Optional[Dict]:
        """Análise ultra-rápida para scalping"""
        try:
            # Obter dados atuais do mercado
            market_data = await self.bitget.get_market_data(self.config.SYMBOL)
            if not market_data:
                return None
            
            current_price = market_data['price']
            volume = market_data['volume']
            
            # Atualizar histórico
            self.price_history.append(current_price)
            self.volume_history.append(volume)
            
            # Manter apenas últimos 100 pontos
            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]
                self.volume_history = self.volume_history[-100:]
            
            # Precisa de pelo menos 20 pontos
            if len(self.price_history) < 20:
                return None
            
            # Análise técnica rápida
            signals = await self.scalping_analysis.quick_analysis(
                self.price_history, 
                self.volume_history,
                market_data
            )
            
            # Calcular confiança do scalping
            confidence = self.calculate_scalping_confidence(signals, market_data)
            
            if confidence >= self.config.MIN_CONFIDENCE_SCORE:
                direction = self.determine_scalp_direction(signals)
                
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'entry_price': current_price,
                    'signals': signals,
                    'market_data': market_data,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de scalping: {e}")
            return None
    
    def calculate_scalping_confidence(self, signals: Dict, market_data: Dict) -> float:
        """Calcula confiança para scalping (mais permissiva)"""
        confidence = 50.0  # Base
        
        # RSI
        rsi = signals.get('rsi', 50)
        if rsi < 30 or rsi > 70:  # Sobrecomprado/vendido
            confidence += 15
        elif 35 < rsi < 65:  # Neutro
            confidence += 5
        
        # EMA Trend
        if signals.get('ema_trend') == 'strong':
            confidence += 20
        elif signals.get('ema_trend') == 'weak':
            confidence += 10
        
        # Volume
        if signals.get('volume_spike'):
            confidence += 15
        
        # MACD
        if signals.get('macd_signal') == 'bullish':
            confidence += 10
        elif signals.get('macd_signal') == 'bearish':
            confidence += 10
        
        # Bollinger Bands
        bb_position = signals.get('bb_position')
        if bb_position in ['lower', 'upper']:
            confidence += 10
        
        # Momentum
        if signals.get('momentum') == 'strong':
            confidence += 10
        
        return min(confidence, 85.0)  # Máximo 85%
    
    def determine_scalp_direction(self, signals: Dict) -> str:
        """Determina direção do scalping"""
        buy_score = 0
        sell_score = 0
        
        # RSI
        rsi = signals.get('rsi', 50)
        if rsi < 30:
            buy_score += 2
        elif rsi > 70:
            sell_score += 2
        
        # EMA
        if signals.get('ema_direction') == 'up':
            buy_score += 2
        elif signals.get('ema_direction') == 'down':
            sell_score += 2
        
        # MACD
        if signals.get('macd_signal') == 'bullish':
            buy_score += 1
        elif signals.get('macd_signal') == 'bearish':
            sell_score += 1
        
        # Bollinger
        if signals.get('bb_position') == 'lower':
            buy_score += 1
        elif signals.get('bb_position') == 'upper':
            sell_score += 1
        
        # Volume + Momentum
        if signals.get('volume_spike') and signals.get('momentum') == 'up':
            buy_score += 1
        elif signals.get('volume_spike') and signals.get('momentum') == 'down':
            sell_score += 1
        
        return 'BUY' if buy_score > sell_score else 'SELL'
    
    async def execute_scalp_trade(self, signal: Dict):
        """Executa trade de scalping rapidamente"""
        try:
            direction = signal['direction']
            confidence = signal['confidence']
            entry_price = signal['entry_price']
            
            # Tamanho de posição menor para scalping
            position_size = self.calculate_scalp_position_size()
            
            # Stop loss e take profit menores
            stop_loss, take_profit = self.calculate_scalp_levels(entry_price, direction)
            
            logger.info(f"⚡ SCALP {direction} | {confidence:.1f}% | ${position_size:.2f}")
            
            # Executar ordem
            if not self.config.PAPER_TRADING:
                order_result = await self.bitget.place_order(
                    symbol=self.config.SYMBOL,
                    side=direction.lower(),
                    size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if order_result:
                    self.daily_trades_count += 1
                    self.last_trade_time = time.time()
                    logger.info(f"✅ SCALP executado! Trade #{self.daily_trades_count}")
                else:
                    logger.error("❌ Falha no scalp")
            else:
                self.daily_trades_count += 1
                self.last_trade_time = time.time()
                logger.info(f"📝 SCALP simulado #{self.daily_trades_count}")
        
        except Exception as e:
            logger.error(f"❌ Erro ao executar scalp: {e}")
    
    def calculate_scalp_position_size(self) -> float:
        """Calcula tamanho menor para scalping"""
        # 10-15% do saldo para cada scalp
        base_size = self.current_balance * 0.12  # 12%
        leveraged_size = base_size * self.config.LEVERAGE
        
        # Máximo seguro
        max_size = self.current_balance * 0.5
        return min(leveraged_size, max_size)
    
    def calculate_scalp_levels(self, entry_price: float, direction: str) -> tuple:
        """Calcula stop/take menores para scalping"""
        if direction == 'BUY':
            stop_loss = entry_price * (1 - self.config.STOP_LOSS_PCT / 100)
            take_profit = entry_price * (1 + self.config.TAKE_PROFIT_PCT / 100)
        else:
            stop_loss = entry_price * (1 + self.config.STOP_LOSS_PCT / 100)
            take_profit = entry_price * (1 - self.config.TAKE_PROFIT_PCT / 100)
        
        return stop_loss, take_profit
    
    async def quick_position_management(self):
        """Gerenciamento rápido de posições"""
        try:
            positions = await self.bitget.get_open_positions()
            
            for position in positions:
                # Verificar se posição está há muito tempo aberta (mais de 5 minutos)
                # Se sim, considerar fechar
                pass
                
        except Exception as e:
            logger.error(f"❌ Erro no gerenciamento: {e}")
    
    def get_stats(self) -> Dict:
        """Estatísticas do scalping"""
        win_rate = (self.successful_trades / max(self.daily_trades_count, 1)) * 100
        
        return {
            'daily_trades': self.daily_trades_count,
            'successful_trades': self.successful_trades,
            'win_rate': win_rate,
            'daily_pnl': self.daily_pnl,
            'current_balance': self.current_balance
        }

# Instância do scalping bot
scalping_bot = AggressiveScalpingBot()

async def start_scalping():
    """Inicia o bot de scalping"""
    await scalping_bot.start_scalping()
