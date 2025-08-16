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
        
        # CORREÇÃO: Valor mínimo da Bitget
        self.MIN_ORDER_USDT = 1.0
        
        # Cache para análises rápidas
        self.price_history = []
        self.volume_history = []
        
        logger.info("🚀 AggressiveScalpingBot - MODO SCALPING ATIVO - CORRIGIDO")

    def validate_min_order_value(self, usdt_amount: float) -> bool:
        """Valida se o valor da ordem atende ao mínimo da exchange"""
        if usdt_amount < self.MIN_ORDER_USDT:
            logger.warning(f"❌ Valor {usdt_amount:.2f} USDT abaixo do mínimo {self.MIN_ORDER_USDT} USDT")
            return False
        return True

    def calculate_scalp_position_size(self) -> float:
        """Calcula tamanho usando 80% do saldo - CORRIGIDO"""
        try:
            # 80% do saldo para cada trade
            usdt_80_percent = self.current_balance * 0.8
            
            # LOGS DE DEBUG
            logger.warning(f"🚨 CÁLCULO SCALP 80% DO SALDO:")
            logger.warning(f"💰 Saldo Atual: ${self.current_balance:.2f} USDT")
            logger.warning(f"🎯 80% Dinâmico: ${usdt_80_percent:.2f} USDT")
            
            # Verificar mínimo da exchange
            if not self.validate_min_order_value(usdt_80_percent):
                logger.warning(f"❌ 80% do saldo (${usdt_80_percent:.2f}) abaixo do mínimo {self.MIN_ORDER_USDT} USDT")
                logger.warning(f"⏰ Aguardando saldo suficiente...")
                return 0  # Não executar trade
            
            return usdt_80_percent  # Retornar valor em USDT
            
        except Exception as e:
            logger.error(f"❌ Erro no cálculo de posição: {e}")
            return 0

    def can_scalp(self) -> bool:
        """Verificação rápida se pode fazer scalping - MELHORADO"""
        current_time = time.time()
        
        # Verificar saldo mínimo
        if self.current_balance < self.MIN_ORDER_USDT * 1.25:  # 25% de margem de segurança
            logger.warning(f"💰 Saldo insuficiente: ${self.current_balance:.2f} < ${self.MIN_ORDER_USDT * 1.25:.2f} USDT")
            return False
        
        # Verificar se 80% do saldo atende ao mínimo
        usdt_80_percent = self.current_balance * 0.8
        if usdt_80_percent < self.MIN_ORDER_USDT:
            logger.warning(f"📉 80% do saldo (${usdt_80_percent:.2f}) < ${self.MIN_ORDER_USDT} USDT mínimo")
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

    async def execute_scalp_trade(self, signal: Dict):
        """Executa trade de scalping - CORRIGIDO"""
        try:
            direction = signal['direction']
            confidence = signal['confidence']
            entry_price = signal['entry_price']
            
            # CORREÇÃO: Usar valor em USDT (80% do saldo)
            position_size_usdt = self.calculate_scalp_position_size()
            
            if position_size_usdt == 0:
                logger.warning(f"❌ Não é possível executar scalp - valor insuficiente")
                return
            
            # Stop loss e take profit menores para scalping
            stop_loss, take_profit = self.calculate_scalp_levels(entry_price, direction)
            
            # LOGS DETALHADOS
            logger.warning(f"⚡ EXECUTANDO SCALP {direction}")
            logger.warning(f"💰 Valor USDT: ${position_size_usdt:.2f}")
            logger.warning(f"🎯 Confiança: {confidence:.1f}%")
            logger.warning(f"💎 Preço: ${entry_price:.2f}")
            logger.warning(f"🛑 Stop Loss: ${stop_loss:.2f}")
            logger.warning(f"🚀 Take Profit: ${take_profit:.2f}")
            
            # Executar ordem
            if not self.config.PAPER_TRADING:
                order_result = await self.bitget.place_order(
                    symbol=self.config.SYMBOL,
                    side=direction.lower(),
                    size=position_size_usdt,  # USAR VALOR USDT
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                if order_result and order_result.get('success'):
                    self.daily_trades_count += 1
                    self.last_trade_time = time.time()
                    logger.info(f"✅ SCALP executado! Trade #{self.daily_trades_count}")
                else:
                    error_msg = order_result.get('error', 'Erro desconhecido') if order_result else 'Falha na comunicação'
                    logger.error(f"❌ SCALP FALHOU: {error_msg}")
            else:
                self.daily_trades_count += 1
                self.last_trade_time = time.time()
                logger.info(f"📝 SCALP simulado #{self.daily_trades_count}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao executar scalp: {e}")

    # Resto dos métodos permanecem iguais...
    def calculate_scalp_levels(self, entry_price: float, direction: str) -> tuple:
        """Calcula stop/take menores para scalping"""
        if direction == 'BUY':
            stop_loss = entry_price * (1 - self.config.STOP_LOSS_PCT/100)
            take_profit = entry_price * (1 + self.config.TAKE_PROFIT_PCT/100)
        else:
            stop_loss = entry_price * (1 + self.config.STOP_LOSS_PCT/100)
            take_profit = entry_price * (1 - self.config.TAKE_PROFIT_PCT/100)
        
        return stop_loss, take_profit
