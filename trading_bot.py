import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List
import threading
from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, bitget_api: BitgetAPI, symbol: str = 'ethusdt_UMCBL', 
                 leverage: int = 10, balance_percentage: float = 100.0,
                 daily_target: int = 200, scalping_interval: int = 2, 
                 paper_trading: bool = False):
        """
        Initialize Trading Bot
        
        Args:
            bitget_api: BitgetAPI instance
            symbol: Trading symbol
            leverage: Leverage multiplier
            balance_percentage: Percentage of balance to use (now 100%)
            daily_target: Number of trades per day
            scalping_interval: Seconds between trades
            paper_trading: If True, simulate trades
        """
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
        self.profit_target = 0.01  # 1% profit target
        
        # Statistics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0.0
        self.start_balance = 0.0
        
        logger.info("✅ APIs inicializadas com sucesso")
        logger.info(f"🤖 Trading Bot configurado:")
        logger.info(f"   Símbolo: {self.symbol}")
        logger.info(f"   Alavancagem: {self.leverage}x")
        logger.info(f"   Uso do saldo: {self.balance_percentage}%")
        logger.info(f"   Meta diária: {self.daily_target} trades")
        logger.info(f"   Scalping: {self.scalping_interval}s entre trades")
        logger.info(f"   Paper Trading: {self.paper_trading}")

    def get_market_data(self) -> Dict:
        """Get current market data"""
        return self.bitget_api.get_market_data(self.symbol)

    def get_account_balance(self) -> float:
        """Get current account balance"""
        return self.bitget_api.get_account_balance()

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
            
            # Executar ordem com cálculo dinâmico
            result = self.bitget_api.place_order(
                symbol=self.symbol,
                side=side,
                size=0,  # Será ignorado - usamos cálculo dinâmico
                price=current_price,
                leverage=self.leverage
            )
            
            if result['success']:
                logger.warning(f"✅ TRADE {side.upper()} EXECUTADO!")
                logger.warning(f"💰 Valor USDT: ${result.get('usdt_amount', 0):.2f}")
                logger.warning(f"📊 Quantidade ETH: {result.get('eth_quantity', 0):.8f}")
                logger.warning(f"💎 Preço: ${result.get('price', 0):.2f}")
                
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
            
            # Determine opposite side
            close_side = 'sell' if self.position_side == 'buy' else 'buy'
            
            logger.warning(f"🔄 FECHANDO POSIÇÃO {self.position_side.upper()}")
            
            result = self.execute_trade(close_side)
            
            if result['success']:
                # Calculate profit
                current_price = result.get('price', 0)
                if self.entry_price and current_price:
                    if self.position_side == 'buy':
                        profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
                    else:
                        profit_pct = ((self.entry_price - current_price) / self.entry_price) * 100
                    
                    logger.warning(f"💰 LUCRO: {profit_pct:.2f}%")
                    
                    if profit_pct > 0:
                        self.profitable_trades += 1
                        self.total_profit += profit_pct
                
                # Reset position
                self.current_position = None
                self.entry_price = None
                self.position_side = None
                
                logger.warning(f"✅ POSIÇÃO FECHADA COM SUCESSO!")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição: {e}")
            return {'success': False, 'error': str(e)}

    def check_profit_target(self) -> bool:
        """Check if profit target is reached"""
        try:
            if not self.current_position or not self.entry_price:
                return False
            
            market_data = self.get_market_data()
            if not market_data:
                return False
            
            current_price = market_data['price']
            
            if self.position_side == 'buy':
                profit_pct = ((current_price - self.entry_price) / self.entry_price)
            else:
                profit_pct = ((self.entry_price - current_price) / self.entry_price)
            
            if profit_pct >= self.profit_target:
                logger.warning(f"🎯 META DE LUCRO ATINGIDA: {profit_pct*100:.2f}%")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"❌ Erro ao verificar meta de lucro: {e}")
            return False

    def scalping_strategy(self):
        """Rapid scalping strategy with 1% profit target"""
        try:
            logger.warning(f"🔥 EXECUTANDO ESTRATÉGIA DE SCALPING")
            
            # If no position, open one
            if not self.current_position:
                # Simple strategy: alternate buy/sell
                side = 'buy' if self.trades_today % 2 == 0 else 'sell'
                
                logger.warning(f"🚀 ABRINDO POSIÇÃO {side.upper()}")
                
                result = self.execute_trade(side)
                
                if result['success']:
                    self.current_position = result['order_id']
                    self.entry_price = result.get('price', 0)
                    self.position_side = side
                    self.trades_today += 1
                    self.total_trades += 1
                    
                    logger.warning(f"✅ POSIÇÃO ABERTA: {side.upper()}")
                    logger.warning(f"💰 Preço entrada: ${self.entry_price:.2f}")
                    logger.warning(f"📊 Trades hoje: {self.trades_today}/{self.daily_target}")
                
            else:
                # Check if profit target reached
                if self.check_profit_target():
                    self.close_position()
                    
        except Exception as e:
            logger.error(f"❌ Erro na estratégia de scalping: {e}")

    def run_trading_loop(self):
        """Main trading loop"""
        logger.warning(f"🚀 Trading bot iniciado")
        
        # Get starting balance
        self.start_balance = self.get_account_balance()
        logger.warning(f"💰 Saldo inicial: ${self.start_balance:.2f} USDT")
        
        while self.is_running:
            try:
                # Check if daily target reached
                if self.trades_today >= self.daily_target:
                    logger.warning(f"🎯 META DIÁRIA ATINGIDA: {self.trades_today} trades")
                    logger.warning(f"😴 Aguardando próximo dia...")
                    time.sleep(60)  # Check every minute
                    
                    # Reset daily counter at midnight (simplified)
                    current_hour = datetime.now().hour
                    if current_hour == 0:
                        self.trades_today = 0
                        logger.warning(f"🌅 NOVO DIA - Contador zerado")
                    
                    continue
                
                # Execute scalping strategy
                self.scalping_strategy()
                
                # Wait for next iteration
                logger.warning(f"⏱️ Aguardando {self.scalping_interval}s para próximo trade...")
                time.sleep(self.scalping_interval)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de trading: {e}")
                time.sleep(5)  # Wait 5 seconds on error
                
            except KeyboardInterrupt:
                logger.warning(f"🛑 Bot interrompido pelo usuário")
                self.stop()
                break

    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning(f"⚠️ Bot já está rodando")
            return
        
        self.is_running = True
        
        # Start trading loop in separate thread
        trading_thread = threading.Thread(target=self.run_trading_loop, daemon=True)
        trading_thread.start()
        
        logger.warning(f"✅ Trading bot iniciado com sucesso!")

    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        
        # Close any open positions
        if self.current_position:
            logger.warning(f"🔄 Fechando posição aberta...")
            self.close_position()
        
        logger.warning(f"🛑 Trading bot parado")
        
        # Print final statistics
        current_balance = self.get_account_balance()
        total_return = ((current_balance - self.start_balance) / self.start_balance) * 100 if self.start_balance > 0 else 0
        
        logger.warning(f"📊 ESTATÍSTICAS FINAIS:")
        logger.warning(f"   💰 Saldo inicial: ${self.start_balance:.2f} USDT")
        logger.warning(f"   💰 Saldo final: ${current_balance:.2f} USDT")
        logger.warning(f"   📈 Retorno total: {total_return:.2f}%")
        logger.warning(f"   🎯 Total de trades: {self.total_trades}")
        logger.warning(f"   ✅ Trades lucrativos: {self.profitable_trades}")
        logger.warning(f"   📊 Taxa de acerto: {(self.profitable_trades/self.total_trades)*100:.1f}%" if self.total_trades > 0 else "   📊 Taxa de acerto: 0%")

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
            'profit_target': self.profit_target * 100  # Convert to percentage
        }

    def update_config(self, **kwargs):
        """Update bot configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.warning(f"✅ Configuração atualizada: {key} = {value}")
