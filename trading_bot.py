"""
Trading Bot Core Logic for Scalping Strategy
Integrates with Bitget API and Gemini AI for automated trading
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

from .bitget_api import BitgetAPI
from .gemini_ai import GeminiAI

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self, config: Dict[str, Any]):
        """Initialize trading bot with configuration"""
        self.config = config
        self.is_running = False
        self.is_paused = False
        self.start_time = datetime.now()
        
        # Trading state
        self.current_position = None
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.total_volume = 0.0
        self.win_rate = 0.0
        self.successful_trades = 0
        self.total_trades = 0
        
        # Initialize APIs
        try:
            self.bitget_api = BitgetAPI(
                api_key=config.get('BITGET_API_KEY'),
                secret_key=config.get('BITGET_SECRET_KEY'),
                passphrase=config.get('BITGET_PASSPHRASE'),
                sandbox=config.get('PAPER_TRADING', False)
            )
            
            self.gemini_ai = GeminiAI(
                api_key=config.get('GEMINI_API_KEY')
            )
            
            logger.info("✅ APIs inicializadas com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar APIs: {e}")
            raise
        
        # Trading parameters
        self.symbol = config.get('SYMBOL', 'ethusdt_UMCBL')
        self.leverage = config.get('LEVERAGE', 10)
        self.target_trades_per_day = config.get('TARGET_TRADES_PER_DAY', 200)
        self.base_currency = config.get('BASE_CURRENCY', 'USDT')
        
        # Risk management - CORREÇÃO: Mudança de 10% para 80%
        self.stop_loss_pct = 0.02  # 2%
        self.take_profit_pct = 0.01  # 1%
        self.position_size_pct = 0.8  # 80% of balance per trade (CORRIGIDO)
        
        # ADIÇÃO: Valor mínimo da exchange
        self.MIN_ORDER_USDT = 1.0
        
        # Activity log
        self.activity_log = []
        self.max_log_entries = 100
        
        logger.info(f"🤖 Trading Bot configurado:")
        logger.info(f"   Símbolo: {self.symbol}")
        logger.info(f"   Alavancagem: {self.leverage}x")
        logger.info(f"   Uso do saldo: {self.position_size_pct*100}% (CORRIGIDO)")  # ADIÇÃO
        logger.info(f"   Meta diária: {self.target_trades_per_day} trades")
        logger.info(f"   Paper Trading: {config.get('PAPER_TRADING', False)}")
    
    # ADIÇÃO: Método de validação
    def validate_min_order_value(self, usdt_amount: float) -> bool:
        """Valida se o valor da ordem atende ao mínimo da exchange"""
        if usdt_amount < self.MIN_ORDER_USDT:
            logger.warning(f"❌ Valor {usdt_amount:.2f} USDT abaixo do mínimo {self.MIN_ORDER_USDT} USDT")
            return False
        return True
    
    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("⚠️ Bot já está em execução")
            return
        
        self.is_running = True
        self.is_paused = False
        self.start_time = datetime.now()
        
        # Start trading loop in separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        self.add_log("SISTEMA", "Trading bot iniciado com sucesso", "success")
        logger.info("🚀 Trading bot iniciado")
    
    def stop(self):
        """Stop the trading bot"""
        self.is_running = False
        self.is_paused = False
        
        # Close any open positions
        if self.current_position:
            self.close_position()
        
        self.add_log("SISTEMA", "Trading bot parado", "info")
        logger.info("🛑 Trading bot parado")
    
    def pause(self):
        """Pause/Resume the trading bot"""
        self.is_paused = not self.is_paused
        status = "pausado" if self.is_paused else "retomado"
        self.add_log("SISTEMA", f"Trading bot {status}", "warning" if self.is_paused else "success")
        logger.info(f"⏸️ Trading bot {status}")
    
    def emergency_stop(self):
        """Emergency stop - close all positions immediately"""
        logger.warning("🚨 PARADA DE EMERGÊNCIA ATIVADA")
        self.stop()
        self.add_log("EMERGÊNCIA", "Parada de emergência ativada - todas as posições fechadas", "error")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(5)
                    continue
                
                # Check if we've reached daily trade limit
                if self.daily_trades >= self.target_trades_per_day:
                    logger.info("📈 Meta diária de trades atingida")
                    time.sleep(300)  # Wait 5 minutes before checking again
                    continue
                
                # Get market analysis
                analysis = self._get_market_analysis()
                
                if analysis:
                    # Execute trading logic based on analysis
                    self._execute_trading_logic(analysis)
                
                # Sleep before next iteration (scalping frequency)
                time.sleep(30)  # 30 seconds between checks
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de trading: {e}")
                self.add_log("ERRO", f"Erro no sistema: {str(e)}", "error")
                time.sleep(60)  # Wait 1 minute on error
    
    def _get_market_analysis(self) -> Optional[Dict]:
        """Get market analysis from Gemini AI and technical indicators"""
        try:
            # Get current price and market data
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data:
                return None
            
            # Get AI analysis
            ai_analysis = self.gemini_ai.analyze_market(
                symbol=self.symbol,
                market_data=market_data
            )
            
            # Combine with technical analysis
            technical_analysis = self._calculate_technical_indicators(market_data)
            
            analysis = {
                'timestamp': datetime.now(),
                'price': market_data.get('price', 0),
                'volume': market_data.get('volume', 0),
                'ai_analysis': ai_analysis,
                'technical': technical_analysis,
                'signal': self._generate_signal(ai_analysis, technical_analysis)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erro na análise de mercado: {e}")
            return None
    
    def _calculate_technical_indicators(self, market_data: Dict) -> Dict:
        """Calculate technical indicators (RSI, MACD, etc.)"""
        # Simplified technical analysis
        # In production, this would use more sophisticated libraries like TA-Lib
        
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        
        # Mock technical indicators for demo
        # Replace with real calculations
        return {
            'rsi': 32.5,  # Oversold
            'macd_signal': 'bullish_crossover',
            'volume_status': 'high' if volume > 1000000 else 'normal',
            'support_level': price * 0.98,
            'resistance_level': price * 1.02,
            'trend': 'bullish'
        }
    
    def _generate_signal(self, ai_analysis: Dict, technical: Dict) -> str:
        """Generate trading signal based on AI and technical analysis"""
        try:
            # Check if we already have a position
            if self.current_position:
                return 'hold'
            
            # AI sentiment
            ai_sentiment = ai_analysis.get('sentiment', 'neutral')
            ai_confidence = ai_analysis.get('confidence', 0)
            
            # Technical indicators
            rsi = technical.get('rsi', 50)
            macd_signal = technical.get('macd_signal', 'neutral')
            volume_status = technical.get('volume_status', 'normal')
            
            # Generate signal
            bullish_signals = 0
            bearish_signals = 0
            
            # AI analysis
            if ai_sentiment == 'bullish' and ai_confidence > 0.7:
                bullish_signals += 2
            elif ai_sentiment == 'bearish' and ai_confidence > 0.7:
                bearish_signals += 2
            
            # RSI
            if rsi < 35:  # Oversold
                bullish_signals += 1
            elif rsi > 65:  # Overbought
                bearish_signals += 1
            
            # MACD
            if macd_signal == 'bullish_crossover':
                bullish_signals += 1
            elif macd_signal == 'bearish_crossover':
                bearish_signals += 1
            
            # Volume confirmation
            if volume_status == 'high':
                if bullish_signals > bearish_signals:
                    bullish_signals += 1
                elif bearish_signals > bullish_signals:
                    bearish_signals += 1
            
            # Decision logic
            if bullish_signals >= 3 and bullish_signals > bearish_signals:
                return 'buy'
            elif bearish_signals >= 3 and bearish_signals > bullish_signals:
                return 'sell'
            else:
                return 'hold'
                
        except Exception as e:
            logger.error(f"❌ Erro ao gerar sinal: {e}")
            return 'hold'
    
    def _execute_trading_logic(self, analysis: Dict):
        """Execute trading based on analysis"""
        signal = analysis.get('signal', 'hold')
        price = analysis.get('price', 0)
        
        if signal == 'buy' and not self.current_position:
            self._execute_buy_order(price, analysis)
        elif signal == 'sell' and self.current_position and self.current_position.get('side') == 'long':
            self._execute_sell_order(price, analysis)
        elif self.current_position:
            self._check_stop_loss_take_profit(price)
    
    def _execute_buy_order(self, price: float, analysis: Dict):
        """Execute buy order - CORRIGIDO APENAS O CÁLCULO"""
        try:
            # Calculate position size
            account_balance = self.bitget_api.get_account_balance()
            
            # CORREÇÃO: Usar 80% do saldo em USDT
            usdt_80_percent = account_balance * self.position_size_pct  # 80% do saldo
            
            # ADIÇÃO: Verificar valor mínimo
            if not self.validate_min_order_value(usdt_80_percent):
                self.add_log("ERRO", f"Valor insuficiente: ${usdt_80_percent:.2f} < ${self.MIN_ORDER_USDT} USDT", "error")
                return
                
            # Calcular quantidade ETH (apenas para logs)
            quantity = usdt_80_percent / price
            
            # ADIÇÃO: Logs detalhados
            logger.warning(f"🚨 CÁLCULO DINÂMICO 80% DO SALDO:")
            logger.warning(f"💰 Saldo Atual: ${account_balance:.2f} USDT")
            logger.warning(f"🎯 80% Dinâmico: ${usdt_80_percent:.2f} USDT")
            logger.warning(f"💎 Preço ETH: ${price:.2f}")
            logger.warning(f"📊 ETH Calculado: {quantity:.6f} ETH")
            logger.warning(f"🚨 Alavancagem: {self.leverage}x")
            logger.warning(f"💥 Exposição Total: ${usdt_80_percent * self.leverage:.2f} USDT")
            logger.warning(f"💰 EXECUTANDO ORDEM FUTURES!")
            
            # Calculate stop loss and take profit
            stop_loss = price * (1 - self.stop_loss_pct)
            take_profit = price * (1 + self.take_profit_pct)
            
            # CORREÇÃO: Executar ordem com valor USDT
            order_result = self.bitget_api.place_order(
                symbol=self.symbol,
                side='buy',
                size=usdt_80_percent,  # USAR VALOR USDT, NÃO QUANTITY
                price=price,
                leverage=self.leverage
            )
            
            if order_result and order_result.get('success'):
                self.current_position = {
                    'side': 'long',
                    'entry_price': price,
                    'quantity': quantity,
                    'usdt_value': usdt_80_percent,  # ADIÇÃO: Guardar valor USDT
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now(),
                    'order_id': order_result.get('order_id')
                }
                
                self.daily_trades += 1
                self.total_trades += 1
                self.total_volume += usdt_80_percent  # CORREÇÃO: Usar valor USDT
                
                self.add_log(
                    "COMPRA EXECUTADA",
                    f"ETH/USDT - Valor: ${usdt_80_percent:.2f} - Preço: ${price:.2f}",  # CORREÇÃO
                    "success",
                    f"Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}"
                )
                
                logger.info(f"✅ Ordem de compra executada: ${usdt_80_percent:.2f} USDT @ ${price:.2f}")
            else:
                # ADIÇÃO: Log de erro detalhado
                error_msg = order_result.get('error', 'Erro desconhecido') if order_result else 'Falha na comunicação'
                logger.error(f"❌ ORDEM FUTURES FALHOU: bitget {error_msg}")
                logger.warning(f"❌ TRADE FUTURES FALHOU")
                
        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem de compra: {e}")
            self.add_log("ERRO", f"Falha na execução da compra: {str(e)}", "error")
    
    def _execute_sell_order(self, price: float, analysis: Dict):
        """Execute sell order"""
        try:
            if not self.current_position:
                return
            
            # CORREÇÃO: Usar valor USDT salvo
            usdt_value = self.current_position.get('usdt_value', 0)
            quantity = self.current_position['quantity']
            entry_price = self.current_position['entry_price']
            
            # Execute order
            order_result = self.bitget_api.place_order(
                symbol=self.symbol,
                side='sell',
                size=usdt_value if usdt_value > 0 else quantity,  # Priorizar valor USDT
                price=price,
                leverage=self.leverage
            )
            
            if order_result and order_result.get('success'):
                # Calculate PnL
                pnl = (price - entry_price) * quantity
                pnl_percentage = ((price - entry_price) / entry_price) * 100
                
                self.daily_pnl += pnl
                
                if pnl > 0:
                    self.successful_trades += 1
                
                self.win_rate = (self.successful_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
                
                self.add_log(
                    "VENDA EXECUTADA",
                    f"ETH/USDT - Quantidade: {quantity:.4f} - Preço: ${price:.2f}",
                    "success" if pnl > 0 else "warning",
                    f"{'Lucro' if pnl > 0 else 'Perda'}: ${pnl:.2f} ({pnl_percentage:+.2f}%)"
                )
                
                logger.info(f"✅ Ordem de venda executada: {quantity:.4f} @ ${price:.2f} - PnL: ${pnl:.2f}")
                
                self.current_position = None
                
        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem de venda: {e}")
            self.add_log("ERRO", f"Falha na execução da venda: {str(e)}", "error")
    
    def _check_stop_loss_take_profit(self, current_price: float):
        """Check if stop loss or take profit should be triggered"""
        if not self.current_position:
            return
        
        stop_loss = self.current_position['stop_loss']
        take_profit = self.current_position['take_profit']
        
        if current_price <= stop_loss:
            logger.info(f"🛑 Stop Loss ativado: ${current_price:.2f} <= ${stop_loss:.2f}")
            self._execute_sell_order(current_price, {'trigger': 'stop_loss'})
        elif current_price >= take_profit:
            logger.info(f"🎯 Take Profit ativado: ${current_price:.2f} >= ${take_profit:.2f}")
            self._execute_sell_order(current_price, {'trigger': 'take_profit'})
    
    def close_position(self):
        """Close current position manually"""
        if not self.current_position:
            logger.info("ℹ️ Nenhuma posição aberta para fechar")
            return
        
        try:
            # Get current market price
            market_data = self.bitget_api.get_market_data(self.symbol)
            if market_data:
                current_price = market_data.get('price', 0)
                self._execute_sell_order(current_price, {'trigger': 'manual_close'})
            else:
                logger.error("❌ Não foi possível obter preço atual para fechamento")
        
        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição: {e}")
    
    def add_log(self, action: str, message: str, log_type: str, details: str = ""):
        """Add entry to activity log"""
        log_entry = {
            'timestamp': datetime.now(),
            'action': action,
            'message': message,
            'type': log_type,
            'details': details
        }
        
        self.activity_log.append(log_entry)
        
        # Keep only recent entries
        if len(self.activity_log) > self.max_log_entries:
            self.activity_log = self.activity_log[-self.max_log_entries:]
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        uptime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'uptime': str(uptime),
            'daily_trades': self.daily_trades,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'win_rate': self.win_rate,
            'daily_pnl': self.daily_pnl,
            'total_volume': self.total_volume,
            'current_position': self.current_position,
            'activity_log': self.activity_log[-10:]  # Last 10 entries
        }
    
    def update_config(self, new_config: Dict):
        """Update bot configuration"""
        try:
            # Update basic parameters
            if 'LEVERAGE' in new_config:
                self.leverage = new_config['LEVERAGE']
            
            if 'TARGET_TRADES_PER_DAY' in new_config:
                self.target_trades_per_day = new_config['TARGET_TRADES_PER_DAY']
            
            # Update risk management
            if 'STOP_LOSS_PCT' in new_config:
                self.stop_loss_pct = new_config['STOP_LOSS_PCT']
            
            if 'TAKE_PROFIT_PCT' in new_config:
                self.take_profit_pct = new_config['TAKE_PROFIT_PCT']
            
            if 'POSITION_SIZE_PCT' in new_config:
                self.position_size_pct = new_config['POSITION_SIZE_PCT']
            
            logger.info("✅ Configuração atualizada")
            self.add_log("CONFIG", "Configurações atualizadas", "info")
            
        except Exception as e:
            logger.error(f"❌ Erro ao atualizar configuração: {e}")
