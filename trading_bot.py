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

from bitget_api import BitgetAPI

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

            # Gemini AI optional
            gemini_key = config.get('GEMINI_API_KEY')
            if gemini_key:
                try:
                    from gemini_handler import GeminiHandler
                    self.gemini_ai = GeminiHandler(api_key=gemini_key)
                except ImportError:
                    logger.warning("⚠️ Gemini AI não disponível - continuando sem IA")
                    self.gemini_ai = None
            else:
                self.gemini_ai = None

            logger.info("✅ APIs inicializadas com sucesso")

        except Exception as e:
            logger.error(f"❌ Erro ao inicializar APIs: {e}")
            raise

        # Trading parameters
        self.symbol = config.get('SYMBOL', 'ethusdt_UMCBL')
        self.leverage = config.get('LEVERAGE', 10)
        self.target_trades_per_day = config.get('TARGET_TRADES_PER_DAY', 200)
        self.base_currency = config.get('BASE_CURRENCY', 'USDT')

        # Risk management - 80% do saldo
        self.stop_loss_pct = 0.02  # 2%
        self.take_profit_pct = 0.01  # 1%
        self.position_size_pct = 0.8  # 80% of balance per trade

        # CORREÇÃO: Scalping ultra rápido
        self.scalping_cooldown = 2  # 2 segundos entre trades
        self.last_trade_time = None

        # Valor mínimo da exchange
        self.MIN_ORDER_USDT = 1.0

        # Activity log
        self.activity_log = []
        self.max_log_entries = 100

        logger.info(f"🤖 Trading Bot configurado:")
        logger.info(f"   Símbolo: {self.symbol}")
        logger.info(f"   Alavancagem: {self.leverage}x")
        logger.info(f"   Uso do saldo: {self.position_size_pct*100}%")
        logger.info(f"   Meta diária: {self.target_trades_per_day} trades")
        logger.info(f"   Scalping: {self.scalping_cooldown}s entre trades")
        logger.info(f"   Paper Trading: {config.get('PAPER_TRADING', False)}")

    def validate_min_order_value(self, usdt_amount: float) -> bool:
        """Valida se o valor da ordem atende ao mínimo da exchange"""
        if usdt_amount < self.MIN_ORDER_USDT:
            logger.warning(f"❌ Valor {usdt_amount:.2f} USDT abaixo do mínimo {self.MIN_ORDER_USDT} USDT")
            return False
        return True

    def add_log(self, action: str, message: str, level: str = "info", details: str = ""):
        """Add log entry"""
        log_entry = {
            'timestamp': datetime.now(),
            'action': action,
            'message': message,
            'level': level,
            'details': details
        }
        
        self.activity_log.append(log_entry)
        
        # Keep only recent entries
        if len(self.activity_log) > self.max_log_entries:
            self.activity_log = self.activity_log[-self.max_log_entries:]

    def can_trade_now(self) -> bool:
        """Verifica se pode fazer um novo trade (cooldown de 2 segundos)"""
        if self.last_trade_time is None:
            return True
        
        time_diff = (datetime.now() - self.last_trade_time).total_seconds()
        return time_diff >= self.scalping_cooldown

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

    def close_position(self):
        """Fechar posição atual manualmente"""
        if not self.current_position:
            return
        
        try:
            # Obter preço atual
            market_data = self.bitget_api.get_market_data(self.symbol)
            if market_data:
                current_price = market_data.get('price', 0)
                self._force_close_position(current_price, "FECHAMENTO_MANUAL")
        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição: {e}")

    def _trading_loop(self):
        """Main trading loop - CORRIGIDO PARA SCALPING RÁPIDO"""
        while self.is_running:
            try:
                if self.is_paused:
                    time.sleep(2)
                    continue

                # Check if we've reached daily trade limit
                if self.daily_trades >= self.target_trades_per_day:
                    logger.info("📈 Meta diária de trades atingida")
                    time.sleep(60)
                    continue

                # Get market analysis
                analysis = self._get_market_analysis()

                if analysis:
                    # Execute trading logic based on analysis
                    self._execute_trading_logic(analysis)

                # CORREÇÃO: Sleep muito menor para scalping ultra rápido
                time.sleep(1)  # 1 segundo entre verificações

            except Exception as e:
                logger.error(f"❌ Erro no loop de trading: {e}")
                self.add_log("ERRO", f"Erro no sistema: {str(e)}", "error")
                time.sleep(10)

    def _get_market_analysis(self) -> Optional[Dict]:
        """Get market analysis from Gemini AI and technical indicators"""
        try:
            # Get current price and market data
            market_data = self.bitget_api.get_market_data(self.symbol)
            if not market_data:
                return None

            # Get AI analysis if available
            ai_analysis = {}
            if self.gemini_ai:
                try:
                    ai_analysis = self.gemini_ai.analyze_market(
                        symbol=self.symbol,
                        market_data=market_data
                    )
                except Exception as e:
                    logger.warning(f"⚠️ IA indisponível: {e}")
                    ai_analysis = {'sentiment': 'neutral', 'confidence': 0.5}
            else:
                # Mock analysis if no AI
                ai_analysis = {'sentiment': 'neutral', 'confidence': 0.5}

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
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)

        # Mock technical indicators for demo
        return {
            'rsi': 32.5,  # Oversold
            'macd_signal': 'bullish_crossover',
            'volume_status': 'high' if volume > 1000000 else 'normal',
            'support_level': price * 0.98,
            'resistance_level': price * 1.02,
            'trend': 'bullish'
        }

    def _generate_signal(self, ai_analysis: Dict, technical: Dict) -> str:
        """Generate trading signal based on AI and technical analysis - CORRIGIDO PARA SCALPING"""
        try:
            # CORREÇÃO: Permitir novos sinais mesmo com posição (para scalping rápido)
            
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
        """Execute trading based on analysis - CORRIGIDO"""
        signal = analysis.get('signal', 'hold')
        price = analysis.get('price', 0)

        # CORREÇÃO: Verificar take profit/stop loss primeiro
        if self.current_position:
            self._check_stop_loss_take_profit(price)

        # CORREÇÃO: Permitir novos trades após cooldown (scalping)
        if signal == 'buy' and self.can_trade_now():
            # Se já tem posição, fecha antes de abrir nova (para scalping)
            if self.current_position:
                self._force_close_position(price, "SCALPING_NOVA_ENTRADA")
                time.sleep(0.5)  # Pequeno delay
            
            self._execute_buy_order(price, analysis)

    def _execute_buy_order(self, price: float, analysis: Dict):
        """Execute buy order - CORRIGIDO PARA 80% DO SALDO"""
        try:
            # Calculate position size
            account_balance = self.bitget_api.get_account_balance()

            # CORREÇÃO: Usar 80% do saldo em USDT
            usdt_80_percent = account_balance * self.position_size_pct  # 80% do saldo

            # Verificar valor mínimo
            if not self.validate_min_order_value(usdt_80_percent):
                self.add_log("ERRO", f"Valor insuficiente: ${usdt_80_percent:.2f} < ${self.MIN_ORDER_USDT} USDT", "error")
                logger.warning(f"⏰ Aguardando saldo suficiente...")
                return

            # Calcular quantidade ETH (apenas para logs)
            quantity = usdt_80_percent / price

            # Logs detalhados
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

            # Execute order com valor USDT
            order_result = self.bitget_api.place_order(
                symbol=self.symbol,
                side='buy',
                size=usdt_80_percent,  # USAR VALOR USDT
                price=price,
                leverage=self.leverage
            )

            if order_result and order_result.get('success'):
                self.current_position = {
                    'side': 'long',
                    'entry_price': price,
                    'quantity': quantity,
                    'usdt_value': usdt_80_percent,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'timestamp': datetime.now(),
                    'order_id': order_result.get('order_id')
                }

                # Atualizar timestamp do último trade
                self.last_trade_time = datetime.now()

                self.daily_trades += 1
                self.total_trades += 1
                self.total_volume += usdt_80_percent

                self.add_log(
                    "COMPRA EXECUTADA",
                    f"ETH/USDT - Valor: ${usdt_80_percent:.2f} - Preço: ${price:.2f}",
                    "success",
                    f"Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}"
                )

                logger.warning(f"✅ ORDEM COMPRA EXECUTADA: ${usdt_80_percent:.2f} USDT @ ${price:.2f}")
                logger.warning(f"🎯 TAKE PROFIT: ${take_profit:.2f} (+1%)")
                logger.warning(f"🛑 STOP LOSS: ${stop_loss:.2f} (-2%)")
            else:
                error_msg = order_result.get('error', 'Erro desconhecido') if order_result else 'Falha na comunicação'
                logger.error(f"❌ ORDEM FUTURES FALHOU: bitget {error_msg}")

        except Exception as e:
            logger.error(f"❌ Erro ao executar ordem de compra: {e}")
            self.add_log("ERRO", f"Falha na execução da compra: {str(e)}", "error")

    def _check_stop_loss_take_profit(self, current_price: float):
        """CORRIGIDO: Verificar stop loss e take profit"""
        if not self.current_position:
            return

        try:
            entry_price = self.current_position.get('entry_price', 0)
            stop_loss = self.current_position.get('stop_loss', 0)
            take_profit = self.current_position.get('take_profit', 0)
            
            # Calcular P&L atual
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            
            logger.info(f"📊 MONITORAMENTO P&L: {pnl_percent:.2f}% | Preço: ${current_price:.2f}")
            
            # TAKE PROFIT: Fechar se atingiu 1% de lucro
            if current_price >= take_profit:
                logger.warning(f"🎯 TAKE PROFIT ATINGIDO! {pnl_percent:.2f}%")
                self._force_close_position(current_price, "TAKE_PROFIT")
                return
            
            # STOP LOSS: Fechar se perdeu 2%
            if current_price <= stop_loss:
                logger.warning(f"🛑 STOP LOSS ATINGIDO! {pnl_percent:.2f}%")
                self._force_close_position(current_price, "STOP_LOSS")
                return
                
        except Exception as e:
            logger.error(f"❌ Erro ao verificar stop/take profit: {e}")

    def _force_close_position(self, price: float, reason: str):
        """Forçar fechamento da posição atual"""
        try:
            if not self.current_position:
                return

            quantity = self.current_position.get('quantity', 0)
            entry_price = self.current_position.get('entry_price', 0)
            usdt_value = self.current_position.get('usdt_value', 0)

            # Executar ordem de venda
            order_result = self.bitget_api.place_order(
                symbol=self.symbol,
                side='sell',
                size=usdt_value,  # Mesmo valor da compra
                price=price,
                leverage=self.leverage
            )

            if order_result and order_result.get('success'):
                # Calcular P&L
                pnl_percent = ((price - entry_price) / entry_price) * 100
                pnl_usdt = usdt_value * (pnl_percent / 100)

                self.daily_pnl += pnl_usdt
                if pnl_usdt > 0:
                    self.successful_trades += 1

                self.add_log(
                    f"VENDA - {reason}",
                    f"ETH/USDT - P&L: ${pnl_usdt:.2f} ({pnl_percent:.2f}%)",
                    "success" if pnl_usdt > 0 else "warning",
                    f"Entrada: ${entry_price:.2f} | Saída: ${price:.2f}"
                )

                logger.warning(f"✅ POSIÇÃO FECHADA - {reason}")
                logger.warning(f"💰 P&L: ${pnl_usdt:.2f} USDT ({pnl_percent:.2f}%)")
                
                # Resetar posição
                self.current_position = None
                self.last_trade_time = datetime.now()

            else:
                error_msg = order_result.get('error', 'Erro desconhecido') if order_result else 'Falha na comunicação'
                logger.error(f"❌ ERRO AO FECHAR POSIÇÃO: {error_msg}")

        except Exception as e:
            logger.error(f"❌ Erro ao fechar posição forçadamente: {e}")

    def get_status(self) -> Dict:
        """Get current bot status"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'uptime': uptime,
            'current_position': self.current_position,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'total_volume': self.total_volume,
            'win_rate': (self.successful_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'successful_trades': self.successful_trades,
            'total_trades': self.total_trades,
            'activity_log': self.activity_log[-10:]  # Last 10 entries
        }
