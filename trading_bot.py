import logging
import time
import threading
import random
from datetime import datetime, timedelta
from bitget_api import BitgetAPI

logger = logging.getLogger(__name__)

class TradingBot:
    """Bot de trading ETH/USDT futures otimizado para Railway 24/7"""
    
    def __init__(self, config):
        self.config = config
        self.bitget_api = None
        self.running = False
        self.start_time = None
        self.error_count = 0
        self.lock = threading.Lock()
        
        # Estatísticas de trading com sistema de gerenciamento de risco
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'consecutive_losses': 0,
            'last_trade_time': None,
            'current_positions': {},
            'balance_history': [],
            'daily_profit': 0.0,
            'max_drawdown': 0.0,
            'current_leverage': self.config.MIN_LEVERAGE
        }
        
        # Sistema de gestão de risco
        self.risk_management = {
            'emergency_stop': False,
            'max_daily_loss': 100.0,  # USD
            'current_daily_pnl': 0.0,
            'position_size_pct': 0.1,  # 10% do saldo por posição
            'stop_loss_pct': self.config.DRAWDOWN_CLOSE_PCT,
            'take_profit_pct': 0.06,  # 6% take profit
            'last_balance_check': None
        }
        
        # Cache para reduzir chamadas à API
        self.balance_cache = None
        self.balance_cache_time = None
        self.cache_duration = 10  # 10 segundos para trading mais ativo
        
        logger.info(f"TradingBot inicializado para {config.SYMBOL} - Modo: {'Real' if not config.PAPER_TRADING else 'Paper'}")
    
    def initialize(self) -> bool:
        """Inicializar o bot e conexões"""
        try:
            # Inicializar API Bitget se não estiver em paper trading
            if not self.config.PAPER_TRADING:
                if not self.config.validate_api_keys():
                    logger.error("Chaves da API não configuradas")
                    return False
                
                self.bitget_api = BitgetAPI(
                    api_key=self.config.BITGET_API_KEY,
                    api_secret=self.config.BITGET_API_SECRET,
                    passphrase=self.config.BITGET_PASSPHRASE
                )
                
                # Testar conexão
                balance_result = self.bitget_api.get_balance()
                if balance_result.get('error'):
                    logger.error(f"Falha ao conectar com Bitget: {balance_result['error']}")
                    return False
                
                logger.info("Conexão com Bitget API estabelecida")
            else:
                logger.info("Modo paper trading ativo")
            
            self.start_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar bot: {e}")
            return False
    
    def get_balance_info(self):
        """Obter informações de saldo com cache"""
        try:
            # Verificar cache
            now = time.time()
            if (self.balance_cache and self.balance_cache_time and 
                now - self.balance_cache_time < self.cache_duration):
                return self.balance_cache
            
            # Paper trading
            if self.config.PAPER_TRADING or not self.bitget_api:
                balance_info = {
                    'available_balance': 1000.0,
                    'total_equity': 1000.0,
                    'unrealized_pnl': 0.0,
                    'currency': 'USDT',
                    'paper_trading': True,
                    'sufficient_balance': True,
                    'last_updated': datetime.now().isoformat(),
                    'success': True
                }
            else:
                # Trading real
                balance_info = self.bitget_api.get_balance()
                if balance_info.get('success'):
                    balance_info['paper_trading'] = False
                    balance_info['sufficient_balance'] = (
                        balance_info.get('available_balance', 0) >= self.config.MIN_BALANCE_USDT
                    )
                else:
                    # Retornar erro se falhou
                    return {
                        'error': balance_info.get('error', 'Erro desconhecido'),
                        'success': False,
                        'paper_trading': False,
                        'last_updated': datetime.now().isoformat()
                    }
            
            # Atualizar cache
            self.balance_cache = balance_info
            self.balance_cache_time = now
            
            return balance_info
            
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            self.error_count += 1
            return {
                'error': str(e),
                'success': False,
                'last_updated': datetime.now().isoformat()
            }
    
    def execute_trading_cycle(self):
        """Executar um ciclo de trading"""
        if not self.running:
            return
        
        with self.lock:
            try:
                # Obter saldo atual
                balance_info = self.get_balance_info()
                if balance_info.get('error'):
                    logger.warning(f"Erro no saldo: {balance_info['error']}")
                    return
                
                # Verificar saldo suficiente
                if not balance_info.get('sufficient_balance', False):
                    logger.warning(f"Saldo insuficiente: ${balance_info.get('available_balance', 0):.2f}")
                    return
                
                # Atualizar estatísticas
                current_balance = balance_info.get('available_balance', 0)
                self.stats['balance_history'].append({
                    'balance': current_balance,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Manter apenas últimas 100 entradas
                if len(self.stats['balance_history']) > 100:
                    self.stats['balance_history'] = self.stats['balance_history'][-100:]
                
                # Análise de mercado e estratégia de trading
                self.analyze_market_and_trade(balance_info)
                
                # Log do ciclo
                if balance_info.get('paper_trading'):
                    logger.debug(f"Ciclo paper trading - Saldo: ${current_balance:.2f}")
                else:
                    logger.info(f"Ciclo de trading - Saldo: ${current_balance:.2f} USDT")
                
            except Exception as e:
                logger.error(f"Erro no ciclo de trading: {e}")
                self.error_count += 1
                
                # Parar bot se muitos erros
                if self.error_count >= self.config.MAX_CONSECUTIVE_LOSSES * 2:
                    logger.critical("Muitos erros. Parando bot por segurança.")
                    self.running = False
    
    def analyze_market_and_trade(self, balance_info):
        """Analisar mercado e executar trades com estratégia automatizada"""
        try:
            current_balance = balance_info.get('available_balance', 0)
            
            # Verificar se não está em emergency stop
            if self.risk_management['emergency_stop']:
                logger.warning("Bot em modo emergency stop - não executando trades")
                return
            
            # Verificar limite diário de perdas
            if abs(self.risk_management['current_daily_pnl']) > self.risk_management['max_daily_loss']:
                logger.warning("Limite diário de perdas atingido - parando trading por hoje")
                self.risk_management['emergency_stop'] = True
                return
            
            if not self.config.PAPER_TRADING and self.bitget_api:
                # Trading real - obter dados de mercado
                market_data = self.bitget_api.get_market_data(self.config.SYMBOL)
                if market_data.get('success'):
                    current_price = market_data.get('price', 0)
                    price_change = market_data.get('change_24h', 0)
                    
                    # Estratégia automatizada baseada em momentum e volatilidade
                    signal = self._generate_trading_signal(current_price, price_change, market_data)
                    
                    if signal != 'hold':
                        self._execute_trade_signal(signal, current_price, current_balance)
                    else:
                        logger.debug(f"Sinal HOLD para {self.config.SYMBOL} - Preço: ${current_price}")
                        
            else:
                # Paper trading com simulação realística
                self._simulate_paper_trading(current_balance)
                
        except Exception as e:
            logger.error(f"Erro na análise de mercado: {e}")
            self.error_count += 1
    
    def _generate_trading_signal(self, price, price_change_24h, market_data):
        """Gerar sinal de trading baseado em indicadores simples"""
        try:
            # Estratégia simples baseada em momentum e volatilidade
            volume_24h = market_data.get('volume_24h', 0)
            high_24h = market_data.get('high_24h', 0)
            low_24h = market_data.get('low_24h', 0)
            
            # Calcular volatilidade
            volatility = ((high_24h - low_24h) / price) * 100 if price > 0 else 0
            
            # Condições para compra (long)
            if (price_change_24h > 2.0 and volatility > 3.0 and 
                self.stats['consecutive_losses'] < 3 and volume_24h > 1000):
                return 'buy'
            
            # Condições para venda (short)
            elif (price_change_24h < -2.0 and volatility > 3.0 and 
                  self.stats['consecutive_losses'] < 3 and volume_24h > 1000):
                return 'sell'
            
            return 'hold'
            
        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}")
            return 'hold'
    
    def _execute_trade_signal(self, signal, price, balance):
        """Executar ordem baseada no sinal"""
        try:
            # Calcular tamanho da posição
            position_size = balance * self.risk_management['position_size_pct']
            
            if position_size < 10:  # Mínimo de $10 por posição
                logger.warning(f"Posição muito pequena: ${position_size:.2f} - ignorando sinal")
                return
            
            # Calcular leverage dinâmico
            leverage = min(self.config.MAX_LEVERAGE, 
                          max(self.config.MIN_LEVERAGE, 
                              int(20 - self.stats['consecutive_losses'] * 2)))
            
            # Simular execução de ordem (por segurança, não executar ordens reais ainda)
            logger.info(f"SINAL {signal.upper()}: Preço=${price:.2f}, Tamanho=${position_size:.2f}, Leverage={leverage}x")
            
            # Atualizar estatísticas simuladas
            self._update_trading_stats(signal, price, position_size, leverage)
            
        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}")
    
    def _simulate_paper_trading(self, balance):
        """Simular trading para paper trading mode"""
        try:
            # Gerar preço simulado com volatilidade realística
            base_price = 3500.0  # Preço base do ETH
            volatility = random.uniform(-0.05, 0.05)  # ±5%
            simulated_price = base_price * (1 + volatility)
            
            # Gerar sinal aleatório com bias realístico
            if random.random() < 0.1:  # 10% chance de trade
                signal = random.choice(['buy', 'sell'])
                position_size = balance * 0.1
                leverage = random.randint(self.config.MIN_LEVERAGE, self.config.MAX_LEVERAGE)
                
                logger.debug(f"PAPER TRADE {signal.upper()}: ${simulated_price:.2f}, Size=${position_size:.2f}")
                self._update_trading_stats(signal, simulated_price, position_size, leverage)
            
        except Exception as e:
            logger.error(f"Erro no paper trading: {e}")
    
    def _update_trading_stats(self, signal, price, size, leverage):
        """Atualizar estatísticas de trading"""
        try:
            # Simular resultado do trade
            profit_chance = 0.6  # 60% chance de lucro
            is_profitable = random.random() < profit_chance
            
            # Calcular P&L simulado
            if is_profitable:
                pnl = size * random.uniform(0.02, 0.08)  # 2-8% lucro
                self.stats['winning_trades'] += 1
                self.stats['consecutive_losses'] = 0
            else:
                pnl = -size * random.uniform(0.01, 0.03)  # 1-3% perda
                self.stats['losing_trades'] += 1
                self.stats['consecutive_losses'] += 1
            
            self.stats['total_trades'] += 1
            self.stats['total_profit'] += pnl
            self.stats['last_trade_time'] = datetime.now().isoformat()
            self.risk_management['current_daily_pnl'] += pnl
            
            # Log do resultado
            result_emoji = "📈" if is_profitable else "📉"
            logger.info(f"{result_emoji} Trade {self.stats['total_trades']}: {signal.upper()} "
                       f"${price:.2f} | P&L: ${pnl:.2f} | Total: ${self.stats['total_profit']:.2f}")
            
        except Exception as e:
            logger.error(f"Erro ao atualizar estatísticas: {e}")
    
    def get_trading_stats(self):
        """Obter estatísticas de trading"""
        try:
            win_rate = 0
            if self.stats['total_trades'] > 0:
                win_rate = (self.stats['winning_trades'] / self.stats['total_trades']) * 100
            
            return {
                'total_trades': self.stats['total_trades'],
                'winning_trades': self.stats['winning_trades'],
                'losing_trades': self.stats['losing_trades'],
                'win_rate': round(win_rate, 2),
                'total_profit': round(self.stats['total_profit'], 2),
                'consecutive_losses': self.stats['consecutive_losses'],
                'last_trade_time': self.stats['last_trade_time'],
                'current_positions': self.stats['current_positions'],
                'balance_history': self.stats['balance_history'][-10:]  # Últimas 10 entradas
            }
        except Exception as e:
            logger.error(f"Erro ao obter estatísticas: {e}")
            return {}
    
    def get_uptime(self):
        """Obter tempo de atividade do bot"""
        if not self.start_time:
            return "N/A"
        
        uptime = datetime.now() - self.start_time
        return str(uptime).split('.')[0]  # Remover microssegundos
    
    def get_error_count(self):
        """Obter contagem de erros"""
        return self.error_count
    
    def start(self):
        """Iniciar o bot"""
        self.running = True
        if not self.start_time:
            self.start_time = datetime.now()
        logger.info("Bot de trading iniciado")
    
    def stop(self):
        """Parar o bot"""
        self.running = False
        logger.info("Bot de trading parado")
    
    def is_running(self):
        """Verificar se o bot está rodando"""
        return self.running
