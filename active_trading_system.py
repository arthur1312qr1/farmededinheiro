import os
import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import threading

from config_updated import EnhancedTradingConfig
from enhanced_bitget_api import EnhancedBitgetAPI
from gemini_handler import GeminiHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    id: str
    timestamp: datetime
    symbol: str
    side: str
    amount: float
    entry_price: float
    exit_price: Optional[float]
    leverage: int
    realized_pnl: Optional[float]
    confidence: float
    status: str  # 'open', 'closed'
    reasoning: str

class ActiveTradingSystem:
    def __init__(self):
        self.config = EnhancedTradingConfig()
        self.gemini_handler = GeminiHandler(self.config.GEMINI_API_KEY)
        self.bitget_api = None
        
        # Estado do sistema
        self.running = False
        self.trades = []
        self.balance_history = []
        self.last_analysis = None
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'current_balance': 10000.0 if self.config.PAPER_TRADING else 0.0
        }
        
        # Configurar API se não for paper trading
        if not self.config.PAPER_TRADING and self.config.validate_api_keys():
            try:
                self.bitget_api = EnhancedBitgetAPI(
                    self.config.BITGET_API_KEY,
                    self.config.BITGET_SECRET,
                    self.config.BITGET_PASSPHRASE
                )
                logger.info("Bitget API conectada para trading real")
            except Exception as e:
                logger.error(f"Erro ao conectar Bitget API: {e}")
                self.config.PAPER_TRADING = True
        
        # Carregar dados salvos
        self.load_trading_data()
        
        logger.info(f"Sistema de trading inicializado - Modo: {'PAPER' if self.config.PAPER_TRADING else 'LIVE'}")
    
    def load_trading_data(self):
        """Carregar dados de trading salvos"""
        try:
            if os.path.exists('trading_data.json'):
                with open('trading_data.json', 'r') as f:
                    data = json.load(f)
                    
                # Carregar trades
                self.trades = []
                for trade_data in data.get('trades', []):
                    trade = Trade(
                        id=trade_data['id'],
                        timestamp=datetime.fromisoformat(trade_data['timestamp']),
                        symbol=trade_data['symbol'],
                        side=trade_data['side'],
                        amount=trade_data['amount'],
                        entry_price=trade_data['entry_price'],
                        exit_price=trade_data.get('exit_price'),
                        leverage=trade_data['leverage'],
                        realized_pnl=trade_data.get('realized_pnl'),
                        confidence=trade_data['confidence'],
                        status=trade_data['status'],
                        reasoning=trade_data['reasoning']
                    )
                    self.trades.append(trade)
                
                # Carregar métricas
                self.performance_metrics.update(data.get('performance_metrics', {}))
                logger.info(f"Dados carregados: {len(self.trades)} trades")
                
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
    
    def save_trading_data(self):
        """Salvar dados de trading"""
        try:
            data = {
                'trades': [
                    {
                        'id': trade.id,
                        'timestamp': trade.timestamp.isoformat(),
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'amount': trade.amount,
                        'entry_price': trade.entry_price,
                        'exit_price': trade.exit_price,
                        'leverage': trade.leverage,
                        'realized_pnl': trade.realized_pnl,
                        'confidence': trade.confidence,
                        'status': trade.status,
                        'reasoning': trade.reasoning
                    }
                    for trade in self.trades
                ],
                'performance_metrics': self.performance_metrics,
                'last_updated': datetime.now().isoformat()
            }
            
            with open('trading_data.json', 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar dados: {e}")
    
    async def get_market_data(self):
        """Obter dados de mercado"""
        try:
            if self.bitget_api and not self.config.PAPER_TRADING:
                return self.bitget_api.get_futures_ticker(self.config.SYMBOL)
            else:
                # Simular dados de mercado para paper trading
                import random
                base_price = 3500.0
                price_change = random.uniform(-0.05, 0.05)
                current_price = base_price * (1 + price_change)
                
                return {
                    'success': True,
                    'symbol': self.config.SYMBOL,
                    'price': current_price,
                    'change_24h_percent': price_change * 100,
                    'volume': random.uniform(1000000, 5000000),
                    'timestamp': time.time() * 1000
                }
        except Exception as e:
            logger.error(f"Erro ao obter dados de mercado: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_balance(self):
        """Obter saldo da conta"""
        try:
            if self.bitget_api and not self.config.PAPER_TRADING:
                return self.bitget_api.get_futures_balance()
            else:
                # Retornar saldo simulado
                return {
                    'success': True,
                    'total_wallet_balance': self.performance_metrics['current_balance'],
                    'available_balance': self.performance_metrics['current_balance'] * 0.8,
                    'margin_balance': self.performance_metrics['current_balance'] * 0.2,
                    'unrealized_pnl': 0.0,
                    'margin_ratio': 0.2
                }
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return {'success': False, 'error': str(e)}
    
    async def analyze_market(self, market_data, balance_data):
        """Analisar mercado com IA"""
        try:
            analysis_context = {
                'symbol': self.config.SYMBOL,
                'current_price': market_data.get('price', 0),
                'price_change_24h': market_data.get('change_24h_percent', 0),
                'volume': market_data.get('volume', 0),
                'available_balance': balance_data.get('available_balance', 0),
                'recent_trades': len([t for t in self.trades if t.timestamp > datetime.now() - timedelta(hours=24)]),
                'win_rate': self.performance_metrics['win_rate'],
                'current_pnl': self.performance_metrics['total_pnl']
            }
            
            analysis = self.gemini_handler.analyze_futures_market(analysis_context)
            self.last_analysis = {
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise de mercado: {e}")
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'reasoning': f'Erro na análise: {str(e)}'
            }
    
    async def execute_trade(self, signal, confidence, market_data):
        """Executar trade baseado no sinal"""
        try:
            if signal == 'hold' or confidence < self.config.MIN_CONFIDENCE_THRESHOLD:
                return {'executed': False, 'reason': 'Sinal insuficiente'}
            
            current_price = market_data.get('price', 0)
            available_balance = self.performance_metrics['current_balance']
            
            # Calcular leverage baseado na confiança
            leverage = int(self.config.MIN_LEVERAGE + 
                          (self.config.MAX_LEVERAGE - self.config.MIN_LEVERAGE) * confidence)
            
            # Calcular tamanho da posição
            position_size = available_balance * self.config.MAX_POSITION_SIZE * confidence
            quantity = position_size / current_price
            
            # Criar trade
            trade_id = f"{self.config.SYMBOL}_{signal}_{int(time.time())}"
            
            new_trade = Trade(
                id=trade_id,
                timestamp=datetime.now(),
                symbol=self.config.SYMBOL,
                side=signal,
                amount=quantity,
                entry_price=current_price,
                exit_price=None,
                leverage=leverage,
                realized_pnl=None,
                confidence=confidence,
                status='open',
                reasoning=self.last_analysis['analysis'].get('reasoning', '')
            )
            
            if self.config.PAPER_TRADING:
                # Simular execução do trade
                logger.info(f"PAPER TRADE: {signal.upper()} {quantity:.4f} {self.config.SYMBOL} "
                           f"a ${current_price:.2f} com {leverage}x leverage")
                
                # Simular fechamento após alguns segundos (para demo)
                threading.Timer(30.0, self.simulate_trade_close, [new_trade]).start()
                
            else:
                # Executar trade real
                result = self.bitget_api.place_futures_market_order(
                    symbol=self.config.SYMBOL,
                    side=signal,
                    amount=quantity,
                    leverage=leverage
                )
                
                if not result.get('success'):
                    return {'executed': False, 'error': result.get('error')}
                
                logger.info(f"LIVE TRADE: {signal.upper()} {quantity:.4f} {self.config.SYMBOL} "
                           f"a ${current_price:.2f} com {leverage}x leverage")
            
            # Adicionar trade à lista
            self.trades.append(new_trade)
            self.performance_metrics['total_trades'] += 1
            
            # Salvar dados
            self.save_trading_data()
            
            return {
                'executed': True,
                'trade_id': trade_id,
                'side': signal,
                'amount': quantity,
                'entry_price': current_price,
                'leverage': leverage
            }
            
        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}")
            return {'executed': False, 'error': str(e)}
    
    def simulate_trade_close(self, trade):
        """Simular fechamento de trade (apenas para paper trading)"""
        try:
            import random
            
            # Simular movimento de preço
            price_change = random.uniform(-0.03, 0.05)  # -3% a +5%
            exit_price = trade.entry_price * (1 + price_change)
            
            # Calcular P&L
            if trade.side == 'buy':
                pnl_percent = (exit_price - trade.entry_price) / trade.entry_price
            else:
                pnl_percent = (trade.entry_price - exit_price) / trade.entry_price
            
            realized_pnl = pnl_percent * trade.amount * trade.entry_price * trade.leverage / 100
            
            # Atualizar trade
            trade.exit_price = exit_price
            trade.realized_pnl = realized_pnl
            trade.status = 'closed'
            
            # Atualizar métricas
            self.performance_metrics['total_pnl'] += realized_pnl
            self.performance_metrics['current_balance'] += realized_pnl
            
            if realized_pnl > 0:
                self.performance_metrics['profitable_trades'] += 1
            
            if self.performance_metrics['total_trades'] > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['profitable_trades'] / 
                    self.performance_metrics['total_trades']
                )
            
            logger.info(f"Trade fechado: {trade.id} - P&L: ${realized_pnl:.2f}")
            
            # Salvar dados
            self.save_trading_data()
            
        except Exception as e:
            logger.error(f"Erro ao simular fechamento de trade: {e}")
    
    async def trading_loop(self):
        """Loop principal de trading"""
        logger.info("Iniciando loop de trading")
        
        while self.running:
            try:
                # Obter dados de mercado
                market_data = await self.get_market_data()
                if not market_data.get('success'):
                    await asyncio.sleep(self.config.POLL_INTERVAL)
                    continue
                
                # Obter saldo
                balance_data = await self.get_balance()
                if not balance_data.get('success'):
                    await asyncio.sleep(self.config.POLL_INTERVAL)
                    continue
                
                # Analisar mercado
                analysis = await self.analyze_market(market_data, balance_data)
                
                # Verificar se deve executar trade
                signal = analysis.get('signal', 'hold')
                confidence = analysis.get('confidence', 0)
                
                if signal in ['buy', 'sell'] and confidence >= self.config.MIN_CONFIDENCE_THRESHOLD:
                    # Verificar se não há trades recentes
                    recent_trades = [t for t in self.trades if 
                                   t.timestamp > datetime.now() - timedelta(minutes=5)]
                    
                    if len(recent_trades) == 0:
                        trade_result = await self.execute_trade(signal, confidence, market_data)
                        if trade_result.get('executed'):
                            logger.info(f"Trade executado: {trade_result}")
                
                await asyncio.sleep(self.config.POLL_INTERVAL)
                
            except Exception as e:
                logger.error(f"Erro no loop de trading: {e}")
                await asyncio.sleep(10)
    
    async def start_trading(self):
        """Iniciar sistema de trading"""
        if self.running:
            return {'message': 'Sistema já está rodando'}
        
        self.running = True
        logger.info("Sistema de trading iniciado")
        
        # Iniciar loop em background
        asyncio.create_task(self.trading_loop())
        
        return {'message': 'Sistema de trading iniciado com sucesso'}
    
    def stop_trading(self):
        """Parar sistema de trading"""
        self.running = False
        logger.info("Sistema de trading parado")
        return {'message': 'Sistema de trading parado'}
    
    def get_status(self):
        """Obter status do sistema"""
        return {
            'running': self.running,
            'paper_trading': self.config.PAPER_TRADING,
            'total_trades': len(self.trades),
            'performance_metrics': self.performance_metrics,
            'last_analysis': self.last_analysis,
            'open_positions': [t for t in self.trades if t.status == 'open']
        }
    
    def get_trade_history(self):
        """Obter histórico de trades"""
        trades_data = []
        for trade in sorted(self.trades, key=lambda x: x.timestamp, reverse=True):
            trades_data.append({
                'id': trade.id,
                'timestamp': trade.timestamp.isoformat(),
                'symbol': trade.symbol,
                'side': trade.side,
                'amount': trade.amount,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'leverage': trade.leverage,
                'realized_pnl': trade.realized_pnl,
                'confidence': trade.confidence,
                'status': trade.status,
                'reasoning': trade.reasoning
            })
        
        return {'trades': trades_data}
    
    def get_positions(self):
        """Obter posições abertas"""
        open_trades = [t for t in self.trades if t.status == 'open']
        positions = []
        
        for trade in open_trades:
            # Para paper trading, simular preço atual
            if self.config.PAPER_TRADING:
                import random
                current_price = trade.entry_price * (1 + random.uniform(-0.02, 0.02))
            else:
                current_price = trade.entry_price  # Seria obtido da API
            
            # Calcular P&L não realizado
            if trade.side == 'buy':
                pnl_percent = (current_price - trade.entry_price) / trade.entry_price
            else:
                pnl_percent = (trade.entry_price - current_price) / trade.entry_price
            
            unrealized_pnl = pnl_percent * trade.amount * trade.entry_price * trade.leverage / 100
            
            positions.append({
                'symbol': trade.symbol,
                'side': trade.side,
                'size': trade.amount,
                'entry_price': trade.entry_price,
                'current_price': current_price,
                'leverage': trade.leverage,
                'unrealized_pnl': unrealized_pnl,
                'timestamp': trade.timestamp.isoformat()
            })
        
        return {'positions': positions}

# Instância global do sistema
trading_system = ActiveTradingSystem()
