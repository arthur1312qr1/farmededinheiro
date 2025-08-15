"""
SocketIO Manager for Real-time Trading Updates
Handles WebSocket connections and real-time data broadcasting
"""

import logging
from datetime import datetime
from flask_socketio import emit, join_room, leave_room
import threading
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SocketIOManager:
    def __init__(self, socketio, trading_bot):
        """Initialize SocketIO manager"""
        self.socketio = socketio
        self.trading_bot = trading_bot
        self.connected_clients = set()
        self.is_broadcasting = False
        
        # Register event handlers
        self._register_handlers()
        
        # Start broadcasting thread
        self.start_broadcasting()
        
        logger.info("🔌 SocketIO Manager inicializado")
    
    def _register_handlers(self):
        """Register SocketIO event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            client_id = str(id(threading.current_thread()))
            self.connected_clients.add(client_id)
            join_room('trading_room')
            
            logger.info(f"🔗 Cliente conectado: {client_id}")
            
            # Send initial data
            self.emit_trading_status()
            self.emit_bot_status()
            
            emit('connection_status', {
                'status': 'connected',
                'timestamp': datetime.now().isoformat(),
                'message': 'Conectado ao servidor de trading'
            })
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            client_id = str(id(threading.current_thread()))
            self.connected_clients.discard(client_id)
            leave_room('trading_room')
            
            logger.info(f"🔌 Cliente desconectado: {client_id}")
        
        @self.socketio.on('get_trading_status')
        def handle_get_trading_status():
            self.emit_trading_status()
        
        @self.socketio.on('get_bot_status')
        def handle_get_bot_status():
            self.emit_bot_status()
        
        @self.socketio.on('toggle_bot')
        def handle_toggle_bot():
            try:
                if self.trading_bot.is_running:
                    if self.trading_bot.is_paused:
                        self.trading_bot.pause()  # Resume
                        action = 'resumed'
                    else:
                        self.trading_bot.pause()  # Pause
                        action = 'paused'
                else:
                    self.trading_bot.start()
                    action = 'started'
                
                self.emit_bot_status()
                emit('bot_action_result', {
                    'success': True,
                    'action': action,
                    'message': f'Bot {action} com sucesso'
                })
                
            except Exception as e:
                logger.error(f"❌ Erro ao controlar bot: {e}")
                emit('bot_action_result', {
                    'success': False,
                    'error': str(e),
                    'message': 'Erro ao controlar bot'
                })
        
        @self.socketio.on('emergency_stop')
        def handle_emergency_stop():
            try:
                self.trading_bot.emergency_stop()
                self.emit_bot_status()
                emit('bot_action_result', {
                    'success': True,
                    'action': 'emergency_stopped',
                    'message': 'Parada de emergência ativada'
                })
                
            except Exception as e:
                logger.error(f"❌ Erro na parada de emergência: {e}")
                emit('bot_action_result', {
                    'success': False,
                    'error': str(e),
                    'message': 'Erro na parada de emergência'
                })
        
        @self.socketio.on('close_position')
        def handle_close_position():
            try:
                success = self.trading_bot.close_position()
                self.emit_trading_status()
                emit('position_action_result', {
                    'success': success,
                    'action': 'closed',
                    'message': 'Posição fechada com sucesso' if success else 'Nenhuma posição para fechar'
                })
                
            except Exception as e:
                logger.error(f"❌ Erro ao fechar posição: {e}")
                emit('position_action_result', {
                    'success': False,
                    'error': str(e),
                    'message': 'Erro ao fechar posição'
                })
    
    def start_broadcasting(self):
        """Start real-time broadcasting thread"""
        if self.is_broadcasting:
            return
        
        self.is_broadcasting = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self.broadcast_thread.start()
        
        logger.info("📡 Broadcasting iniciado")
    
    def stop_broadcasting(self):
        """Stop real-time broadcasting"""
        self.is_broadcasting = False
        logger.info("📡 Broadcasting parado")
    
    def _broadcast_loop(self):
        """Main broadcasting loop"""
        while self.is_broadcasting:
            try:
                if self.connected_clients:
                    # Broadcast trading updates every 5 seconds
                    self.emit_trading_status()
                    self.emit_price_update()
                    
                    # Broadcast bot status every 10 seconds
                    if int(time.time()) % 10 == 0:
                        self.emit_bot_status()
                
                time.sleep(5)  # 5 second intervals
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de broadcasting: {e}")
                time.sleep(10)  # Wait longer on error
    
    def emit_trading_status(self):
        """Emit current trading status"""
        try:
            status = self.trading_bot.get_status()
            
            self.socketio.emit('trading_status_update', {
                'account': {
                    'balance': 12847.32,  # Would get from actual API
                    'daily_pnl': status['daily_pnl'],
                    'total_volume': status['total_volume']
                },
                'stats': {
                    'trades_count': f"{status['daily_trades']}/{status['target_trades']}",
                    'win_rate': f"{status['win_rate']:.1f}%",
                    'uptime': status['uptime']
                },
                'position': status['current_position'],
                'activity_log': status['activity_log']
            }, room='trading_room')
            
        except Exception as e:
            logger.error(f"❌ Erro ao emitir status de trading: {e}")
    
    def emit_bot_status(self):
        """Emit bot status"""
        try:
            status = self.trading_bot.get_status()
            
            self.socketio.emit('bot_status_update', {
                'is_running': status['is_running'],
                'is_paused': status['is_paused'],
                'uptime': status['uptime'],
                'config': status['config']
            }, room='trading_room')
            
        except Exception as e:
            logger.error(f"❌ Erro ao emitir status do bot: {e}")
    
    def emit_price_update(self):
        """Emit price updates"""
        try:
            # Get current market data
            market_data = self.trading_bot.bitget_api.get_market_data(self.trading_bot.symbol)
            
            if market_data:
                self.socketio.emit('price_update', {
                    'eth_price': market_data['price'],
                    'change_24h': market_data.get('change_24h', 0),
                    'volume': market_data.get('volume', 0),
                    'timestamp': market_data.get('timestamp', int(time.time() * 1000))
                }, room='trading_room')
            
        except Exception as e:
            logger.error(f"❌ Erro ao emitir atualização de preço: {e}")
    
    def emit_trade_execution(self, trade_data: Dict):
        """Emit trade execution notification"""
        try:
            self.socketio.emit('trade_executed', trade_data, room='trading_room')
            logger.info(f"📢 Trade executado notificado: {trade_data}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao notificar execução de trade: {e}")
    
    def emit_analysis_update(self, analysis_data: Dict):
        """Emit AI analysis update"""
        try:
            self.socketio.emit('analysis_update', analysis_data, room='trading_room')
            
        except Exception as e:
            logger.error(f"❌ Erro ao emitir análise AI: {e}")
    
    def emit_log_entry(self, log_entry: Dict):
        """Emit new log entry"""
        try:
            self.socketio.emit('new_log_entry', log_entry, room='trading_room')
            
        except Exception as e:
            logger.error(f"❌ Erro ao emitir entrada de log: {e}")
    
    def broadcast_message(self, message_type: str, data: Dict):
        """Broadcast custom message to all clients"""
        try:
            self.socketio.emit(message_type, data, room='trading_room')
            
        except Exception as e:
            logger.error(f"❌ Erro ao broadcast mensagem: {e}")
