from flask_socketio import emit, disconnect
from app import socketio, db
from models import BotStatus, Trade, Signal
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to WebSocket")
    
    # Send initial data to client
    try:
        # Send bot status
        bot_status = BotStatus.query.first()
        if bot_status:
            emit('status_update', bot_status.to_dict())
        
        # Send recent trades
        recent_trades = Trade.query.order_by(Trade.entry_time.desc()).limit(5).all()
        emit('trades_update', [trade.to_dict() for trade in recent_trades])
        
        # Send recent signals
        recent_signals = Signal.query.order_by(Signal.timestamp.desc()).limit(5).all()
        emit('signals_update', [signal.to_dict() for signal in recent_signals])
        
    except Exception as e:
        logger.error(f"Error sending initial data: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from WebSocket")

@socketio.on('request_status')
def handle_status_request():
    """Handle status update request"""
    try:
        bot_status = BotStatus.query.first()
        if bot_status:
            emit('status_update', bot_status.to_dict())
        else:
            emit('status_update', {
                'is_active': False,
                'balance': 0.0,
                'daily_pnl': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'current_price': 0.0
            })
    except Exception as e:
        logger.error(f"Error handling status request: {e}")
        emit('error', {'message': 'Failed to get status'})

@socketio.on('request_trades')
def handle_trades_request(data=None):
    """Handle trades data request"""
    try:
        limit = data.get('limit', 10) if data else 10
        trades = Trade.query.order_by(Trade.entry_time.desc()).limit(limit).all()
        emit('trades_update', [trade.to_dict() for trade in trades])
    except Exception as e:
        logger.error(f"Error handling trades request: {e}")
        emit('error', {'message': 'Failed to get trades'})

@socketio.on('request_signals')
def handle_signals_request(data=None):
    """Handle signals data request"""
    try:
        limit = data.get('limit', 5) if data else 5
        signals = Signal.query.order_by(Signal.timestamp.desc()).limit(limit).all()
        emit('signals_update', [signal.to_dict() for signal in signals])
    except Exception as e:
        logger.error(f"Error handling signals request: {e}")
        emit('error', {'message': 'Failed to get signals'})

@socketio.on('ping')
def handle_ping():
    """Handle ping request for connection testing"""
    emit('pong', {'timestamp': datetime.utcnow().isoformat()})

# Global functions for emitting updates from other modules

def emit_status_update(status_data):
    """Emit status update to all connected clients"""
    try:
        socketio.emit('status_update', status_data)
    except Exception as e:
        logger.error(f"Error emitting status update: {e}")

def emit_trade_update(trade_data):
    """Emit trade update to all connected clients"""
    try:
        socketio.emit('trade_update', trade_data)
    except Exception as e:
        logger.error(f"Error emitting trade update: {e}")

def emit_new_trade(trade_data):
    """Emit new trade to all connected clients"""
    try:
        socketio.emit('new_trade', trade_data)
    except Exception as e:
        logger.error(f"Error emitting new trade: {e}")

def emit_trade_closed(data):
    """Emit trade closed event to all connected clients"""
    try:
        socketio.emit('trade_closed', data)
    except Exception as e:
        logger.error(f"Error emitting trade closed: {e}")

def emit_signal_update(signal_data):
    """Emit signal update to all connected clients"""
    try:
        socketio.emit('signal_update', signal_data)
    except Exception as e:
        logger.error(f"Error emitting signal update: {e}")

def emit_price_update(price_data):
    """Emit price update to all connected clients"""
    try:
        socketio.emit('price_update', price_data)
    except Exception as e:
        logger.error(f"Error emitting price update: {e}")

def emit_error(error_message):
    """Emit error message to all connected clients"""
    try:
        socketio.emit('error', {'message': error_message, 'timestamp': datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"Error emitting error message: {e}")
