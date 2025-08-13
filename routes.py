from flask import render_template, request, jsonify, redirect, url_for
from app import app, db
from models import Trade, BotStatus, Signal
from trading_engine import trading_engine
from config import Config
import logging

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get bot status
        bot_status = BotStatus.query.first()
        if not bot_status:
            # Create default bot status
            bot_status = BotStatus()
            bot_status.balance = 1000.0 if Config.PAPER_TRADING else 0.0
            db.session.add(bot_status)
            db.session.commit()
        
        # Get recent trades
        recent_trades = Trade.query.order_by(Trade.entry_time.desc()).limit(10).all()
        
        # Get recent signals
        recent_signals = Signal.query.order_by(Signal.timestamp.desc()).limit(5).all()
        
        return render_template('index.html', 
                             bot_status=bot_status,
                             recent_trades=recent_trades,
                             recent_signals=recent_signals,
                             paper_trading=Config.PAPER_TRADING)
    
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        return render_template('error.html', error=str(e))

@app.route('/api/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    try:
        if trading_engine.start():
            return jsonify({'success': True, 'message': 'Bot started successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to start bot'})
    
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    try:
        trading_engine.stop()
        return jsonify({'success': True, 'message': 'Bot stopped successfully'})
    
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot_status')
def get_bot_status():
    """Get current bot status"""
    try:
        bot_status = BotStatus.query.first()
        if bot_status:
            return jsonify(bot_status.to_dict())
        else:
            return jsonify({'error': 'No bot status found'})
    
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    try:
        limit = request.args.get('limit', 50, type=int)
        trades = Trade.query.order_by(Trade.entry_time.desc()).limit(limit).all()
        return jsonify([trade.to_dict() for trade in trades])
    
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/signals')
def get_signals():
    """Get recent signals"""
    try:
        limit = request.args.get('limit', 20, type=int)
        signals = Signal.query.order_by(Signal.timestamp.desc()).limit(limit).all()
        return jsonify([signal.to_dict() for signal in signals])
    
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/engine_status')
def get_engine_status():
    """Get trading engine status"""
    try:
        status = trading_engine.get_status()
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/close_position/<int:trade_id>', methods=['POST'])
def close_position(trade_id):
    """Manually close a position"""
    try:
        trade = Trade.query.get(trade_id)
        if not trade:
            return jsonify({'success': False, 'error': 'Trade not found'})
        
        if trade.status != 'open':
            return jsonify({'success': False, 'error': 'Trade is not open'})
        
        # Get current price
        from bitget_api import BitgetAPI
        api = BitgetAPI()
        price_result = api.get_current_price()
        
        if not price_result['success']:
            return jsonify({'success': False, 'error': 'Failed to get current price'})
        
        current_price = price_result['price']
        
        # Close position via API
        close_result = api.close_position(trade.side, trade.quantity)
        
        if close_result['success']:
            # Update trade record
            trade.exit_price = current_price
            trade.status = 'closed'
            trade.exit_time = db.func.now()
            
            # Calculate P&L
            if trade.side == 'long':
                pnl = (current_price - trade.entry_price) * trade.quantity * trade.leverage
            else:
                pnl = (trade.entry_price - current_price) * trade.quantity * trade.leverage
            
            trade.pnl = pnl
            trade.pnl_percentage = (pnl / (trade.entry_price * trade.quantity)) * 100
            
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Position closed successfully'})
        else:
            return jsonify({'success': False, 'error': close_result.get('error')})
    
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/config')
def get_config():
    """Get current configuration"""
    try:
        config_data = {
            'symbol': Config.SYMBOL,
            'leverage': Config.LEVERAGE,
            'max_position_percent': Config.MAX_POSITION_PERCENT,
            'stop_loss_percent': Config.STOP_LOSS_PERCENT,
            'take_profit_percent': Config.TAKE_PROFIT_PERCENT,
            'paper_trading': Config.PAPER_TRADING,
            'min_signal_confidence': Config.MIN_SIGNAL_CONFIDENCE,
            'timeframes': Config.TIMEFRAMES,
            'analysis_interval': Config.ANALYSIS_INTERVAL
        }
        return jsonify(config_data)
    
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({'error': str(e)})

@app.route('/settings')
def settings():
    """Settings page"""
    try:
        return render_template('settings.html', config=Config)
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        return render_template('error.html', error=str(e))

@app.route('/trades')
def trades_page():
    """Trades history page"""
    try:
        trades = Trade.query.order_by(Trade.entry_time.desc()).limit(100).all()
        return render_template('trades.html', trades=trades)
    except Exception as e:
        logger.error(f"Error loading trades page: {e}")
        return render_template('error.html', error=str(e))

@app.route('/analytics')
def analytics():
    """Analytics dashboard page"""
    try:
        # Get performance metrics
        total_trades = Trade.query.count()
        winning_trades = Trade.query.filter(Trade.pnl > 0).count()
        losing_trades = Trade.query.filter(Trade.pnl < 0).count()
        
        total_pnl = db.session.query(db.func.sum(Trade.pnl)).scalar() or 0
        
        win_rate = (winning_trades / max(total_trades, 1)) * 100
        
        # Get recent performance
        recent_trades = Trade.query.order_by(Trade.entry_time.desc()).limit(30).all()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'recent_trades': [trade.to_dict() for trade in recent_trades]
        }
        
        return render_template('analytics.html', metrics=metrics)
    
    except Exception as e:
        logger.error(f"Error loading analytics: {e}")
        return render_template('error.html', error=str(e))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('error.html', error='Internal server error'), 500
