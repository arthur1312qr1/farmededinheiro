from app import app, trading_bot, socketio
from flask import render_template, jsonify
from flask_socketio import emit, join_room, leave_room
import logging

logger = logging.getLogger(__name__)

# Rota para a página principal (a interface do bot)
@app.route('/')
def index():
    return render_template('index.html')

# Rota para o status do bot
@app.route('/api/bot/status')
def get_bot_status():
    if trading_bot:
        stats = trading_bot.get_statistics()
        return jsonify({
            'success': True,
            'data': {
                'is_running': stats['is_running'],
                'total_trades': stats['total_trades'],
                'winning_trades': stats['winning_trades'],
                'win_rate': stats['win_rate'],
                'total_pnl': stats['total_pnl'],
                'consecutive_losses': stats['consecutive_losses']
            }
        })
    return jsonify({'success': False, 'error': 'Bot not initialized'})

# Rota para o saldo da conta
@app.route('/api/balance')
def get_balance():
    if trading_bot:
        balance_data = trading_bot.api.get_account_balance()
        return jsonify(balance_data)
    return jsonify({'success': False, 'error': 'Bot not initialized'})

# Rota para o preço atual
@app.route('/api/price/<symbol>')
def get_price(symbol):
    if trading_bot:
        price_data = trading_bot.api.get_current_price(symbol)
        return jsonify(price_data)
    return jsonify({'success': False, 'error': 'Bot not initialized'})

# Rota para iniciar o bot
@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    if trading_bot:
        if trading_bot.start():
            return jsonify({'success': True, 'message': 'Bot started'})
    return jsonify({'success': False, 'error': 'Failed to start bot'})

# Rota para parar o bot
@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    if trading_bot:
        trading_bot.stop()
        return jsonify({'success': True, 'message': 'Bot stopped'})
    return jsonify({'success': False, 'error': 'Bot not initialized'})

# Rotas para SocketIO (websocket_handler)
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected via WebSocket')
    emit('bot_status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')
