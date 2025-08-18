from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Criar aplicação Flask
app = Flask(__name__)
CORS(app)

# Variáveis globais para simular estado do bot
bot_state = {
    'is_running': False,
    'is_paused': False,
    'trades_today': 0,
    'profitable_trades': 0,
    'total_trades': 0,
    'total_profit': 0.0,
    'consecutive_wins': 0,
    'current_position': None,
    'position_side': None,
    'position_size': 0.0,
    'entry_price': None,
    'symbol': 'ETH/USDT:USDT',
    'leverage': 10,
    'paper_trading': True,
    'balance': 1000.0
}

activity_logs = [
    {
        'timestamp': datetime.now().isoformat(),
        'action': 'Sistema Iniciado',
        'message': 'Trading Bot Dashboard carregado com sucesso',
        'type': 'info',
        'details': 'Modo simulação ativo'
    }
]

@app.route('/')
def index():
    """Servir página principal"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Retornar status atual do bot"""
    current_time = datetime.now()
    current_hour = current_time.hour
    
    # Calcular progresso diário
    hours_passed = max(1, current_hour - 8) if current_hour >= 8 else max(1, current_hour + 16)
    expected_trades_by_now = (240 / 16) * hours_passed
    trade_deficit = max(0, expected_trades_by_now - bot_state['trades_today'])
    
    urgency_level = "NORMAL"
    if trade_deficit > 30:
        urgency_level = "CRÍTICO"
    elif trade_deficit > 15:
        urgency_level = "ALTO"
    elif trade_deficit > 5:
        urgency_level = "MÉDIO"
    
    win_rate = (bot_state['profitable_trades'] / max(1, bot_state['total_trades'])) * 100 if bot_state['total_trades'] > 0 else 95.0
    
    return jsonify({
        'bot_status': {
            'is_running': bot_state['is_running'],
            'is_paused': bot_state['is_paused'],
            'symbol': bot_state['symbol'],
            'leverage': bot_state['leverage'],
            'paper_trading': bot_state['paper_trading']
        },
        'daily_progress': {
            'trades_today': bot_state['trades_today'],
            'min_target': 240,
            'target': 280,
            'progress_percent': round((bot_state['trades_today'] / 240) * 100, 1),
            'expected_by_now': round(expected_trades_by_now),
            'deficit': round(trade_deficit),
            'urgency_level': urgency_level
        },
        'performance': {
            'profitable_trades': bot_state['profitable_trades'],
            'losing_trades': bot_state['total_trades'] - bot_state['profitable_trades'],
            'win_rate': round(win_rate, 2),
            'target_win_rate': 95.0,
            'total_profit': round(bot_state['total_profit'], 4),
            'consecutive_wins': bot_state['consecutive_wins']
        },
        'current_position': {
            'active': bot_state['current_position'] is not None,
            'side': bot_state['position_side'],
            'size': bot_state['position_size'],
            'entry_price': bot_state['entry_price'],
            'unrealized_pnl': 0.0
        },
        'market_data': {
            'price': 2500.0 + (hash(str(datetime.now().second)) % 100 - 50),
            'volume': 1000000,
            'high': 2550.0,
            'low': 2450.0,
            'change': 1.2
        },
        'timestamp': current_time.isoformat()
    })

@app.route('/api/balance')
def get_balance():
    """Retornar saldo da conta"""
    return jsonify({
        'balance': bot_state['balance'],
        'free': bot_state['balance'],
        'used': 0.0,
        'total': bot_state['balance']
    })

@app.route('/api/logs')
def get_logs():
    """Retornar logs de atividade"""
    return jsonify({
        'logs': activity_logs[-50:]  # Últimos 50 logs
    })

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Iniciar o bot"""
    try:
        bot_state['is_running'] = True
        bot_state['is_paused'] = False
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'Bot Iniciado',
            'message': 'Trading bot iniciado com sucesso',
            'type': 'success',
            'details': f'Símbolo: {bot_state["symbol"]}, Leverage: {bot_state["leverage"]}x'
        })
        
        logger.info("🚀 Bot iniciado")
        return jsonify({'success': True, 'message': 'Bot iniciado com sucesso'})
        
    except Exception as e:
        logger.error(f"Erro ao iniciar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pausar o bot"""
    try:
        bot_state['is_paused'] = True
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'Bot Pausado',
            'message': 'Trading bot pausado temporariamente',
            'type': 'warning',
            'details': 'Todas as operações suspensas'
        })
        
        logger.info("⏸️ Bot pausado")
        return jsonify({'success': True, 'message': 'Bot pausado'})
        
    except Exception as e:
        logger.error(f"Erro ao pausar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Parar o bot"""
    try:
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'Bot Parado',
            'message': 'Trading bot parado completamente',
            'type': 'error',
            'details': 'Todas as operações encerradas'
        })
        
        logger.info("🛑 Bot parado")
        return jsonify({'success': True, 'message': 'Bot parado'})
        
    except Exception as e:
        logger.error(f"Erro ao parar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Parada de emergência"""
    try:
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        bot_state['current_position'] = None
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'PARADA DE EMERGÊNCIA',
            'message': 'Parada de emergência ativada - todas as posições fechadas',
            'type': 'error',
            'details': 'Sistema em modo segurança'
        })
        
        logger.warning("🚨 Parada de emergência ativada")
        return jsonify({'success': True, 'message': 'Parada de emergência executada'})
        
    except Exception as e:
        logger.error(f"Erro na parada de emergência: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market/<symbol>')
def get_market_data(symbol):
    """Retornar dados de mercado para um símbolo"""
    return jsonify({
        'symbol': symbol,
        'price': 2500.0 + (hash(str(datetime.now().second)) % 100 - 50),
        'volume': 1000000,
        'high': 2550.0,
        'low': 2450.0,
        'change': 1.2,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
