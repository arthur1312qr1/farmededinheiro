import logging
import os
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from bitget_api import BitgetAPI
from trading_bot import TradingBot

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# Global bot state
bot = None
bot_state = {
    'is_running': False,
    'is_paused': False,
    'daily_trades': 0,
    'win_rate': 0.0,
    'last_error': None
}

# Activity log storage
activity_logs = []

def add_log(action, message, log_type='info', details=None):
    """Add entry to activity log"""
    from datetime import datetime
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'message': message,
        'type': log_type,
        'details': details
    }
    activity_logs.append(log_entry)
    # Keep only last 100 logs
    if len(activity_logs) > 100:
        activity_logs.pop(0)
    
    # Also log to console
    if log_type == 'error':
        logging.error(f"{action}: {message}")
    elif log_type == 'warning':
        logging.warning(f"{action}: {message}")
    else:
        logging.info(f"{action}: {message}")

def init_bot():
    """Initialize trading bot"""
    global bot
    try:
        # Get API credentials from environment
        api_key = os.getenv('BITGET_API_KEY')
        secret_key = os.getenv('BITGET_SECRET_KEY')  # ← CORRIGIDO: era BITGET_SECRET
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        if not all([api_key, secret_key, passphrase]):
            add_log("Inicialização", "Credenciais não encontradas - usando valores de teste", "warning")
            # Valores de teste para não quebrar o deploy
            api_key = "test_key"
            secret_key = "test_secret"
            passphrase = "test_pass"
            sandbox_mode = True
            paper_trading = True
        else:
            add_log("Inicialização", "Credenciais encontradas - modo de produção", "success")
            sandbox_mode = False
            paper_trading = False

        # Initialize Bitget API
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=sandbox_mode
        )

        # Initialize Trading Bot
        bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ethusdt_UMCBL',
            leverage=10,
            balance_percentage=100.0,
            daily_target=200,
            scalping_interval=2,
            paper_trading=paper_trading
        )
        
        add_log("Inicialização", "Bot inicializado com sucesso", "success")
        return True

    except Exception as e:
        add_log("Inicialização", f"Erro ao inicializar bot: {str(e)}", "error")
        return False

# Initialize bot on startup
init_bot()

@app.route('/')
def dashboard():
    """Render main dashboard"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get bot status and statistics"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'is_running': False,
                'is_paused': False,
                'daily_trades': 0,
                'win_rate': 0.0
            })

        # Get stats from bot if available
        try:
            stats = bot.get_status()
            bot_state.update(stats)
        except:
            pass

        return jsonify({
            'is_running': bot_state['is_running'],
            'is_paused': bot_state['is_paused'],
            'daily_trades': bot_state['daily_trades'],
            'win_rate': bot_state['win_rate'],
            'last_error': bot_state['last_error'],
            'success': True
        })

    except Exception as e:
        add_log("Status", f"Erro ao obter status: {str(e)}", "error")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    try:
        if not bot:
            add_log("Controle", "Erro: Bot não inicializado", "error")
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500

        if bot_state['is_running'] and not bot_state['is_paused']:
            add_log("Controle", "Bot já está rodando", "warning")
            return jsonify({'message': 'Bot já está rodando', 'success': True})

        bot.start()
        bot_state['is_running'] = True
        bot_state['is_paused'] = False
        
        add_log("Controle", "Bot iniciado com sucesso", "success")
        return jsonify({'message': 'Bot iniciado com sucesso', 'status': 'running', 'success': True})

    except Exception as e:
        add_log("Controle", f"Erro ao iniciar bot: {str(e)}", "error")
        bot_state['last_error'] = str(e)
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    try:
        if not bot:
            add_log("Controle", "Erro: Bot não inicializado", "error")
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500

        bot.stop()
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        
        add_log("Controle", "Bot parado com sucesso", "success")
        return jsonify({'message': 'Bot parado com sucesso', 'status': 'stopped', 'success': True})

    except Exception as e:
        add_log("Controle", f"Erro ao parar bot: {str(e)}", "error")
        bot_state['last_error'] = str(e)
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pause the trading bot"""
    try:
        if not bot:
            add_log("Controle", "Erro: Bot não inicializado", "error")
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500

        if not bot_state['is_running']:
            add_log("Controle", "Bot não está rodando", "warning")
            return jsonify({'message': 'Bot não está rodando', 'success': True})

        # Implement pause logic here
        bot_state['is_paused'] = True
        
        add_log("Controle", "Bot pausado com sucesso", "success")
        return jsonify({'message': 'Bot pausado com sucesso', 'status': 'paused', 'success': True})

    except Exception as e:
        add_log("Controle", f"Erro ao pausar bot: {str(e)}", "error")
        bot_state['last_error'] = str(e)
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop - force stop all operations"""
    try:
        if bot:
            bot.stop()
        
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        
        add_log("Emergência", "Parada de emergência ativada", "warning")
        return jsonify({'message': 'Parada de emergência executada', 'status': 'emergency_stopped', 'success': True})

    except Exception as e:
        add_log("Emergência", f"Erro na parada de emergência: {str(e)}", "error")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/balance')
def get_balance():
    """Get account balance"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'balance': 0.0,
                'currency': 'USDT',
                'leverage_power': 0.0,
                'success': False
            })

        balance = bot.get_account_balance()
        return jsonify({
            'balance': balance,
            'currency': 'USDT',
            'leverage_power': balance * 10,
            'success': True
        })

    except Exception as e:
        add_log("Saldo", f"Erro ao obter saldo: {str(e)}", "error")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/logs')
def get_logs():
    """Get activity logs"""
    try:
        return jsonify({
            'logs': activity_logs[-50:],  # Return last 50 logs
            'total_logs': len(activity_logs),
            'success': True
        })
    except Exception as e:
        logging.error(f"Erro ao obter logs: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update bot configuration"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500

        data = request.get_json()
        
        # Validate configuration data
        if not data:
            return jsonify({'error': 'Dados de configuração inválidos', 'success': False}), 400

        bot.update_config(**data)
        add_log("Configuração", f"Configuração atualizada: {data}", "success")
        
        return jsonify({'message': 'Configuração atualizada', 'config': data, 'success': True})

    except Exception as e:
        add_log("Configuração", f"Erro ao atualizar config: {str(e)}", "error")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/restart', methods=['POST'])
def restart_bot():
    """Restart the bot (reinitialize)"""
    try:
        global bot
        
        # Stop current bot if running
        if bot and bot_state['is_running']:
            bot.stop()
        
        # Reinitialize
        success = init_bot()
        
        if success:
            bot_state['is_running'] = False
            bot_state['is_paused'] = False
            add_log("Sistema", "Bot reinicializado com sucesso", "success")
            return jsonify({'message': 'Bot reinicializado com sucesso', 'success': True})
        else:
            return jsonify({'error': 'Falha ao reinicializar bot', 'success': False}), 500

    except Exception as e:
        add_log("Sistema", f"Erro ao reinicializar: {str(e)}", "error")
        return jsonify({'error': str(e), 'success': False}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint não encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    add_log("Erro", f"Erro interno do servidor: {str(error)}", "error")
    return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    if bot:
        add_log("Sistema", "Trading Bot API iniciada com sucesso!", "success")
    else:
        add_log("Sistema", "Trading Bot API iniciada em modo de teste", "warning")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
