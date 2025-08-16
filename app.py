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

# Initialize APIs and Bot
def init_bot():
    try:
        # Get API credentials from environment
        api_key = os.getenv('BITGET_API_KEY')
        secret_key = os.getenv('BITGET_SECRET_KEY')  # ← CORRIGIDO: era BITGET_SECRET
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        if not all([api_key, secret_key, passphrase]):
            logging.error("❌ Credenciais não encontradas - usando valores de teste")
            # Valores de teste para não quebrar o deploy
            api_key = "test_key"
            secret_key = "test_secret"
            passphrase = "test_pass"
        
        # Initialize Bitget API
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=True  # Usar sandbox se não tem credenciais reais
        )
        
        # Initialize Trading Bot
        trading_bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ethusdt_UMCBL',
            leverage=10,
            balance_percentage=100.0,
            daily_target=200,
            scalping_interval=2,
            paper_trading=True  # Paper trading se não tem credenciais
        )
        
        return trading_bot
        
    except Exception as e:
        logging.error(f"❌ Erro ao inicializar bot: {e}")
        return None

# Initialize bot globally
bot = init_bot()

@app.route('/')
def home():
    """Serve the dashboard website"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get real bot status"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'status': 'error'}), 500
        
        stats = bot.get_status()
        return jsonify(stats)
        
    except Exception as e:
        logging.error(f"❌ Erro ao obter status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Start the real trading bot"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        bot.start()
        return jsonify({'message': 'Bot iniciado com sucesso', 'status': 'running', 'success': True})
        
    except Exception as e:
        logging.error(f"❌ Erro ao iniciar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the real trading bot"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        bot.stop()
        return jsonify({'message': 'Bot parado com sucesso', 'status': 'stopped', 'success': True})
        
    except Exception as e:
        logging.error(f"❌ Erro ao parar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pause the real trading bot"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        # Call real pause method if exists
        if hasattr(bot, 'pause'):
            bot.pause()
            return jsonify({'message': 'Bot pausado com sucesso', 'status': 'paused', 'success': True})
        else:
            return jsonify({'error': 'Método pause não disponível', 'success': False}), 400
            
    except Exception as e:
        logging.error(f"❌ Erro ao pausar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop the bot"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        bot.stop()  # Force stop
        return jsonify({'message': 'Parada de emergência executada', 'status': 'emergency_stopped', 'success': True})
        
    except Exception as e:
        logging.error(f"❌ Erro na parada de emergência: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/balance')
def get_balance():
    """Get real account balance"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 500
        
        balance = bot.get_account_balance()
        return jsonify({
            'balance': balance,
            'currency': 'USDT',
            'leverage_power': balance * 10,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao obter saldo: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/logs')
def get_logs():
    """Get logs from bot"""
    try:
        # Return basic response since we don't simulate logs
        return jsonify({
            'message': 'Verifique os logs no console do servidor',
            'status': 'active',
            'logs': [],
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update bot configuration"""
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado', 'success': False}), 500
        
        data = request.get_json()
        bot.update_config(**data)
        return jsonify({'message': 'Configuração atualizada', 'config': data, 'success': True})
        
    except Exception as e:
        logging.error(f"❌ Erro ao atualizar config: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    if bot:
        logging.info("🚀 Trading Bot API iniciada com sucesso!")
    else:
        logging.error("❌ Falha ao inicializar Trading Bot - usando modo de teste")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
