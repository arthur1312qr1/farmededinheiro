import logging
import os
from flask import Flask, jsonify, request
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
        secret_key = os.getenv('BITGET_SECRET_KEY') 
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        if not all([api_key, secret_key, passphrase]):
            raise ValueError("Credenciais da API não encontradas nas variáveis de ambiente")
        
        # Initialize Bitget API
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=False
        )
        
        # Initialize Trading Bot - CORREÇÃO AQUI
        trading_bot = TradingBot(
            bitget_api=bitget_api,  # Passar a instância da API
            symbol='ethusdt_UMCBL',
            leverage=10,
            balance_percentage=100.0,  # 100% do saldo
            daily_target=200,
            scalping_interval=2,
            paper_trading=False
        )
        
        return trading_bot
        
    except Exception as e:
        logging.error(f"❌ Erro ao inicializar bot: {e}")
        return None

# Initialize bot globally
bot = init_bot()

@app.route('/')
def home():
    return jsonify({
        'status': 'Trading Bot ativo',
        'version': '2.0',
        'endpoints': ['/api/status', '/api/start', '/api/stop', '/api/balance', '/api/logs']
    })

@app.route('/api/status')
def get_status():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 500
            
        stats = bot.get_status()
        return jsonify(stats)
    except Exception as e:
        logging.error(f"❌ Erro ao obter status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_bot():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 500
            
        bot.start()
        return jsonify({'message': 'Bot iniciado com sucesso', 'status': 'running'})
    except Exception as e:
        logging.error(f"❌ Erro ao iniciar bot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 500
            
        bot.stop()
        return jsonify({'message': 'Bot parado com sucesso', 'status': 'stopped'})
    except Exception as e:
        logging.error(f"❌ Erro ao parar bot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/balance')
def get_balance():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 500
            
        balance = bot.get_account_balance()
        return jsonify({
            'balance': balance,
            'currency': 'USDT',
            'leverage_power': balance * 10
        })
    except Exception as e:
        logging.error(f"❌ Erro ao obter saldo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/logs')
def get_logs():
    return jsonify({
        'message': 'Verifique os logs no console do servidor',
        'status': 'active'
    })

@app.route('/api/config', methods=['POST'])
def update_config():
    try:
        if not bot:
            return jsonify({'error': 'Bot não inicializado'}), 500
            
        data = request.get_json()
        bot.update_config(**data)
        return jsonify({'message': 'Configuração atualizada', 'config': data})
    except Exception as e:
        logging.error(f"❌ Erro ao atualizar config: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if bot:
        logging.info("🚀 Trading Bot API iniciada com sucesso!")
    else:
        logging.error("❌ Falha ao inicializar Trading Bot")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
