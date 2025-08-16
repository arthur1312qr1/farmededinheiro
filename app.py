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
        # Get API credentials from environment - USAR NOMES CORRETOS DO RENDER
        api_key = os.getenv('BITGET_API_KEY')
        secret_key = os.getenv('BITGET_SECRET')  # ← CORRIGIDO: usar BITGET_SECRET
        passphrase = os.getenv('BITGET_PASSPHRASE')
        
        # Debug: Log se as variáveis estão sendo carregadas
        logging.info(f"🔍 Verificando credenciais:")
        logging.info(f"   API_KEY: {'✅ OK' if api_key else '❌ VAZIO'}")
        logging.info(f"   SECRET: {'✅ OK' if secret_key else '❌ VAZIO'}")
        logging.info(f"   PASSPHRASE: {'✅ OK' if passphrase else '❌ VAZIO'}")
        
        # FAIL HARD if no credentials
        if not all([api_key, secret_key, passphrase]):
            logging.error("❌ ERRO CRÍTICO: Credenciais obrigatórias não encontradas")
            logging.error("❌ Variáveis necessárias: BITGET_API_KEY, BITGET_SECRET, BITGET_PASSPHRASE")
            raise Exception("Credenciais não configuradas no ambiente")
        
        # Validate that credentials are not test values
        if api_key == "test_key" or secret_key == "test_secret" or passphrase == "test_pass":
            logging.error("❌ ERRO: Credenciais de teste detectadas - configure credenciais reais")
            raise Exception("Credenciais de teste não são permitidas")
        
        logging.info("✅ Credenciais reais encontradas - INICIANDO MODO PRODUÇÃO")
        
        # Initialize Bitget API - PRODUCTION MODE ONLY
        bitget_api = BitgetAPI(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=False  # PRODUCTION MODE
        )
        
        # Initialize Trading Bot - REAL TRADING ONLY
        trading_bot = TradingBot(
            bitget_api=bitget_api,
            symbol='ethusdt_UMCBL',
            leverage=10,
            balance_percentage=100.0,
            daily_target=200,
            scalping_interval=2,
            paper_trading=False  # REAL TRADING
        )
        
        logging.info("🚀 Bot inicializado em MODO PRODUÇÃO - Trading Real")
        return trading_bot
        
    except Exception as e:
        logging.error(f"❌ Falha na inicialização: {e}")
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
            return jsonify({
                'error': 'Bot não inicializado - verifique credenciais no Render.com', 
                'status': 'error',
                'is_running': False,
                'is_paused': False,
                'daily_trades': 0,
                'win_rate': 0.0
            }), 500
        
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
            return jsonify({
                'error': 'Bot não inicializado - configure credenciais no Render.com',
                'success': False
            }), 500
        
        bot.start()
        logging.info("🟢 Bot INICIADO - Trading Real Ativo")
        return jsonify({
            'message': 'Bot iniciado - Trading Real Ativo',
            'status': 'running',
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao iniciar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Stop the real trading bot"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        bot.stop()
        logging.info("🔴 Bot PARADO")
        return jsonify({
            'message': 'Bot parado',
            'status': 'stopped',
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao parar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pause the real trading bot"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        if hasattr(bot, 'pause'):
            bot.pause()
            logging.info("⏸️ Bot PAUSADO")
            return jsonify({
                'message': 'Bot pausado',
                'status': 'paused',
                'success': True
            })
        else:
            return jsonify({
                'error': 'Método pause não disponível',
                'success': False
            }), 400
            
    except Exception as e:
        logging.error(f"❌ Erro ao pausar bot: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Emergency stop the bot"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        bot.stop()
        logging.warning("🚨 PARADA DE EMERGÊNCIA ATIVADA")
        return jsonify({
            'message': 'Parada de emergência executada',
            'status': 'emergency_stopped',
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro na parada de emergência: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/balance')
def get_balance():
    """Get real account balance"""
    try:
        if not bot:
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
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
            return jsonify({
                'error': 'Bot não inicializado',
                'success': False
            }), 500
        
        data = request.get_json()
        bot.update_config(**data)
        logging.info(f"⚙️ Configuração atualizada: {data}")
        return jsonify({
            'message': 'Configuração atualizada',
            'config': data,
            'success': True
        })
        
    except Exception as e:
        logging.error(f"❌ Erro ao atualizar config: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    if bot:
        logging.info("🚀 Trading Bot API iniciada - MODO PRODUÇÃO")
        logging.info("💰 TRADING REAL ATIVO")
    else:
        logging.error("❌ FALHA CRÍTICA: Bot não pôde ser inicializado")
        logging.error("❌ Configure as variáveis no Render.com:")
        logging.error("❌   BITGET_API_KEY")
        logging.error("❌   BITGET_SECRET") 
        logging.error("❌   BITGET_PASSPHRASE")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
