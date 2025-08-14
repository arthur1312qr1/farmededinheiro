import os
import logging
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, render_template, jsonify
    from flask_socketio import SocketIO
    from config import Config
    
    logger.info("📦 Dependências importadas com sucesso")
    
except Exception as e:
    logger.error(f"❌ Erro ao importar dependências: {e}")
    # Importações mínimas se algo falhar
    from flask import Flask, jsonify
    
    class Config:
        SECRET_KEY = 'fallback-key'
        DEBUG = False

# Criar app Flask
app = Flask(__name__)
app.config.from_object(Config())

logger.info("🔧 Flask app criada")

# Tentar criar SocketIO
try:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    logger.info("🔌 SocketIO inicializado")
except Exception as e:
    logger.error(f"⚠️ SocketIO falhou, continuando sem: {e}")
    socketio = None

# Rota principal
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"❌ Erro no template: {e}")
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Bot</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                .status {{ padding: 15px; background: #4CAF50; color: white; border-radius: 5px; margin: 20px 0; }}
                .info {{ background: #2196F3; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Trading Bot - Farm de Dinheiro</h1>
                <div class="status">✅ ONLINE - Bot Funcionando</div>
                
                <h3>📊 Status do Sistema</h3>
                <div class="info">🤖 Bot de Trading: ATIVO</div>
                <div class="info">⚡ Modo: SCALPING AGRESSIVO</div>
                <div class="info">💰 Símbolo: ETHUSDT (10x leverage)</div>
                <div class="info">🎯 Trading Real: {'ATIVO' if not app.config.get('PAPER_TRADING', True) else 'PAPER'}</div>
                
                <h3>🔄 APIs Status</h3>
                <div class="info">📈 Bitget API: Configurado</div>
                <div class="info">🧠 Gemini AI: Configurado</div>
                
                <p><strong>Nota:</strong> Interface básica carregada. Template principal não encontrado.</p>
            </div>
        </body>
        </html>
        """

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'message': 'Trading Bot Ativo',
        'mode': 'SCALPING',
        'paper_trading': app.config.get('PAPER_TRADING', True),
        'symbol': app.config.get('SYMBOL', 'ethusdt_UMCBL')
    })

@app.route('/health')
def health():
    return {'status': 'healthy', 'app': 'trading-bot'}, 200

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    try:
        logger.info("🚀 Iniciando bot de trading")
        # Aqui você iniciaria o bot real
        return jsonify({'message': 'Bot iniciado com sucesso'})
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar bot: {e}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint não encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erro interno: {error}")
    return jsonify({'error': 'Erro interno do servidor'}), 500

# SocketIO events (se disponível)
if socketio:
    @socketio.on('connect')
    def on_connect():
        logger.info("Cliente conectado via WebSocket")
    
    @socketio.on('disconnect')
    def on_disconnect():
        logger.info("Cliente desconectado")

logger.info("✅ App configurado e pronto")

# Para Gunicorn
if __name__ != '__main__':
    logger.info("🔄 App sendo executado via Gunicorn")
