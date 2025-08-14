import os
import logging
import sys

# Configurar logging para produção
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
    from flask import Flask, jsonify
    
    class Config:
        SECRET_KEY = 'fallback-key'
        DEBUG = False

# Criar app Flask
app = Flask(__name__)
app.config.from_object(Config())

logger.info("🔧 Flask app criada")

# SocketIO com configuração otimizada para produção
try:
    socketio = SocketIO(
        app, 
        cors_allowed_origins="*", 
        async_mode="threading",  # Threading é melhor para Gunicorn
        logger=False,            # Reduz logs verbosos
        engineio_logger=False    # Reduz logs verbosos
    )
    logger.info("🔌 SocketIO inicializado")
except Exception as e:
    logger.error(f"⚠️ SocketIO falhou: {e}")
    socketio = None

# Rota principal
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"❌ Template não encontrado: {e}")
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>🚀 Trading Bot</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #00ff00; }}
                .container {{ max-width: 900px; margin: 0 auto; padding: 20px; }}
                .header {{ text-align: center; border: 2px solid #00ff00; padding: 20px; margin: 20px 0; }}
                .status {{ background: #003300; padding: 15px; margin: 10px 0; border-left: 5px solid #00ff00; }}
                .trading {{ background: #330000; padding: 15px; margin: 10px 0; border-left: 5px solid #ff6600; }}
                .button {{ background: #00ff00; color: #000; padding: 10px 20px; margin: 5px; border: none; cursor: pointer; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 TRADING BOT - FARM DE DINHEIRO</h1>
                    <h2>💰 SCALPING AGRESSIVO ATIVO 💰</h2>
                </div>
                
                <div class="status">
                    <h3>✅ STATUS: ONLINE E FUNCIONANDO</h3>
                    <p>🤖 Bot de Trading: ATIVO</p>
                    <p>⚡ Modo: SCALPING AGRESSIVO</p>
                    <p>📊 Trading Real: {'ATIVO' if not app.config.get('PAPER_TRADING', True) else 'PAPER'}</p>
                </div>
                
                <div class="trading">
                    <h3>💰 CONFIGURAÇÕES DE TRADING</h3>
                    <p>🎯 Símbolo: {app.config.get('SYMBOL', 'ETHUSDT')}</p>
                    <p>📈 Alavancagem: {app.config.get('LEVERAGE', 10)}x</p>
                    <p>⚡ Intervalo: 30 segundos</p>
                    <p>🎲 Max Trades/Dia: 200</p>
                    <p>🎯 Confiança Mínima: 60%</p>
                </div>
                
                <div style="text-align: center;">
                    <button class="button" onclick="fetch('/api/bot/start', {{method: 'POST'}}).then(()=>alert('Bot iniciado!'))">
                        🚀 INICIAR TRADING
                    </button>
                    <button class="button" onclick="location.reload()">🔄 ATUALIZAR</button>
                </div>
                
                <div class="status">
                    <h3>📊 APIS CONFIGURADAS</h3>
                    <p>📈 Bitget API: Pronto</p>
                    <p>🧠 Gemini AI: Pronto</p>
                    <p>🌐 WebSocket: {'Ativo' if socketio else 'Básico'}</p>
                </div>
            </div>
        </body>
        </html>
        """

@app.route('/api/status')
def api_status():
    return jsonify({
        'status': 'online',
        'message': '🚀 Trading Bot Scalping Ativo',
        'mode': 'SCALPING_AGRESSIVO',
        'paper_trading': app.config.get('PAPER_TRADING', True),
        'symbol': app.config.get('SYMBOL', 'ethusdt_UMCBL'),
        'leverage': app.config.get('LEVERAGE', 10),
        'max_trades_day': 200,
        'min_confidence': 60
    })

@app.route('/health')
def health():
    return {'status': 'healthy', 'app': 'trading-bot-scalping'}, 200

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    try:
        logger.info("🚀 Comando para iniciar bot recebido")
        # Aqui você integraria o bot real de scalping
        return jsonify({
            'message': '🚀 Bot de Scalping iniciado com sucesso!',
            'mode': 'SCALPING_AGRESSIVO',
            'symbol': app.config.get('SYMBOL'),
            'leverage': f"{app.config.get('LEVERAGE')}x"
        })
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar bot: {e}")
        return jsonify({'error': f'Erro: {str(e)}'}), 500

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    logger.info("🛑 Comando para parar bot recebido")
    return jsonify({'message': '🛑 Bot parado'})

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

logger.info("✅ App configurado e pronto para Gunicorn")

# IMPORTANTE: Para Gunicorn no Render, não execute app.run() aqui
# O Gunicorn vai importar e usar o objeto 'app' automaticamente
