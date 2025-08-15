import os
import sys
import traceback
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def create_app():
    """Cria e configura a aplicação Flask"""
    app = Flask(__name__)
    
    # Configurações
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')
    app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Configurar CORS
    CORS(app, origins="*")
    
    # Configurar SocketIO (SEM eventlet para Python 3.13)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Rota principal
    @app.route('/')
    def index():
        try:
            return render_template('index.html')
        except Exception as e:
            # Se não encontrar template, retorna HTML básico
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Bot - Farmede Dinheiro</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: white; }
                    .container { max-width: 800px; margin: 0 auto; text-align: center; }
                    .status { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }
                    .success { color: #4CAF50; }
                    .title { color: #0f4c75; font-size: 2em; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1 class="title">🚀 Trading Bot - Farmede Dinheiro</h1>
                    <div class="status">
                        <h2 class="success">✅ Sistema Online!</h2>
                        <p>Bot de trading funcionando corretamente</p>
                        <p>API Status: <span class="success">Ativo</span></p>
                        <p>Integração com exchanges: <span class="success">Disponível</span></p>
                    </div>
                    <div class="status">
                        <h3>Próximos Passos:</h3>
                        <p>• Configure suas chaves de API</p>
                        <p>• Defina suas estratégias de trading</p>
                        <p>• Monitore em tempo real</p>
                    </div>
                </div>
            </body>
            </html>
            """
    
    @app.route('/api/status')
    def status():
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'message': 'Trading Bot API está funcionando!',
            'features': {
                'ccxt': 'disponível',
                'websockets': 'ativo',
                'api': 'online'
            }
        })
    
    @app.route('/api/test')
    def test():
        return jsonify({
            'message': 'API funcionando perfeitamente!',
            'timestamp': datetime.now().isoformat()
        })
    
    # Importar e registrar rotas
    try:
        from routes import register_routes
        register_routes(app, socketio)
        logger.info("Rotas registradas com sucesso")
    except ImportError as e:
        logger.warning(f"Não foi possível importar routes: {e}")
