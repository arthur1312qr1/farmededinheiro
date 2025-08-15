import os
import sys
import traceback
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import eventlet

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
    
    # Configurar SocketIO
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
    
    # Rota principal
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/status')
    def status():
        return jsonify({
            'status': 'online',
            'timestamp': datetime.now().isoformat(),
            'message': 'Trading Bot API está funcionando!'
        })
    
    # Importar e registrar rotas (corrigido)
    try:
        from routes import register_routes  # ✅ CORRETO
        register_routes(app, socketio)
        logger.info("Rotas registradas com sucesso")
    except ImportError as e:
        logger.warning(f"Não foi possível importar routes: {e}")
        logger.info("Aplicação iniciará sem rotas adicionais")
    
    # Event handlers do SocketIO
    @socketio.on('connect')
    def handle_connect():
        logger.info('Cliente conectado via WebSocket')
        emit('response', {'data': 'Conectado ao Trading Bot!'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info('Cliente desconectado')
    
    @socketio.on('ping')
    def handle_ping():
        emit('pong', {'timestamp': datetime.now().isoformat()})
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint não encontrado'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Erro interno: {error}")
        return jsonify({'error': 'Erro interno do servidor'}), 500
    
    app.socketio = socketio
    return app

if __name__ == '__main__':
    try:
        app = create_app()
        port = int(os.environ.get('PORT', 5000))
        
        logger.info(f"Iniciando Trading Bot na porta {port}")
        logger.info("Aplicação configurada com sucesso")
        
        # Usar eventlet para melhor performance com SocketIO
        app.socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=app.config['DEBUG']
        )
        
    except Exception as e:
        logger.error(f"Erro ao iniciar aplicação: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
