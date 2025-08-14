"""
Entrada principal otimizada para Gunicorn no Render
"""
import os
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Importar app
try:
    from app import app, socketio
    logger.info("✅ App importado para Gunicorn")
    
    # Para Gunicorn, não usar socketio.run()
    # O Gunicorn vai usar o objeto 'app' diretamente
    
except Exception as e:
    logger.error(f"❌ Erro crítico: {e}")
    # App de fallback mínimo
    from flask import Flask
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "<h1>🤖 Trading Bot</h1><p>Modo de emergência ativo</p>"
    
    @app.route('/health')
    def health():
        return {'status': 'emergency'}, 200

# Esta linha é importante para o Gunicorn encontrar o app
if __name__ == "__main__":
    # Apenas para desenvolvimento local
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🔧 Executando localmente na porta {port}")
    
    # Para desenvolvimento local, usar socketio se disponível
    if 'socketio' in globals():
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    else:
        app.run(host='0.0.0.0', port=port, debug=False)
