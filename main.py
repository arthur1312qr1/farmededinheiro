"""
Arquivo principal otimizado para Render
"""
import os
import logging

# Configurar logging antes de tudo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

try:
    # Importar app somente se necessário
    from app import app, socketio
    logger.info("✅ App importado com sucesso")
    
    if __name__ == "__main__":
        # Para execução local
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"🚀 Iniciando servidor local na porta {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=False)
    
except Exception as e:
    logger.error(f"❌ Erro ao importar app: {e}")
    
    # Fallback - App mínimo caso algo dê errado
    from flask import Flask
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return """
        <h1>🤖 Trading Bot</h1>
        <p>Status: Online (Modo Fallback)</p>
        <p>O bot está funcionando mas com configuração mínima.</p>
        <p>Verifique os logs para mais detalhes.</p>
        """
    
    @app.route('/health')
    def health():
        return {'status': 'ok'}, 200
    
    if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
