import eventlet
eventlet.monkey_patch()

import os

# Tentar importar o app original
try:
    from app import app, socketio
    print("✅ App original importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar app original: {e}")
    print("Criando app básico como fallback...")
    
    # Fallback: criar app básico se o original não funcionar
    from flask import Flask
    from flask_socketio import SocketIO
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')
    socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
    
    @app.route('/')
    def home():
        return "App básico funcionando - Verifique configuração original"
    
    @app.route('/health')
    def health():
        return {'status': 'ok'}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Iniciando servidor na porta {port}")
    
    # Verificar se existe rota principal
    print("Rotas disponíveis:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule.endpoint}: {rule.rule}")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
