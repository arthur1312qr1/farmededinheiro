import eventlet
eventlet.monkey_patch()

import os
from flask import Flask
from flask_socketio import SocketIO

# Criar app Flask simples
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')

# Configurar SocketIO
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

@app.route('/')
def home():
    return "Bot rodando no Render! ✅"

@app.route('/health')
def health():
    return {'status': 'ok', 'message': 'Bot funcionando'}, 200

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado via WebSocket')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando bot na porta {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
