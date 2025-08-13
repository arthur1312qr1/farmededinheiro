import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, render_template
from flask_socketio import SocketIO
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Configurar SocketIO para Render
socketio = SocketIO(
    app, 
    async_mode='eventlet',
    cors_allowed_origins="*",
    logger=False,
    engineio_logger=False
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return {'status': 'healthy', 'service': 'farmededinheiro-bot'}, 200

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

# Importar suas rotas aqui
try:
    from routes import *
except ImportError:
    print("Routes não encontrado, usando rotas básicas")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
