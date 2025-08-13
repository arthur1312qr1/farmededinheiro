import eventlet
eventlet.monkey_patch()

# Resto das importações
from flask import Flask, render_template
from flask_socketio import SocketIO
# ... outras importações

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Suas rotas originais aqui
@app.route('/')
def index():
    return render_template('index.html')  # ou sua página principal

# ... resto do código original
