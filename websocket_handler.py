from flask_socketio import emit
from extensions import socketio

def setup_websocket(socketio):
    @socketio.on("connect")
    def handle_connect():
        emit("mensagem", {"data": "Conectado ao servidor WebSocket"})
