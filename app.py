import os
import logging
import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from config import Config

# Logging básico
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object(Config)

# Banco de dados (AGORA EXISTE!)
db = SQLAlchemy(app)

# Socket.IO com eventlet
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Cria tabelas se ainda não existirem
with app.app_context():
    try:
        db.create_all()
        logger.info("✅ Tabelas verificadas/criadas com sucesso.")
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar o banco: {e}")

# IMPORTANTE: importe as rotas e websocket DEPOIS de criar app/db/socketio
# (assim evita importação circular e mantém seu site original)
import routes      # noqa: F401
import websocket_handler  # noqa: F401

if __name__ == "__main__":
    host = getattr(Config, "HOST", "0.0.0.0")
    port = int(getattr(Config, "PORT", os.environ.get("PORT", 5000)))
    logger.info(f"🚀 Subindo servidor em http://{host}:{port}")
    socketio.run(app, host=host, port=port)
