import os
import logging
import asyncio
from threading import Thread
import eventlet

eventlet.monkey_patch()

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from config import Config

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Flask
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object(Config())

# Database
db = SQLAlchemy(app)

# Socket.IO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Criar tabelas
with app.app_context():
    try:
        db.create_all()
        logger.info("✅ Database inicializada")
    except Exception as e:
        logger.error(f"❌ Erro no database: {e}")

# Importar bot de scalping
from scalping_bot import scalping_bot, start_scalping

# Estado global
bot_running = False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/status")
def api_status():
    stats = scalping_bot.get_stats()
    return jsonify({
        "status": "online" if scalping_bot.is_running else "offline",
        "mode": "SCALPING AGRESSIVO",
        "daily_trades": stats['daily_trades'],
        "win_rate": f"{stats['win_rate']:.1f}%",
        "daily_pnl": f"${stats['daily_pnl']:.2f}",
        "balance": f"${stats['current_balance']:.2f}"
    })

@app.route("/api/start", methods=['POST'])
def start_bot():
    global bot_running
    if not bot_running:
        bot_running = True
        # Iniciar bot em thread separada
        Thread(target=lambda: asyncio.run(start_scalping())).start()
        logger.info("🚀 Bot de scalping iniciado")
        return jsonify({"message": "Bot iniciado"})
    return jsonify({"message": "Bot já está rodando"})

@app.route("/api/stop", methods=['POST'])
def stop_bot():
    global bot_running
    if bot_running:
        scalping_bot.is_running = False
        bot_running = False
        logger.info("🔴 Bot parado")
        return jsonify({"message": "Bot parado"})
    return jsonify({"message": "Bot já estava parado"})

# Socket.IO events
@socketio.on('connect')
def on_connect():
    emit('connected', {'message': 'Conectado ao Scalping Bot'})
    logger.info("Cliente conectado via WebSocket")

# Auto-start bot
def auto_start_bot():
    global bot_running
    if not bot_running:
        bot_running = True
        asyncio.run(start_scalping())

# Iniciar bot automaticamente em produção
if not app.config.get('DEBUG'):
    Thread(target=auto_start_bot).start()
    logger.info("🚀 Auto-iniciando bot de scalping")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Servidor rodando na porta {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
