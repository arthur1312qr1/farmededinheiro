from flask import Flask
from extensions import db, socketio
from routes import routes
from websocket_handler import setup_websocket
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "minha_chave_secreta")
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL", "sqlite:///meubanco.db")

    db.init_app(app)
    socketio.init_app(app)

    app.register_blueprint(routes)

    setup_websocket(socketio)

    return app

app = create_app()

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
