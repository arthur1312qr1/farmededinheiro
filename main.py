# IMPORTANTE: eventlet deve ser importado PRIMEIRO
import eventlet
eventlet.monkey_patch()

import os
from app import app, socketio

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando servidor na porta: {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
