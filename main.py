from app import app, socketio
import os

if __name__ == '__main__':
    # Usa PORT do ambiente ou 5000 como fallback
    port = int(os.environ.get('PORT', 5000))
    print(f"Iniciando servidor na porta: {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
