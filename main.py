#!/usr/bin/env python3
"""
Main entry point for Farme de Dinheiro Trading Bot
Production-ready Flask application with robust error handling
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Configure logging before importing Flask
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log') if os.path.exists('/tmp') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point with comprehensive error handling"""
    try:
        logger.info("🚀 Iniciando Farme de Dinheiro Trading Bot")
        
        # Import app after logging setup
        from app import create_app, socketio
        
        # Create Flask application
        app = create_app()
        
        # Get port from environment (Render sets this automatically)
        port = int(os.environ.get('PORT', 5000))
        host = '0.0.0.0'
        
        logger.info(f"🌐 Configurando servidor - Host: {host}, Port: {port}")
        logger.info(f"🔧 Modo: {'Produção' if not app.config.get('DEBUG') else 'Desenvolvimento'}")
        
        # Start the application with SocketIO
        socketio.run(
            app,
            host=host,
            port=port,
            debug=False,  # Always False in production
            use_reloader=False,  # Prevent double startup
            allow_unsafe_werkzeug=True
        )
        
    except ImportError as e:
        logger.error(f"❌ Erro de importação: {e}")
        logger.error("Verifique se todas as dependências estão instaladas")
        sys.exit(1)
        
    except OSError as e:
        logger.error(f"❌ Erro de sistema/rede: {e}")
        logger.error("Verifique se a porta está disponível")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        logger.exception("Stack trace completo:")
        sys.exit(1)

if __name__ == '__main__':
    main()
