import os
import sys
import logging
from app import create_app

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    """Entrada principal otimizada para Gunicorn"""
    try:
        logger.info("🚀 Iniciando ETH Bot 80% - CORRIGIDO")
        
        # Criar app
        app = create_app()
        
        # Porta
        port = int(os.environ.get('PORT', 5000))
        
        logger.info(f"🌐 Servidor iniciando na porta {port}")
        logger.warning("🚨 ETH BOT 80% - MÉTODO CORRIGIDO ATIVO!")
        
        # Iniciar servidor
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
