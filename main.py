import os
import logging
from app import app

# Configure logging otimizado para Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Iniciando Trading Bot Server na porta {port} - Railway Deploy")
    
    # Railway production settings otimizadas
    try:
        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False  # Importante para Railway
        )
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar servidor: {e}")
        raise
