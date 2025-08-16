import logging
import os
from app import app

if __name__ == '__main__':
    logging.info("🚀 Iniciando Trading Bot API")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
