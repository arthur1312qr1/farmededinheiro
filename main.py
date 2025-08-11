#!/usr/bin/env python3
"""
Entry point for Railway deployment - Trading Bot
"""
import os
import logging
from app import app

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Trading Bot server on port {port}")
    logger.info(f"Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'development')}")
    logger.info(f"Paper Trading Mode: {os.environ.get('PAPER_TRADING', 'true')}")
    
    app.run(host="0.0.0.0", port=port, debug=False)
