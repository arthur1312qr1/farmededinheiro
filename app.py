import os
import logging
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.middleware.proxy_fix import ProxyFix

from config import Config
from trading_bot import TradingBot
from gemini_handler import GeminiErrorHandler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log")
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize components
config = Config()
gemini_handler = GeminiErrorHandler(config.GEMINI_API_KEY)
trading_bot = TradingBot(config, gemini_handler)

# Global bot state
bot_thread = None
bot_running = False

@app.route('/')
def index():
    """Main dashboard page"""
    bot_state = trading_bot.get_state()
    balance_info = trading_bot.get_balance_info()
    
    return render_template('index.html', 
                         bot_state=bot_state,
                         balance_info=balance_info,
                         config=config,
                         bot_running=bot_running)

@app.route('/start', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global bot_thread, bot_running
    
    try:
        if bot_running:
            flash("Bot is already running!", "warning")
            return redirect(url_for('index'))
        
        # Validate API configuration
        if not config.validate_api_keys():
            flash("Missing required API keys. Please check environment variables.", "error")
            return redirect(url_for('index'))
        
        # Start bot in separate thread
        bot_running = True
        bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
        bot_thread.start()
        
        flash("Trading bot started successfully!", "success")
        logger.info("Trading bot started by user")
        
    except Exception as e:
        logger.error(f"Erro ao iniciar bot: {e}")
        flash(f"Erro ao iniciar bot: {str(e)}", "error")
        bot_running = False
    
    return redirect(url_for('index'))

@app.route('/stop', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global bot_running
    
    try:
        if not bot_running:
            flash("Bot is not running!", "warning")
            return redirect(url_for('index'))
        
        bot_running = False
        trading_bot.stop()
        flash("Trading bot stopped successfully!", "success")
        logger.info("Trading bot stopped by user")
        
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        flash(f"Error stopping bot: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/status')
def status():
    """Get bot status page"""
    return redirect(url_for('index'))

@app.route('/api/status')
def api_status():
    """API endpoint for bot status"""
    try:
        bot_state = trading_bot.get_state()
        balance_info = trading_bot.get_balance_info()
        
        return jsonify({
            "success": True,
            "bot_running": bot_running,
            "bot_state": bot_state,
            "balance_info": balance_info,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/logs')
def api_logs():
    """Get recent log entries"""
    try:
        with open("bot.log", "r") as f:
            lines = f.readlines()
            recent_logs = lines[-50:]  # Last 50 lines
        
        return jsonify({
            "success": True,
            "logs": recent_logs,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "logs": [],
            "timestamp": datetime.now().isoformat()
        })

@app.route('/api/gemini-fix', methods=['POST'])
def gemini_fix():
    """Trigger Gemini AI error analysis and fix"""
    try:
        data = request.get_json()
        error_description = data.get('error', '')
        
        if not error_description:
            return jsonify({
                "success": False,
                "error": "No error description provided"
            }), 400
        
        # Use Gemini to analyze and suggest fixes
        fix_suggestion = gemini_handler.analyze_and_fix_error(error_description)
        
        return jsonify({
            "success": True,
            "fix_suggestion": fix_suggestion,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in Gemini fix: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

def run_bot_loop():
    """Main bot execution loop"""
    global bot_running
    
    logger.info("Bot loop started")
    
    try:
        while bot_running:
            try:
                trading_bot.execute_trading_cycle()
                time.sleep(config.POLL_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                
                # Use Gemini AI to analyze and potentially fix the error
                try:
                    error_analysis = gemini_handler.analyze_and_fix_error(str(e))
                    logger.info(f"Gemini error analysis: {error_analysis}")
                    
                    # Apply automatic fixes if suggested
                    if "RESTART_REQUIRED" in error_analysis:
                        logger.info("Gemini suggests restarting - stopping bot")
                        bot_running = False
                        break
                        
                except Exception as gemini_error:
                    logger.error(f"Gemini error analysis failed: {gemini_error}")
                
                # Wait before retrying
                time.sleep(config.POLL_INTERVAL * 2)
                
    except KeyboardInterrupt:
        logger.info("Bot loop interrupted")
    except Exception as e:
        logger.error(f"Fatal error in bot loop: {e}")
    finally:
        bot_running = False
        logger.info("Bot loop ended")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Render.com configuration
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
