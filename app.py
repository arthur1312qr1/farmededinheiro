import os
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.middleware.proxy_fix import ProxyFix

from config import RailwayConfig
from trading_bot import TradingBot
from gemini_handler import GeminiHandler

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback-secret-key-for-development")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize configuration
config = RailwayConfig()

# Initialize services
gemini_handler = GeminiHandler(config.GEMINI_API_KEY)
trading_bot = TradingBot(config, gemini_handler)

# Global bot state
bot_thread = None
bot_running = False

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html', config=config)

@app.route('/dashboard')
def dashboard():
    """Trading dashboard"""
    try:
        # Get bot state
        bot_state = trading_bot.get_state()
        
        # Get balance information
        balance_info = trading_bot.get_balance_info()
        
        # Combine data for template
        dashboard_data = {
            'bot_state': bot_state,
            'balance_info': balance_info,
            'config': {
                'paper_trading': config.PAPER_TRADING,
                'symbol': config.SYMBOL,
                'poll_interval': config.POLL_INTERVAL,
                'min_balance': config.MIN_BALANCE_USDT
            },
            'bot_running': bot_running,
            'api_connected': trading_bot.bitget_api is not None
        }
        
        return render_template('dashboard.html', **dashboard_data)
    except Exception as e:
        logger.error(f"Error loading dashboard: {e}")
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/balance')
def api_balance():
    """API endpoint for balance information"""
    try:
        balance_info = trading_bot.get_balance_info()
        return jsonify(balance_info)
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/start', methods=['POST'])
def api_start_bot():
    """Start the trading bot"""
    global bot_thread, bot_running
    
    try:
        if bot_running:
            return jsonify({'error': 'Bot is already running'}), 400
        
        # Validate API keys if not in paper trading mode
        if not config.PAPER_TRADING and not config.validate_api_keys():
            return jsonify({'error': 'API keys not properly configured'}), 400
        
        # Start bot in separate thread
        bot_running = True
        trading_bot.start()
        
        def run_bot():
            global bot_running
            while bot_running:
                try:
                    trading_bot.execute_trading_cycle()
                    threading.Event().wait(config.POLL_INTERVAL)
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    # Continue running despite errors
                    threading.Event().wait(10)  # Wait 10 seconds before retry
        
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        
        logger.info("Trading bot started successfully")
        return jsonify({'message': 'Bot started successfully', 'status': 'running'})
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        bot_running = False
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def api_stop_bot():
    """Stop the trading bot"""
    global bot_running
    
    try:
        bot_running = False
        trading_bot.stop()
        
        logger.info("Trading bot stopped")
        return jsonify({'message': 'Bot stopped successfully', 'status': 'stopped'})
        
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/status')
def api_bot_status():
    """Get current bot status"""
    try:
        state = trading_bot.get_state()
        state['running'] = bot_running
        state['thread_alive'] = bot_thread is not None and bot_thread.is_alive()
        return jsonify(state)
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config')
def api_config():
    """Get current configuration"""
    return jsonify({
        'paper_trading': config.PAPER_TRADING,
        'symbol': config.SYMBOL,
        'poll_interval': config.POLL_INTERVAL,
        'min_balance': config.MIN_BALANCE_USDT,
        'api_keys_configured': config.validate_api_keys(),
        'environment': config.RAILWAY_ENVIRONMENT
    })

@app.route('/health')
def health_check():
    """Health check endpoint for Railway"""
    try:
        # Basic health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'environment': config.RAILWAY_ENVIRONMENT,
            'paper_trading': config.PAPER_TRADING,
            'bot_running': bot_running
        }
        
        # Check API connectivity if not in paper trading mode
        if not config.PAPER_TRADING:
            try:
                balance_info = trading_bot.get_balance_info()
                health_status['api_connected'] = not balance_info.get('error')
            except:
                health_status['api_connected'] = False
        else:
            health_status['api_connected'] = True
        
        return jsonify(health_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.errorhandler(404)
def not_found(error):
    """404 error handler"""
    return render_template('index.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    """500 error handler"""
    logger.error(f"Internal error: {error}")
    return render_template('index.html', error='Internal server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
