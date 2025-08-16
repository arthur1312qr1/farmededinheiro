from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging
import threading
import time
from datetime import datetime
from typing import Dict

from trading_bot import TradingBot
from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Initialize trading bot
    config = get_config()
    bot = TradingBot(config)
    
    @app.route('/')
    def index():
        """Main dashboard"""
        return render_template('index.html')
    
    @app.route('/api/status')
    def get_status():
        """Get bot status and statistics"""
        stats = bot.get_stats()
        return jsonify(stats)
    
    @app.route('/api/start', methods=['POST'])
    def start_bot():
        """Start the trading bot"""
        try:
            bot.start()
            return jsonify({'success': True, 'message': 'Bot iniciado'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/stop', methods=['POST'])
    def stop_bot():
        """Stop the trading bot"""
        try:
            bot.stop()
            return jsonify({'success': True, 'message': 'Bot parado'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/pause', methods=['POST'])
    def pause_bot():
        """Pause/Resume the trading bot"""
        try:
            bot.pause()
            status = 'pausado' if bot.is_paused else 'retomado'
            return jsonify({'success': True, 'message': f'Bot {status}'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/emergency_stop', methods=['POST'])
    def emergency_stop():
        """Emergency stop"""
        try:
            bot.emergency_stop()
            return jsonify({'success': True, 'message': 'Parada de emergência ativada'})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/api/balance')
    def get_balance():
        """Get account balance"""
        try:
            balance = bot.bitget_api.get_account_balance()
            return jsonify({'balance': balance})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/logs')
    def get_logs():
        """Get recent activity logs"""
        try:
            stats = bot.get_stats()
            return jsonify({'logs': stats['activity_log']})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)
