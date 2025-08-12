import os
import logging
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.middleware.proxy_fix import ProxyFix

from config import TradingConfig
from trading_bot import TradingBot

# Configure logging para Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "railway-trading-bot-secret-key-2024")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize configuration and bot
config = TradingConfig()
trading_bot = TradingBot(config)

# Global bot state com lock para thread safety
bot_thread = None
bot_running = False
bot_lock = threading.Lock()
bot_stats_lock = threading.Lock()

@app.route('/')
def index():
    """Página principal com controles do bot"""
    try:
        bot_status = get_bot_status()
        balance_info = trading_bot.get_balance_info()
        
        return render_template('index.html', 
                             bot_status=bot_status, 
                             balance_info=balance_info,
                             config=config.get_display_config())
    except Exception as e:
        logger.error(f"Erro na página principal: {e}")
        return render_template('index.html', 
                             error=f"Erro ao carregar dados: {str(e)}",
                             config=config.get_display_config())

@app.route('/dashboard')
def dashboard():
    """Dashboard detalhado do bot"""
    try:
        bot_status = get_bot_status()
        balance_info = trading_bot.get_balance_info()
        trading_stats = trading_bot.get_trading_stats()
        
        return render_template('dashboard.html',
                             bot_status=bot_status,
                             balance_info=balance_info,
                             trading_stats=trading_stats,
                             config=config.get_display_config())
    except Exception as e:
        logger.error(f"Erro no dashboard: {e}")
        flash(f"Erro ao carregar dashboard: {str(e)}", "error")
        return redirect(url_for('index'))

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Iniciar o bot de trading"""
    global bot_thread, bot_running
    
    with bot_lock:
        try:
            if bot_running:
                return jsonify({'error': 'Bot já está rodando'}), 400
            
            # Validar configuração
            if not config.validate_configuration():
                return jsonify({'error': 'Configuração inválida - verifique as variáveis de ambiente'}), 400
            
            # Inicializar bot
            if not trading_bot.initialize():
                return jsonify({'error': 'Falha ao inicializar bot'}), 500
            
            # Iniciar thread do bot
            bot_running = True
            bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
            bot_thread.start()
            
            logger.info("Bot de trading iniciado com sucesso")
            return jsonify({
                'message': 'Bot iniciado com sucesso',
                'status': 'running',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Erro ao iniciar bot: {e}")
            bot_running = False
            return jsonify({'error': f'Erro ao iniciar bot: {str(e)}'}), 500

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Parar o bot de trading"""
    global bot_running
    
    with bot_lock:
        try:
            if not bot_running:
                return jsonify({'error': 'Bot não está rodando'}), 400
            
            bot_running = False
            trading_bot.stop()
            
            logger.info("Bot de trading parado")
            return jsonify({
                'message': 'Bot parado com sucesso',
                'status': 'stopped',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Erro ao parar bot: {e}")
            return jsonify({'error': f'Erro ao parar bot: {str(e)}'}), 500

@app.route('/api/bot/status')
def bot_status():
    """Status atual do bot"""
    try:
        status = get_bot_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/balance')
def balance():
    """Informações de saldo"""
    try:
        balance_info = trading_bot.get_balance_info()
        return jsonify(balance_info)
    except Exception as e:
        logger.error(f"Erro ao obter saldo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check para Railway"""
    try:
        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'bot_running': bot_running,
            'thread_alive': bot_thread is not None and bot_thread.is_alive(),
            'paper_trading': config.PAPER_TRADING,
            'environment': 'railway'
        }
        
        # Verificar conectividade da API se não estiver em paper trading
        if not config.PAPER_TRADING:
            try:
                balance_info = trading_bot.get_balance_info()
                status['api_connected'] = not balance_info.get('error')
            except:
                status['api_connected'] = False
        else:
            status['api_connected'] = True
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check falhou: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def get_bot_status():
    """Obter status completo do bot"""
    global bot_running, bot_thread
    
    return {
        'running': bot_running,
        'thread_alive': bot_thread is not None and bot_thread.is_alive(),
        'last_update': datetime.now().isoformat(),
        'uptime': trading_bot.get_uptime(),
        'error_count': trading_bot.get_error_count()
    }

def run_bot_loop():
    """Loop principal do bot otimizado para Railway 24/7"""
    global bot_running
    
    logger.info("🚀 Iniciando loop do bot de trading para Railway")
    consecutive_errors = 0
    max_consecutive_errors = config.MAX_CONSECUTIVE_LOSSES
    cycle_count = 0
    
    while bot_running:
        try:
            cycle_count += 1
            
            # Log de ciclo a cada 100 iterações para não poluir logs
            if cycle_count % 100 == 0:
                logger.info(f"🔄 Ciclo {cycle_count} - Bot rodando estável")
            
            # Executar ciclo de trading
            trading_bot.execute_trading_cycle()
            consecutive_errors = 0
            
            # Aguardar próximo ciclo
            time.sleep(config.POLL_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("🛑 Interrupção manual detectada - parando bot")
            bot_running = False
            break
            
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"❌ Erro no ciclo {cycle_count} ({consecutive_errors}/{max_consecutive_errors}): {e}")
            
            # Parar bot se muitos erros consecutivos
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(f"🚨 ALERTA: {consecutive_errors} erros consecutivos - PARANDO BOT por segurança")
                bot_running = False
                
                # Tentar notificar sobre o erro crítico
                try:
                    trading_bot.risk_management['emergency_stop'] = True
                    logger.info("✅ Emergency stop ativado com sucesso")
                except:
                    logger.error("❌ Falha ao ativar emergency stop")
                break
            
            # Delay progressivo com backoff exponencial
            retry_delay = min(60, consecutive_errors * 10)  # Máx 60s
            logger.info(f"⏳ Aguardando {retry_delay}s antes do retry...")
            
            # Aguardar com verificação de interrupção
            for i in range(retry_delay):
                if not bot_running:
                    break
                time.sleep(1)
    
    logger.info("🏁 Loop do bot finalizado - Railway deployment estável")

@app.errorhandler(404)
def not_found(error):
    """Handler para erro 404"""
    return render_template('index.html', error='Página não encontrada'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handler para erro 500"""
    logger.error(f"Erro interno: {error}")
    return render_template('index.html', error='Erro interno do servidor'), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
