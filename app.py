# -*- coding: utf-8 -*-
#
# Arquivo principal para a aplicação Flask do bot de negociação.
# Este script inicializa o aplicativo, gerencia o estado do bot
# e define as rotas para a interface da web e a API.

import os
import logging
import threading
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.middleware.proxy_fix import ProxyFix

# Importa as classes personalizadas do bot
from config import Config
from trading_bot import TradingBot
from gemini_handler import GeminiErrorHandler

# Configura o logging para a aplicação
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log")
    ]
)
logger = logging.getLogger(__name__)

# Cria a instância do aplicativo Flask
app = Flask(__name__)
# Configura uma chave secreta para as sessões do Flask
app.secret_key = os.environ.get("SESSION_SECRET", "fallback-secret-key-change-in-production")
# Adiciona o middleware ProxyFix para lidar com proxies (útil em ambientes como o Render)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Inicializa os componentes do bot
config = Config()
gemini_handler = GeminiErrorHandler(config.GEMINI_API_KEY)
trading_bot = TradingBot(config, gemini_handler)

# Variáveis globais para controlar o estado do bot
bot_thread = None
bot_running = False

@app.route('/')
def index():
    """Rota principal para a página do painel"""
    bot_state = trading_bot.get_state()
    balance_info = trading_bot.get_balance_info()
    
    return render_template('index.html',
                           bot_state=bot_state,
                           balance_info=balance_info,
                           config=config,
                           bot_running=bot_running)

@app.route('/start', methods=['POST'])
def start_bot():
    """Inicia o bot de negociação"""
    global bot_thread, bot_running
    
    try:
        if bot_running:
            flash("Bot já está em execução!", "warning")
            return redirect(url_for('index'))
        
        # Valida a configuração das chaves de API
        if not config.validate_api_keys():
            flash("Faltam chaves de API necessárias. Por favor, verifique as variáveis de ambiente.", "error")
            return redirect(url_for('index'))
        
        # Inicia o bot em uma thread separada
        bot_running = True
        bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
        bot_thread.start()
        
        flash("Bot de negociação iniciado com sucesso!", "success")
        logger.info("Bot de negociação iniciado pelo usuário")
        
    except Exception as e:
        logger.exception(f"Erro ao iniciar o bot: {e}")
        flash(f"Erro ao iniciar o bot: {str(e)}", "error")
        bot_running = False
    
    return redirect(url_for('index'))

@app.route('/stop', methods=['POST'])
def stop_bot():
    """Para o bot de negociação"""
    global bot_running
    
    try:
        if not bot_running:
            flash("Bot não está em execução!", "warning")
            return redirect(url_for('index'))
        
        bot_running = False
        trading_bot.stop()
        flash("Bot de negociação parado com sucesso!", "success")
        logger.info("Bot de negociação parado pelo usuário")
        
    except Exception as e:
        logger.exception(f"Erro ao parar o bot: {e}")
        flash(f"Erro ao parar o bot: {str(e)}", "error")
    
    return redirect(url_for('index'))

@app.route('/status')
def status():
    """Redireciona para a página de status do bot (página principal)"""
    return redirect(url_for('index'))

@app.route('/api/status')
def api_status():
    """Endpoint da API para obter o status do bot"""
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
        logger.exception(f"Erro ao obter o status: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/logs')
def api_logs():
    """Obtém as entradas de log mais recentes"""
    try:
        with open("bot.log", "r") as f:
            lines = f.readlines()
            recent_logs = lines[-50:]  # Últimas 50 linhas
        
        return jsonify({
            "success": True,
            "logs": recent_logs,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.exception(f"Erro ao ler os logs: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "logs": [],
            "timestamp": datetime.now().isoformat()
        })

@app.route('/api/gemini-fix', methods=['POST'])
def gemini_fix():
    """Aciona a análise e correção de erros pelo Gemini AI"""
    try:
        data = request.get_json()
        error_description = data.get('error', '')
        
        if not error_description:
            return jsonify({
                "success": False,
                "error": "Nenhuma descrição de erro fornecida"
            }), 400
        
        # Usa Gemini para analisar e sugerir correções
        fix_suggestion = gemini_handler.analyze_and_fix_error(error_description)
        
        return jsonify({
            "success": True,
            "fix_suggestion": fix_suggestion,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.exception(f"Erro na correção do Gemini: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

def run_bot_loop():
    """Loop principal de execução do bot"""
    global bot_running
    
    logger.info("Loop do bot iniciado")
    
    try:
        while bot_running:
            try:
                trading_bot.execute_trading_cycle()
                time.sleep(config.POLL_INTERVAL)
                
            except Exception as e:
                logger.exception(f"Erro no ciclo de negociação: {e}")
                
                # Usa Gemini AI para analisar e potencialmente corrigir o erro
                try:
                    error_analysis = gemini_handler.analyze_and_fix_error(str(e))
                    logger.info(f"Análise de erro do Gemini: {error_analysis}")
                    
                    # Aplica correções automáticas se sugeridas
                    if "RESTART_REQUIRED" in error_analysis:
                        logger.info("Gemini sugere reiniciar - parando o bot")
                        bot_running = False
                        break
                        
                except Exception as gemini_error:
                    logger.exception(f"A análise de erro do Gemini falhou: {gemini_error}")
                
                # Espera antes de tentar novamente
                time.sleep(config.POLL_INTERVAL * 2)
                
    except KeyboardInterrupt:
        logger.info("Loop do bot interrompido")
    except Exception as e:
        logger.exception(f"Erro fatal no loop do bot: {e}")
    finally:
        bot_running = False
        logger.info("Loop do bot encerrado")

@app.errorhandler(404)
def not_found_error(error):
    """Manipulador para erros 404 (Não Encontrado)"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Manipulador para erros 500 (Erro Interno do Servidor)"""
    logger.exception(f"Erro interno do servidor: {error}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Configuração para o ambiente Render.com ou local
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Iniciando o aplicativo Flask em {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
