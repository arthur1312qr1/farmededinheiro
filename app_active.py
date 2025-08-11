from flask import Flask, render_template, jsonify, request
import logging
import asyncio
import threading
from datetime import datetime
import os

from active_trading_system import trading_system

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def dashboard():
    """Dashboard principal"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """Status do bot"""
    try:
        status = trading_system.get_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Erro ao obter status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_data')
def api_market_data():
    """Dados de mercado"""
    try:
        # Executar função assíncrona
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        market_data = loop.run_until_complete(trading_system.get_market_data())
        loop.close()
        
        return jsonify({'market_data': market_data})
    except Exception as e:
        logger.error(f"Erro ao obter dados de mercado: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/balance')
def api_balance():
    """Saldo da conta"""
    try:
        # Executar função assíncrona
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        balance_data = loop.run_until_complete(trading_system.get_balance())
        loop.close()
        
        return jsonify(balance_data)
    except Exception as e:
        logger.error(f"Erro ao obter saldo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/positions')
def api_positions():
    """Posições abertas"""
    try:
        positions = trading_system.get_positions()
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Erro ao obter posições: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade_history')
def api_trade_history():
    """Histórico de trades"""
    try:
        trades = trading_system.get_trade_history()
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Erro ao obter histórico: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/last_analysis')
def api_last_analysis():
    """Última análise da IA"""
    try:
        last_analysis = trading_system.last_analysis
        if last_analysis:
            return jsonify(last_analysis)
        else:
            return jsonify({'message': 'Nenhuma análise disponível'})
    except Exception as e:
        logger.error(f"Erro ao obter análise: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_bot', methods=['POST'])
def api_start_bot():
    """Iniciar bot"""
    try:
        # Executar função assíncrona em thread separada
        def start_trading_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(trading_system.start_trading())
            loop.close()
        
        thread = threading.Thread(target=start_trading_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Bot iniciado com sucesso'})
    except Exception as e:
        logger.error(f"Erro ao iniciar bot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_bot', methods=['POST'])
def api_stop_bot():
    """Parar bot"""
    try:
        result = trading_system.stop_trading()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Erro ao parar bot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def api_emergency_stop():
    """Parada de emergência"""
    try:
        result = trading_system.stop_trading()
        
        # Tentar fechar todas as posições abertas
        positions = trading_system.get_positions()
        if positions.get('positions'):
            logger.warning(f"Parada de emergência: {len(positions['positions'])} posições abertas")
            # Aqui você adicionaria lógica para fechar posições na exchange real
        
        return jsonify({'message': 'Parada de emergência ativada'})
    except Exception as e:
        logger.error(f"Erro na parada de emergência: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_market', methods=['POST'])
def api_analyze_market():
    """Forçar análise de mercado"""
    try:
        # Executar análise
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        market_data = loop.run_until_complete(trading_system.get_market_data())
        balance_data = loop.run_until_complete(trading_system.get_balance())
        
        if market_data.get('success') and balance_data.get('success'):
            analysis = loop.run_until_complete(trading_system.analyze_market(market_data, balance_data))
            loop.close()
            
            return jsonify({
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            })
        else:
            loop.close()
            return jsonify({'error': 'Erro ao obter dados para análise'}), 500
            
    except Exception as e:
        logger.error(f"Erro na análise forçada: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint não encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Erro interno do servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info("="*50)
    logger.info("BOT DE TRADING ATIVO - INICIANDO")
    logger.info("="*50)
    logger.info(f"Modo: {'PAPER TRADING' if trading_system.config.PAPER_TRADING else 'LIVE TRADING'}")
    logger.info(f"Symbol: {trading_system.config.SYMBOL}")
    logger.info(f"Porta: {port}")
    logger.info("="*50)
    
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
