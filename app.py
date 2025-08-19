from flask import Flask
from flask_cors import CORS
import os
import logging
from datetime import datetime
import sys
import time
import threading
import traceback

# Importar seus arquivos existentes
try:
    from bitget_api import BitgetAPI
    from config import get_config
    from trading_bot import TradingBot
    
    config = get_config()
    
    # Inicializar API
    bitget_api = BitgetAPI(
        api_key=config.get('BITGET_API_KEY'),
        secret_key=config.get('BITGET_SECRET_KEY'),
        passphrase=config.get('BITGET_PASSPHRASE')
    )
    
    # Inicializar TradingBot com os parâmetros corretos
    trading_bot = TradingBot(
        api=bitget_api,
        symbol=config.get('SYMBOL', 'ETH/USDT:USDT'),
        leverage=int(config.get('LEVERAGE', 10)),
        paper_trading=config.get('PAPER_TRADING', 'true').lower() == 'true'
    )
    
    API_CONNECTED = True
    print("✅ Sistema inicializado:")
    print(f"   📊 Símbolo: {config.get('SYMBOL', 'ETH/USDT:USDT')}")
    print(f"   📈 Alavancagem: {config.get('LEVERAGE', 10)}x")
    print(f"   📋 Paper Trading: {config.get('PAPER_TRADING', 'true')}")
    
except Exception as e:
    print(f"❌ Erro na inicialização: {e}")
    traceback.print_exc()
    bitget_api = None
    trading_bot = None
    API_CONNECTED = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Estado do bot
bot_state = {
    'is_running': False,
    'is_paused': False,
    'trades_today': 0,
    'profitable_trades': 0,
    'total_trades': 0,
    'total_profit': 0.0,
    'thread': None,
    'last_error': None,
    'last_analysis': None,
    'current_prediction': None
}

def run_trading_loop():
    """Loop principal do bot de trading"""
    print("🚀 Iniciando loop principal do trading bot...")
    
    while bot_state['is_running']:
        if bot_state['is_paused']:
            print("⏸️ Bot pausado, aguardando...")
            time.sleep(5)
            continue
            
        try:
            if trading_bot and API_CONNECTED:
                print("🔄 Executando análise e trading...")
                
                # Chamar o método principal do seu trading_bot
                if hasattr(trading_bot, 'execute'):
                    result = trading_bot.execute()
                elif hasattr(trading_bot, 'run'):
                    result = trading_bot.run()
                elif hasattr(trading_bot, 'analyze_and_trade'):
                    result = trading_bot.analyze_and_trade()
                else:
                    # Tentar métodos individuais
                    analysis = None
                    prediction = None
                    trade_result = None
                    
                    # Fazer análise
                    if hasattr(trading_bot, 'analyze_market'):
                        analysis = trading_bot.analyze_market()
                        bot_state['last_analysis'] = analysis
                        print(f"📊 Análise: {analysis}")
                    
                    # Fazer previsão
                    if hasattr(trading_bot, 'predict') or hasattr(trading_bot, 'make_prediction'):
                        if hasattr(trading_bot, 'predict'):
                            prediction = trading_bot.predict()
                        else:
                            prediction = trading_bot.make_prediction()
                        bot_state['current_prediction'] = prediction
                        print(f"🔮 Previsão: {prediction}")
                    
                    # Executar trade baseado na análise
                    if hasattr(trading_bot, 'execute_trade'):
                        trade_result = trading_bot.execute_trade(analysis, prediction)
                    elif hasattr(trading_bot, 'trade'):
                        trade_result = trading_bot.trade()
                    
                    result = {
                        'analysis': analysis,
                        'prediction': prediction,
                        'trade': trade_result
                    }
                
                # Processar resultado
                if result:
                    print(f"✅ Resultado: {result}")
                    
                    # Se foi um trade bem sucedido
                    if isinstance(result, dict):
                        if result.get('trade_executed') or result.get('success'):
                            bot_state['trades_today'] += 1
                            bot_state['total_trades'] += 1
                            
                            if result.get('profitable', True):
                                bot_state['profitable_trades'] += 1
                            
                            if result.get('profit'):
                                bot_state['total_profit'] += result['profit']
                            
                            print(f"🎉 Trade executado! Total hoje: {bot_state['trades_today']}")
                    
                    elif result:  # Se retornou algo truthy
                        bot_state['trades_today'] += 1
                        print(f"🎉 Ação executada! Total hoje: {bot_state['trades_today']}")
                
                bot_state['last_error'] = None
                
            else:
                error_msg = "TradingBot ou API não disponível"
                print(f"❌ {error_msg}")
                bot_state['last_error'] = error_msg
                time.sleep(10)
                
        except Exception as e:
            error_msg = f"Erro no loop de trading: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            logger.error(error_msg)
            bot_state['last_error'] = error_msg
            time.sleep(5)  # Pausa antes de tentar novamente
        
        # Intervalo entre execuções (configurável)
        time.sleep(2)  # 2 segundos entre tentativas
    
    print("🛑 Loop de trading finalizado")

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: white;
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .control-panel {
            background: #333;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            margin: 5px;
            transition: all 0.3s ease;
            min-width: 120px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-start {
            background: #4CAF50;
            color: white;
        }
        
        .btn-start:hover {
            background: #45a049;
        }
        
        .btn-pause {
            background: #ff9800;
            color: white;
        }
        
        .btn-pause:hover {
            background: #e68900;
        }
        
        .btn-stop {
            background: #f44336;
            color: white;
        }
        
        .btn-stop:hover {
            background: #da190b;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: #333;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .metric-label {
            color: #ccc;
            font-size: 0.9em;
        }
        
        .analysis-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .info-card {
            background: #333;
            padding: 20px;
            border-radius: 8px;
        }
        
        .status-indicator {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }
        
        .status-running {
            background: #4CAF50;
            color: white;
        }
        
        .status-paused {
            background: #ff9800;
            color: white;
        }
        
        .status-stopped {
            background: #f44336;
            color: white;
        }
        
        .progress-bar {
            width: 100%;
            background: #555;
            height: 20px;
            border-radius: 10px;
            margin: 10px 0;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background: #4CAF50;
            border-radius: 10px;
            transition: width 0.5s ease;
            width: 0%;
        }
        
        .debug-info {
            font-size: 0.8em;
            color: #888;
            margin-top: 10px;
        }
        
        @media (max-width: 768px) {
            .analysis-grid {
                grid-template-columns: 1fr;
            }
            
            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Trading Bot Dashboard Avançado</h1>
        </div>
        
        <div class="control-panel">
            <h2>Status: <span id="status" class="status-indicator status-stopped">PARADO</span></h2>
            <div style="margin-top: 20px;">
                <button id="startBtn" class="btn btn-start" onclick="startBot()">🚀 Iniciar Bot</button>
                <button id="pauseBtn" class="btn btn-pause" onclick="pauseBot()">⏸️ Pausar</button>
                <button id="stopBtn" class="btn btn-stop" onclick="stopBot()">🛑 Parar</button>
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" style="color: #2196F3;" id="trades">0</div>
                <div class="metric-label">Trades Hoje</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #4CAF50;" id="win-rate">0%</div>
                <div class="metric-label">Taxa de Sucesso</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #9C27B0;" id="balance">$0.00</div>
                <div class="metric-label">Saldo USDT</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color: #FF5722;" id="price">$0.00</div>
                <div class="metric-label">Preço ETH</div>
            </div>
        </div>
        
        <div class="analysis-grid">
            <div class="info-card">
                <h3>📊 Análise Atual</h3>
                <div id="analysis" style="color: #00BCD4;">Aguardando análise...</div>
            </div>
            <div class="info-card">
                <h3>🔮 Previsão</h3>
                <div id="prediction" style="color: #FF9800;">Aguardando previsão...</div>
            </div>
        </div>
        
        <div class="info-card">
            <h3>📊 Meta Diária: 240 Trades</h3>
            <div>Atual: <span id="progress-trades">0</span> | Restam: <span id="remaining">240</span> | Lucro: $<span id="profit">0.0000</span></div>
            <div class="progress-bar">
                <div class="progress-fill" id="progress-bar"></div>
            </div>
            <div id="progress-text">0% da meta diária</div>
        </div>
        
        <div class="info-card">
            <h3>🔧 Sistema</h3>
            <div id="info">Carregando...</div>
            <div class="debug-info" id="debug">Sistema inicializando...</div>
        </div>
        
        <div class="info-card">
            <h3>⚠️ Status e Erros</h3>
            <div id="errors" style="color: #4CAF50;">Sistema funcionando</div>
        </div>
    </div>
    
    <script>
        console.log('🚀 Dashboard carregado!');
        
        function updateData() {
            console.log('📡 Atualizando dados...');
            
            fetch('/api/data')
                .then(response => {
                    console.log('📡 Resposta recebida:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('📊 Dados:', data);
                    
                    // Atualizar métricas
                    document.getElementById('trades').textContent = data.trades || 0;
                    document.getElementById('balance').textContent = '$' + (data.balance || 0).toFixed(4);
                    document.getElementById('price').textContent = '$' + (data.price || 0).toFixed(2);
                    document.getElementById('profit').textContent = (data.profit || 0).toFixed(4);
                    
                    // Taxa de sucesso
                    const winRate = data.total_trades > 0 ? ((data.profitable_trades / data.total_trades) * 100) : 0;
                    document.getElementById('win-rate').textContent = winRate.toFixed(1) + '%';
                    
                    // Status com classes CSS
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status || 'PARADO';
                    statusEl.className = 'status-indicator';
                    
                    if (data.status === 'RODANDO') {
                        statusEl.classList.add('status-running');
                    } else if (data.status === 'PAUSADO') {
                        statusEl.classList.add('status-paused');
                    } else {
                        statusEl.classList.add('status-stopped');
                    }
                    
                    // Análise e previsão
                    document.getElementById('analysis').textContent = data.analysis || 'Aguardando análise...';
                    document.getElementById('prediction').textContent = data.prediction || 'Aguardando previsão...';
                    
                    // Progresso
                    const trades = data.trades || 0;
                    const progress = Math.min((trades / 240) * 100, 100);
                    document.getElementById('progress-trades').textContent = trades;
                    document.getElementById('remaining').textContent = Math.max(240 - trades, 0);
                    document.getElementById('progress-bar').style.width = progress + '%';
                    document.getElementById('progress-text').textContent = progress.toFixed(1) + '% da meta diária';
                    
                    document.getElementById('info').textContent = data.info || 'Sistema ativo';
                    document.getElementById('debug').textContent = 'Última atualização: ' + new Date().toLocaleTimeString();
                    
                    // Erros
                    const errorEl = document.getElementById('errors');
                    if (data.error) {
                        errorEl.textContent = '❌ ERRO: ' + data.error;
                        errorEl.style.color = '#ff6b6b';
                    } else {
                        errorEl.textContent = '✅ Sistema funcionando - Bot executando análises e trades';
                        errorEl.style.color = '#4CAF50';
                    }
                })
                .catch(error => {
                    console.error('❌ Erro na requisição:', error);
                    document.getElementById('errors').textContent = '❌ ERRO DE CONEXÃO: ' + error.message;
                    document.getElementById('errors').style.color = '#ff6b6b';
                });
        }
        
        function startBot() {
            console.log('🚀 Iniciando bot...');
            document.getElementById('startBtn').disabled = true;
            
            fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('✅ Resposta start:', data);
                document.getElementById('startBtn').disabled = false;
                setTimeout(updateData, 1000);
            })
            .catch(error => {
                console.error('❌ Erro ao iniciar:', error);
                document.getElementById('startBtn').disabled = false;
            });
        }
        
        function pauseBot() {
            console.log('⏸️ Pausando bot...');
            document.getElementById('pauseBtn').disabled = true;
            
            fetch('/api/pause', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('⏸️ Resposta pause:', data);
                document.getElementById('pauseBtn').disabled = false;
                setTimeout(updateData, 1000);
            })
            .catch(error => {
                console.error('❌ Erro ao pausar:', error);
                document.getElementById('pauseBtn').disabled = false;
            });
        }
        
        function stopBot() {
            console.log('🛑 Parando bot...');
            document.getElementById('stopBtn').disabled = true;
            
            fetch('/api/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                console.log('🛑 Resposta stop:', data);
                document.getElementById('stopBtn').disabled = false;
                setTimeout(updateData, 1000);
            })
            .catch(error => {
                console.error('❌ Erro ao parar:', error);
                document.getElementById('stopBtn').disabled = false;
            });
        }
        
        // Inicializar
        console.log('🔄 Iniciando atualizações automáticas...');
        updateData();
        setInterval(updateData, 3000);
    </script>
</body>
</html>
'''

@app.route('/api/data')
def get_data():
    """Dados completos do sistema"""
    try:
        # Dados da API
        balance = 0
        price = 0
        
        if bitget_api and API_CONNECTED:
            try:
                balance_info = bitget_api.get_balance()
                if balance_info:
                    balance = balance_info.get('free', 0)
            except:
                pass
                
            try:
                price = bitget_api.get_eth_price() or 0
            except:
                pass
        
        # Status
        if bot_state['is_running'] and not bot_state['is_paused']:
            status = 'RODANDO'
        elif bot_state['is_paused']:
            status = 'PAUSADO'
        else:
            status = 'PARADO'
        
        # Thread status
        thread_active = bot_state['thread'] and bot_state['thread'].is_alive()
        
        return {
            'trades': bot_state['trades_today'],
            'profitable_trades': bot_state['profitable_trades'],
            'total_trades': bot_state['total_trades'],
            'balance': balance,
            'price': price,
            'profit': bot_state['total_profit'],
            'status': status,
            'analysis': str(bot_state['last_analysis']) if bot_state['last_analysis'] else 'Aguardando análise...',
            'prediction': str(bot_state['current_prediction']) if bot_state['current_prediction'] else 'Aguardando previsão...',
            'info': f'API: {API_CONNECTED}, Thread: {"Ativa" if thread_active else "Inativa"}, Total: {bot_state["total_trades"]}',
            'error': bot_state['last_error']
        }
        
    except Exception as e:
        return {
            'trades': 0, 'balance': 0, 'price': 0, 'profit': 0,
            'status': 'ERRO', 'error': str(e),
            'analysis': 'Erro na análise', 'prediction': 'Erro na previsão'
        }

@app.route('/api/start', methods=['POST'])
def start():
    """Iniciar o sistema de trading"""
    try:
        if not trading_bot:
            return {'success': False, 'error': 'TradingBot não disponível'}
        
        bot_state['is_running'] = True
        bot_state['is_paused'] = False
        bot_state['last_error'] = None
        
        if not bot_state['thread'] or not bot_state['thread'].is_alive():
            bot_state['thread'] = threading.Thread(target=run_trading_loop, daemon=True)
            bot_state['thread'].start()
        
        print("🚀 Sistema de trading iniciado!")
        return {'success': True, 'message': 'Sistema iniciado - executando análises e trades'}
        
    except Exception as e:
        error_msg = f"Erro ao iniciar: {e}"
        bot_state['last_error'] = error_msg
        return {'success': False, 'error': error_msg}

@app.route('/api/pause', methods=['POST'])
def pause():
    bot_state['is_paused'] = True
    print("⏸️ Sistema pausado")
    return {'success': True, 'message': 'Sistema pausado'}

@app.route('/api/stop', methods=['POST'])
def stop():
    bot_state['is_running'] = False
    bot_state['is_paused'] = False
    print("🛑 Sistema parado")
    return {'success': True, 'message': 'Sistema parado'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Iniciando Trading Bot Dashboard...")
    print(f"🔗 API: {'Conectada' if API_CONNECTED else 'Desconectada'}")
    print(f"🤖 TradingBot: {'Disponível' if trading_bot else 'Indisponível'}")
    
    app.run(host='0.0.0.0', port=port, debug=True)
