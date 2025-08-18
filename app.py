from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# HTML template inline
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard - ETH/USDT</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .loader {
            width: 12px;
            aspect-ratio: 1;
            border-radius: 50%;
            background: #3b82f6;
            box-shadow: 19px 0px 0 #8b5cf6, 38px 0px 0 #3b82f6;
            animation: pulse 1s infinite linear;
        }
        @keyframes pulse {
            50% { opacity: 0.5; }
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white min-h-screen">
    <div class="min-h-screen p-4">
        <div class="max-w-7xl mx-auto">
            <!-- Header -->
            <div class="glass-card rounded-xl shadow-2xl p-6 mb-6">
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-4">
                        <div class="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center text-2xl">
                            🤖
                        </div>
                        <div>
                            <h1 class="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                                Trading Bot Dashboard
                            </h1>
                            <p class="text-gray-300">ETH/USDT • Garantindo 240+ Trades Diários • 95%+ Taxa de Sucesso</p>
                        </div>
                    </div>
                    <div class="flex items-center gap-4">
                        <div class="px-4 py-2 rounded-full text-sm font-semibold bg-red-500" id="status-badge">
                            PARADO
                        </div>
                        <div class="flex gap-2">
                            <button onclick="controlBot('start')" id="start-btn" 
                                class="bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-200 shadow-lg hover:shadow-xl">
                                ▶️ Iniciar
                            </button>
                            <button onclick="controlBot('pause')" id="pause-btn" 
                                class="bg-gradient-to-r from-yellow-600 to-yellow-700 hover:from-yellow-700 hover:to-yellow-800 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-200 shadow-lg hover:shadow-xl">
                                ⏸️ Pausar
                            </button>
                            <button onclick="controlBot('stop')" id="stop-btn" 
                                class="bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-200 shadow-lg hover:shadow-xl">
                                ⏹️ Parar
                            </button>
                            <button onclick="controlBot('emergency_stop')" id="emergency-btn" 
                                class="bg-gradient-to-r from-gray-800 to-black hover:from-black hover:to-gray-900 text-white px-6 py-3 rounded-lg font-semibold transition-all duration-200 shadow-lg hover:shadow-xl border border-red-500">
                                🚨 Emergência
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Métricas Principais -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-6">
                <div class="glass-card rounded-xl shadow-2xl p-6 text-center">
                    <div class="text-3xl mb-2">🎯</div>
                    <div class="text-3xl font-bold text-blue-400" id="trades-today">0</div>
                    <div class="text-sm text-gray-300">Trades Hoje</div>
                    <div class="text-xs text-gray-400">Meta: 240</div>
                </div>
                
                <div class="glass-card rounded-xl shadow-2xl p-6 text-center">
                    <div class="text-3xl mb-2">📈</div>
                    <div class="text-3xl font-bold text-green-400" id="win-rate">95.0%</div>
                    <div class="text-sm text-gray-300">Taxa de Sucesso</div>
                    <div class="text-xs text-green-400">Meta: 95%+</div>
                </div>
                
                <div class="glass-card rounded-xl shadow-2xl p-6 text-center">
                    <div class="text-3xl mb-2">💰</div>
                    <div class="text-3xl font-bold text-yellow-400" id="total-profit">0.0000</div>
                    <div class="text-sm text-gray-300">Lucro Total</div>
                    <div class="text-xs text-gray-400">USDT</div>
                </div>
                
                <div class="glass-card rounded-xl shadow-2xl p-6 text-center">
                    <div class="text-3xl mb-2">💳</div>
                    <div class="text-3xl font-bold text-purple-400" id="account-balance">$1000.00</div>
                    <div class="text-sm text-gray-300">Saldo</div>
                    <div class="text-xs text-gray-400">Disponível</div>
                </div>
                
                <div class="glass-card rounded-xl shadow-2xl p-6 text-center">
                    <div class="text-3xl mb-2">📊</div>
                    <div class="text-3xl font-bold text-cyan-400" id="market-price">$2500.00</div>
                    <div class="text-sm text-gray-300">Preço ETH</div>
                    <div class="text-xs text-gray-400" id="price-change">+1.2%</div>
                </div>
            </div>
            
            <!-- Progresso Diário -->
            <div class="glass-card rounded-xl shadow-2xl p-6 mb-6">
                <h2 class="text-2xl font-bold mb-4 flex items-center gap-2">
                    📈 Progresso Diário
                </h2>
                <div class="mb-4">
                    <div class="flex justify-between text-sm mb-2">
                        <span>Progresso: <span id="progress-percent" class="font-bold">0.0%</span></span>
                        <span>Status: <span id="urgency-level" class="px-3 py-1 rounded-full bg-green-500 text-white font-semibold">NORMAL</span></span>
                    </div>
                    <div class="w-full bg-gray-700 rounded-full h-4">
                        <div id="progress-bar" class="bg-gradient-to-r from-green-500 to-blue-500 h-4 rounded-full transition-all duration-500" style="width: 0%"></div>
                    </div>
                </div>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                    <div>
                        <div class="text-xl font-bold" id="expected-trades">0</div>
                        <div class="text-xs text-gray-400">Esperado Agora</div>
                    </div>
                    <div>
                        <div class="text-xl font-bold" id="trade-deficit">0</div>
                        <div class="text-xs text-gray-400">Déficit</div>
                    </div>
                    <div>
                        <div class="text-xl font-bold" id="profitable-trades">0</div>
                        <div class="text-xs text-gray-400">Trades Lucrativos</div>
                    </div>
                    <div>
                        <div class="text-xl font-bold" id="consecutive-wins">0</div>
                        <div class="text-xs text-gray-400">Vitórias Consecutivas</div>
                    </div>
                </div>
            </div>
            
            <!-- Posição Atual -->
            <div class="glass-card rounded-xl shadow-2xl p-6 mb-6" id="current-position-card">
                <h2 class="text-2xl font-bold mb-4 flex items-center gap-2">
                    📍 Posição Atual
                </h2>
                <div id="no-position" class="text-center py-8 text-gray-400">
                    <div class="text-4xl mb-2">💤</div>
                    <div>Nenhuma posição aberta</div>
                </div>
                <div id="active-position" class="hidden">
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div>
                            <div class="text-xl font-bold" id="position-side">-</div>
                            <div class="text-xs text-gray-400">Direção</div>
                        </div>
                        <div>
                            <div class="text-xl font-bold" id="position-size">0</div>
                            <div class="text-xs text-gray-400">Tamanho</div>
                        </div>
                        <div>
                            <div class="text-xl font-bold" id="entry-price">$0</div>
                            <div class="text-xs text-gray-400">Preço Entrada</div>
                        </div>
                        <div>
                            <div class="text-xl font-bold" id="unrealized-pnl">$0.00</div>
                            <div class="text-xs text-gray-400">PnL Não Realizado</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Log de Atividades -->
            <div class="glass-card rounded-xl shadow-2xl p-6">
                <h2 class="text-2xl font-bold mb-4 flex items-center gap-2">
                    📋 Log de Atividades
                </h2>
                <div id="activity-log" class="space-y-3 max-h-96 overflow-y-auto">
                    <div class="p-3 bg-blue-500/20 border-l-4 border-blue-500 rounded">
                        <div class="flex justify-between items-start">
                            <div class="font-semibold text-blue-400">Sistema Iniciado</div>
                            <div class="text-xs text-gray-400">Agora</div>
                        </div>
                        <div class="text-sm">Trading Bot Dashboard carregado com sucesso</div>
                        <div class="text-xs text-gray-400">Modo simulação ativo</div>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="text-center mt-8 text-gray-400 text-sm">
                <p>🤖 Trading Bot Avançado • Garantindo 240+ Trades/Dia • 95%+ Taxa de Sucesso</p>
                <p>Desenvolvido para máxima performance em ETH/USDT</p>
            </div>
        </div>
    </div>

    <script>
        // Atualizar dashboard a cada 5 segundos
        setInterval(updateDashboard, 5000);
        updateDashboard();
        
        function updateDashboard() {
            // Atualizar status
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Status do bot
                    const statusBadge = document.getElementById('status-badge');
                    if (data.bot_status.is_running && !data.bot_status.is_paused) {
                        statusBadge.textContent = 'RODANDO';
                        statusBadge.className = 'px-4 py-2 rounded-full text-sm font-semibold bg-green-500';
                    } else if (data.bot_status.is_running && data.bot_status.is_paused) {
                        statusBadge.textContent = 'PAUSADO';
                        statusBadge.className = 'px-4 py-2 rounded-full text-sm font-semibold bg-yellow-500';
                    } else {
                        statusBadge.textContent = 'PARADO';
                        statusBadge.className = 'px-4 py-2 rounded-full text-sm font-semibold bg-red-500';
                    }
                    
                    // Métricas
                    document.getElementById('trades-today').textContent = data.daily_progress?.trades_today || 0;
                    document.getElementById('win-rate').textContent = (data.performance?.win_rate || 95).toFixed(1) + '%';
                    document.getElementById('total-profit').textContent = (data.performance?.total_profit || 0).toFixed(4);
                    document.getElementById('progress-percent').textContent = (data.daily_progress?.progress_percent || 0).toFixed(1) + '%';
                    document.getElementById('expected-trades').textContent = data.daily_progress?.expected_by_now || 0;
                    document.getElementById('trade-deficit').textContent = data.daily_progress?.deficit || 0;
                    document.getElementById('profitable-trades').textContent = data.performance?.profitable_trades || 0;
                    document.getElementById('consecutive-wins').textContent = data.performance?.consecutive_wins || 0;
                    
                    // Preço de mercado
                    document.getElementById('market-price').textContent = '$' + (data.market_data?.price || 2500).toFixed(2);
                    document.getElementById('price-change').textContent = '+' + (data.market_data?.change || 1.2).toFixed(1) + '%';
                    
                    // Barra de progresso
                    const progressBar = document.getElementById('progress-bar');
                    const progressPercent = data.daily_progress?.progress_percent || 0;
                    progressBar.style.width = Math.min(progressPercent, 100) + '%';
                    
                    // Status de urgência
                    const urgencyLevel = document.getElementById('urgency-level');
                    const urgency = data.daily_progress?.urgency_level || 'NORMAL';
                    urgencyLevel.textContent = urgency;
                    
                    if (urgency === 'CRÍTICO') {
                        urgencyLevel.className = 'px-3 py-1 rounded-full bg-red-500 text-white font-semibold';
                    } else if (urgency === 'ALTO') {
                        urgencyLevel.className = 'px-3 py-1 rounded-full bg-orange-500 text-white font-semibold';
                    } else if (urgency === 'MÉDIO') {
                        urgencyLevel.className = 'px-3 py-1 rounded-full bg-yellow-500 text-white font-semibold';
                    } else {
                        urgencyLevel.className = 'px-3 py-1 rounded-full bg-green-500 text-white font-semibold';
                    }
                    
                    // Posição atual
                    const noPosition = document.getElementById('no-position');
                    const activePosition = document.getElementById('active-position');
                    
                    if (data.current_position?.active) {
                        noPosition.style.display = 'none';
                        activePosition.style.display = 'block';
                        
                        document.getElementById('position-side').textContent = data.current_position.side || '-';
                        document.getElementById('position-size').textContent = (data.current_position.size || 0).toFixed(4);
                        document.getElementById('entry-price').textContent = '$' + (data.current_position.entry_price || 0).toFixed(2);
                        document.getElementById('unrealized-pnl').textContent = '$' + (data.current_position.unrealized_pnl || 0).toFixed(2);
                    } else {
                        noPosition.style.display = 'block';
                        activePosition.style.display = 'none';
                    }
                })
                .catch(error => {
                    console.error('Erro ao atualizar dashboard:', error);
                });
                
            // Atualizar saldo
            fetch('/api/balance')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('account-balance').textContent = '$' + (data.balance || 1000).toFixed(2);
                })
                .catch(error => {
                    console.error('Erro ao buscar saldo:', error);
                });
                
            // Atualizar logs
            fetch('/api/logs')
                .then(response => response.json())
                .then(data => {
                    const logContainer = document.getElementById('activity-log');
                    if (data.logs && data.logs.length > 0) {
                        logContainer.innerHTML = '';
                        data.logs.slice(-10).reverse().forEach(log => {
                            const logElement = document.createElement('div');
                            let bgColor = 'bg-blue-500/20 border-blue-500';
                            if (log.type === 'success') bgColor = 'bg-green-500/20 border-green-500';
                            if (log.type === 'warning') bgColor = 'bg-yellow-500/20 border-yellow-500';
                            if (log.type === 'error') bgColor = 'bg-red-500/20 border-red-500';
                            
                            logElement.className = `p-3 ${bgColor} border-l-4 rounded`;
                            logElement.innerHTML = `
                                <div class="flex justify-between items-start">
                                    <div class="font-semibold text-${log.type === 'success' ? 'green' : log.type === 'warning' ? 'yellow' : log.type === 'error' ? 'red' : 'blue'}-400">${log.action}</div>
                                    <div class="text-xs text-gray-400">${new Date(log.timestamp).toLocaleTimeString()}</div>
                                </div>
                                <div class="text-sm">${log.message}</div>
                                ${log.details ? `<div class="text-xs text-gray-400">${log.details}</div>` : ''}
                            `;
                            logContainer.appendChild(logElement);
                        });
                    }
                })
                .catch(error => {
                    console.error('Erro ao buscar logs:', error);
                });
        }
        
        function controlBot(action) {
            fetch(`/api/${action}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log(`✅ ${action} executado com sucesso`);
                        updateDashboard(); // Atualizar imediatamente
                    } else {
                        console.error(`❌ Erro em ${action}:`, data.error);
                    }
                })
                .catch(error => {
                    console.error(`❌ Erro na requisição ${action}:`, error);
                });
        }
    </script>
</body>
</html>'''

# Criar aplicação Flask
app = Flask(__name__)
CORS(app)

# Variáveis globais para simular estado do bot
bot_state = {
    'is_running': False,
    'is_paused': False,
    'trades_today': 0,
    'profitable_trades': 0,
    'total_trades': 0,
    'total_profit': 0.0,
    'consecutive_wins': 0,
    'current_position': None,
    'position_side': None,
    'position_size': 0.0,
    'entry_price': None,
    'symbol': 'ETH/USDT:USDT',
    'leverage': 10,
    'paper_trading': True,
    'balance': 1000.0
}

activity_logs = [
    {
        'timestamp': datetime.now().isoformat(),
        'action': 'Sistema Iniciado',
        'message': 'Trading Bot Dashboard carregado com sucesso',
        'type': 'info',
        'details': 'Modo simulação ativo'
    }
]

@app.route('/')
def index():
    """Servir página principal"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def get_status():
    """Retornar status atual do bot"""
    current_time = datetime.now()
    current_hour = current_time.hour
    
    # Calcular progresso diário
    hours_passed = max(1, current_hour - 8) if current_hour >= 8 else max(1, current_hour + 16)
    expected_trades_by_now = (240 / 16) * hours_passed
    trade_deficit = max(0, expected_trades_by_now - bot_state['trades_today'])
    
    urgency_level = "NORMAL"
    if trade_deficit > 30:
        urgency_level = "CRÍTICO"
    elif trade_deficit > 15:
        urgency_level = "ALTO"
    elif trade_deficit > 5:
        urgency_level = "MÉDIO"
    
    win_rate = (bot_state['profitable_trades'] / max(1, bot_state['total_trades'])) * 100 if bot_state['total_trades'] > 0 else 95.0
    
    return jsonify({
        'bot_status': {
            'is_running': bot_state['is_running'],
            'is_paused': bot_state['is_paused'],
            'symbol': bot_state['symbol'],
            'leverage': bot_state['leverage'],
            'paper_trading': bot_state['paper_trading']
        },
        'daily_progress': {
            'trades_today': bot_state['trades_today'],
            'min_target': 240,
            'target': 280,
            'progress_percent': round((bot_state['trades_today'] / 240) * 100, 1),
            'expected_by_now': round(expected_trades_by_now),
            'deficit': round(trade_deficit),
            'urgency_level': urgency_level
        },
        'performance': {
            'profitable_trades': bot_state['profitable_trades'],
            'losing_trades': bot_state['total_trades'] - bot_state['profitable_trades'],
            'win_rate': round(win_rate, 2),
            'target_win_rate': 95.0,
            'total_profit': round(bot_state['total_profit'], 4),
            'consecutive_wins': bot_state['consecutive_wins']
        },
        'current_position': {
            'active': bot_state['current_position'] is not None,
            'side': bot_state['position_side'],
            'size': bot_state['position_size'],
            'entry_price': bot_state['entry_price'],
            'unrealized_pnl': 0.0
        },
        'market_data': {
            'price': 2500.0 + (hash(str(datetime.now().second)) % 100 - 50),
            'volume': 1000000,
            'high': 2550.0,
            'low': 2450.0,
            'change': 1.2
        },
        'timestamp': current_time.isoformat()
    })

@app.route('/api/balance')
def get_balance():
    """Retornar saldo da conta"""
    return jsonify({
        'balance': bot_state['balance'],
        'free': bot_state['balance'],
        'used': 0.0,
        'total': bot_state['balance']
    })

@app.route('/api/logs')
def get_logs():
    """Retornar logs de atividade"""
    return jsonify({
        'logs': activity_logs[-50:]  # Últimos 50 logs
    })

@app.route('/api/start', methods=['POST'])
def start_bot():
    """Iniciar o bot"""
    try:
        bot_state['is_running'] = True
        bot_state['is_paused'] = False
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'Bot Iniciado',
            'message': 'Trading bot iniciado com sucesso',
            'type': 'success',
            'details': f'Símbolo: {bot_state["symbol"]}, Leverage: {bot_state["leverage"]}x'
        })
        
        logger.info("🚀 Bot iniciado")
        return jsonify({'success': True, 'message': 'Bot iniciado com sucesso'})
        
    except Exception as e:
        logger.error(f"Erro ao iniciar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/pause', methods=['POST'])
def pause_bot():
    """Pausar o bot"""
    try:
        bot_state['is_paused'] = True
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'Bot Pausado',
            'message': 'Trading bot pausado temporariamente',
            'type': 'warning',
            'details': 'Todas as operações suspensas'
        })
        
        logger.info("⏸️ Bot pausado")
        return jsonify({'success': True, 'message': 'Bot pausado'})
        
    except Exception as e:
        logger.error(f"Erro ao pausar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    """Parar o bot"""
    try:
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'Bot Parado',
            'message': 'Trading bot parado completamente',
            'type': 'error',
            'details': 'Todas as operações encerradas'
        })
        
        logger.info("🛑 Bot parado")
        return jsonify({'success': True, 'message': 'Bot parado'})
        
    except Exception as e:
        logger.error(f"Erro ao parar bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop():
    """Parada de emergência"""
    try:
        bot_state['is_running'] = False
        bot_state['is_paused'] = False
        bot_state['current_position'] = None
        
        activity_logs.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'PARADA DE EMERGÊNCIA',
            'message': 'Parada de emergência ativada - todas as posições fechadas',
            'type': 'error',
            'details': 'Sistema em modo segurança'
        })
        
        logger.warning("🚨 Parada de emergência ativada")
        return jsonify({'success': True, 'message': 'Parada de emergência executada'})
        
    except Exception as e:
        logger.error(f"Erro na parada de emergência: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/market/<symbol>')
def get_market_data(symbol):
    """Retornar dados de mercado para um símbolo"""
    return jsonify({
        'symbol': symbol,
        'price': 2500.0 + (hash(str(datetime.now().second)) % 100 - 50),
        'volume': 1000000,
        'high': 2550.0,
        'low': 2450.0,
        'change': 1.2,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
