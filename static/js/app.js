// JavaScript principal para o Trading Bot ETH/USDT
// Otimizado para Railway deployment

class TradingBotInterface {
    constructor() {
        this.isLoading = false;
        this.statusCheckInterval = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.startStatusMonitoring();
        this.showWelcomeMessage();
    }

    bindEvents() {
        // Event listeners para botões
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (startBtn) {
            startBtn.addEventListener('click', (e) => this.handleStartBot(e));
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', (e) => this.handleStopBot(e));
        }

        // Event listeners para navegação
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.clearAllAlerts();
            }
        });

        // Prevent double-click issues
        document.addEventListener('click', (e) => {
            if (e.target.matches('.btn') && this.isLoading) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
    }

    async handleStartBot(event) {
        event.preventDefault();
        
        if (this.isLoading) return;
        
        const button = event.target;
        this.setLoadingState(button, true);
        
        try {
            const result = await this.makeRequest('/api/bot/start', 'POST');
            
            if (result.error) {
                this.showAlert('Erro ao iniciar bot: ' + result.error, 'danger');
                this.logError('Start bot error', result.error);
            } else {
                this.showAlert('Bot iniciado com sucesso! Redirecionando...', 'success');
                this.logSuccess('Bot started successfully');
                
                // Aguardar um pouco antes de recarregar
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            }
        } catch (error) {
            this.showAlert('Erro de conexão: ' + error.message, 'danger');
            this.logError('Connection error', error);
        } finally {
            this.setLoadingState(button, false);
        }
    }

    async handleStopBot(event) {
        event.preventDefault();
        
        if (this.isLoading) return;
        
        const button = event.target;
        this.setLoadingState(button, true);
        
        try {
            const result = await this.makeRequest('/api/bot/stop', 'POST');
            
            if (result.error) {
                this.showAlert('Erro ao parar bot: ' + result.error, 'danger');
                this.logError('Stop bot error', result.error);
            } else {
                this.showAlert('Bot parado com sucesso! Redirecionando...', 'success');
                this.logSuccess('Bot stopped successfully');
                
                // Aguardar um pouco antes de recarregar
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            }
        } catch (error) {
            this.showAlert('Erro de conexão: ' + error.message, 'danger');
            this.logError('Connection error', error);
        } finally {
            this.setLoadingState(button, false);
        }
    }

    async makeRequest(url, method = 'GET', data = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    setLoadingState(button, loading) {
        this.isLoading = loading;
        
        const spinner = button.querySelector('.loading-spinner');
        const icon = button.querySelector('i:not(.loading-spinner i)');
        const text = button.querySelector('.btn-text') || button.childNodes[1];
        
        if (loading) {
            button.disabled = true;
            button.classList.add('loading');
            
            if (spinner) spinner.style.display = 'inline-block';
            if (icon) icon.style.display = 'none';
            
            // Adicionar indicador visual
            button.style.cursor = 'wait';
            button.style.opacity = '0.8';
        } else {
            button.disabled = false;
            button.classList.remove('loading');
            
            if (spinner) spinner.style.display = 'none';
            if (icon) icon.style.display = 'inline';
            
            // Restaurar visual
            button.style.cursor = 'pointer';
            button.style.opacity = '1';
        }
    }

    showAlert(message, type = 'info', duration = 5000) {
        // Remover alertas existentes
        this.clearAllAlerts();
        
        const alertClass = this.getAlertClass(type);
        const icon = this.getAlertIcon(type);
        
        const alertHtml = `
            <div class="alert ${alertClass} alert-custom alert-dismissible fade show" role="alert">
                <i class="fas ${icon} me-2"></i>
                <span class="alert-message">${message}</span>
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Fechar"></button>
            </div>
        `;
        
        // Inserir no topo da página
        const container = document.querySelector('.container') || document.body;
        container.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-remover
        if (duration > 0) {
            setTimeout(() => {
                this.clearAllAlerts();
            }, duration);
        }
        
        // Scroll suave para o topo
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        // Flash effect
        const alert = document.querySelector('.alert');
        if (alert) {
            alert.classList.add(type === 'success' ? 'success-flash' : 'error-flash');
        }
    }

    clearAllAlerts() {
        const alerts = document.querySelectorAll('.alert-dismissible');
        alerts.forEach(alert => {
            alert.classList.remove('show');
            setTimeout(() => alert.remove(), 150);
        });
    }

    getAlertClass(type) {
        const classes = {
            'success': 'alert-success',
            'danger': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        };
        return classes[type] || 'alert-info';
    }

    getAlertIcon(type) {
        const icons = {
            'success': 'fa-check-circle',
            'danger': 'fa-exclamation-triangle',
            'warning': 'fa-exclamation-circle',
            'info': 'fa-info-circle'
        };
        return icons[type] || 'fa-info-circle';
    }

    startStatusMonitoring() {
        // Verificar status a cada 30 segundos
        this.statusCheckInterval = setInterval(async () => {
            try {
                await this.updateStatus();
            } catch (error) {
                this.logError('Status check failed', error);
            }
        }, 30000);
    }

    async updateStatus() {
        try {
            const status = await this.makeRequest('/api/bot/status', 'GET');
            
            if (!status.error) {
                this.updateStatusIndicators(status);
            }
        } catch (error) {
            // Silently fail for status updates
            console.warn('Status update failed:', error);
        }
    }

    updateStatusIndicators(status) {
        // Atualizar indicador de status
        const statusIndicator = document.querySelector('.status-indicator');
        const statusBadge = document.querySelector('.badge');
        
        if (statusIndicator) {
            statusIndicator.className = `status-indicator ${status.running ? 'status-running' : 'status-stopped'}`;
        }
        
        if (statusBadge) {
            statusBadge.className = `ms-2 badge ${status.running ? 'bg-success' : 'bg-secondary'}`;
            statusBadge.textContent = status.running ? 'RODANDO' : 'PARADO';
        }
        
        // Atualizar uptime se disponível
        const uptimeElement = document.querySelector('.uptime-display');
        if (uptimeElement && status.uptime) {
            uptimeElement.textContent = status.uptime;
        }
    }

    showWelcomeMessage() {
        // Mostrar mensagem de boas-vindas apenas uma vez por sessão
        if (!sessionStorage.getItem('welcomeShown')) {
            setTimeout(() => {
                this.showAlert('Trading Bot ETH/USDT carregado com sucesso. Pronto para operar no Railway!', 'info', 3000);
                sessionStorage.setItem('welcomeShown', 'true');
            }, 1000);
        }
    }

    logSuccess(message, data = null) {
        console.log(`✅ [TradingBot] ${message}`, data || '');
    }

    logError(message, error = null) {
        console.error(`❌ [TradingBot] ${message}`, error || '');
    }

    logInfo(message, data = null) {
        console.info(`ℹ️ [TradingBot] ${message}`, data || '');
    }

    destroy() {
        // Cleanup
        if (this.statusCheckInterval) {
            clearInterval(this.statusCheckInterval);
        }
        
        this.clearAllAlerts();
    }
}

// Dashboard specific functionality
class DashboardInterface extends TradingBotInterface {
    constructor() {
        super();
        this.charts = {};
        this.initDashboard();
    }

    initDashboard() {
        this.initializeCharts();
        this.startDataRefresh();
    }

    initializeCharts() {
        // Chart.js configuration for dark theme
        if (typeof Chart !== 'undefined') {
            Chart.defaults.color = '#adb5bd';
            Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
        }
    }

    startDataRefresh() {
        // Refresh dashboard data every 60 seconds
        setInterval(async () => {
            await this.refreshDashboardData();
        }, 60000);
    }

    async refreshDashboardData() {
        try {
            const [balanceData, statusData] = await Promise.all([
                this.makeRequest('/api/balance'),
                this.makeRequest('/api/bot/status')
            ]);

            if (!balanceData.error) {
                this.updateBalanceDisplay(balanceData);
            }

            if (!statusData.error) {
                this.updateStatusIndicators(statusData);
            }
        } catch (error) {
            this.logError('Dashboard data refresh failed', error);
        }
    }

    updateBalanceDisplay(balanceData) {
        const balanceElements = document.querySelectorAll('[data-balance]');
        
        balanceElements.forEach(element => {
            const field = element.getAttribute('data-balance');
            if (balanceData[field] !== undefined) {
                element.textContent = `$${balanceData[field].toFixed(2)}`;
            }
        });
    }
}

// Utility functions
const utils = {
    formatCurrency: (value) => {
        return new Intl.NumberFormat('pt-BR', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    },

    formatPercentage: (value) => {
        return `${value.toFixed(2)}%`;
    },

    formatTime: (timestamp) => {
        return new Date(timestamp).toLocaleString('pt-BR');
    },

    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on dashboard page
    const isDashboard = window.location.pathname === '/dashboard';
    
    // Initialize appropriate interface
    window.tradingBotApp = isDashboard ? 
        new DashboardInterface() : 
        new TradingBotInterface();
    
    console.log('🚀 Trading Bot interface initialized');
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (window.tradingBotApp) {
        window.tradingBotApp.destroy();
    }
});

// Error handling for uncaught errors
window.addEventListener('error', function(event) {
    console.error('❌ Uncaught error:', event.error);
});

// Promise rejection handling
window.addEventListener('unhandledrejection', function(event) {
    console.error('❌ Unhandled promise rejection:', event.reason);
    event.preventDefault();
});
