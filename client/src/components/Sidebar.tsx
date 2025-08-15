import { useTradingConfig, useTradingStatus } from '@/hooks/useTradingData';

export default function Sidebar() {
  const { data: config } = useTradingConfig();
  const { data: stats } = useTradingStatus();

  return (
    <aside className="w-80 bg-charcoal border-r border-dark-gray p-6">
      <nav className="space-y-4 mb-8">
        <a 
          href="#" 
          className="flex items-center space-x-3 text-text-primary hover:text-warning transition-colors"
          data-testid="link-dashboard"
        >
          <i className="fas fa-chart-line w-5"></i>
          <span>Dashboard</span>
        </a>
        <a 
          href="#" 
          className="flex items-center space-x-3 text-text-secondary hover:text-warning transition-colors"
          data-testid="link-bot-config"
        >
          <i className="fas fa-robot w-5"></i>
          <span>Configurações Bot</span>
        </a>
        <a 
          href="#" 
          className="flex items-center space-x-3 text-text-secondary hover:text-warning transition-colors"
          data-testid="link-history"
        >
          <i className="fas fa-history w-5"></i>
          <span>Histórico</span>
        </a>
      </nav>

      {/* Trading Configuration Panel */}
      <div className="bg-dark-gray rounded-lg p-4 mb-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <i className="fas fa-sliders-h mr-2 text-warning"></i>
          Configuração Atual
        </h3>
        
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-text-secondary">Par de Trading:</span>
            <span className="font-mono font-semibold" data-testid="text-trading-pair">
              {config?.symbol || 'ETHUSDT'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-secondary">Alavancagem:</span>
            <span className="font-mono font-semibold text-warning" data-testid="text-leverage">
              {config?.leverage || 10}x
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-secondary">Modo:</span>
            <span className="font-semibold text-profit" data-testid="text-trading-mode">
              {config?.paper_trading ? 'DEMO' : 'REAL'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-secondary">Estratégia:</span>
            <span className="font-semibold" data-testid="text-strategy">
              {config?.strategy || 'Scalping'}
            </span>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="bg-dark-gray rounded-lg p-4">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <i className="fas fa-chart-bar mr-2 text-warning"></i>
          Stats Rápidas
        </h3>
        
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-text-secondary">Trades Hoje:</span>
            <span className="font-mono font-semibold" data-testid="text-daily-trades">
              {stats?.daily_trades || 147}/{stats?.target_trades || 200}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-secondary">Win Rate:</span>
            <span className="font-mono font-semibold text-profit" data-testid="text-win-rate">
              {stats?.win_rate?.toFixed(1) || '68.3'}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-secondary">Uptime:</span>
            <span className="font-mono font-semibold text-profit" data-testid="text-uptime">
              {stats?.uptime || '23h 47m'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
}
