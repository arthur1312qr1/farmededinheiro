import PerformanceCards from './PerformanceCards';
import ActivityLog from './ActivityLog';
import CurrentPosition from './CurrentPosition';
import BotControls from './BotControls';

export default function MainContent() {
  return (
    <main className="flex-1 p-6 overflow-auto">
      {/* Performance Cards Row */}
      <PerformanceCards />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trading Activity Log */}
        <ActivityLog />

        {/* Current Position & Controls */}
        <div className="space-y-6">
          <CurrentPosition />
          <BotControls />

          {/* AI Analysis */}
          <div className="bg-charcoal rounded-lg border border-dark-gray p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <i className="fas fa-brain mr-2 text-warning"></i>
              Análise AI
            </h3>
            
            <div className="space-y-3 text-sm">
              <div className="p-3 bg-dark-gray rounded">
                <p className="font-medium text-profit mb-1">Tendência: BULLISH</p>
                <p className="text-text-secondary">
                  Confluência de indicadores sugere movimento de alta. RSI oversold com divergência positiva.
                </p>
              </div>
              
              <div className="p-3 bg-dark-gray rounded">
                <p className="font-medium text-warning mb-1">Confiança: 78%</p>
                <p className="text-text-secondary">
                  Volume acima da média confirmando breakout.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chart Placeholder */}
      <div className="mt-8 bg-charcoal rounded-lg border border-dark-gray p-6">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <i className="fas fa-chart-candlestick mr-3 text-warning"></i>
          Gráfico ETH/USDT
        </h2>
        <div className="h-96 bg-dark-gray rounded-lg flex items-center justify-center">
          <div className="text-center">
            <i className="fas fa-chart-line text-6xl text-text-secondary mb-4"></i>
            <p className="text-text-secondary">Gráfico em tempo real será carregado aqui</p>
            <p className="text-sm text-text-secondary mt-2">
              Integração com TradingView Widget
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
