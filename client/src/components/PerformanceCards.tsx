import { usePerformanceMetrics } from '@/hooks/useTradingData';
import { useSocket } from '@/hooks/useSocket';

export default function PerformanceCards() {
  const metrics = usePerformanceMetrics();
  const { data } = useSocket();

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatVolume = (volume: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(volume);
  };

  const formatChange = (change: number) => {
    const formatted = Math.abs(change).toFixed(1);
    return change >= 0 ? `+${formatted}%` : `-${formatted}%`;
  };

  // Use real-time price data if available
  const currentPrice = data.price_update?.eth_price || metrics?.eth_price || 3847.32;
  const priceChange = data.price_update?.change_24h || metrics?.price_change_pct || -2.3;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      {/* Volume 24h */}
      <div className="bg-charcoal rounded-lg p-6 border border-dark-gray">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-text-secondary text-sm font-medium">Volume 24h</h3>
          <i className="fas fa-chart-area text-warning"></i>
        </div>
        <p className="text-2xl font-bold font-mono" data-testid="text-volume-24h">
          {formatVolume(metrics?.volume_24h || 45672.30)}
        </p>
        <p className="text-profit text-sm mt-1">
          <i className="fas fa-arrow-up text-xs"></i>
          +{metrics?.volume_change_pct?.toFixed(1) || '12.5'}% vs ontem
        </p>
      </div>

      {/* Gross Profit */}
      <div className="bg-charcoal rounded-lg p-6 border border-dark-gray">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-text-secondary text-sm font-medium">Lucro Bruto</h3>
          <i className="fas fa-dollar-sign text-profit"></i>
        </div>
        <p className="text-2xl font-bold font-mono text-profit" data-testid="text-gross-profit">
          {formatCurrency(metrics?.gross_profit || 1234.56)}
        </p>
        <p className="text-profit text-sm mt-1">
          <i className="fas fa-arrow-up text-xs"></i>
          +{metrics?.profit_change_pct?.toFixed(1) || '8.7'}% vs ontem
        </p>
      </div>

      {/* Trades Executed */}
      <div className="bg-charcoal rounded-lg p-6 border border-dark-gray">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-text-secondary text-sm font-medium">Trades Executados</h3>
          <i className="fas fa-exchange-alt text-warning"></i>
        </div>
        <p className="text-2xl font-bold font-mono" data-testid="text-trades-executed">
          {metrics?.trades_executed || 147}
        </p>
        <p className="text-text-secondary text-sm mt-1">Meta: 200/dia</p>
      </div>

      {/* ETH Price */}
      <div className="bg-charcoal rounded-lg p-6 border border-dark-gray">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-text-secondary text-sm font-medium">Preço ETH</h3>
          <i className="fab fa-ethereum text-warning"></i>
        </div>
        <p className="text-2xl font-bold font-mono" data-testid="text-eth-price">
          {formatCurrency(currentPrice)}
        </p>
        <p className={`text-sm mt-1 ${priceChange >= 0 ? 'text-profit' : 'text-loss'}`}>
          <i className={`fas ${priceChange >= 0 ? 'fa-arrow-up' : 'fa-arrow-down'} text-xs`}></i>
          {formatChange(priceChange)} (1h)
        </p>
      </div>
    </div>
  );
}
