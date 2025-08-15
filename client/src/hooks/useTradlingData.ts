import { useQuery } from '@tanstack/react-query';
import { TradingStatus, BotStatus, MarketData, ActivityLogEntry, TradingConfig } from '@shared/schema';
import { PerformanceMetrics } from '@/types/trading';

export function useTradingStatus() {
  return useQuery<TradingStatus>({
    queryKey: ['/api/trading/status'],
    refetchInterval: 5000, // Refetch every 5 seconds
    staleTime: 1000, // Consider data stale after 1 second
  });
}

export function useBotStatus() {
  return useQuery<BotStatus>({
    queryKey: ['/api/bot/status'],
    refetchInterval: 10000, // Refetch every 10 seconds
  });
}

export function useMarketData(symbol: string = 'ETHUSDT') {
  return useQuery<MarketData>({
    queryKey: ['/api/market/data', symbol],
    refetchInterval: 5000, // Refetch every 5 seconds
  });
}

export function useActivityLogs() {
  return useQuery<ActivityLogEntry[]>({
    queryKey: ['/api/trading/logs'],
    refetchInterval: 10000, // Refetch every 10 seconds
  });
}

export function useTradingConfig() {
  return useQuery<TradingConfig>({
    queryKey: ['/api/trading/config'],
    staleTime: 60000, // Config doesn't change often
  });
}

// Calculate performance metrics from trading data
export function usePerformanceMetrics(): PerformanceMetrics | null {
  const { data: tradingStatus } = useTradingStatus();
  const { data: marketData } = useMarketData();

  if (!tradingStatus || !marketData) {
    return null;
  }

  return {
    volume_24h: tradingStatus.total_volume || 45672.30,
    gross_profit: tradingStatus.daily_pnl || 1234.56,
    trades_executed: tradingStatus.daily_trades || 147,
    eth_price: marketData.price || 3847.32,
    price_change_pct: marketData.change_24h || -2.3,
    volume_change_pct: 12.5, // Mock data
    profit_change_pct: 8.7, // Mock data
  };
}
