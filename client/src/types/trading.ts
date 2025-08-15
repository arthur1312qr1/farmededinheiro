export interface TradingPosition {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entry_price: number;
  unrealized_pnl: number;
  stop_loss: number;
  take_profit: number;
  timestamp: Date;
}

export interface ActivityLog {
  timestamp: Date;
  action: string;
  message: string;
  type: 'success' | 'error' | 'warning' | 'info';
  details?: string;
}

export interface TradingStats {
  daily_trades: number;
  target_trades: number;
  win_rate: number;
  daily_pnl: number;
  total_volume: number;
  uptime: string;
}

export interface BotStatus {
  is_running: boolean;
  is_paused: boolean;
  uptime: string;
  config: {
    symbol: string;
    leverage: number;
    paper_trading: boolean;
    strategy: string;
  };
}

export interface MarketData {
  symbol: string;
  price: number;
  change_24h: number;
  volume: number;
  high_24h: number;
  low_24h: number;
  timestamp: number;
}

export interface PerformanceMetrics {
  volume_24h: number;
  gross_profit: number;
  trades_executed: number;
  eth_price: number;
  price_change_pct: number;
  volume_change_pct: number;
  profit_change_pct: number;
}

export interface AIAnalysis {
  sentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  trend: string;
  key_factors: string[];
  recommendation: string;
  risk_level: 'low' | 'medium' | 'high';
}
