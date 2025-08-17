import  { useState, useEffect } from 'react';
import { Activity, ArrowUp, ArrowDown, Zap, Target, Clock, Star, AlertTriangle, CheckCircle, TrendingUp, Database, Settings } from 'lucide-react';

function App() {
  const [botStatus, setBotStatus] = useState({
    is_running: false,
    trades_today: 0,
    win_rate: 0,
    total_profit: 0,
    current_position: null,
    deficit: 0,
    urgency_level: 'NORMAL'
  });

  const [stats, setStats] = useState({
    profitable_trades: 0,
    losing_trades: 0,
    consecutive_wins: 0,
    last_trade_seconds_ago: 0,
    boost_mode_active: false
  });

  const [isStarting, setIsStarting] = useState(false);

  useEffect(() => {
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    try {
      // Simulated status for demo
      const mockStatus = {
        bot_status: {
          is_running: botStatus.is_running,
          symbol: 'ETH/USDT',
          leverage: 10,
          paper_trading: true,
          aggressive_mode: true
        },
        daily_progress: {
          trades_today: botStatus.trades_today + (botStatus.is_running ? Math.floor(Math.random() * 2) : 0),
          min_target: 240,
          target: 280,
          progress_percent: Math.min(100, ((botStatus.trades_today + 1) / 240) * 100),
          expected_by_now: 45,
          deficit: Math.max(0, 45 - botStatus.trades_today),
          urgency_level: botStatus.trades_today < 30 ? 'HIGH' : 'NORMAL',
          trades_per_hour_current: 15.2,
          trades_per_hour_needed: 15
        },
        performance: {
          profitable_trades: Math.floor(botStatus.trades_today * 0.95),
          losing_trades: Math.floor(botStatus.trades_today * 0.05),
          win_rate: 95.2,
          target_win_rate: 95.0,
          total_profit: botStatus.total_profit + (botStatus.is_running ? Math.random() * 0.5 : 0),
          consecutive_wins: 12
        },
        current_position: botStatus.is_running ? {
          active: Math.random() > 0.7,
          side: Math.random() > 0.5 ? 'long' : 'short',
          size: 0.5,
          entry_price: 2234.56,
          duration_seconds: 45,
          max_duration: 180,
          unrealized_pnl: (Math.random() - 0.5) * 0.02
        } : { active: false },
        timing_control: {
          last_trade_seconds_ago: Math.floor(Math.random() * 210),
          max_gap_allowed: 210,
          next_trade_urgency: Math.random() > 0.7 ? 'HIGH' : 'NORMAL',
          boost_mode_active: Math.random() > 0.8
        }
      };

      setBotStatus(prev => ({
        ...prev,
        is_running: mockStatus.bot_status.is_running,
        trades_today: mockStatus.daily_progress.trades_today,
        win_rate: mockStatus.performance.win_rate,
        total_profit: mockStatus.performance.total_profit,
        current_position: mockStatus.current_position,
        deficit: mockStatus.daily_progress.deficit,
        urgency_level: mockStatus.daily_progress.urgency_level
      }));

      setStats({
        profitable_trades: mockStatus.performance.profitable_trades,
        losing_trades: mockStatus.performance.losing_trades,
        consecutive_wins: mockStatus.performance.consecutive_wins,
        last_trade_seconds_ago: mockStatus.timing_control.last_trade_seconds_ago,
        boost_mode_active: mockStatus.timing_control.boost_mode_active
      });
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  const startBot = async () => {
    setIsStarting(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      setBotStatus(prev => ({ ...prev, is_running: true }));
    } catch (error) {
      console.error('Error starting bot:', error);
    }
    setIsStarting(false);
  };

  const stopBot = async () => {
    try {
      setBotStatus(prev => ({ ...prev, is_running: false }));
    } catch (error) {
      console.error('Error stopping bot:', error);
    }
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'CRITICAL': return 'text-red-600 bg-red-50 border-red-200';
      case 'HIGH': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'MEDIUM': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-green-600 bg-green-50 border-green-200';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur border-b border-slate-700">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-emerald-500 rounded-lg">
                <Activity className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">Trading Bot 240+</h1>
                <p className="text-slate-400 text-sm">AI-Powered High-Frequency Trading</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${botStatus.is_running ? 'bg-emerald-500 text-white' : 'bg-slate-600 text-slate-300'}`}>
                {botStatus.is_running ? 'ACTIVE' : 'STOPPED'}
              </div>
              
              {botStatus.is_running ? (
                <button
                  onClick={stopBot}
                  className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-all"
                >
                  Stop Bot
                </button>
              ) : (
                <button
                  onClick={startBot}
                  disabled={isStarting}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-600 text-white rounded-lg font-medium transition-all flex items-center space-x-2"
                >
                  {isStarting && <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />}
                  <span>{isStarting ? 'Starting...' : 'Start Bot'}</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {/* Daily Progress Card */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-bold text-white flex items-center space-x-2">
              <Target className="w-5 h-5 text-emerald-500" />
              <span>Daily Progress - 240+ Trades Guarantee</span>
            </h2>
            <div className={`px-3 py-1 rounded-full text-xs font-bold border ${getUrgencyColor(botStatus.urgency_level)}`}>
              {botStatus.urgency_level} URGENCY
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-white mb-1">{botStatus.trades_today}</div>
              <div className="text-slate-400 text-sm">Trades Today</div>
              <div className="text-emerald-400 text-xs mt-1">Target: 240+</div>
            </div>

            <div className="text-center">
              <div className="text-3xl font-bold text-white mb-1">{((botStatus.trades_today / 240) * 100).toFixed(1)}%</div>
              <div className="text-slate-400 text-sm">Progress</div>
              <div className="w-full bg-slate-700 rounded-full h-2 mt-2">
                <div 
                  className="bg-emerald-500 h-2 rounded-full transition-all"
                  style={{ width: `${Math.min(100, (botStatus.trades_today / 240) * 100)}%` }}
                />
              </div>
            </div>

            <div className="text-center">
              <div className="text-3xl font-bold text-white mb-1">{botStatus.deficit}</div>
              <div className="text-slate-400 text-sm">Trade Deficit</div>
              <div className={`text-xs mt-1 ${botStatus.deficit > 20 ? 'text-red-400' : botStatus.deficit > 10 ? 'text-orange-400' : 'text-emerald-400'}`}>
                {botStatus.deficit > 0 ? 'Behind Schedule' : 'On Track'}
              </div>
            </div>

            <div className="text-center">
              <div className="text-3xl font-bold text-white mb-1">15.2</div>
              <div className="text-slate-400 text-sm">Trades/Hour</div>
              <div className="text-slate-400 text-xs mt-1">Need: 15/hour</div>
            </div>
          </div>

          {stats.boost_mode_active && (
            <div className="mt-4 p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-orange-400" />
                <span className="text-orange-400 font-medium text-sm">BOOST MODE ACTIVE - Accelerating to meet daily target</span>
              </div>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Stats */}
          <div className="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
              <Star className="w-5 h-5 text-yellow-500" />
              <span>Performance Analytics</span>
            </h3>

            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Win Rate</span>
                <div className="flex items-center space-x-2">
                  <span className="text-emerald-400 font-bold">{botStatus.win_rate.toFixed(1)}%</span>
                  <div className="w-20 bg-slate-700 rounded-full h-2">
                    <div className="bg-emerald-500 h-2 rounded-full" style={{ width: `${botStatus.win_rate}%` }} />
                  </div>
                </div>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-slate-400">Profitable Trades</span>
                <span className="text-white font-medium">{stats.profitable_trades}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-slate-400">Losing Trades</span>
                <span className="text-white font-medium">{stats.losing_trades}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-slate-400">Consecutive Wins</span>
                <span className="text-emerald-400 font-bold">{stats.consecutive_wins}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-slate-400">Total Profit</span>
                <span className={`font-bold ${botStatus.total_profit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {botStatus.total_profit >= 0 ? '+' : ''}{botStatus.total_profit.toFixed(2)}%
                </span>
              </div>
            </div>
          </div>

          {/* Current Position */}
          <div className="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
            <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
              <TrendingUp className="w-5 h-5 text-blue-500" />
              <span>Current Position</span>
            </h3>

            {botStatus.current_position?.active ? (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Position Side</span>
                  <div className="flex items-center space-x-2">
                    {botStatus.current_position.side === 'long' ? (
                      <ArrowUp className="w-4 h-4 text-emerald-400" />
                    ) : (
                      <ArrowDown className="w-4 h-4 text-red-400" />
                    )}
                    <span className={`font-bold uppercase ${botStatus.current_position.side === 'long' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {botStatus.current_position.side}
                    </span>
                  </div>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Entry Price</span>
                  <span className="text-white font-medium">${botStatus.current_position.entry_price?.toFixed(2)}</span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Duration</span>
                  <span className="text-white font-medium">{botStatus.current_position.duration_seconds}s / 180s</span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-slate-400">Unrealized P&L</span>
                  <span className={`font-bold ${botStatus.current_position.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {botStatus.current_position.unrealized_pnl >= 0 ? '+' : ''}{(botStatus.current_position.unrealized_pnl * 100).toFixed(2)}%
                  </span>
                </div>

                <div className="w-full bg-slate-700 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${(botStatus.current_position.duration_seconds / 180) * 100}%` }}
                  />
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <Database className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                <p className="text-slate-400">No active position</p>
                <p className="text-slate-500 text-sm mt-1">Scanning for opportunities...</p>
              </div>
            )}
          </div>
        </div>

        {/* Timing Control */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
            <Clock className="w-5 h-5 text-purple-500" />
            <span>Timing Control & Trade Frequency</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-2xl font-bold text-white mb-1">{stats.last_trade_seconds_ago}s</div>
              <div className="text-slate-400 text-sm">Since Last Trade</div>
              <div className="text-purple-400 text-xs mt-1">Max: 210s (3.5min)</div>
            </div>

            <div className="text-center">
              <div className="text-2xl font-bold text-white mb-1">
                {stats.last_trade_seconds_ago > 180 ? 'HIGH' : 'NORMAL'}
              </div>
              <div className="text-slate-400 text-sm">Trade Urgency</div>
              <div className={`text-xs mt-1 ${stats.last_trade_seconds_ago > 180 ? 'text-red-400' : 'text-emerald-400'}`}>
                {stats.last_trade_seconds_ago > 180 ? 'Overdue' : 'On Schedule'}
              </div>
            </div>

            <div className="text-center">
              <div className="text-2xl font-bold text-white mb-1">274</div>
              <div className="text-slate-400 text-sm">Max Possible/Day</div>
              <div className="text-slate-400 text-xs mt-1">@ 3.5min intervals</div>
            </div>
          </div>

          <div className="mt-4 w-full bg-slate-700 rounded-full h-3">
            <div 
              className={`h-3 rounded-full transition-all ${stats.last_trade_seconds_ago > 180 ? 'bg-red-500' : stats.last_trade_seconds_ago > 120 ? 'bg-orange-500' : 'bg-emerald-500'}`}
              style={{ width: `${Math.min(100, (stats.last_trade_seconds_ago / 210) * 100)}%` }}
            />
          </div>
        </div>

        {/* Key Features */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl border border-slate-700 p-6">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
            <Settings className="w-5 h-5 text-slate-400" />
            <span>Bot Configuration & Features</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <div>
                <div className="text-white font-medium">240+ Trades Guarantee</div>
                <div className="text-slate-400 text-sm">Adaptive frequency control</div>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <div>
                <div className="text-white font-medium">95% Success Rate</div>
                <div className="text-slate-400 text-sm">AI prediction algorithms</div>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <div>
                <div className="text-white font-medium">10x Leverage</div>
                <div className="text-slate-400 text-sm">Maximum capital efficiency</div>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <div>
                <div className="text-white font-medium">3.5min Max Gap</div>
                <div className="text-slate-400 text-sm">High-frequency trading</div>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <div>
                <div className="text-white font-medium">Boost Mode</div>
                <div className="text-slate-400 text-sm">Auto-acceleration</div>
              </div>
            </div>

            <div className="flex items-center space-x-3 p-3 bg-slate-700/30 rounded-lg">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <div>
                <div className="text-white font-medium">Risk Management</div>
                <div className="text-slate-400 text-sm">1% TP / 2% SL</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
 
