from app import db
from datetime import datetime
from sqlalchemy import JSON

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(20), nullable=False, default='ETHUSDT')
    side = db.Column(db.String(10), nullable=False)  # 'long' or 'short'
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float, nullable=True)
    quantity = db.Column(db.Float, nullable=False)
    leverage = db.Column(db.Float, nullable=False, default=10.0)
    stop_loss = db.Column(db.Float, nullable=True)
    take_profit = db.Column(db.Float, nullable=True)
    pnl = db.Column(db.Float, nullable=True)
    pnl_percentage = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(20), nullable=False, default='open')  # 'open', 'closed', 'cancelled'
    entry_time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime, nullable=True)
    signals = db.Column(JSON, nullable=True)  # Store signal data as JSON
    confidence_score = db.Column(db.Float, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'status': self.status,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'confidence_score': self.confidence_score
        }

class BotStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    is_active = db.Column(db.Boolean, nullable=False, default=False)
    balance = db.Column(db.Float, nullable=False, default=0.0)
    daily_pnl = db.Column(db.Float, nullable=False, default=0.0)
    total_trades = db.Column(db.Integer, nullable=False, default=0)
    winning_trades = db.Column(db.Integer, nullable=False, default=0)
    losing_trades = db.Column(db.Integer, nullable=False, default=0)
    consecutive_losses = db.Column(db.Integer, nullable=False, default=0)
    last_updated = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    current_price = db.Column(db.Float, nullable=True)
    
    def to_dict(self):
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        return {
            'is_active': self.is_active,
            'balance': self.balance,
            'daily_pnl': self.daily_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'consecutive_losses': self.consecutive_losses,
            'win_rate': round(win_rate, 2),
            'current_price': self.current_price,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class Signal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    timeframe = db.Column(db.String(10), nullable=False)
    signal_type = db.Column(db.String(10), nullable=False)  # 'buy' or 'sell'
    confidence = db.Column(db.Float, nullable=False)
    indicators = db.Column(JSON, nullable=False)
    price = db.Column(db.Float, nullable=False)
    executed = db.Column(db.Boolean, nullable=False, default=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'signal_type': self.signal_type,
            'confidence': self.confidence,
            'indicators': self.indicators,
            'price': self.price,
            'executed': self.executed
        }
