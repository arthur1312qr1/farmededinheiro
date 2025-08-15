import type { Express } from "express";
import { createServer, type Server } from "http";
import { Server as SocketIOServer } from "socket.io";

export async function registerRoutes(app: Express): Promise<Server> {
  // Health check endpoint
  app.get("/api/health", (req, res) => {
    res.json({ 
      status: "healthy",
      timestamp: new Date().toISOString(),
      service: "farme-de-dinheiro"
    });
  });

  // Trading status endpoint
  app.get("/api/trading/status", (req, res) => {
    // This would get data from the Python trading bot
    res.json({
      is_running: true,
      daily_trades: 147,
      daily_pnl: 234.56,
      win_rate: 68.3,
      current_position: {
        symbol: "ETHUSDT",
        side: "long",
        size: 0.0234,
        entry_price: 3847.32,
        unrealized_pnl: 8.45
      }
    });
  });

  // Trading configuration endpoint
  app.get("/api/trading/config", (req, res) => {
    res.json({
      symbol: "ETHUSDT",
      leverage: 10,
      paper_trading: false,
      strategy: "Scalping",
      target_trades: 200
    });
  });

  // Market data endpoint
  app.get("/api/market/data", (req, res) => {
    const symbol = req.query.symbol || "ETHUSDT";
    
    // Mock market data - in production this would come from Bitget API
    res.json({
      symbol,
      price: 3847.32,
      change_24h: -2.3,
      volume: 45672.30,
      high_24h: 3920.15,
      low_24h: 3798.44,
      timestamp: Date.now()
    });
  });

  // Activity log endpoint
  app.get("/api/trading/logs", (req, res) => {
    // Mock activity logs
    res.json([
      {
        timestamp: new Date().toISOString(),
        action: "COMPRA EXECUTADA",
        message: "ETH/USDT - Quantidade: 0.0234 - Preço: $3,847.32",
        type: "success",
        details: "Stop Loss: $3,800.00 | Take Profit: $3,890.00"
      },
      {
        timestamp: new Date(Date.now() - 60000).toISOString(),
        action: "ANÁLISE TÉCNICA",
        message: "RSI: 32.5 (Oversold) | MACD: Bullish Crossover | Volume: Alto",
        type: "info",
        details: "✓ Condições de compra atendidas"
      }
    ]);
  });

  // Bot control endpoints
  app.post("/api/bot/toggle", (req, res) => {
    // This would interface with the Python trading bot
    res.json({ success: true, action: "toggled" });
  });

  app.post("/api/bot/emergency-stop", (req, res) => {
    res.json({ success: true, action: "emergency_stopped" });
  });

  app.post("/api/position/close", (req, res) => {
    res.json({ success: true, action: "position_closed" });
  });

  const httpServer = createServer(app);

  // Initialize Socket.IO for real-time updates
  const io = new SocketIOServer(httpServer, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"]
    }
  });

  io.on("connection", (socket) => {
    console.log("Client connected:", socket.id);

    // Join trading room
    socket.join("trading_room");

    // Send initial data
    socket.emit("connection_status", {
      status: "connected",
      timestamp: new Date().toISOString()
    });

    // Handle bot controls
    socket.on("toggle_bot", () => {
      socket.emit("bot_action_result", {
        success: true,
        action: "toggled"
      });
    });

    socket.on("emergency_stop", () => {
      socket.emit("bot_action_result", {
        success: true,
        action: "emergency_stopped"
      });
    });

    socket.on("close_position", () => {
      socket.emit("position_action_result", {
        success: true,
        action: "closed"
      });
    });

    socket.on("disconnect", () => {
      console.log("Client disconnected:", socket.id);
    });
  });

  // Simulate real-time updates
  setInterval(() => {
    io.to("trading_room").emit("price_update", {
      eth_price: 3847.32 + (Math.random() - 0.5) * 100,
      change_24h: -2.3 + (Math.random() - 0.5) * 2,
      volume: 45672.30 + Math.random() * 10000,
      timestamp: Date.now()
    });
  }, 5000);

  return httpServer;
}
