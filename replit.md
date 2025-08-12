# Trading Bot for ETH/USDT Futures

## Overview

This is an automated cryptocurrency trading bot designed for ETH/USDT futures trading with fixed 10x leverage. The bot integrates with the Bitget cryptocurrency exchange API for executing real trades automatically. Features web interface for monitoring and control.

## User Preferences

Preferred communication style: Simple, everyday language.
Trading mode: Real trading (não simulação) - opera com saldo real da conta Bitget
Bot requirements: Execução automática de transações ETH/USDT com alavancagem fixa 10x
Interface: Web dashboard para monitoramento

## System Architecture

### Core Trading Engine
- **Fixed Leverage System**: Uses fixed 10x leverage for all ETH/USDT futures trades
- **Real Trading**: Operates with actual account balance on Bitget exchange
- **Auto Execution**: Automatically detects opportunities and executes trades
- **Web Interface**: Flask-based dashboard for monitoring and control

### Risk Management Framework
- **Position Tracking**: Monitors positions and performance in real-time
- **Conservative Settings**: Optimized for real money trading
- **Error Handling**: Comprehensive error management and recovery
- **Emergency Controls**: Stop functionality and risk limits

### External Dependencies
- **Bitget API**: Cryptocurrency exchange for real trading
- **Flask**: Web interface framework
- **Threading**: Concurrent execution support
