#!/bin/bash
# This script sets up the project structure for Smart Market Analyzer

# Create root directory
mkdir -p smart-market-analyzer

# Create directory structure
mkdir -p smart-market-analyzer/src/api
mkdir -p smart-market-analyzer/src/core
mkdir -p smart-market-analyzer/src/infrastructure
mkdir -p smart-market-analyzer/src/trading
mkdir -p smart-market-analyzer/config
mkdir -p smart-market-analyzer/deployment
mkdir -p smart-market-analyzer/data/market
mkdir -p smart-market-analyzer/data/models
mkdir -p smart-market-analyzer/data/backtest
mkdir -p smart-market-analyzer/logs

# Copy files to their respective directories
# API
cp src/api/main.py smart-market-analyzer/src/api/

# Core services
cp src/core/data_service.py smart-market-analyzer/src/core/
cp src/core/model_service.py smart-market-analyzer/src/core/
cp src/core/signal_service.py smart-market-analyzer/src/core/
cp src/core/backtest_service.py smart-market-analyzer/src/core/
cp src/core/strategy_manager.py smart-market-analyzer/src/core/

# Infrastructure
cp src/infrastructure/ninja_trader_client.py smart-market-analyzer/src/infrastructure/
cp src/infrastructure/data_store.py smart-market-analyzer/src/infrastructure/
cp src/infrastructure/monitoring.py smart-market-analyzer/src/infrastructure/

# Trading
cp src/trading/multi_market_manager.py smart-market-analyzer/src/trading/
cp src/trading/portfolio_manager.py smart-market-analyzer/src/trading/
cp src/trading/trade_executor.py smart-market-analyzer/src/trading/

# Configuration
cp config/config.json smart-market-analyzer/config/

# Deployment
cp deployment/Dockerfile smart-market-analyzer/deployment/
cp deployment/docker-compose.yml smart-market-analyzer/deployment/
cp deployment/prometheus.yml smart-market-analyzer/deployment/

# Root files
cp requirements.txt smart-market-analyzer/
cp src/worker.py smart-market-analyzer/

# Create __init__.py files for Python packages
touch smart-market-analyzer/src/__init__.py
touch smart-market-analyzer/src/api/__init__.py
touch smart-market-analyzer/src/core/__init__.py
touch smart-market-analyzer/src/infrastructure/__init__.py
touch smart-market-analyzer/src/trading/__init__.py

echo "Project structure set up successfully!"
