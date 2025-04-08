**ShambhoAlgoTrader

**Smart Market Analyzer** is a modular, microservices-based system for analyzing historical and real-time market data, predicting stock prices, generating trading signals, and backtesting custom trading strategies. Built with flexibility in mind, it supports any stock or instrument, integrates Shannon entropy for uncertainty quantification, and leverages the NinjaTrader API for real-time data. Users can customize models (CNN, FNN, LSTM), data splits, sampling rates, stop-loss/profit-taking strategies, and backtesting metrics like Sharpe and Sortino ratios.

## Features
- **Instrument Flexibility**: Analyze any stock or instrument (e.g., NQ, AAPL, BTC/USD) with historical data.
- **Prediction Models**: CNN for pattern recognition, FNN as a baseline, and LSTM for time-series forecasting.
- **Timeframes**: Predict and generate signals for 1-minute, 5-minute, 15-minute, and 1-hour intervals.
- **Shannon Entropy**: Quantify prediction uncertainty and confidence using self-information and entropy.
- **Real-Time Integration**: Fetch live data via NinjaTrader API for signal generation.
- **Custom Data Handling**: Define train/test/validation splits and sampling rates (e.g., 1-min to 5-min).
- **Trading Signals**: Generate Buy/Sell/Hold signals with Decision Trees, enhanced by entropy metrics.
- **Backtesting**: Test strategies with metrics like Sharpe Ratio, Sortino Ratio, and max drawdown.
- **Custom Strategies**: Configure stop-loss (e.g., ATR-based) and profit-taking (e.g., Fibonacci levels).
- **Modular Architecture**: Microservices design for scalability and easy model/strategy swapping.

## Prerequisites
- Python 3.8+
- Docker (for containerization)
- NinjaTrader API access (for real-time data)
- Libraries: TensorFlow/PyTorch, Scikit-learn, FastAPI, Kafka, Pandas, NumPy

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/smart-market-analyzer.git
   cd smart-market-analyzer
Install Dependencies:
bash
pip install -r requirements.txt
Set Up Environment Variables:
Create a .env file in the root directory:
NINJATRADER_API_KEY=your_api_key
KAFKA_BROKER=localhost:9092
Build and Run with Docker:
bash
docker-compose up --build
Usage
1. Preprocess Data
Load historical data for your chosen instrument and configure splits/sampling:
bash
curl -X POST "http://localhost:8000/data" -H "Content-Type: application/json" -d '{
  "instrument": "AAPL",
  "data_path": "path/to/aapl_Pillars of Creation.csv",
  "split": {"train": 0.7, "val": 0.15, "test": 0.15},
  "sample_rate": "1-minute"
}'
2. Train a Model
Train a model (e.g., LSTM) for a specific timeframe:
bash
curl -X POST "http://localhost:8000/train" -d '{
  "instrument": "AAPL",
  "model_type": "LSTM",
  "timeframe": "1-minute",
  "window_size": 60,
  "bins": 10
}'
3. Get Predictions
Predict prices for multiple timeframes using live NinjaTrader data:
bash
curl -X POST "http://localhost:8000/predict" -d '{
  "model_id": "lstm_1min",
  "timeframe": "1-minute",
  "data": "ninjatrader_live_data"
}'
4. Generate Signals
Get trading signals with entropy-based confidence:
bash
curl -X POST "http://localhost:8000/signal" -d '{
  "prediction": {"price": 150.5, "entropy": 0.8},
  "current_candle": "ninjatrader_candle"
}'
5. Backtest a Strategy
Run a backtest with custom stop-loss and profit-taking:
bash
curl -X POST "http://localhost:8000/backtest" -d '{
  "strategy": "Custom",
  "stop_loss": {"type": "ATR", "multiplier": 2, "period": 14},
  "profit_take": {"type": "Fibonacci", "level": 0.618},
  "data_id": "aapl_data"
}'
Example Output
Instrument: AAPL
Timeframe: 1-minute
  Prediction: 150.5, Entropy: 0.8 bits, Confidence: 0.85
  Signal: Buy, Confidence: 0.88
Backtest Results:
  Sharpe Ratio: 1.6
  Sortino Ratio: 2.1
  Total Profit: $4500
Architecture
Data Service: Handles historical and live data preprocessing.
Model Service: Trains CNN, FNN, LSTM models with entropy-based confidence.
Signal Service: Generates trading signals using Decision Trees.
Backtest Service: Simulates trades with custom strategies.
Evaluation Service: Compares performance across models and strategies.
API Gateway: Exposes RESTful endpoints.
Customization
Instruments: Add any stock/instrument by providing historical data in CSV format (e.g., Time, Open, High, Low, Close, Volume, EMAs).
Models: Swap CNN, FNN, LSTM via config (e.g., "model_type": "CNN").
Strategies: Define stop-loss (e.g., "type": "ATR") and profit-taking (e.g., "type": "Fibonacci") in JSON.
Metrics: Extend backtesting with additional metrics in backtest_service.py.
Configuration
Edit config.json to adjust settings:
json
{
  "instrument": "NQ",
  "split": {"train": 0.8, "val": 0.1, "test": 0.1},
  "sample_rate": "1-minute",
  "strategy": {
    "stop_loss": {"type": "Trailing", "distance": 0.5},
    "profit_take": {"type": "Fixed", "amount": 1.0}
  }
}
Contributing
Pull requests are welcome! Please open an issue to discuss changes or enhancements.
License
MIT License - see LICENSE for details.
Contact
For questions, reach out to your.email@example.com (mailto:your.email@example.com).

---

### Fixes Applied
1. **Headings**: Added proper Markdown headings (`#`, `##`) for hierarchy.
2. **Code Blocks**: Used triple backticks (```) with language specifiers (e.g., `bash`, `json`) for syntax highlighting.
3. **Lists**: Formatted features, prerequisites, and architecture as bullet points with proper indentation.
4. **Inline Code**: Used single backticks for file names (e.g., `.env`) and variables.
5. **Spacing**: Added line breaks for readability between sections.
6. **Consistency**: Ensured uniform formatting for commands and examples.

### How to Use
- Copy the entire content above into a `README.md` file.
- Save it in your project’s root directory.
- Replace placeholders like `yourusername` and `your.email@example.com` with your actual details.
- Preview it in a Markdown viewer (e.g., GitHub) to confirm formatting.

Let me know if you need further adjustments!
# Smart Market Analyzer

A modular, microservices-based system for analyzing market data, predicting stock prices, generating trading signals, and backtesting custom trading strategies.

## Features

- **Instrument Flexibility**: Analyze any stock or instrument with historical data
- **Prediction Models**: Transformer with self-attention for price prediction, CNN for pattern recognition
- **Timeframes**: Support for 1-minute, 5-minute, 15-minute, and 1-hour intervals
- **Shannon Entropy**: Quantify prediction uncertainty and confidence
- **Custom Data Handling**: User-defined train/test/validation splits and sampling rates
- **Trading Signals**: Generate Buy/Sell/Hold signals with Decision Trees
- **Backtesting**: Test strategies with metrics like Sharpe Ratio, Sortino Ratio, etc.
- **Custom Strategies**: Configure stop-loss (e.g., ATR-based) and profit-taking (e.g., Fibonacci levels)
- **Modular Architecture**: Microservices design for scalability and model swapping

## Project Structure

```
smart-market-analyzer/
├── src/
│   ├── api/              # API endpoints
│   ├── core/             # Core services
│   ├── infrastructure/   # Infrastructure services
│   └── trading/          # Trading services
├── config/               # Configuration files
├── deployment/           # Deployment files
├── data/                 # Data storage
├── logs/                 # Log files
├── requirements.txt      # Python dependencies
└── worker.py             # Background worker
```

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerization)
- Redis (for caching and message broker)
- MongoDB (for persistent storage)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/smart-market-analyzer.git
   cd smart-market-analyzer
   ```

2. Set up the project structure:
   ```bash
   chmod +x setup_project.sh
   ./setup_project.sh
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Fix imports:
   ```bash
   python fix_imports.py
   ```

## Configuration

Edit `config/config.json` to configure the system according to your needs:

```json
{
  "data": {
    "csv_path": "data/nq_1min.csv",
    "instrument": "NQ",
    "split": {
      "train": 0.7,
      "val": 0.15,
      "test": 0.15
    },
    "sample_rate": "1-minute"
  },
  "model": {
    "price_model": "Transformer",
    "pattern_model": "CNN",
    "window_size": 60,
    "bins": 10,
    "num_heads": 4,
    "d_model": 64,
    "entropy_threshold": 0.75
  },
  "backtest": {
    "strategy": {
      "stop_loss": {
        "type": "ATR",
        "multiplier": 2,
        "period": 14
      },
      "profit_take": {
        "type": "Fibonacci",
        "level": 0.618
      }
    },
    "slippage": 0.001,
    "max_position": 0.02,
    "max_daily_loss": 0.05
  }
}
```

## Running the Application

### Development Mode

1. Start the API server:
   ```bash
   chmod +x start.sh
   ./start.sh api
   ```

2. Start the worker service:
   ```bash
   ./start.sh worker
   ```

3. Or start both:
   ```bash
   ./start.sh both
   ```

### Production Mode (Docker)

```bash
./start.sh docker
```

## Usage

1. Access the API documentation:
   ```
   http://localhost:8000/docs
   ```

2. Process historical data:
   ```bash
   curl -X POST "http://localhost:8000/data" -H "Content-Type: application/json" -d '{"csv_path": "data/sample.csv"}'
   ```

3. Check training status:
   ```bash
   curl "http://localhost:8000/training-status"
   ```

4. Generate predictions:
   ```bash
   curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"data": [{"Open": 4323.5, "High": 4323.75, "Low": 4323.25, "Close": 4323.5, "Volume": 19, "EMA9": 4323.66, "EMA21": 4322.29, "EMA220": 4318.87}, ...], "window_size": 60}'
   ```

5. Generate trading signals:
   ```bash
   curl -X POST "http://localhost:8000/signal" -H "Content-Type: application/json" -d '{"prediction": {"price_prediction": [4324.0], "pattern": [2], "pattern_labels": ["Positive"], "entropy": [0.6], "confidence": [0.9]}, "latest_candle": {"Close": 4323.5, "Volume": 19, "EMA9": 4323.66, "EMA21": 4322.29, "EMA220": 4318.87}}'
   ```

6. Run backtesting:
   ```bash
   curl -X POST "http://localhost:8000/backtest"
   ```

## Monitoring

Access Grafana dashboards for monitoring:
```
http://localhost:3000
Username: admin
Password: smartmarket
```

## Troubleshooting

1. Check logs:
   ```bash
   tail -f smart-market-analyzer/logs/worker.log
   ```

2. Check Docker container status:
   ```bash
   docker-compose -f smart-market-analyzer/deployment/docker-compose.yml ps
   ```

3. Reset the system:
   ```bash
   docker-compose -f smart-market-analyzer/deployment/docker-compose.yml down -v
   ```

## License

[MIT License](LICENSE)

## Contributing

Pull requests are welcome! Please open an issue first to discuss changes.