# CryptoTrading Bot

An advanced cryptocurrency trading bot with technical analysis, sentiment analysis, and risk management capabilities.

## Features

- Real-time market data from Alpaca Markets API
- Technical analysis with multiple indicators (RSI, MACD, EMA, Bollinger Bands, etc.)
- Sentiment analysis from news and social media
- Risk management with position sizing and drawdown protection
- User-friendly Streamlit interface for monitoring and control

## Project Structure

```
CryptoTrading/
├── api/                # API integrations
├── technical/          # Technical indicators and signals
├── sentiment/          # Sentiment analysis modules
├── on_chain/           # On-chain data analysis
├── ml_models/          # Machine learning models
├── utils/              # Utility modules
├── config/             # Configuration files
├── trading_app.py      # Streamlit application
├── trading_strategy.py # Core trading logic
└── requirements.txt    # Project dependencies
```

## Getting Started

### Prerequisites

- Python 3.9+
- Alpaca Markets API key
- Finnhub API key
- OpenAI API key (optional)

### Installation

1. Clone the repository
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your API keys:
   ```
   ALPACA_API_KEY=your_alpaca_api_key
   ALPACA_API_SECRET=your_alpaca_api_secret
   FINNHUB_API_KEY=your_finnhub_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Application

To start the trading dashboard:

```
streamlit run trading_app.py
```

## Configuration

Modify the `config/trading_config.json` file to customize:

- Trading pairs
- Risk parameters
- Technical indicators
- Signal thresholds
- UI settings

## Disclaimer

This trading bot is for educational purposes only. Use at your own risk. Always thoroughly test any trading strategy in a paper trading environment before using real funds. 