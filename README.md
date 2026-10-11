# Trading Signal ML

Data-science oriented market signal generation project.

Adapted from trading-agent concepts into a pure ML/feature-engineering focus:

- Feature engineering from OHLCV data
- Market regime classification
- Ensemble of simple strategies (momentum, mean-reversion, breakout)
- Backtest-style evaluation
- Risk filters (spread, volatility, drawdown proxies)

## Quick Start

```bash
pip install -r requirements.txt
python src/generate_signals.py
```

No live trading. Educational / research only.

Author: Muhammad AbuHuraira
