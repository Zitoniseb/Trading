import importlib.util
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Load module from file with space in name
module_path = Path(__file__).resolve().parents[1] / 'backtest 1.py'
spec = importlib.util.spec_from_file_location("backtest", module_path)
backtest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backtest)

calculate_rsi = backtest.calculate_rsi
backtest_ticker = backtest.backtest_ticker


def test_calculate_rsi_monotonic_increase():
    prices = pd.Series(range(1, 21))
    rsi = calculate_rsi(prices)
    assert rsi.iloc[-1] == 100


def test_backtest_ticker_tp_hit():
    index = pd.date_range('2021-01-01', periods=6)
    data = pd.DataFrame({
        'Open': [100, 101, 109, 108, 107, 105],
        'High': [100, 104, 110, 108, 107, 105],
        'Low': [99, 100, 108, 107, 106, 104],
        'Close': [100, 103, 109, 108, 107, 105],
        'Volume': [1000] * 6,
    }, index=index)

    with patch('yfinance.download', return_value=data):
        result = backtest_ticker('TEST', '2021-01-01', 5)

    assert result['Final Profit (€)'] == 8
    assert result['Exit Reason'] == 'TP Hit'
    assert result['Hold & Wait Profit (€)'] == 5
