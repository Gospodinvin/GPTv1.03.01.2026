import numpy as np

def market_regime(candles):
    closes = np.array([c["close"] for c in candles])
    returns = np.diff(closes) / closes[:-1]

    vol = np.std(returns)
    slope = np.polyfit(range(len(closes)), closes, 1)[0]

    if vol < 0.001:
        return "flat"
    if abs(slope) > vol * 2:
        return "trend"
    return "volatile"


def trend_signal(candles):
    closes = np.array([c["close"] for c in candles])
    ma_fast = closes[-5:].mean()
    ma_slow = closes[-20:].mean()

    if ma_fast > ma_slow:
        return 0.65
    elif ma_fast < ma_slow:
        return 0.35
    return 0.5
