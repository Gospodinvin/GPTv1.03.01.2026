import numpy as np

def compute_rsi(closes, period=5):  # Укоротили для micro-moves
    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gain[-period:]) if len(gain) >= period else 0.5
    avg_loss = np.mean(loss[-period:]) if len(loss) >= period else 0.5
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow:
        return 0.0
    ema_fast = np.mean(closes[-fast:])
    ema_slow = np.mean(closes[-slow:])
    macd_line = ema_fast - ema_slow
    return macd_line

def compute_bollinger(closes, period=20, std_dev=2):
    if len(closes) < period:
        return "neutral"
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    upper, lower = sma + std_dev * std, sma - std_dev * std
    price = closes[-1]
    if price > upper:
        return "overbought"
    elif price < lower:
        return "oversold"
    return "neutral"

def compute_ema(closes, period=9):
    if len(closes) == 0:
        return 0.0
    alpha = 2 / (period + 1)
    ema = closes[0]
    for p in closes[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema

def compute_stochastic(closes, highs, lows, period=14):
    if len(closes) < period:
        return 50.0
    low_min = np.min(lows[-period:])
    high_max = np.max(highs[-period:])
    current_close = closes[-1]
    if high_max == low_min:
        return 50.0
    return 100 * (current_close - low_min) / (high_max - low_min)

def compute_adx_strength(highs, lows, closes, period=14):
    if len(highs) < period + 1:
        return 20.0
    
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1]))
    )
    atr = np.mean(tr[-period:])
    
    up_move = highs[1:] - highs[:-1]
    down_move = lows[:-1] - lows[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    return dx

def compute_atr(highs, lows, closes, period=14):
    if len(highs) < 2:
        return 0.0
    tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
    return np.mean(tr[-period:]) if len(tr) >= period else 0.0

def compute_cci(highs, lows, closes, period=20):
    if len(closes) < period:
        return 0.0
    tp = (highs + lows + closes) / 3
    sma_tp = np.mean(tp[-period:])
    mean_dev = np.mean(np.abs(tp[-period:] - sma_tp))
    return (tp[-1] - sma_tp) / (0.015 * mean_dev + 1e-9)

def compute_parabolic_sar(highs, lows, closes, af_step=0.015, af_max=0.2):
    if len(highs) < 2:
        return "neutral"
    sar = [lows[0]]
    ep = highs[0]
    af = 0.015
    trend = 1  # 1 up, -1 down
    for i in range(1, len(highs)):
        sar.append(sar[-1] + af * (ep - sar[-1]))
        if trend > 0:
            sar[-1] = min(sar[-1], lows[i-1])
            if highs[i] > ep:
                ep = highs[i]
                af = min(af + af_step, af_max)
        else:
            sar[-1] = max(sar[-1], highs[i-1])
            if lows[i] < ep:
                ep = lows[i]
                af = min(af + af_step, af_max)
        if trend > 0 and lows[i] < sar[-1]:
            trend = -1
            sar[-1] = ep
            ep = lows[i]
            af = 0.015
        elif trend < 0 and highs[i] > sar[-1]:
            trend = 1
            sar[-1] = ep
            ep = highs[i]
            af = 0.015
    if trend > 0:
        return "up"
    elif trend < 0:
        return "down"
    return "neutral"

def compute_roc(closes, period=5):  # Новый: Rate of Change
    if len(closes) < period + 1:
        return 0.0
    return (closes[-1] - closes[-period-1]) / closes[-period-1] * 100  # В %

def compute_obv(closes, volumes):  # Новый: On-Balance Volume
    obv = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    return obv[-1] / max(volumes[-10:]) if max(volumes[-10:]) > 0 else 0  # Нормализованный

def scalping_strategy(indicators, patterns, regime):
    adj = 0.0
    rsi = indicators['rsi']
    macd = indicators['macd']
    bb = indicators['bb']
    ema = indicators['ema']
    price = indicators['closes'][-1]
    stoch = indicators.get('stoch', 50)
    adx = indicators.get('adx', 20)
    atr = indicators.get('atr', 0.01)
    cci = indicators.get('cci', 0)
    psar = indicators.get('psar', 'neutral')
    roc = indicators.get('roc', 0)  # Новый
    obv = indicators.get('obv', 0)  # Новый

    # Buy signals
    if rsi < 30 and any(p in patterns for p in ["Hammer", "Pinbar", "Morning Star", "Bullish Harami"]):
        adj += 0.25
    if macd > 0 and "Engulfing" in patterns:
        adj += 0.20
    if bb == "oversold" and price > ema:
        adj += 0.15
    if stoch < 20 and adx > 25:
        adj += 0.12
    if cci < -100:
        adj += 0.10
    if psar == "up":
        adj += 0.08
    if atr > 0.005:
        adj += 0.05
    if roc > 0.1:  # Новый: positive momentum
        adj += 0.15
    if obv > 0.5:  # Новый: volume bias up
        adj += 0.10

    # Sell signals
    if rsi > 70 and any(p in patterns for p in ["Shooting Star", "Evening Star", "Bearish Harami"]):
        adj -= 0.25
    if macd < 0 and "Engulfing" in patterns:
        adj -= 0.20
    if bb == "overbought" and price < ema:
        adj -= 0.15
    if stoch > 80 and adx > 25:
        adj -= 0.12
    if cci > 100:
        adj -= 0.10
    if psar == "down":
        adj -= 0.08
    if atr > 0.005:
        if regime == "volatile":
            adj -= 0.05
    if roc < -0.1:  # Новый: negative momentum
        adj -= 0.15
    if obv < -0.5:  # Новый: volume bias down
        adj -= 0.10

    if regime == "volatile":
        adj *= 1.2
    elif regime == "flat":
        adj *= 0.8

    return np.clip(adj, -0.4, 0.4)