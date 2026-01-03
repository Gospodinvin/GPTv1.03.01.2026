# features.py — версия для RandomForest (2D фичи)
import numpy as np
from indicators import (
    compute_rsi, compute_macd, compute_bollinger, compute_ema,
    compute_stochastic, compute_adx_strength, compute_atr,
    compute_cci, compute_parabolic_sar, compute_roc, compute_obv
)

def build_features(candles, tf):
    if len(candles) < 2:
        return np.array([])

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])

    indicators = {
        "rsi": compute_rsi(closes),
        "macd": compute_macd(closes),
        "bb": 1 if compute_bollinger(closes) == "overbought" else -1 if compute_bollinger(closes) == "oversold" else 0,
        "ema": compute_ema(closes),
        "stoch": compute_stochastic(closes, highs, lows),
        "adx": compute_adx_strength(highs, lows, closes),
        "atr": compute_atr(highs, lows, closes),
        "cci": compute_cci(highs, lows, closes),
        "psar": 1 if compute_parabolic_sar(highs, lows, closes) == "up" else -1 if "down" else 0,
        "roc": compute_roc(closes),
        "obv": compute_obv(closes, volumes),
    }

    X = []
    scale = {"1": 1.0, "2": 1.2, "5": 1.5, "10": 2.0}.get(tf, 1.0)
    
    for i in range(1, len(candles)):
        c = candles[i]
        body = abs(c["close"] - c["open"]) * scale
        direction = np.sign(c["close"] - c["open"])
        vol = (c["high"] - c["low"]) * scale
        
        feat = [
            body, direction, vol,
            indicators["rsi"]/100,
            indicators["macd"],
            indicators["bb"],
            (c["close"] - indicators["ema"]) / indicators["ema"] if indicators["ema"] > 0 else 0,
            indicators["stoch"]/100,
            indicators["adx"]/100,
            indicators["atr"],
            indicators["cci"]/200,
            indicators["psar"],
            indicators["roc"]/10,
            indicators["obv"]/1000
        ]
        X.append(feat)
    
    return np.array(X) if X else np.array([])