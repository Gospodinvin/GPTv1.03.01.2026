# features_lstm.py — специально для LSTM: возвращает 3D sequences (samples, seq_len, features)
import numpy as np
from indicators import (
    compute_rsi, compute_macd, compute_bollinger, compute_ema,
    compute_stochastic, compute_adx_strength, compute_atr,
    compute_cci, compute_parabolic_sar, compute_roc, compute_obv
)

SEQ_LEN = 20  # Длина последовательности

def build_lstm_features(candles):
    if len(candles) < SEQ_LEN:
        return np.array([])

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])
    opens = np.array([c["open"] for c in candles])

    # Вычисляем индикаторы для всей истории
    rsi_vals = []
    macd_vals = []
    bb_vals = []
    ema_vals = []
    stoch_vals = []
    adx_vals = []
    atr_vals = []
    cci_vals = []
    psar_vals = []
    roc_vals = []
    obv_vals = []

    for i in range(len(candles)):
        sub_closes = closes[:i+1]
        sub_highs = highs[:i+1]
        sub_lows = lows[:i+1]
        sub_volumes = volumes[:i+1]

        rsi_vals.append(compute_rsi(sub_closes))
        macd_vals.append(compute_macd(sub_closes))
        bb_vals.append(1 if compute_bollinger(sub_closes) == "overbought" else -1 if compute_bollinger(sub_closes) == "oversold" else 0)
        ema_vals.append(compute_ema(sub_closes))
        stoch_vals.append(compute_stochastic(sub_closes, sub_highs, sub_lows))
        adx_vals.append(compute_adx_strength(sub_highs, sub_lows, sub_closes))
        atr_vals.append(compute_atr(sub_highs, sub_lows, sub_closes))
        cci_vals.append(compute_cci(sub_highs, sub_lows, sub_closes))
        psar_vals.append(1 if compute_parabolic_sar(sub_highs, sub_lows, sub_closes) == "up" else -1 if "down" else 0)
        roc_vals.append(compute_roc(sub_closes))
        obv_vals.append(compute_obv(sub_closes, sub_volumes))

    rsi_vals = np.array(rsi_vals)
    macd_vals = np.array(macd_vals)
    bb_vals = np.array(bb_vals)
    ema_vals = np.array(ema_vals)
    stoch_vals = np.array(stoch_vals)
    adx_vals = np.array(adx_vals)
    atr_vals = np.array(atr_vals)
    cci_vals = np.array(cci_vals)
    psar_vals = np.array(psar_vals)
    roc_vals = np.array(roc_vals)
    obv_vals = np.array(obv_vals)

    X = []
    for i in range(SEQ_LEN - 1, len(candles)):
        seq = []
        for j in range(i - SEQ_LEN + 1, i + 1):
            c = candles[j]
            body = abs(c["close"] - c["open"])
            direction = 1 if c["close"] > c["open"] else -1 if c["close"] < c["open"] else 0
            range_ = c["high"] - c["low"]

            relative_to_ema = (c["close"] - ema_vals[j]) / ema_vals[j] if ema_vals[j] > 0 else 0

            feat = [
                body / c["close"] if c["close"] > 0 else 0,  # нормализуем
                direction,
                range_ / c["close"] if c["close"] > 0 else 0,
                rsi_vals[j] / 100,
                macd_vals[j],
                bb_vals[j],
                relative_to_ema,
                stoch_vals[j] / 100,
                adx_vals[j] / 100,
                atr_vals[j] / c["close"] if c["close"] > 0 else 0,
                cci_vals[j] / 200,
                psar_vals[j],
                roc_vals[j] / 10,
                obv_vals[j] / 1000 if abs(obv_vals[j]) > 0 else 0,
            ]
            seq.append(feat)
        X.append(seq)

    return np.array(X)