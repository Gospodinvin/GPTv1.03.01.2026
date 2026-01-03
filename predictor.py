import httpx
import os
import logging
import numpy as np

from features import build_features
from patterns import detect_patterns
from trend import trend_signal, market_regime
from confidence import confidence_from_probs
from model_registry import get_model
from data_provider import get_candles
from cv_extractor import extract_candles
from indicators import (
    compute_rsi,
    compute_macd,
    compute_bollinger,
    compute_ema,
    compute_stochastic,
    compute_adx_strength,
    scalping_strategy,
    compute_atr,
    compute_cci,
    compute_parabolic_sar,
    compute_roc,  # Новый
    compute_obv   # Новый
)

XAI_API_KEY = os.getenv("XAI_API_KEY")
GROK_MODEL = "grok-4"

async def call_grok(candles, patterns, regime, tf, symbol, indicators):
    if not XAI_API_KEY:
        logging.warning("Grok отключён (нет ключа)")
        return 0.5

    recent = candles[-10:]
    desc = []
    for i, c in enumerate(recent):
        dir_ = "🟢" if c["close"] > c["open"] else "🔴"
        body = abs(c["close"] - c["open"])
        desc.append(f"{i+1}: {dir_} O:{c['open']:.4f} H:{c['high']:.4f} L:{c['low']:.4f} C:{c['close']:.4f} (body {body:.4f})")

    ind_desc = (
        f"RSI: {indicators['rsi']:.1f}\n"
        f"Stoch: {indicators.get('stoch', 50):.1f}\n"
        f"ADX: {indicators.get('adx', 20):.1f}\n"
        f"MACD: {indicators.get('macd', 0):.5f}\n"
        f"Bollinger: {indicators['bb']}\n"
        f"ROC: {indicators.get('roc', 0):.2f}\n"  # Новый
        f"OBV: {indicators.get('obv', 0):.0f}\n"  # Новый
    )

    prompt = f"""Ты скальпер на Forex/крипте. Анализируй {symbol} {tf}мин | Режим: {regime}.

Шаг 1: Опиши последние 3 свечи: momentum, volume spikes, micro-changes.
Шаг 2: Проверь индикаторы на overbought/oversold (RSI<30 buy, >70 sell; Stoch cross; MACD sign change).
Шаг 3: Учти паттерны и тренд для short-term bias.
Шаг 4: Предскажи вероятность роста на 1-2 минуты (даже на 0.05%): 0.00-1.00.

Свечи (последние 10): {", ".join([f"{i+1}: O{recent[i]['open']:.4f} C{recent[i]['close']:.4f} V{recent[i]['volume']:.0f}" for i in range(len(recent))])}
Паттерны: {", ".join(patterns) or "нет"}
Индикаторы: RSI{indicators['rsi']:.1f}, Stoch{indicators['stoch']:.1f}, ADX{indicators['adx']:.1f}, MACD{indicators['macd']:.4f}, BB{indicators['bb']}, ATR{indicators['atr']:.4f}, CCI{indicators['cci']:.1f}, PSAR{indicators['psar']}, ROC{indicators['roc']:.2f}, OBV{indicators['obv']:.0f}

Только финальное число: вероятность роста (up) на 1-2 мин."""

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROK_MODEL, "messages": [{"role": "user", "content": prompt}]}
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            try:
                return float(content)
            except ValueError:
                logging.warning(f"Grok вернул не число: {content}")
                return 0.5
    except Exception as e:
        logging.error(f"Grok ошибка: {e}")
        return 0.5

async def analyze(symbol=None, tf="1", image_bytes=None):
    source = "API" if symbol else "Скриншот"
    quality = 1.0
    if image_bytes:
        try:
            candles, quality = extract_candles(image_bytes)
            source = "Скриншот"
            if len(candles) < 10:
                return None, "Мало свечей на скрине"
        except Exception as e:
            return None, f"Ошибка анализа скрина: {e}"
    else:
        try:
            interval = {"1": "1m", "2": "2m", "5": "5m", "10": "10m"}.get(tf, "1m")
            candles = get_candles(symbol, interval=interval)
        except Exception as e:
            return None, f"Ошибка получения данных: {e}"

    if len(candles) < 10:
        return None, "Мало свечей"

    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    volumes = np.array([c["volume"] for c in candles])  # Для OBV

    indicators = {
        "rsi": compute_rsi(closes),
        "macd": compute_macd(closes),
        "bb": compute_bollinger(closes),
        "ema": compute_ema(closes),
        "stoch": compute_stochastic(closes, highs, lows),
        "adx": compute_adx_strength(highs, lows, closes),
        "atr": compute_atr(highs, lows, closes),
        "cci": compute_cci(highs, lows, closes),
        "psar": compute_parabolic_sar(highs, lows, closes),
        "closes": closes,  # Для strategy
        "roc": compute_roc(closes),  # Новый
        "obv": compute_obv(closes, volumes),  # Новый
    }

    features = build_features(candles, tf)  # Теперь sequences: (1, seq_len, features) для последней
    if features is None or features.size == 0:
        features = np.zeros((1, SEQ_LEN, 14))  # Подставь реальное num_features

    X = features[-1:]  # Последняя последовательность (1, seq_len, features)
    ml_change = get_model(tf).predict(X)  # Теперь scalar % change
    ml_prob_up = max(0, min(1, (ml_change / 2 + 0.5)))  # Нормализуем
    ml_prob_down = 1 - ml_prob_up

    patterns, pattern_score = detect_patterns(candles)

    regime = market_regime(candles)
    scalp_adj = scalping_strategy(indicators, patterns, regime)
    pattern_score = np.clip(pattern_score + scalp_adj, 0.0, 1.0)

    trend_prob = trend_signal(candles)

    grok_prob = await call_grok(candles, patterns, regime, tf, symbol or "Неизвестно", indicators)

    # Взвешивание вероятностей (усилено для Grok на коротких TF)
    if int(tf or 0) <= 2:
        weights = [0.15, 0.25, 0.15, 0.45]  # Больше Grok
    elif int(tf or 0) <= 5:
        weights = [0.20, 0.30, 0.20, 0.30]
    elif regime == "trend":
        weights = [0.30, 0.25, 0.20, 0.25]
    elif regime == "flat":
        weights = [0.15, 0.40, 0.20, 0.25]
    else:
        weights = [0.20, 0.30, 0.25, 0.25]

    final_prob_up = np.dot(weights, [ml_prob_up, pattern_score, trend_prob, grok_prob])
    final_prob_down = np.dot(weights, [ml_prob_down, 1 - pattern_score, 1 - trend_prob, 1 - grok_prob])
    final_prob = final_prob_up - final_prob_down + 0.5

    conf_label, conf_score = confidence_from_probs([ml_prob_up, pattern_score, trend_prob, grok_prob, ml_prob_down])

    return {
        "prob": round(final_prob, 3),
        "down_prob": round(final_prob_down, 3),
        "up_prob": round(final_prob_up, 3),
        "neutral_prob": round(1 - final_prob_up - final_prob_down, 3),
        "confidence": conf_label,
        "confidence_score": conf_score,
        "regime": regime,
        "patterns": patterns,
        "tf": tf,
        "symbol": symbol or "Скриншот",
        "source": source,
        "quality": quality,
        "indicators": indicators
    }, None