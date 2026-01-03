import numpy as np
from trend import trend_signal  # добавляем импорт

def detect_patterns(candles):
    patterns = []
    score = 0.0
    if len(candles) < 3:
        return patterns, score

    for i in range(-3, 0):
        prev = candles[i-1] if i > -3 else candles[0]
        c = candles[i]
        o, cl, h, l = c["open"], c["close"], c["high"], c["low"]
        body = abs(cl - o)
        range_ = h - l
        upper_wick = h - max(o, cl)
        lower_wick = min(o, cl) - l
        direction = 1 if cl > o else -1

        # Engulfing
        if i > -3:
            prev_body = abs(prev["close"] - prev["open"])
            if body > prev_body * 1.5 and direction == -np.sign(prev["close"] - prev["open"]):
                patterns.append("Engulfing")
                score += 0.25

        # Marubozu
        if body > range_ * 0.85:
            patterns.append("Marubozu")
            score += 0.20

        # Hammer / Shooting Star
        if body < range_ * 0.4 and lower_wick > body * 2 and upper_wick < body:
            patterns.append("Hammer" if direction > 0 else "Shooting Star")
            score += 0.22

        # Pinbar
        if (lower_wick > body * 2 and upper_wick < body * 0.5) or (upper_wick > body * 2 and lower_wick < body * 0.5):
            patterns.append("Pinbar")
            score += 0.18

        # Doji
        if body < range_ * 0.1:
            patterns.append("Doji")
            score += 0.10

    # 3-свечные
    if len(candles) >= 3:
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]

        # Morning Star
        if (c1["close"] < c1["open"] and
            abs(c2["close"] - c2["open"]) < abs(c1["close"] - c1["open"]) * 0.3 and
            c3["close"] > c3["open"] and
            c3["close"] > (c1["open"] + c1["close"]) / 2):
            patterns.append("Morning Star")
            score += 0.28

        # Evening Star
        if (c1["close"] > c1["open"] and
            abs(c2["close"] - c2["open"]) < abs(c1["close"] - c1["open"]) * 0.3 and
            c3["close"] < c3["open"] and
            c3["close"] < (c1["open"] + c1["close"]) / 2):
            patterns.append("Evening Star")
            score += 0.28

        # Three White Soldiers
        if all(c["close"] > c["open"] for c in [c1, c2, c3]) and c2["close"] > c1["close"] and c3["close"] > c2["close"]:
            patterns.append("Three White Soldiers")
            score += 0.30

        # Three Black Crows
        if all(c["close"] < c["open"] for c in [c1, c2, c3]) and c2["close"] < c1["close"] and c3["close"] < c2["close"]:
            patterns.append("Three Black Crows")
            score += 0.30

        # Bullish Harami
        if c1["close"] < c1["open"] and c3["close"] > c3["open"] and c3["open"] > c1["close"] and c3["close"] < c1["open"]:
            patterns.append("Bullish Harami")
            score += 0.20

        # Bearish Harami
        if c1["close"] > c1["open"] and c3["close"] < c3["open"] and c3["open"] < c1["close"] and c3["close"] > c1["open"]:
            patterns.append("Bearish Harami")
            score += 0.20

    patterns = list(set(patterns))

    # Улучшение: усиление паттернов в соответствии с трендом
    trend_prob = trend_signal(candles)
    if trend_prob > 0.65 and any("bull" in p.lower() or "hammer" in p.lower() or "morning" in p.lower() for p in patterns):
        score = min(score * 1.3, 1.0)
    elif trend_prob < 0.35 and any("bear" in p.lower() or "shooting" in p.lower() or "evening" in p.lower() for p in patterns):
        score = min(score * 1.3, 1.0)

    return patterns, min(score, 1.0)
