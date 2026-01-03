# train_models.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import joblib
import os
import logging

from sklearn.model_selection import train_test_split, GridSearchCV

# Импортируем нужные функции!
from binance_data import get_candles as get_candles_binance
from features import build_features  # ← ЭТО БЫЛО ПРОПУЩЕНО!

logging.basicConfig(level=logging.INFO)

# Список символов
SYMBOLS = ["BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD"]

# Только поддерживаемые Binance интервалы!
TIMEFRAMES = ["1", "5"]  # Убрали 2 и 10, потому что Binance их не поддерживает
INTERVALS = {"1": "1m", "5": "5m"}

PROFIT_THRESHOLD = 0.05  # Для micro-moves

LIMIT = 10000

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_data(candles, tf):
    if len(candles) < 50:
        return None, None

    X_raw = build_features(candles, tf)
    if len(X_raw) == 0:
        return None, None

    df = pd.DataFrame(candles)
    y = []

    # Определяем, с какой свечи начинаются фичи
    # В текущей features.py: for i in range(1, len(candles)) → X_raw[0] = candles[1]
    start_idx = 1  # Это важно! Подгони под свою features.py

    for i in range(len(X_raw)):
        candle_idx = start_idx + i
        if candle_idx + 3 >= len(df):
            break
        current_close = df.iloc[candle_idx]["close"]
        future_close = df.iloc[candle_idx + 3]["close"]
        change = (future_close - current_close) / current_close * 100
        y.append(change)

    y = np.array(y)

    # Обрезаем X, если y короче (на случай ошибок)
    min_len = min(len(X_raw), len(y))
    return X_raw[:min_len], y[:min_len]

def train_and_save():
    for tf in TIMEFRAMES:
        print(f"\n=== Обучение модели для {tf}-минутного таймфрейма ===")
        interval = INTERVALS[tf]
        all_X = []
        all_y = []

        for symbol in SYMBOLS:
            try:
                logging.info(f"Загрузка {symbol} {interval}...")
                # Binance сам заменит USD → USDT
                candles = get_candles_binance(symbol, interval=interval, limit=LIMIT)
                if len(candles) < 100:
                    print(f"  {symbol}: мало свечей ({len(candles)})")
                    continue

                X, y = prepare_data(candles, tf)
                if X is not None and len(X) > 0:
                    all_X.extend(X)
                    all_y.extend(y)
                    print(f"  {symbol}: +{len(X)} примеров (всего: {len(all_X)})")
            except Exception as e:
                logging.error(f"Ошибка с {symbol}: {e}")
                print(f"  {symbol}: ошибка — {e}")

        if len(all_X) < 500:
            print(f"Недостаточно данных для {tf}m — пропускаем")
            continue

        X = np.array(all_X)
        y = np.array(all_y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [400, 600],
            'max_depth': [10, 12],
            'min_samples_split': [8, 10]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print(f"\n{tf}m — MSE на тестовой: {mse:.6f}")

        path = os.path.join(MODEL_DIR, f"model_{tf}m.joblib")
        joblib.dump(model, path)
        print(f"Модель сохранена: {path}\n")

    print("Обучение всех моделей завершено! Модели лежат в папке 'models/'")

if __name__ == "__main__":
    train_and_save()