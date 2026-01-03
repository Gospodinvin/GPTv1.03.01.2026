# train_lstm.py — Обучение LSTM моделей на PyTorch (совместимо с текущим features.py)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import os
import logging

from binance_data import get_candles as get_candles_binance
from features import build_features  # Используем текущий 2D features.py

logging.basicConfig(level=logging.INFO)

SYMBOLS = ["BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD"]
TIMEFRAMES = ["1", "5"]
INTERVALS = {"1": "1m", "5": "5m"}

SEQ_LEN = 20        # Длина последовательности свечей для LSTM
LOOKAHEAD = 3       # Предсказываем изменение через 3 свечи
LIMIT = 20000       # Максимум свечей с Binance

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Предсказываем % изменение

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(out[:, -1, :])  # Берём выход с последнего шага


def create_sequences(features_2d, lookahead=LOOKAHEAD, seq_len=SEQ_LEN):
    """
    Преобразует 2D фичи (samples, features) в 3D последовательности (samples, seq_len, features)
    """
    if len(features_2d) < seq_len:
        return np.array([])

    sequences = []
    for i in range(seq_len - 1, len(features_2d)):
        seq = features_2d[i - seq_len + 1 : i + 1]
        sequences.append(seq)

    return np.array(sequences)


def prepare_data(candles, tf):
    if len(candles) < SEQ_LEN + LOOKAHEAD + 10:
        return None, None, None

    # Получаем 2D фичи из текущего features.py
    features_2d = build_features(candles, tf)
    if len(features_2d) == 0:
        return None, None, None

    # Создаём последовательности
    X_seq = create_sequences(features_2d, seq_len=SEQ_LEN)
    if len(X_seq) == 0:
        return None, None, None

    df = pd.DataFrame(candles)

    y = []
    start_idx = SEQ_LEN - 1  # Индекс свечи, соответствующей последнему элементу в первой последовательности

    for i in range(len(X_seq)):
        current_idx = start_idx + i
        if current_idx + LOOKAHEAD >= len(df):
            break
        current_close = df.iloc[current_idx]["close"]
        future_close = df.iloc[current_idx + LOOKAHEAD]["close"]
        change_pct = (future_close - current_close) / current_close * 100
        y.append(change_pct)

    y = np.array(y)

    # Обрезаем до одинаковой длины
    min_len = min(len(X_seq), len(y))
    X_seq = X_seq[:min_len]
    y = y[:min_len]

    if len(X_seq) < 50:  # Слишком мало данных
        return None, None, None

    # Нормализация
    scaler_X = MinMaxScaler()
    X_flat = X_seq.reshape(-1, X_seq.shape[-1])
    X_scaled_flat = scaler_X.fit_transform(X_flat)
    X_scaled = X_scaled_flat.reshape(X_seq.shape)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    print(f"  Подготовлено {len(X_seq)} последовательностей для {tf}m")
    return X_scaled, y_scaled, scaler_y


def train_lstm(tf, all_X, all_y, scaler_y):
    train_size = int(0.8 * len(all_X))
    X_train, X_test = all_X[:train_size], all_X[train_size:]
    y_train, y_test = all_y[:train_size], all_y[train_size:]

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_size = all_X.shape[-1]
    model = LSTMModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    patience = 7
    best_loss = float('inf')
    counter = 0
    best_state = None

    epochs = 60
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                output = model(X_batch)
                val_loss += criterion(output.squeeze(), y_batch).item()

        val_loss /= len(test_loader)
        train_loss /= len(train_loader)

        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping!")
                break

    # Загружаем лучшую модель
    model.load_state_dict(best_state)

    # Оценка на оригинальной шкале
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
        preds_orig = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        mse = mean_squared_error(y_test_orig, preds_orig)
        print(f"\n{tf}m LSTM — MSE на тестовой (оригинальная шкала): {mse:.6f}")

    # Сохранение
    path = os.path.join(MODEL_DIR, f"lstm_{tf}m.pth")
    torch.save(model.state_dict(), path)
    scaler_path = os.path.join(MODEL_DIR, f"scaler_y_lstm_{tf}m.joblib")
    joblib.dump(scaler_y, scaler_path)

    print(f"LSTM модель сохранена: {path}")
    print(f"Scaler сохранён: {scaler_path}\n")


def train_and_save():
    for tf in TIMEFRAMES:
        print(f"\n=== Обучение LSTM для {tf}-минутного таймфрейма ===")
        interval = INTERVALS[tf]
        all_X = []
        all_y = []
        final_scaler_y = None

        for symbol in SYMBOLS:
            try:
                logging.info(f"Загрузка {symbol} {interval}...")
                candles = get_candles_binance(symbol, interval=interval, limit=LIMIT)
                if len(candles) < SEQ_LEN + LOOKAHEAD + 50:
                    print(f"  {symbol}: мало свечей ({len(candles)})")
                    continue

                X, y, scaler_y = prepare_data(candles, tf)
                if X is not None:
                    all_X.append(X)
                    all_y.append(y)
                    final_scaler_y = scaler_y  # Берём от последнего символа (или можно усреднить)
            except Exception as e:
                logging.error(f"Ошибка с {symbol}: {e}")
                print(f"  {symbol}: ошибка — {e}")

        if len(all_X) == 0:
            print(f"Недостаточно данных для {tf}m — пропускаем")
            continue

        all_X = np.vstack(all_X)
        all_y = np.hstack(all_y)

        print(f"Всего последовательностей для {tf}m: {len(all_X)}")
        train_lstm(tf, all_X, all_y, final_scaler_y)

    print("Обучение всех LSTM моделей завершено!")


if __name__ == "__main__":
    train_and_save()