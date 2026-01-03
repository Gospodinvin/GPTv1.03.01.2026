# model.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import logging

class CandleModel:
    def __init__(self, tf: str):
        self.tf = tf
        self.rf_path = f"models/model_{tf}m.joblib"  # RF в приоритете
        self.lstm_path = f"models/lstm_{tf}m.pth"
        self.scaler_path = f"models/scaler_y_lstm_{tf}m.joblib"
        self.model = None
        self.is_rf = True
        self.fallback = True
        self.load_model()

    def load_model(self):
        # Сначала пробуем RF
        if os.path.exists(self.rf_path):
            try:
                self.model = joblib.load(self.rf_path)
                self.is_rf = True
                self.fallback = False
                logging.info(f"Загружена RandomForest модель для {self.tf}m (основная)")
                return
            except Exception as e:
                logging.error(f"Ошибка загрузки RF {self.tf}m: {e}")

        # Если нет RF — пробуем LSTM (но пока не рекомендуется)
        # if os.path.exists(self.lstm_path):
        #     ... (код LSTM)

        logging.info(f"Нет модели для {self.tf}m — fallback")

    def predict(self, X):
        if X.shape[0] == 0:
            return 0.0

        if self.fallback or self.model is None:
            return 0.0  # Нейтраль

        return self.model.predict(X)[0]  # % change