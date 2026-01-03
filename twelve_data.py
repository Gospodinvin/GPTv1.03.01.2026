import requests
import logging
from typing import List, Dict, Optional
from config import TWELVE_DATA_API_KEY

class TwelveDataClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        self.session.params = {"apikey": api_key}

    def get_candles(self, symbol: str, interval: str, outputsize: int = 50) -> Optional[List[Dict]]:
        try:
            url = f"{self.base_url}/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "format": "JSON"
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if "values" not in data or not data["values"]:
                logging.warning(f"Нет данных для {symbol} {interval}: {data.get('message', 'пустой ответ')}")
                return None
                
            candles = []
            for candle in data["values"][:outputsize]:
                candles.append({
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "volume": float(candle.get("volume", 0))
                })
            
            return candles[::-1]  # от старых к новым
            
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"Twelve Data HTTP error для {symbol} {interval}: {http_err} | {response.text}")
            return None
        except Exception as e:
            logging.error(f"Twelve Data unexpected error для {symbol} {interval}: {e}")
            return None

# Глобальный клиент
client = None

def get_client() -> Optional[TwelveDataClient]:
    global client
    if client is None and TWELVE_DATA_API_KEY is not None:
        try:
            client = TwelveDataClient(TWELVE_DATA_API_KEY)
        except Exception as e:
            logging.error(f"Не удалось создать TwelveDataClient: {e}")
            client = "error"
    return client if client != "error" else None
