import requests
import logging

BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision"
]

# Добавляем реалистичный User-Agent
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
}

def get_candles(symbol, interval="1m", limit=70):
    symbol = symbol.replace("USD", "USDT")  # На всякий случай, хотя вызывающий код уже может это делать

    for base_url in BINANCE_ENDPOINTS:
        url = f"{base_url}/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        try:
            r = requests.get(url, params=params, timeout=8, headers=HEADERS)
            if r.status_code == 200:
                data = r.json()
                if not data:  # Пустой ответ
                    continue
                candles = [
                    {
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5])
                    }
                    for c in data
                ]
                return candles
            else:
                logging.error(f"Binance {r.status_code} {r.text} via {base_url}")
        except Exception as e:
            logging.error(f"Binance error via {base_url}: {e}")

    raise RuntimeError("Binance data unavailable (all endpoints failed)")
