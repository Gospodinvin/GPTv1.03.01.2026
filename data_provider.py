# data_provider.py
from binance_data import get_candles as get_candles_binance
from twelve_data import get_client
import logging

def get_candles(symbol: str, interval: str = "1m", limit: int = 70):
    original_symbol = symbol.upper()
    binance_symbol = original_symbol.replace("USD", "USDT")

    is_forex_like = len(original_symbol) == 6 and '/' not in original_symbol

    if is_forex_like:
        if original_symbol.endswith("USD"):
            forex_symbol = original_symbol[:-3] + "/USD"
        else:
            forex_symbol = original_symbol[:3] + "/" + original_symbol[3:]
    else:
        forex_symbol = original_symbol

    td_interval_map = {
        "1m": "1min",
        "2m": "1min",
        "5m": "5min",
        "10m": "5min",
    }
    td_interval = td_interval_map.get(interval, interval)

    client = get_client()
    if client:
        logging.info(f"Пытаемся получить {forex_symbol} {td_interval} через Twelve Data...")
        candles = client.get_candles(symbol=forex_symbol, interval=td_interval, outputsize=limit)
        if candles:
            logging.info(f"Успешно получены данные через Twelve Data ({len(candles)} свечей)")
            return candles

        logging.warning("Twelve Data не вернула данные, переходим на Binance (если применимо)...")

    if not is_forex_like or original_symbol.endswith("USD"):
        logging.info(f"Пытаемся получить {binance_symbol} {interval} через Binance...")
        try:
            candles = get_candles_binance(binance_symbol, interval=interval, limit=limit)
            if candles:
                logging.info(f"Успешно получены данные через Binance ({len(candles)} свечей)")
                # Real-time update: обновляем close последней свечи текущей ценой
                # spot_client = Spot()
                # current_price = float(spot_client.ticker_price(binance_symbol)['price'])
                # candles[-1]['close'] = current_price
                return candles
        except Exception as e:
            logging.error(f"Binance не сработал для {binance_symbol}: {e}")

        logging.info(f"Пытаемся получить {original_symbol} {interval} через Binance (оригинальный символ)...")
        try:
            candles = get_candles_binance(original_symbol, interval=interval, limit=limit)
            if candles:
                logging.info(f"Успешно получены данные через Binance с оригинальным символом ({len(candles)} свечей)")
                spot_client = Spot()
                current_price = float(spot_client.ticker_price(original_symbol)['price'])
                candles[-1]['close'] = current_price
                return candles
        except Exception as e:
            logging.error(f"Binance не сработал и для оригинального символа {original_symbol}: {e}")


    raise RuntimeError("Не удалось получить данные ни с Twelve Data, ни с Binance")
