import os
import logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()  # Изменено на DEBUG для лучших логов

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")  # Новый
STATE_TTL_SECONDS = 15 * 60
