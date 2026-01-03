from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
import datetime

# Категории рынков
MARKET_CATEGORIES = {
    "forex": {
        "asian": ["AUDUSD", "NZDUSD", "USDJPY", "AUDJPY", "USDCNH", "EURJPY", "GBPAUD", "CHFJPY", "AUDNZD", "NZDJPY"],
        "london": ["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY", "USDCHF", "EURCAD", "GBPCAD", "EURCHF", "GBPCHF"],
        "newyork": ["EURUSD", "GBPUSD", "USDCAD", "USDJPY", "AUDCAD", "EURCHF", "GBPCHF", "GBPJPY", "EURJPY"],
        "overlap": ["EURUSD", "GBPUSD", "USDCAD", "USDJPY", "EURCHF", "GBPCHF", "GBPJPY", "EURJPY", "USDCHF", "EURCAD"],
    },
    "crypto": ["BTCUSD", "ETHUSD", "BNBUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "AVAXUSD", "DOTUSD", "LTCUSD"],  # 24/7, без сессий
    "metals": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "HGUSD", "SIUSD", "PAUSD", "PLUSD", "ALUSD", "ZNUSD"],
    "stocks": ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "NFLX", "INTC", "AMD"],  # Индексы/акции
}

def get_current_session():
    # Московское время (UTC+3)
    msk_hour = (datetime.datetime.utcnow() + datetime.timedelta(hours=3)).hour
    
    if 3 <= msk_hour < 11:
        return "asian", "🌏 Азиатская сессия (03:00–11:00 MSK)"
    elif 11 <= msk_hour < 16:
        return "london", "🇬🇧 Лондонская сессия (11:00–19:00 MSK)"
    elif 16 <= msk_hour < 19:
        return "overlap", "🔥 Пересечение Лондон + Нью-Йорк (16:00–19:00 MSK) — максимальная волатильность!"
    elif 19 <= msk_hour < 24 or 0 <= msk_hour < 3:
        return "newyork", "🇺🇸 Нью-Йоркская сессия (16:00–00:00 MSK)"
    else:
        return "closed", "🌙 Рынок спит (выходные или ночь)"


def market_keyboard():
    buttons = [
        [
            InlineKeyboardButton(text="💱 Forex", callback_data="market:forex"),
            InlineKeyboardButton(text="🪙 Crypto (24/7)", callback_data="market:crypto"),
        ],
        [
            InlineKeyboardButton(text="🛡️ Metals", callback_data="market:metals"),
            InlineKeyboardButton(text="📈 Stocks", callback_data="market:stocks"),
        ],
        [
            InlineKeyboardButton(text="📸 Анализ по скриншоту", callback_data="mode:image"),
        ]
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def tickers_keyboard(market: str):
    session_key, session_text = get_current_session()
    
    # Для крипты и металлов/акций — нет разделения по сессиям, берём весь список напрямую
    if market == "crypto":
        tickers = MARKET_CATEGORIES["crypto"]
        session_text = "🪙 Крипта работает 24/7"
    elif market in ["metals", "stocks"]:
        tickers = MARKET_CATEGORIES[market]
        session_text = session_text  # оставляем текущую сессию как информацию
    elif session_key == "closed":
        tickers = []  # на выходных forex не показываем ничего или можно показать все
        session_text = "🌙 Рынок спит — Forex недоступен"
    else:
        # Для forex — выбираем по текущей сессии
        tickers = MARKET_CATEGORIES.get("forex", {}).get(session_key, [])
    
    buttons = []
    row = []
    for t in tickers:
        row.append(InlineKeyboardButton(text=t, callback_data=f"ticker:{t}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    
    buttons.append([InlineKeyboardButton(text="🔙 Назад к рынкам", callback_data="back:markets")])
    
    info = f"Текущая сессия: {session_text}\nРекомендуемые пары для {market.upper()}:\n\nВыберите тикер:"
    return InlineKeyboardMarkup(inline_keyboard=buttons), info


def timeframe_keyboard():
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="1 минута", callback_data="tf:1"),
            InlineKeyboardButton(text="2 минуты", callback_data="tf:2"),
            InlineKeyboardButton(text="5 минут", callback_data="tf:5"),
        ],
        [
            InlineKeyboardButton(text="10 минут", callback_data="tf:10"),
        ]
    ])
