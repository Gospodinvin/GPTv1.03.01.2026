from io import BytesIO
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.filters import CommandStart
from aiogram.enums import ContentType
from config import TELEGRAM_BOT_TOKEN, STATE_TTL_SECONDS
from keyboards import market_keyboard, tickers_keyboard, timeframe_keyboard
from state import TTLState
from predictor import analyze
import logging
import os  # ← Добавлено для Railway PORT
from flask import Flask
import threading

state = TTLState(STATE_TTL_SECONDS)

app = Flask(__name__)

@app.route('/health')
def health():
    return "OK", 200

def run_flask():
    port = int(os.environ.get("PORT", 8080))  # Динамический порт для Railway
    app.run(host="0.0.0.0", port=port)

threading.Thread(target=run_flask, daemon=True).start()

async def start(m: Message):
    await m.answer(
        "🤖 Боттрейд — анализ графиков с индикаторами и скальпинг-стратегией\n\n"
        "Выберите рынок:",
        reply_markup=market_keyboard()
    )

async def image_handler(m: Message):
    bio = BytesIO()
    file_id = m.photo[-1].file_id if m.photo else m.document.file_id
    file = await m.bot.get_file(file_id)
    await m.bot.download_file(file.file_path, bio)
    await state.set(m.from_user.id, "data", bio.getvalue())
    await state.set(m.from_user.id, "mode", "image")
    await m.answer("Выберите таймфрейм:", reply_markup=timeframe_keyboard())

async def send_result(message: Message, res: dict):
    recommendation = "🟢 BUY" if res["prob"] > 0.6 else "🔴 SELL" if res["prob"] < 0.4 else "⚪ Нейтрал"
    color = "🟢" if res["prob"] > 0.6 else "🔴" if res["prob"] < 0.4 else "⚪"

    growth_percent = round(res.get("up_prob", 0) * 100, 1)
    down_percent = round(res.get("down_prob", 0) * 100, 1)
    neutral_percent = round(res.get("neutral_prob", 0) * 100, 1)

    html_txt = (
        f"📊 <b>{res['symbol']} | {res['tf']} мин</b>\n\n"
        f"{color} <b>Рекомендация:</b> {recommendation}\n"
        f"Рост (1–2 свечи): <b>{growth_percent}%</b>\n"
        f"Падение: <b>{down_percent}%</b>\n"
        f"Нейтрал: <b>{neutral_percent}%</b>\n"
        f"Уверенность: <b>{res['confidence']}</b> ({res['confidence_score']})\n"
        f"Режим рынка: <b>{res['regime'].capitalize()}</b>\n"
        f"Источник: <i>{res['source']}</i>\n"
    )

    if res.get("quality", 1.0) < 0.9:
        html_txt += f"⚠ <b>Качество скрина:</b> {res['quality']:.2f} (может влиять на точность)\n"

    if res["patterns"]:
        html_txt += f"🔥 <b>Паттерны:</b> {', '.join(res['patterns'])}\n"

    ind = res.get("indicators", {})
    html_txt += (
        f"\n📈 <b>Индикаторы:</b>\n"
        f"• RSI: <code>{ind.get('rsi', 50):.1f}</code>\n"
        f"• Stoch: <code>{ind.get('stoch', 50):.1f}</code>\n"
        f"• ADX: <code>{ind.get('adx', 20):.1f}</code>\n"
        f"• MACD: <code>{ind.get('macd', 0):.5f}</code>\n"
        f"• Bollinger: <code>{ind.get('bb', 'neutral').capitalize()}</code>\n"
        f"• ATR: <code>{ind.get('atr', 0.01):.4f}</code>\n"
        f"• CCI: <code>{ind.get('cci', 0):.1f}</code>\n"
        f"• PSAR: <code>{ind.get('psar', 'neutral').capitalize()}</code>\n"
        f"• ROC: <code>{ind.get('roc', 0):.2f}</code>\n"
        f"• OBV: <code>{ind.get('obv', 0):.0f}</code>\n"
    )

    html_txt += "\n⚠️ <b>Не финансовая рекомендация!</b> Торгуйте на свой страх и риск. SL рекомендуется на уровне ATR×2."

    # Клавиатура с тремя кнопками
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🔄 Обновить анализ", callback_data=f"refresh:{res['symbol']}:{res['tf']}"),
            InlineKeyboardButton(text="⏱ Другой таймфрейм", callback_data="change_tf")
        ],
        [
            InlineKeyboardButton(text="🔙 Назад к рынкам", callback_data="back:markets")
        ]
    ])

    await message.answer(html_txt, parse_mode="HTML", reply_markup=keyboard)

async def callback_handler(cb: CallbackQuery):
    if not cb.data:
        await cb.answer()
        return

    data = cb.data
    user_id = cb.from_user.id
    logging.info(f"Callback: '{data}' от {user_id}")

    if data.startswith("market:"):
        market = data.split(":")[1]
        kb, info = tickers_keyboard(market)
        await cb.message.edit_text(info, reply_markup=kb)
        await state.set(user_id, "market", market)
        await cb.answer()
        return

    if data.startswith("ticker:"):
        ticker = data.split(":")[1]
        logging.info(f"Выбран тикер: {ticker}")
        await state.set(user_id, "ticker", ticker)
        await state.set(user_id, "mode", "api")
        await cb.message.edit_text(f"Инструмент: {ticker}\n\nВыберите таймфрейм:", reply_markup=timeframe_keyboard())
        await cb.answer()
        return

    if data.startswith("tf:"):
        tf = data.split(":")[1]
        logging.info(f"Выбран TF: {tf}")
        await state.set(user_id, "tf", tf)

        res, err = await analyze(user_id, state, cb.bot)
        if err:
            await cb.message.answer(f"Ошибка: {err}")
        else:
            await send_result(cb.message, res)
        await cb.answer()
        return

    # Новые обработчики кнопок из результата
    if data.startswith("refresh:"):
        _, symbol, tf = data.split(":")
        await state.set(user_id, "ticker", symbol)
        await state.set(user_id, "tf", tf)
        await state.set(user_id, "mode", "api")

        res, err = await analyze(user_id, state, cb.bot)
        if err:
            await cb.message.edit_text(f"Ошибка при обновлении: {err}")
        else:
            await send_result(cb.message, res)
        await cb.answer()
        return

    if data == "change_tf":
        ticker = await state.get(user_id, "ticker") or "Неизвестно"
        await cb.message.edit_text(
            f"Инструмент: {ticker}\n\nВыберите таймфрейм:",
            reply_markup=timeframe_keyboard()
        )
        await cb.answer()
        return

    if data.startswith("back:"):
        await cb.message.edit_text("Выберите рынок:", reply_markup=market_keyboard())
        await state.clear(user_id)
        await cb.answer()
        return

    await cb.answer()

def main():
    bot = Bot(TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()

    dp.message.register(start, CommandStart())
    dp.message.register(image_handler, F.content_type.in_({ContentType.PHOTO, ContentType.DOCUMENT}))
    dp.callback_query.register(callback_handler)

    print("Бот запущен — версия со скальпингом и индикаторами!")
    logging.info("Bot polling started...")

    dp.run_polling(bot)

if __name__ == "__main__":
    main()
