import os
import threading
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from huggingface_hub import InferenceClient
from flask import Flask

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# Инициализация клиента HF
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

SYSTEM_PROMPT = (
    "Ты — эксперт по российскому и советскому кинематографу. "
    "Отвечай точно, кратко и по делу. Если не знаешь ответа — так и скажи. "
    "Используй только проверенные факты. Не выдумывай сюжеты или имена."
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🎬 Привет! Я бот-эксперт по российскому кино. "
        "Задавай вопросы о фильмах, режиссёрах, студиях, фестивалях и индустрии."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text.strip()
    if not user_text:
        return

    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            max_tokens=512,
            temperature=0.7
        )
        reply = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка LLM: {e}")
        reply = "⚠️ Модель временно недоступна или перегружена. Попробуй через минуту."

    await update.message.reply_text(reply)

def run_bot():
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    logger.info("🤖 Запуск Telegram бота (polling mode)...")
    application.run_polling(drop_pending_updates=True)

# Минимальный веб-сервер для Render (Free Tier требует HTTP-порт)
web_app = Flask(__name__)

@web_app.route("/")
def health():
    return {"status": "ok", "bot": "running"}

if __name__ == "__main__":
    # Запуск бота в фоновом потоке
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    # Запуск веб-сервера, чтобы Render не убивал процесс
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🌐 Запуск веб-сервера на порту {port}...")
    web_app.run(host="0.0.0.0", port=port)
