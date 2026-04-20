"""
بوت تيليغرام لتحديد بحر الشعر العربي
Arabic Poetry Meter Detection Telegram Bot
"""

import os
import logging
import numpy as np
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MODEL_PATH = "best_bilstm_fixed.keras"

try:
    import keras
    model = keras.models.load_model(MODEL_PATH)
    logger.info("✅ تم تحميل النموذج بنجاح")
    logger.info(f"📐 Input shape: {model.input_shape}")
    logger.info(f"📐 Output shape: {model.output_shape}")
except Exception as e:
    logger.error(f"❌ فشل تحميل النموذج: {e}")
    model = None

METERS = [
    "الخفيف",
    "الرجز",
    "الرمل",
    "السريع",
    "الطويل",
    "الكامل",
    "المتقارب",
    "المجتث",
    "المديد",
    "المنسرح",
    "المواليا",
    "الهزج",
    "الوافر",
    "البسيط",
]

MAX_LEN = 128
ARABIC_CHARS = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأإةىؤئ"

def normalize_arabic(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text.strip()

def text_to_sequence(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    text = normalize_arabic(text)
    vocab = list(ARABIC_CHARS)
    char2idx = {c: i + 1 for i, c in enumerate(vocab)}
    seq = [char2idx.get(c, 0) for c in text if c != ' ']
    seq = seq[:max_len]
    seq = seq + [0] * (max_len - len(seq))
    return np.array([seq], dtype=np.int32)

def predict_meter(verse: str):
    if model is None:
        return None, 0.0
    seq = text_to_sequence(verse)
    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100
    meter = METERS[idx] if idx < len(METERS) else f"صنف {idx}"
    return meter, confidence

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = "✅ النموذج محمّل وجاهز" if model else "❌ النموذج لم يُحمَّل"
    msg = (
        "👋 أهلاً! أنا بوت تحليل بحور الشعر العربي 🎵\n\n"
        f"الحالة: {status}\n\n"
        "أرسل لي بيتًا من الشعر وسأخبرك ببحره.\n\n"
        "مثال:\n"
        "_قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ_\n\n"
        "الأوامر:\n"
        "/start — بدء البوت\n"
        "/help — المساعدة\n"
        "/meters — قائمة البحور المدعومة"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📖 *كيفية الاستخدام:*\n\n"
        "1. أرسل بيتًا من الشعر مباشرةً.\n"
        "2. سيحدد البوت البحر ودرجة الثقة.\n\n"
        "💡 *نصائح للنتائج الأفضل:*\n"
        "• أرسل بيتًا كاملاً (شطران).\n"
        "• يمكنك إرسال البيت بالتشكيل أو بدونه.\n"
        "• تجنّب الأبيات القصيرة جداً."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def list_meters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["🎼 *البحور المدعومة:*\n"]
    for i, m in enumerate(METERS, 1):
        lines.append(f"{i}. {m}")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

async def analyze_verse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    verse = update.message.text.strip()

    if len(verse) < 5:
        await update.message.reply_text("⚠️ الرجاء إرسال بيت شعري كامل.")
        return

    if model is None:
        await update.message.reply_text("❌ النموذج غير محمّل. تحقق من Logs على Render.")
        return

    await update.message.reply_text("⏳ جارٍ التحليل...")

    try:
        meter, confidence = predict_meter(verse)
        filled = int(confidence / 10)
        bar = "█" * filled + "░" * (10 - filled)
        response = (
            f"📜 *البيت:*\n_{verse}_\n\n"
            f"🎵 *البحر:* {meter}\n"
            f"📊 *الثقة:* {confidence:.1f}%\n"
            f"`{bar}`"
        )
        await update.message.reply_text(response, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"خطأ في التحليل: {e}")
        await update.message.reply_text(f"❌ خطأ في التحليل:\n`{e}`", parse_mode="Markdown")

def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("❌ لم يُعثر على TELEGRAM_BOT_TOKEN في متغيرات البيئة!")

    logger.info(f"📁 مسار العمل: {os.getcwd()}")
    logger.info(f"📁 الملفات في المجلد: {os.listdir('.')}")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("meters", list_meters))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_verse))

    logger.info("🤖 البوت يعمل الآن...")
    app.run_polling()

if __name__ == "__main__":
    main()
