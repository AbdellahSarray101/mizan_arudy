"""
بوت تيليغرام لتحديد بحر الشعر العربي
"""

import os
import logging
import numpy as np
import re
import threading
import http.server
import socketserver
import os

def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()

# تشغيل السيرفر الوهمي في خلفية الكود
threading.Thread(target=run_dummy_server, daemon=True).start()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

MAX_LEN = 128

METERS_AR = {
    'saree'    : 'السريع',
    'kamel'    : 'الكامل',
    'mutakareb': 'المتقارب',
    'mutadarak': 'المتدارك',
    'munsareh' : 'المنسرح',
    'madeed'   : 'المديد',
    'mujtath'  : 'المجتث',
    'ramal'    : 'الرمل',
    'baseet'   : 'البسيط',
    'khafeef'  : 'الخفيف',
    'taweel'   : 'الطويل',
    'wafer'    : 'الوافر',
    'hazaj'    : 'الهزج',
    'rajaz'    : 'الرجز',
}

# ── دوال التنظيف ──────────────────────────────────────────────────────
def normalize_arabic(text):
    text = text.replace('#', ' ')
    text = re.sub('[أإآٱ]', 'ا', text)
    text = re.sub('ى', 'ي', text)
    text = re.sub('ـ', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_chars(text):
    return ' '.join(list(text))

def extract_rhyme(verse):
    diacritics = re.compile(
        r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]'
    )
    SKIP_CHARS = set('اىهوي')
    lines = [l.strip() for l in verse.split('\n') if l.strip()]
    last_line = lines[-1] if lines else verse
    last_line = diacritics.sub('', last_line)
    words = last_line.split()
    if not words:
        return '—'
    last_word = words[-1]
    arabic_chars = [c for c in last_word if '\u0600' <= c <= '\u06FF']
    if not arabic_chars:
        return '—'
    i = len(arabic_chars) - 1
    while i > 0 and arabic_chars[i] in SKIP_CHARS:
        i -= 1
    return arabic_chars[i]

# ── بناء التوكنايزر من train.txt ──────────────────────────────────────
logger.info("Building tokenizer from train.txt...")
train_verses, train_labels = [], []

with open("train.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split(' ', 1)
            train_labels.append(int(parts[0]))
            train_verses.append(parts[1])

train_clean = [normalize_arabic(v) for v in train_verses]
train_char  = [split_chars(v) for v in train_clean]

tokenizer = Tokenizer(oov_token="<OOV>", char_level=False)
tokenizer.fit_on_texts(train_char)

# ── بناء LabelEncoder من labels.txt ───────────────────────────────────
logger.info("Building label encoder from labels.txt...")
with open("labels.txt", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

le = LabelEncoder()
le.fit(train_labels)

# ── تحميل النموذج ─────────────────────────────────────────────────────
logger.info("Loading model...")
try:
    import keras
    def build_model():
        m = keras.Sequential([
            keras.layers.Embedding(45, 128),
            keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True)),
            keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            keras.layers.Dropout(0.5),
            keras.layers.Bidirectional(keras.layers.LSTM(64)),
            keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(14, activation='softmax'),
        ])
        m(tf.zeros((1, 128)))
        return m

    model = build_model()
    model.load_weights("model.weights.h5")
    logger.info("✅ تم تحميل النموذج بنجاح")
except Exception as e:
    logger.error(f"❌ فشل تحميل النموذج: {e}")
    model = None

# ── دالة التوقع ───────────────────────────────────────────────────────
def predict_meter(verse):
    cleaned  = normalize_arabic(verse)
    charred  = split_chars(cleaned)
    seq      = tokenizer.texts_to_sequences([charred])
    padded   = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred     = model.predict(padded, verbose=0)
    class_id = np.argmax(pred)
    meter_en = class_names[le.inverse_transform([class_id])[0]]
    meter_ar = METERS_AR.get(meter_en, meter_en)
    rhyme    = extract_rhyme(verse)
    return meter_ar, rhyme

# ── معالجات البوت ─────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = "✅ النموذج محمّل وجاهز" if model else "❌ النموذج لم يُحمَّل"
    msg = (
        "👋 أهلاً! أنا بوت تحليل بحور الشعر العربي 🎵\n\n"
        f"الحالة: {status}\n\n"
        "أرسل لي بيتًا من الشعر وسأخبرك ببحره وقافيته.\n\n"
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
        "2. سيحدد البوت البحر والقافية ودرجة الثقة.\n\n"
        "💡 *نصائح للنتائج الأفضل:*\n"
        "• أرسل بيتًا كاملاً (شطران).\n"
        "• يمكنك إرسال البيت بالتشكيل أو بدونه.\n"
        "• تجنّب الأبيات القصيرة جداً."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def list_meters(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["🎼 *البحور المدعومة:*\n"]
    for i, m in enumerate(METERS_AR.values(), 1):
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
        meter, rhyme = predict_meter(verse)
        response = (
            f"📜 *البيت:*\n_{verse}_\n\n"
            f"🎵 *البحر:* {meter}\n"
            f"🔤 *القافية:* {rhyme}"
        )
        await update.message.reply_text(response, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"خطأ في التحليل: {e}")
        await update.message.reply_text(f"❌ خطأ في التحليل:\n`{e}`", parse_mode="Markdown")

# ── نقطة الدخول ───────────────────────────────────────────────────────
def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("❌ لم يُعثر على TELEGRAM_BOT_TOKEN في متغيرات البيئة!")

    logger.info(f"📁 الملفات في المجلد: {os.listdir('.')}")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("meters", list_meters))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_verse))

    logger.info("🤖 البوت يعمل الآن...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
