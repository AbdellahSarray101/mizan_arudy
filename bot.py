import os
import logging
import numpy as np
import re
import threading
import http.server
import socketserver
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# --- 1. الخادم الوهمي ---
def run_dummy_server():
    port = int(os.environ.get("PORT", 8080))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        httpd.serve_forever()

threading.Thread(target=run_dummy_server, daemon=True).start()

# --- 2. الإعدادات العامة ---
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_LEN = 128
METERS_AR = {
    'saree': 'السريع', 'kamel': 'الكامل', 'mutakareb': 'المتقارب',
    'mutadarak': 'المتدارك', 'munsareh': 'المنسرح', 'madeed': 'المديد',
    'mujtath': 'المجتث', 'ramal': 'الرمل', 'baseet': 'البسيط',
    'khafeef': 'الخفيف', 'taweel': 'الطويل', 'wafer': 'الوافر',
    'hazaj': 'الهزج', 'rajaz': 'الرجز',
}

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

# --- 3. تحميل الموارد (خارج الدوال) ---
# قمنا بتعريفها هنا لتكون مرئية لكل الملف
tokenizer = Tokenizer(oov_token="<OOV>", char_level=False)
le = LabelEncoder()
class_names = []
model = None

try:
    # تحميل البيانات للتوكنايزر
    train_verses, train_labels = [], []
    with open("train.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    train_labels.append(int(parts[0]))
                    train_verses.append(parts[1])
    
    train_clean = [normalize_arabic(v) for v in train_verses]
    train_char  = [split_chars(v) for v in train_clean]
    tokenizer.fit_on_texts(train_char)
    
    with open("labels.txt", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    le.fit(train_labels)

    # التحميل الصحيح للنموذج (تم حذف حرف k الزائد)
    model = tf.keras.models.load_model("best_bilstm.keras", compile=False)
    logger.info("✅ تم تحميل النموذج والملفات بنجاح")
except Exception as e:
    logger.error(f"❌ فشل حرج في التحميل: {e}")

# --- 4. الوظائف والمعالجات ---
def extract_rhyme(verse):
    diacritics = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
    SKIP_CHARS = set('اىهوي')
    lines = [l.strip() for l in verse.split('\n') if l.strip()]
    last_line = lines[-1] if lines else verse
    last_line = diacritics.sub('', last_line)
    words = last_line.split()
    if not words: return '—'
    last_word = words[-1]
    arabic_chars = [c for c in last_word if '\u0600' <= c <= '\u06FF']
    i = len(arabic_chars) - 1
    while i > 0 and arabic_chars[i] in SKIP_CHARS: i -= 1
    return arabic_chars[i] if arabic_chars else '—'

async def analyze_verse(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # استخدام المتغيرات العالمية
    global model, tokenizer, class_names, le
    
    verse = update.message.text.strip()
    if model is None:
        await update.message.reply_text("❌ عذراً، النموذج لم يُحمل بعد بشكل صحيح.")
        return

    try:
        cleaned = normalize_arabic(verse)
        charred = split_chars(cleaned)
        seq = tokenizer.texts_to_sequences([charred])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        pred = model.predict(padded, verbose=0)
        
        class_id = np.argmax(pred)
        confidence = np.max(pred) * 100
        label_val = le.inverse_transform([class_id])[0]
        meter_en = class_names[label_val]
        meter_ar = METERS_AR.get(meter_en, meter_en)
        rhyme = extract_rhyme(verse)

        response = (
            f"📜 *البيت:* \n_{verse}_\n\n"
            f"🎵 *البحر:* {meter_ar}\n"
            f"🎯 *الروي:* {rhyme}\n"
            f"📊 *الثقة:* {confidence:.1f}%"
        )
        await update.message.reply_text(response, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        await update.message.reply_text("❌ حدث خطأ أثناء التحليل.")

# --- 5. التشغيل ---
if __name__ == "__main__":
    token = "8402505295:AAGplKwmq7GBB_dSYE64eGlSXQ6BpVYxXKQ"
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("أهلاً بك! أرسل بيتاً لتحليله.")))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_verse))
    app.run_polling()
