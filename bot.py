import json
import os
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

TOKEN = "7713136112:AAFN3c3bPTIzOzlJK73YOGwZPbbdw-lFPio"

LANGUAGE_FILE = "user_languages.json"

# إنشاء ملف اللغات لو مش موجود
if not os.path.exists(LANGUAGE_FILE):
    with open(LANGUAGE_FILE, "w") as f:
        json.dump({}, f)

# حفظ لغة المستخدم
def save_user_language(user_id, language):
    with open(LANGUAGE_FILE, "r") as f:
        data = json.load(f)
    data[str(user_id)] = language
    with open(LANGUAGE_FILE, "w") as f:
        json.dump(data, f)

# الحصول على لغة المستخدم
def get_user_language(user_id):
    with open(LANGUAGE_FILE, "r") as f:
        data = json.load(f)
    return data.get(str(user_id), None)

# دالة بدء البوت
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [['🇸🇦 العربية', '🇺🇸 English']]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    text = "\ud83d\udc4b أهلاً بك!\n\nWelcome!\n\nاختر لغتك / Choose your language:"
    await update.message.reply_text(text, reply_markup=reply_markup)

# دالة معالجة الرسائل
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    text = update.message.text

    if text == '🇸🇦 العربية':
        save_user_language(user_id, 'ar')
        await update.message.reply_text('✅ تم اختيار اللغة العربية.')
    elif text == '🇺🇸 English':
        save_user_language(user_id, 'en')
        await update.message.reply_text('✅ English language selected.')
    else:
        lang = get_user_language(user_id)
        if not lang:
            keyboard = [['🇸🇦 العربية', '🇺🇸 English']]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
            await update.message.reply_text("❗ اختر لغتك أولاً:", reply_markup=reply_markup)
            return

        await process_user_message(update, context, lang)

# دالة الرد حسب اللغة
async def process_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE, lang: str):
    message_text = update.message.text

    if "aliexpress.com" in message_text:
        if lang == 'ar':
            await update.message.reply_text('🔗 تم استلام الرابط! جاري جلب الخصومات...')
        else:
            await update.message.reply_text('🔗 Link received! Fetching discounts...')
        # ضع هنا كود جلب بيانات الرابط من علي اكسبرس
    else:
        if lang == 'ar':
            await update.message.reply_text('❗ الرجاء إرسال رابط منتج من AliExpress.')
        else:
            await update.message.reply_text('❗ Please send a valid AliExpress product link.')

# تشغيل البوت
app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

app.run_polling()
