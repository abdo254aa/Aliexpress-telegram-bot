# -*- coding: utf-8 -*-
import logging
import os
import re
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# --- 1. الإعدادات ---
# تأكد من وضع التوكن في ملف .env
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# إعداد اللوج (السجلات) لمتابعة الأخطاء
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- 2. دالة تنسيق الرسالة ---
def get_formatted_text(product_title):
    # نستخدم f-string مع نصوص عربية مباشرة
    text = f"📝 إسم المنتج : {product_title}\n\n"
    text += "✳️ قارن الأسعار واكتشف أرخص سعر للمنتج ⬇️ 😂\n\n"
    text += "🟨 رابط الشراء بالعملات 🥇 :\n(ضع رابطك هنا)\n\n"
    text += "✅ شارك البوت مع أصدقاء ليستفيد الجميع ⚡ 🤖"
    return text

# --- 3. معالج الرسائل ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    # فحص بسيط لوجود رابط
    if "aliexpress" in text:
        # هنا ستضع كود جلب التفاصيل
        product_title = "مثال لاسم المنتج" 
        reply_text = get_formatted_text(product_title)
        
        await update.message.reply_text(reply_text)

# --- 4. الدالة الرئيسية (نقطة الانطلاق) ---
async def post_init(application):
    """تنظيف أي اتصالات قديمة لتجنب خطأ الـ Conflict"""
    await application.bot.delete_webhook(drop_pending_updates=True)

if __name__ == '__main__':
    if not TOKEN:
        print("خطأ: يرجى التأكد من إضافة TELEGRAM_BOT_TOKEN في إعدادات البيئة (Environment Variables)")
        exit()

    # بناء التطبيق مع زيادة الـ Timeouts لتجنب خطأ TimedOut
    app = ApplicationBuilder() \
        .token(TOKEN) \
        .read_timeout(60) \
        .write_timeout(60) \
        .connect_timeout(60) \
        .post_init(post_init) \
        .build()

    # إضافة المعالج
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    print("البوت يعمل الآن بدون أخطاء...")
    app.run_polling()
