# -*- coding: utf-8 -*-
import re
import json
import os
import logging
from urllib.parse import urlencode
from pathlib import Path
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)
from dotenv import load_dotenv

# تحميل المتغيرات البيئية
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# إعدادات التسجيل
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# إعدادات التخزين
LANGUAGE_FILE = Path("user_languages.json")

# تحميل لغات المستخدمين
def load_languages():
    if LANGUAGE_FILE.exists():
        with open(LANGUAGE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_languages(data):
    with open(LANGUAGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

user_languages = load_languages()

# نصوص متعددة اللغات
LANGUAGES = {
    'en': {
        'welcome': "🌟 Welcome! Send any AliExpress product link",
        'choose_lang': "🌍 Please choose your language:",
        'error': "❌ Invalid link detected!",
        'processing': "⏳ Processing your link...",
        'product_id': "Product ID",
        'price': "Price",
        'currency': "USD",
        'offers': "🔥 Hot Offers"
    },
    'ar': {
        'welcome': "🌟 مرحبًا! أرسل رابط منتج من علي إكسبريس",
        'choose_lang': "🌍 الرجاء اختيار اللغة:",
        'error': "❌ تم اكتشاف رابط غير صحيح!",
        'processing': "⏳ جاري معالجة الرابط...",
        'product_id': "رقم المنتج",
        'price': "السعر",
        'currency': "دولار",
        'offers': "🔥 عروض ساخنة"
    }
}

# أزرار اختيار اللغة
language_keyboard = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🇺🇸 English", callback_data='lang_en'),
        InlineKeyboardButton("🇸🇦 العربية", callback_data='lang_ar')
    ]
])

# العروض الترويجية
OFFER_PARAMS = {
    "coin": {
        "name": {
            "en": "🪙 Coin Offers", 
            "ar": "🪙 عروض العملات"
        },
        "params": {"sourceType": "620", "channel": "coin"}
    },
    "super": {
        "name": {
            "en": "🔥 Super Deals", 
            "ar": "🔥 صفقات سوبر"
        },
        "params": {"sourceType": "562", "channel": "sd"}
    }
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة أمر /start"""
    user_id = str(update.effective_user.id)
    if user_id in user_languages:
        lang = user_languages[user_id]
        await update.message.reply_text(LANGUAGES[lang]['welcome'])
    else:
        await update.message.reply_text(
            LANGUAGES['en']['choose_lang'],
            reply_markup=language_keyboard
        )

async def handle_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة اختيار اللغة"""
    query = update.callback_query
    await query.answer()
    lang = query.data.split('_')[1]
    user_id = str(query.from_user.id)
    user_languages[user_id] = lang
    save_languages(user_languages)
    await query.edit_message_text(f"✅ Language set to {lang.upper()}!")
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=LANGUAGES[lang]['welcome']
    )

def generate_offers(base_url: str, lang: str) -> tuple:
    """إنشاء عروض ترويجية"""
    offers_text = []
    buttons = []
    for key, offer in OFFER_PARAMS.items():
        offer_url = f"{base_url}?{urlencode(offer['params'])}"
        offers_text.append(f"{offer['name'][lang]}: {offer_url}")
        buttons.append(InlineKeyboardButton(offer['name'][lang], url=offer_url))
    keyboard = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
    return "\n".join(offers_text), InlineKeyboardMarkup(keyboard)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالجة الرسائل"""
    user_id = str(update.effective_user.id)
    if user_id not in user_languages:
        await update.message.reply_text("🌍 Please choose language first / الرجاء اختيار اللغة أولاً!")
        return
    
    lang = user_languages[user_id]
    message_text = update.message.text
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )
    
    if 'aliexpress.com/item/' in message_text:
        try:
            product_id = re.search(r'/item/(\d+)', message_text).group(1)
            base_url = f"https://www.aliexpress.com/item/{product_id}.html"
            
            offers_text, offers_markup = generate_offers(base_url, lang)
            
            response_text = (
                f"📦 {LANGUAGES[lang]['product_id']}: {product_id}\n"
                f"💵 {LANGUAGES[lang]['price']}: 99.99 {LANGUAGES[lang]['currency']}\n\n"
                f"{LANGUAGES[lang]['offers']}:\n{offers_text}"
            )
            
            await update.message.reply_text(
                text=response_text,
                reply_markup=offers_markup,
                disable_web_page_preview=True
            )
            
        except Exception as e:
            logger.error(f"Error processing link: {e}")
            await update.message.reply_text(LANGUAGES[lang]['error'])
    else:
        await update.message.reply_text(LANGUAGES[lang]['error'])

def main():
    """تشغيل البوت"""
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(handle_language, pattern='^lang_'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == '__main__':
    main()
