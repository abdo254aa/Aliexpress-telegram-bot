# -*- coding: utf-8 -*-
import logging
import os
import re
import asyncio
from urllib.parse import urlencode
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Language configuration
user_language = {}
LANGUAGES = {
    'en': {
        'welcome': "🌟 Welcome! Send me any AliExpress product link!",
        'choose_lang': "Please choose your language:",
        'processing': "⏳ Processing your link...",
        'error': "❌ Invalid link detected!",
        'price': "Price:",
        'currency': "USD",
        'offers': "🔥 Special Offers",
        'product_id': "Product ID",
        'rate': "Rate Us"
    },
    'ar': {
        'welcome': "🌟 مرحبًا! أرسل رابط منتج من AliExpress",
        'choose_lang': "الرجاء اختيار اللغة:",
        'processing': "⏳ جاري معالجة الرابط...",
        'error': "❌ رابط غير صحيح!",
        'price': "السعر:",
        'currency': "دولار",
        'offers': "🔥 عروض خاصة",
        'product_id': "رقم المنتج",
        'rate': "قيمنا"
    }
}

# Promotional offers configuration
OFFER_PARAMS = {
    "coin": {
        "name": {"en": "🪙 Coin Offers", "ar": "🪙 عروض العملات"},
        "params": {"sourceType": "620", "channel": "coin"}
    },
    "super": {
        "name": {"en": "🔥 Super Deals", "ar": "🔥 صفقات سوبر"},
        "params": {"sourceType": "562", "channel": "sd"}
    },
    "limited": {
        "name": {"en": "⏳ Limited Offers", "ar": "⏳ عروض محدودة"},
        "params": {"sourceType": "561", "channel": "limitedoffers"}
    }
}

LANGUAGE_KEYBOARD = InlineKeyboardMarkup([
    [
        InlineKeyboardButton("🇺🇸 English", callback_data='lang_en'),
        InlineKeyboardButton("🇸🇦 العربية", callback_data='lang_ar')
    ]
])

def generate_offer_links(base_url: str, lang: str) -> tuple:
    """Generate promotional offer links with buttons"""
    offers_text = []
    buttons = []
    
    for key, offer in OFFER_PARAMS.items():
        offer_url = f"{base_url}?{urlencode(offer['params'])}"
        offers_text.append(f"{offer['name'][lang]}: {offer_url}")
        buttons.append(InlineKeyboardButton(offer['name'][lang], url=offer_url))
    
    keyboard = [buttons[i:i+2] for i in range(0, len(buttons), 2)]
    return "\n".join(offers_text), InlineKeyboardMarkup(keyboard)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command"""
    await update.message.reply_text(
        text=LANGUAGES['en']['choose_lang'],
        reply_markup=LANGUAGE_KEYBOARD
    )

async def handle_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle language selection"""
    query = update.callback_query
    await query.answer()
    
    lang = query.data.split('_')[1]
    user_id = query.from_user.id
    user_language[user_id] = lang
    
    await query.edit_message_text(LANGUAGES[lang]['welcome'])

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main message handler"""
    user_id = update.message.from_user.id
    lang = user_language.get(user_id, 'en')
    message_text = update.message.text
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )
    
    if re.search(r'aliexpress\.com/item/\d+', message_text, re.IGNORECASE):
        product_id = re.search(r'/item/(\d+)', message_text).group(1)
        base_url = f"https://www.aliexpress.com/item/{product_id}.html"
        
        # Generate offers
        offers_text, offer_buttons = generate_offer_links(base_url, lang)
        
        response_text = (
            f"📦 {LANGUAGES[lang]['product_id']}: {product_id}\n"
            f"💵 {LANGUAGES[lang]['price']}: 29.99 {LANGUAGES[lang]['currency']}\n\n"
            f"{LANGUAGES[lang]['offers']}:\n{offers_text}"
        )
        
        offer_buttons.inline_keyboard.append([
            InlineKeyboardButton(f"⭐ {LANGUAGES[lang]['rate']}", callback_data='rate')
        ])
        
        await update.message.reply_text(
            text=response_text,
            reply_markup=offer_buttons,
            disable_web_page_preview=True
        )
    else:
        await update.message.reply_text(LANGUAGES[lang]['error'])

def main() -> None:
    """Start the bot"""
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(handle_language))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    application.run_polling()

if __name__ == '__main__':
    main()
