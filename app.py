# -*- coding: utf-8 -*-
import logging
import os
import re
import json
import asyncio
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse, urlencode
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.constants import ParseMode

import iop
# تأكد أن هذا الملف موجود في مجلد البوت
try:
    from aliexpress_utils import get_product_details_by_id
except ImportError:
    get_product_details_by_id = None

load_dotenv()

# --- الإعدادات ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALIEXPRESS_APP_KEY = os.getenv('ALIEXPRESS_APP_KEY')
ALIEXPRESS_APP_SECRET = os.getenv('ALIEXPRESS_APP_SECRET')
ALIEXPRESS_TRACKING_ID = os.getenv('ALIEXPRESS_TRACKING_ID', 'default')
ALIEXPRESS_API_URL = 'https://api-sg.aliexpress.com/sync'

# --- التنسيق المطلوب (نفس الصورة الأولى) ---
def format_bot_reply(product_title: str, links_dict: dict) -> str:
    text = f"📝 إسم المنتج : {product_title}\n\n"
    text += "✳️ قارن الأسعار واكتشف أرخص سعر للمنتج ⬇️ 😂\n\n"
    
    if links_dict.get("coin"):
        text += f"🟨 رابط الشراء بالعملات 🥇 :\n{links_dict['coin']}\n\n"
    if links_dict.get("super"):
        text += f"🟥 المنتج في SuperDeals 🚀 :\n{links_dict['super']}\n\n"
    if links_dict.get("limited"):
        text += f"⏰ المنتج في العرض المحدود بـ : 🔥\n{links_dict['limited']}\n\n"
    if links_dict.get("choice"):
        text += f"🏆 المنتج في عرض choice 🌟 :\n{links_dict['choice']}\n\n"
        
    text += "✅ شارك البوت مع أصدقاء ليستفيد الجميع ⚡ 🤖"
    return text

# --- تهيئة العميل ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
executor = ThreadPoolExecutor(max_workers=10)

def extract_product_id(url: str) -> str | None:
    match = re.search(r'/item/(\d+)\.html', url)
    if match: return match.group(1)
    alt = re.search(r'product/([0-9]+)', url)
    return alt.group(1) if alt else None

async def get_affiliate_links(product_url: str):
    """توليد روابط الأفلييت المختلفة"""
    # هنا نضع البارامترات الخاصة بـ Coin و SuperDeals كما في الكود الأصلي
    offers = {
        "coin": "620%26channel=coin",
        "super": "562",
        "limited": "561",
        "choice": "680"
    }
    generated_links = {}
    
    for key, val in offers.items():
        # ملاحظة: هنا يجب استدعاء aliexpress.affiliate.link.generate 
        # لتبسيط الكود، سنفترض أن الرابط يتم إنشاؤه بنجاح
        generated_links[key] = f"https://s.click.aliexpress.com/e/_example_{key}"
    
    return generated_links

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    
    url_match = re.search(r'https?://[^\s<>"]+', update.message.text)
    if not url_match: return
    
    raw_url = url_match.group(0)
    if 'aliexpress' not in raw_url: return

    # إظهار أن البوت "يكتب" أو "يرسل صورة"
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")

    product_id = extract_product_id(raw_url)
    if not product_id: return

    # جلب البيانات وتوليد الروابط
    def fetch_api():
        req = iop.IopRequest('aliexpress.affiliate.productdetail.get')
        req.add_api_param('product_ids', product_id)
        req.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
        return aliexpress_client.execute(req)

    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(executor, fetch_api)
    
    try:
        data = json.loads(res.body)
        product = data['aliexpress_affiliate_productdetail_get_response']['resp_result']['result']['products']['product'][0]
        title = product.get('product_title')
        image = product.get('product_main_image_url')
        
        links = await get_affiliate_links(raw_url)
        reply = format_bot_reply(title, links)
        
        await update.message.reply_photo(photo=image, caption=reply)
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text("عذراً، حدث خطأ في جلب بيانات هذا المنتج.")

def main():
    if not TELEGRAM_BOT_TOKEN:
        print("خطأ: TELEGRAM_BOT_TOKEN غير موجود في ملف .env")
        return
        
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("البوت يعمل الآن... أرسل رابطاً لتجربته.")
    app.run_polling()

if __name__ == '__main__':
    main()
