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
import html 

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
# تأكد من وجود هذا الملف في مجلد البوت
from aliexpress_utils import get_product_details_by_id

load_dotenv()

# --- الإعدادات والمتغيرات ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALIEXPRESS_APP_KEY = os.getenv('ALIEXPRESS_APP_KEY')
ALIEXPRESS_APP_SECRET = os.getenv('ALIEXPRESS_APP_SECRET')
TARGET_CURRENCY = os.getenv('TARGET_CURRENCY', 'USD')
TARGET_LANGUAGE = os.getenv('TARGET_LANGUAGE', 'en')
QUERY_COUNTRY = os.getenv('QUERY_COUNTRY', 'MA') # تم التعديل للمغرب أو حسب دولتك
ALIEXPRESS_TRACKING_ID = os.getenv('ALIEXPRESS_TRACKING_ID', 'default')
ALIEXPRESS_API_URL = 'https://api-sg.aliexpress.com/sync'
QUERY_FIELDS = 'product_main_image_url,target_sale_price,product_title,target_sale_price_currency'
CACHE_EXPIRY_SECONDS = 24 * 60 * 60
MAX_WORKERS = 10

# --- إعداد اللوج الإفتراضي ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- التحقق من المتغيرات الأساسية ---
if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET]):
    logger.error("Missing required environment variables.")
    exit()

# --- تهيئة عميل AliExpress ---
try:
    aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
except Exception as e:
    logger.exception(f"Error initializing API: {e}")
    exit()

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- كلاس التخزين المؤقت (Cache) لسرعة الأداء ---
class CacheWithExpiry:
    def __init__(self, expiry_seconds):
        self.cache = {}
        self.expiry_seconds = expiry_seconds
        self._lock = asyncio.Lock()

    async def get(self, key):
        async with self._lock:
            if key in self.cache:
                item, timestamp = self.cache[key]
                if time.time() - timestamp < self.expiry_seconds:
                    return item
                del self.cache[key]
            return None

    async def set(self, key, value):
        async with self._lock:
            self.cache[key] = (value, time.time())

product_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)

# --- دوال الاستخراج والتنسيق (تعديل لتطابق الصورة) ---

def format_bot_reply(product_title: str, links_dict: dict) -> str:
    """تنسيق الرسالة لتطابق الصورة المطلوبة تماماً"""
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

def extract_product_id(url: str) -> str | None:
    match = re.search(r'/item/(\d+)\.html', url)
    if match: return match.group(1)
    alt_match = re.search(r'product/([0-9]+)', url)
    return alt_match.group(1) if alt_match else None

# --- دالة جلب التفاصيل من AliExpress ---
async def fetch_product_details(product_id: str) -> dict | None:
    cached = await product_cache.get(product_id)
    if cached: return cached

    def _api_call():
        request = iop.IopRequest('aliexpress.affiliate.productdetail.get')
        request.add_api_param('fields', QUERY_FIELDS)
        request.add_api_param('product_ids', product_id)
        request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
        return aliexpress_client.execute(request)

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _api_call)
    
    try:
        data = json.loads(response.body)
        res = data['aliexpress_affiliate_productdetail_get_response']['resp_result']['result']['products']['product'][0]
        product_info = {
            'title': res.get('product_title', 'AliExpress Product'),
            'image': res.get('product_main_image_url'),
            'price': res.get('target_sale_price')
        }
        await product_cache.set(product_id, product_info)
        return product_info
    except:
        # محاولة السحب (Scraping) في حال فشل API
        try:
            title, image = get_product_details_by_id(product_id)
            return {'title': title, 'image': image, 'price': None}
        except: return None

# --- معالج الرسائل الرئيسي ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if not text: return
    
    # البحث عن روابط داخل الرسالة
    urls = re.findall(r'https?://[^\s<>"]+', text)
    for url in urls:
        if 'aliexpress' in url:
            # هنا يمكنك إضافة دالة لتحويل الروابط لروابط أفلييت خاصة بك
            # سأستخدم روابط تجريبية لتوضيح الشكل النهائي
            product_id = extract_product_id(url)
            if product_id:
                details = await fetch_product_details(product_id)
                if details:
                    # هذه الروابط يجب توليدها عبر API الأفلييت الخاص بك
                    fake_links = {
                        "coin": f"https://s.click.aliexpress.com/e/_links1",
                        "super": f"https://s.click.aliexpress.com/e/_links2",
                        "choice": f"https://s.click.aliexpress.com/e/_links3"
                    }
                    reply_text = format_bot_reply(details['title'], fake_links)
                    
                    if details['image']:
                        await update.message.reply_photo(photo=details['image'], caption=reply_text)
                    else:
                        await update.message.reply_text(reply_text)

# --- تشغيل البوت ---
def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot started...")
    application.run_polling()

if __name__ == '__main__':
    main()
