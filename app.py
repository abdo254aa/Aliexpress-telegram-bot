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
    JobQueue,
)
from telegram.constants import ParseMode, ChatAction

import iop
from aliexpress_utils import get_product_details_by_id

# --- تحميل الإعدادات ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALIEXPRESS_APP_KEY = os.getenv('ALIEXPRESS_APP_KEY')
ALIEXPRESS_APP_SECRET = os.getenv('ALIEXPRESS_APP_SECRET')
TARGET_CURRENCY = os.getenv('TARGET_CURRENCY', 'USD')
TARGET_LANGUAGE = os.getenv('TARGET_LANGUAGE', 'en')
QUERY_COUNTRY = os.getenv('QUERY_COUNTRY', 'US')
ALIEXPRESS_TRACKING_ID = os.getenv('ALIEXPRESS_TRACKING_ID', 'default')
ALIEXPRESS_API_URL = 'https://api-sg.aliexpress.com/sync'
QUERY_FIELDS = 'product_main_image_url,target_sale_price,product_title,target_sale_price_currency'
CACHE_EXPIRY_DAYS = 1
CACHE_EXPIRY_SECONDS = CACHE_EXPIRY_DAYS * 24 * 60 * 60
MAX_WORKERS = 10

# --- إعداد الـ Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID]):
    logger.error("Missing required environment variables.")
    exit()

# --- تهيئة عميل AliExpress ---
try:
    aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
    logger.info("AliExpress API client initialized.")
except Exception as e:
    logger.exception(f"Error initializing AliExpress API client: {e}")
    exit()

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- أنماط Regex المتقدمة ---
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)

# --- كلاس الكاش الاحترافي ---
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
                else:
                    del self.cache[key]
            return None

    async def set(self, key, value):
        async with self._lock:
            self.cache[key] = (value, time.time())

    async def clear_expired(self):
        async with self._lock:
            current_time = time.time()
            expired_keys = [k for k, (_, t) in self.cache.items() if current_time - t >= self.expiry_seconds]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)

product_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)
link_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)
resolved_url_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)

# --- الدوال المساعدة للتنسيق والروابط ---
def build_url_with_offer_params(base_url: str, params_to_add: dict) -> str:
    parsed_base = urlparse(base_url)
    query_string = urlencode(params_to_add)
    redirect_url = urlunparse((parsed_base.scheme, parsed_base.netloc, parsed_base.path, '', query_string, ''))
    final_params = {"platform": "AE", "businessType": "ProductDetail", "redirectUrl": redirect_url}
    return urlunparse(('https', 'star.aliexpress.com', '/share/share.htm', '', urlencode(final_params), ''))

def get_formatted_text(product_info, links_dict):
    text = f"📝 إسم المنتج : {product_info.get('title', 'غير معروف')}\n"
    text += f"💰 السعر : {product_info.get('price', 'N/A')} {product_info.get('currency', '')}\n\n"
    
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

# --- معالجة الروابط واستخراج البيانات ---
async def resolve_short_link(short_url: str, session: aiohttp.ClientSession) -> str | None:
    cached = await resolved_url_cache.get(short_url)
    if cached: return cached
    try:
        async with session.get(short_url, allow_redirects=True, timeout=10) as response:
            if response.status == 200:
                final_url = str(response.url)
                await resolved_url_cache.set(short_url, final_url)
                return final_url
    except Exception as e:
        logger.error(f"Error resolving {short_url}: {e}")
    return None

async def fetch_product_details_v2(product_id: str) -> dict | None:
    cached = await product_cache.get(product_id)
    if cached: return cached

    def _call():
        request = iop.IopRequest('aliexpress.affiliate.productdetail.get')
        request.add_api_param('fields', QUERY_FIELDS)
        request.add_api_param('product_ids', product_id)
        return aliexpress_client.execute(request)

    response = await asyncio.get_event_loop().run_in_executor(executor, _call)
    if not response or not response.body: return None
    
    data = json.loads(response.body)
    res = data.get('aliexpress_affiliate_productdetail_get_response', {}).get('resp_result', {}).get('result', {}).get('products', {}).get('product', [])
    
    if res:
        p = res[0]
        info = {'title': p.get('product_title'), 'price': p.get('target_sale_price'), 'currency': p.get('target_sale_price_currency')}
        await product_cache.set(product_id, info)
        return info
    return None

# --- معالج الرسائل الرئيسي ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    
    urls = URL_REGEX.findall(update.message.text)
    if not urls: return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    async with aiohttp.ClientSession() as session:
        for url in urls:
            full_url = url
            if 'a.aliexpress.com' in url or 's.click' in url:
                full_url = await resolve_short_link(url, session)
            
            if not full_url: continue
            
            product_id_match = PRODUCT_ID_REGEX.search(full_url)
            if product_id_match:
                pid = product_id_match.group(1)
                product = await fetch_product_details_v2(pid)
                if product:
                    base = f"https://www.aliexpress.com/item/{pid}.html"
                    links = {
                        "coin": build_url_with_offer_params(base, {"sourceType": "620"}),
                        "super": build_url_with_offer_params(base, {"sourceType": "562"}),
                        "limited": build_url_with_offer_params(base, {"sourceType": "561"}),
                        "choice": build_url_with_offer_params(base, {"sourceType": "680"})
                    }
                    await update.message.reply_text(get_formatted_text(product, links))
                    break

# --- تشغيل البوت ---
async def post_init(application: Application):
    await application.bot.delete_webhook(drop_pending_updates=True)

if __name__ == '__main__':
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.run_polling()
