# --- START OF ULTIMATE PRICE + IMAGE LOCKED BOT (FIXED & SECURED) ---

import logging
import os
import re
import json
import asyncio
import time
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from dotenv import load_dotenv

# مكتبات معالجة ومطابقة الصور
from PIL import Image
import imagehash
from io import BytesIO

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
# تأكد من وجود ملف aliexpress_utils.py في نفس المجلد
from aliexpress_utils import get_product_details_by_id

load_dotenv()

# --- Environment Variables Loading ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALIEXPRESS_APP_KEY = os.getenv('ALIEXPRESS_APP_KEY')
ALIEXPRESS_APP_SECRET = os.getenv('ALIEXPRESS_APP_SECRET')
TARGET_CURRENCY = os.getenv('TARGET_CURRENCY', 'USD')
TARGET_LANGUAGE = os.getenv('TARGET_LANGUAGE', 'en')
QUERY_COUNTRY = os.getenv('QUERY_COUNTRY', 'US')
ALIEXPRESS_TRACKING_ID = os.getenv('ALIEXPRESS_TRACKING_ID', 'default')
ALIEXPRESS_API_URL = os.getenv('ALIEXPRESS_API_URL', 'https://api-sg.aliexpress.com/sync')

QUERY_FIELDS = 'product_main_image_url,target_sale_price,product_title,target_sale_price_currency'
CACHE_EXPIRY_SECONDS = 1 * 24 * 60 * 60
MAX_WORKERS = 10

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID]):
    logger.error("Error: Missing required environment variables.")
    exit()

try:
    aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
except Exception as e:
    logger.exception(f"Error initializing AliExpress API client: {e}")
    exit()

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)
SHORT_LINK_DOMAIN_REGEX = re.compile(r'https?://(?:s\.click\.aliexpress\.com/e/|a\.aliexpress\.com/_)[a-zA-Z0-9_-]+/?', re.IGNORECASE)
COMBINED_DOMAIN_REGEX = re.compile(r'aliexpress\.com|s\.click\.aliexpress\.com|a\.aliexpress\.com', re.IGNORECASE)

class CacheWithExpiry:
    def __init__(self, expiry_seconds):
        self.cache = {}
        self.expiry_seconds = expiry_seconds
        self._lock = asyncio.Lock()
    async def get(self, key):
        async with self._lock:
            if key in self.cache:
                item, timestamp = self.cache[key]
                if time.time() - timestamp < self.expiry_seconds: return item
                else: del self.cache[key]
            return None
    async def set(self, key, value):
        async with self._lock: self.cache[key] = (value, time.time())
    async def clear_expired(self):
        async with self._lock:
            current_time = time.time()
            expired_keys = [k for k, (_, t) in self.cache.items() if current_time - t >= self.expiry_seconds]
            for key in expired_keys:
                try: del self.cache[key]
                except KeyError: pass

product_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)
link_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)
resolved_url_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)

# --- دالة استخراج بصمة الصورة الذكية (تم إصلاح ثغرة الرابط البديل) ---
async def get_image_hash(image_url: str, session: aiohttp.ClientSession) -> imagehash.ImageHash or None:
    if not image_url: return None
    
    # 🛠️ إصلاح فوري لروابط ألي إكسبريس التي تأتي بدون بروتوكول حماية
    if image_url.startswith('//'):
        image_url = 'https:' + image_url
        
    try:
        async with session.get(image_url, timeout=8) as response:
            if response.status == 200:
                image_bytes = await response.read()
                image = Image.open(BytesIO(image_bytes))
                return imagehash.dhash(image)
    except Exception as e:
        logger.error(f"Error hashing image {image_url}: {e}")
    return None

def extract_clean_price(price_raw) -> float:
    if not price_raw: return 0.0
    try:
        price_str = str(price_raw).replace(',', '.')
        clean_str = re.sub(r'[^\d.]', '', price_str)
        if clean_str.count('.') > 1:
            parts = clean_str.split('.')
            clean_str = parts[0] + '.' + ''.join(parts[1:])
        return float(clean_str) if clean_str else 0.0
    except Exception: return 0.0

def clean_keywords(title: str) -> str:
    title_clean = re.sub(r'\[.*?\]|\(.*?\)', '', title.lower())
    title_clean = re.sub(r'[^\w\s-]', ' ', title_clean)
    words = title_clean.split()
    stop_words = {
        'with', 'for', 'from', 'and', 'the', 'new', 'original', 'version', 'global', 
        'shipping', 'free', 'led', 'lcd', 'to', 'official', 'store', 'brand', 'top', 
        'hot', 'sale', 'promotion', 'choice', 'high', 'quality', '2024', '2025', '2026',
        'in', 'on', 'at', 'by', 'an', 'a', 'of', 'fast', 'quick', 'charging', 'phone', 'compatible'
    }
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    
    # 🛠️ تم زيادة الكلمات المفتاحية إلى 6 كلمات لضمان التقاط نوع المنتج (شاحن، كابل، إلخ)
    if len(filtered_words) >= 2: return " ".join(filtered_words[:6])
    return " ".join(words[:6])

# --- نظام الفلترة الفولاذي المطور والمانع للإكسسوارات البديلة ---
async def filter_strict_alternatives_v3(orig_title: str, orig_price_raw, orig_image_url: str, search_products: list, orig_id: str, session: aiohttp.ClientSession) -> list:
    if not search_products: return []
    
    orig_price = extract_clean_price(orig_price_raw)
    if orig_price <= 0.5: return []

    orig_hash = await get_image_hash(orig_image_url, session)
    valid_products = []
    seen_ids = set()
    orig_title_lower = orig_title.lower()
    
    strict_negative_keywords = {
        'bag', 'case', 'box', 'sticker', 'strap', 'holder', 'protector', 'rope', 'organizer', 'cover', 'pouch', 'film', 'glass', 'stand',
        'حقيبة', 'كفر', 'ملصق', 'حامل', 'حماية', 'جراب', 'منظم', 'صندوق', 'حبل', 'سير', 'شاشة', 'زجاج'
    }

    for p in search_products:
        p_id = str(p.get('product_id') or '')
        if not p_id or p_id in seen_ids or p_id == orig_id: continue
            
        p_title_lower = p.get('product_title', '').lower()
        p_price = extract_clean_price(p.get('target_sale_price'))
        p_image_url = p.get('product_main_image_url')
        
        if p_price <= 0.1: continue

        # 1. فلتر السعر الصارم (12%)
        min_allowed_price = orig_price * 0.88  
        max_allowed_price = orig_price * 1.10  
        if p_price < min_allowed_price or p_price > max_allowed_price: continue

        # 2. فلتر الكلمات المانعة الأساسية
        if any(neg in p_title_lower for neg in strict_negative_keywords):
            if not any(neg in orig_title_lower for neg in strict_negative_keywords): continue

        # 3. 🛡️ فحص الفئة النصية الصارم (منع كابل يحاكي شاحن)
        if 'charger' in orig_title_lower or 'شاحن' in orig_title_lower:
            if not any(w in p_title_lower for w in ['charger', 'شاحن', 'adapter', 'power', 'gan', '🔌']):
                continue

        if 'cable' in orig_title_lower or 'كابل' in orig_title_lower or 'سلك' in orig_title_lower:
            # إذا كان المنتج الأصلي عبارة عن كابل فقط، والبديل ليس فيه كابل
            if not any(w in p_title_lower for w in ['cable', 'كابل', 'سلك', 'cord', 'line']):
                continue

        # 4. 👁️ حصن الأمان البصري المشدد (تم تضييق النطاق إلى 8 لمطابقة صارمة جداً)
        if orig_hash and p_image_url:
            p_hash = await get_image_hash(p_image_url, session)
            if p_hash:
                if (orig_hash - p_hash) > 8:
                    logger.info(f"Product {p_id} rejected due to image mismatch hash distance: {orig_hash - p_hash}")
                    continue

        valid_products.append(p)
        seen_ids.add(p_id)

    valid_products.sort(key=lambda x: extract_clean_price(x.get('target_sale_price')))
    return valid_products

async def resolve_short_link(short_url: str, session: aiohttp.ClientSession) -> str | None:
    cached_final_url = await resolved_url_cache.get(short_url)
    if cached_final_url: return cached_final_url
    try:
        async with session.get(short_url, allow_redirects=True, timeout=10) as response:
            if response.status == 200 and response.url:
                final_url = str(response.url)
                if '.aliexpress.us' in final_url: final_url = final_url.replace('.aliexpress.us', '.aliexpress.com')
                product_id = extract_product_id(final_url)
                if STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(final_url) and product_id:
                    await resolved_url_cache.set(short_url, final_url)
                    return final_url
            return None
    except Exception: return None

def extract_product_id(url: str) -> str | None:
    if '.aliexpress.us' in url: url = url.replace('.aliexpress.us', '.aliexpress.com')
    match = PRODUCT_ID_REGEX.search(url)
    if match: return match.group(1)
    alt_patterns = [r'/p/[^/]+/([0-9]+)\.html', r'product/([0-9]+)']
    for pattern in alt_patterns:
        alt_match = re.search(pattern, url)
        if alt_match: return alt_match.group(1)
    return None

def extract_potential_aliexpress_urls(text: str) -> list[str]:
    return URL_REGEX.findall(text)

def clean_aliexpress_url(url: str, product_id: str) -> str | None:
    try:
        parsed_url = urlparse(url)
        path_segment = f'/item/{product_id}.html'
        return urlunparse((parsed_url.scheme or 'https', "www.aliexpress.com", path_segment, '', '', ''))
    except ValueError: return None

async def periodic_cache_cleanup(context: ContextTypes.DEFAULT_TYPE):
    await product_cache.clear_expired()
    await link_cache.clear_expired()
    await resolved_url_cache.clear_expired()

async def fetch_product_details_v2(product_id: str) -> dict | None:
    cached_data = await product_cache.get(product_id)
    if cached_data: return cached_data

    def _execute_api_call():
        try:
            request = iop.IopRequest('aliexpress.affiliate.productdetail.get')
            request.add_api_param('fields', QUERY_FIELDS)
            request.add_api_param('product_ids', product_id)
            request.add_api_param('target_currency', TARGET_CURRENCY)
            request.add_api_param('target_language', TARGET_LANGUAGE)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            request.add_api_param('country', QUERY_COUNTRY)
            return aliexpress_client.execute(request)
        except Exception: return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_api_call)
    if not response or not response.body: return None

    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)
        if 'error_response' in response_data: return None
        result = response_data.get('aliexpress_affiliate_productdetail_get_response', {}).get('resp_result', {})
        if result.get('resp_code') != 200: return None
        products = result.get('result', {}).get('products', {}).get('product', [])
        if not products:
            try:
                 scraped_name, scraped_image = await loop.run_in_executor(executor, get_product_details_by_id, product_id)
                 if scraped_name:
                      product_info = {'title': scraped_name, 'image_url': scraped_image, 'price': None, 'currency': TARGET_CURRENCY, 'source': 'Scraped'}
                      await product_cache.set(product_id, product_info)
                      return product_info
                 return None
            except Exception: return None

        product_data = products[0]
        product_info = {
            'image_url': product_data.get('product_main_image_url'),
            'price': product_data.get('target_sale_price'), 
            'currency': product_data.get('target_sale_price_currency', TARGET_CURRENCY),
            'title': product_data.get('product_title', f'منتج {product_id}'),
            'source': 'API'
        }
        await product_cache.set(product_id, product_info)
        return product_info
    except Exception: return None

async def fetch_alternative_cheapest_products(title: str, sort_mode: str = 'SALE_PRICE_ASC') -> list:
    cleaned_query = clean_keywords(title)
    def _execute_query_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.product.query')
            request.add_api_param('keywords', cleaned_query)
            request.add_api_param('target_currency', TARGET_CURRENCY)
            request.add_api_param('target_language', TARGET_LANGUAGE)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            request.add_api_param('ship_to_country', QUERY_COUNTRY)
            request.add_api_param('sort', sort_mode)
            request.add_api_param('page_size', '40')
            return aliexpress_client.execute(request)
        except Exception: return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_query_api)
    if not response or not response.body: return []
    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)
        result = response_data.get('aliexpress_affiliate_product_query_response', {}).get('resp_result', {})
        if result.get('resp_code') != 200: return []
        return result.get('result', {}).get('products', {}).get('product', [])
    except Exception: return []

async def generate_affiliate_links_batch(target_urls: list[str]) -> dict[str, str | None]:
    results_dict = {url: await link_cache.get(url) for url in target_urls}
    uncached_urls = [url for url, cached in results_dict.items() if not cached]
    if not uncached_urls: return results_dict
    source_values_str = ",".join(uncached_urls)
    def _execute_batch_link_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.link.generate')
            request.add_api_param('promotion_link_type', '0')
            request.add_api_param('source_values', source_values_str)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            return aliexpress_client.execute(request)
        except Exception: return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_batch_link_api)
    if not response or not response.body: return results_dict
    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)
        links_data = response_data.get('aliexpress_affiliate_link_generate_response', {}).get('resp_result', {}).get('result', {}).get('promotion_links', {}).get('promotion_link', [])
        api_returned_links_map = {}
        for link_info in links_data:
            if isinstance(link_info, dict):
                src = link_info.get('source_value')
                promo = link_info.get('promotion_link')
                if src and promo: api_returned_links_map[src] = promo
        for url in uncached_urls:
            if url in api_returned_links_map:
                promo_link = api_returned_links_map[url]
                results_dict[url] = promo_link
                await link_cache.set(url, promo_link)
        return results_dict
    except Exception: return results_dict

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("<b>👋 مرحبًا بك في بوت المقارنة البصري المطور لـ AliExpress!🚀</b>", parse_mode=ParseMode.HTML)

async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE, session: aiohttp.ClientSession):
    chat_id = update.effective_chat.id
    import html
    try:
        product_details = await fetch_product_details_v2(product_id)
        if not product_details or product_details.get('source') == 'None':
             await context.bot.send_message(chat_id=chat_id, text=f"<b>❌ تعذر استرداد بيانات المنتج من AliExpress.</b>", parse_mode=ParseMode.HTML)
             return

        title = product_details.get('title', 'منتج AliExpress')
        orig_price_raw = product_details.get('price')
        orig_image_url = product_details.get('image_url')
        orig_price_num = extract_clean_price(orig_price_raw)

        raw_alternatives = await fetch_alternative_cheapest_products(title, sort_mode='SALE_PRICE_ASC')
        filtered_alternatives = await filter_strict_alternatives_v3(title, orig_price_raw, orig_image_url, raw_alternatives, product_id, session)
        
        if len(filtered_alternatives) < 3:
            raw_backups = await fetch_alternative_cheapest_products(title, sort_mode='VOLUME_HIGH_2_LOW')
            backups_filtered = await filter_strict_alternatives_v3(title, orig_price_raw, orig_image_url, raw_backups, product_id, session)
            for b in backups_filtered:
                if b not in filtered_alternatives: filtered_alternatives.append(b)

        final_offers_data = []
        for p in filtered_alternatives[:4]:
            p_url = p.get('product_detail_url')
            p_price = extract_clean_price(p.get('target_sale_price'))
            if p_url: final_offers_data.append({"url": p_url, "price": p_price, "is_original": False})

        if not any(item['url'] == base_url for item in final_offers_data):
            final_offers_data.append({"url": base_url, "price": orig_price_num, "is_original": True})

        final_offers_data.sort(key=lambda x: x['price'] if x['price'] > 0 else 999999)
        urls_to_convert = [item['url'] for item in final_offers_data[:4]]
        generated_links_batch = await generate_affiliate_links_batch(urls_to_convert)

        labels_pool = [
            "<b>🥇 الخيار الأول (أفضل بائع بديل ومطابق) 🏆 بـ : ({price_val} $) 🔥</b>",
            "<b>🥈 الخيار الثاني (عرض متجر بديل) 🚀 بـ : ({price_val} $) 🔥</b>",
            "<b>🥉 الخيار الثالث (عرض منافس ممتاز) ⚡️ بـ : ({price_val} $) 🔥</b>",
            "<b>🏅 الخيار الرابع (عرض متوفر إضافي) ✨ بـ : ({price_val} $) 🔥</b>"
        ]

        final_offers = []
        for i, item in enumerate(final_offers_data[:4]):
            aff_link = generated_links_batch.get(item['url']) or item['url']
            price_display = f"{item['price']:.2f}" if item['price'] > 0 else "غير معروف"
            if item.get('is_original'):
                label_text = f"<b>📍 الرابط الأصلي الذي أرسلته أنت 🌟 بـ : ({price_display} $) 🔥</b>"
            else:
                label_text = labels_pool[i].format(price_val=price_display)
            final_offers.append({"label": label_text, "link": aff_link})

        message_lines = []
        product_title = html.escape(title)
        message_lines.append(f"<b>📝 إسم المنتج : {product_title[:250]}</b>")
        if orig_price_num <= 0.5:
             message_lines.append("<b>\n⚠️ لم نتمكن من التقاط سعر المنتج الأصلي لتصفية العروض بدقة.</b>")
        else:
             message_lines.append("<b>\n🎯 تم جلب هذه الأسعار بمطابقة صارمة للسعر (12%) والفئة والصورة البصرية ⬇️🤩\n</b>")
             for offer in final_offers:
                 message_lines.append(f"{offer['label']}\n{html.escape(offer['link'])}\n")
            
        message_lines.append("<b>✅ شارك البوت مع أصدقائك ليستفيد الجميع⚡️🤖</b>")
        
        if orig_image_url:
            try: 
                if orig_image_url.startswith('//'): orig_image_url = 'https:' + orig_image_url
                await context.bot.send_photo(chat_id=chat_id, photo=orig_image_url, caption="\n".join(message_lines), parse_mode=ParseMode.HTML)
            except Exception: await context.bot.send_message(chat_id=chat_id, text="\n".join(message_lines), parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        else:
            await context.bot.send_message(chat_id=chat_id, text="\n".join(message_lines), parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Error in process_product_telegram: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    message_text = update.message.text
    chat_id = update.effective_chat.id
    potential_urls = extract_potential_aliexpress_urls(message_text)
    if not potential_urls: return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loading_sticker_msg = None
    try: loading_sticker_msg = await context.bot.send_sticker(chat_id, "CAACAgIAAxkBAAIU1GYOk5jWvCvtykd7TZkeiFFZRdUYAAIjAAMoD2oUJ1El54wgpAY0BA")
    except Exception: pass

    processed_product_ids = set()
    async with aiohttp.ClientSession() as session:
        for url in potential_urls:
            product_id = None
            base_url = None
            if not url.startswith(('http://', 'https://')):
                 if COMBINED_DOMAIN_REGEX.search(url): url = f"https://{url}"
                 else: continue

            if STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(url):
                product_id = extract_product_id(url)
                if product_id: base_url = clean_aliexpress_url(url, product_id)
            elif SHORT_LINK_DOMAIN_REGEX.match(url):
                final_url = await resolve_short_link(url, session)
                if final_url:
                    product_id = extract_product_id(final_url)
                    if product_id: base_url = clean_aliexpress_url(final_url, product_id)

            if product_id and base_url and product_id not in processed_product_ids:
                processed_product_ids.add(product_id)
                await process_product_telegram(product_id, base_url, update, context, session)

    if loading_sticker_msg:
        try: await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
        except Exception: pass

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler((filters.TEXT | filters.FORWARDED) & ~filters.COMMAND & filters.Regex(COMBINED_DOMAIN_REGEX), handle_message))
    
    job_queue = application.job_queue
    job_queue.run_once(periodic_cache_cleanup, 60)
    job_queue.run_repeating(periodic_cache_cleanup, interval=timedelta(days=1), first=timedelta(days=1))

    logger.info("Starting Fixed Image + Price Locked Bot (12%)...")
    application.run_polling()

if __name__ == "__main__":
    main()

# --- END OF ULTIMATE PRICE + IMAGE LOCKED BOT (FIXED & SECURED) ---
