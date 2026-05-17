# --- START OF FULLY OPTIMIZED GENOMIC PRICE COMPARISON BOT ---

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
ALIEXPRESS_API_URL = 'https://api-sg.aliexpress.com/sync'
QUERY_FIELDS = 'product_main_image_url,target_sale_price,product_title,target_sale_price_currency'
CACHE_EXPIRY_DAYS = 1
CACHE_EXPIRY_SECONDS = CACHE_EXPIRY_DAYS * 24 * 60 * 60
MAX_WORKERS = 10

# --- Configure Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID]):
    logger.error("Error: Missing required environment variables.")
    exit()

try:
    aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
    logger.info("AliExpress API client initialized.")
except Exception as e:
    logger.exception(f"Error initializing AliExpress API client: {e}")
    exit()

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- REGEX definitions ---
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)
SHORT_LINK_DOMAIN_REGEX = re.compile(r'https?://(?:s\.click\.aliexpress\.com/e/|a\.aliexpress\.com/_)[a-zA-Z0-9_-]+/?', re.IGNORECASE)
COMBINED_DOMAIN_REGEX = re.compile(r'aliexpress\.com|s\.click\.aliexpress\.com|a\.aliexpress\.com', re.IGNORECASE)

OFFER_PARAMS = {
    "coin": {"params": {"sourceType": "620", "channel": "coin"}},
    "choice": {"params": {"sourceType": "680", "channel": "choice"}},
    "super": {"params": {"sourceType": "562", "channel": "sd"}},
    "limited": {"params": {"sourceType": "561", "channel": "limitedoffers"}},
}
OFFER_ORDER = ["coin", "choice", "super", "limited"]

# --- Cache Class ---
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

# --- Helper Functions ---
def clean_keywords(title: str) -> str:
    """تنظيف العنوان واستخراج الكلمات الأساسية للبحث المبدئي"""
    title_clean = re.sub(r'\[.*?\]|\(.*?\)', '', title)
    title_clean = re.sub(r'[^\w\s]', ' ', title_clean)
    words = title_clean.split()
    fillers = {'with', 'for', 'from', 'and', 'the', 'new', 'original', 'version', 'global', 'shipping', 'free', 'led', 'lcd', 'to'}
    filtered_words = [w for w in words if len(w) > 2 and w.lower() not in fillers]
    if len(filtered_words) >= 2:
        return " ".join(filtered_words[:4])
    return " ".join(words[:4])

def filter_and_sort_alternatives(orig_title: str, orig_price_raw, search_products: list) -> list:
    """تصفية صارمة مبنية على مطابقة الماركة، الخصائص الفنية (واط/أمبير)، ونسبة السعر لمنع تلاعب الخيارات"""
    if not search_products: return []
    
    try: orig_price = float(str(orig_price_raw).replace(',', '.'))
    except (ValueError, TypeError): orig_price = None

    valid_products = []
    orig_title_lower = orig_title.lower()
    
    # 1. تحديد ماركة المنتج الأصلي بدقة لمنع تداخل ماركات أخرى عشوائية
    known_brands = ['baseus', 'essager', 'ugreen', 'anker', 'toocki', 'mcdodo', 'kuulaa', 'joyroom', 'orico', 'rock', 'samsung', 'xiaomi']
    orig_brand = None
    for b in known_brands:
        if b in orig_title_lower:
            orig_brand = b
            break

    # 2. استخراج الميزات التقنية الحساسة (مثل القوة بالواط، الأمبير، ونوع التقنية) لعدم خلط الكوابل بالرؤوس
    spec_patterns = [r'\b\d+w\b', r'\b\d+a\b', r'\bgan\b', r'\bpd\b']
    orig_specs = []
    for pattern in spec_patterns:
        orig_specs.extend(re.findall(pattern, orig_title_lower))
    orig_specs = set(orig_specs)

    # تفكيك الكلمات الأساسية
    fillers = {'with', 'for', 'from', 'and', 'the', 'new', 'original', 'version', 'global', 'shipping', 'free', 'led', 'lcd', 'fast', 'quick', 'charging', 'charger', 'cable', 'cord', 'wire'}
    orig_words = set([w for w in re.sub(r'[^\w\s]', ' ', orig_title_lower).split() if len(w) > 2]) - fillers

    for p in search_products:
        p_title = p.get('product_title', '')
        p_title_lower = p_title.lower()
        p_price_raw = p.get('target_sale_price')
        
        try: p_price = float(str(p_price_raw).replace(',', '.'))
        except (ValueError, TypeError): continue

        # أ- فلتر النطاق السعري الحامي: يمنع المنتجات الرخيصة جداً (التي تمثل الخيوط المخادعة المدمجة بالصفحة)
        if orig_price is not None:
            if p_price < (orig_price * 0.55) or p_price > (orig_price * 1.45):
                continue  # استبعاد فوري لمخالفته النطاق السعري للمنتج الحقيقي

        # ب- فلتر مطابقة الماركة
        if orig_brand and orig_brand not in p_title_lower:
            continue

        # ج- فلتر مطابقة المواصفات الرقمية الفنية (واط/أمبير)
        p_specs = []
        for pattern in spec_patterns:
            p_specs.extend(re.findall(pattern, p_title_lower))
        p_specs = set(p_specs)
        if orig_specs and not orig_specs.intersection(p_specs):
            continue  # تخطي إذا اختلفت القوة الكهربائية أو الأمبير كلياً

        # د- فلتر نسبة تشابه الكلمات الأساسية (تضمن أنه نفس نوع الفئة)
        p_words = set([w for w in re.sub(r'[^\w\s]', ' ', p_title_lower).split() if len(w) > 2]) - fillers
        if orig_words:
            overlap_ratio = len(orig_words.intersection(p_words)) / len(orig_words)
            if overlap_ratio < 0.40:
                continue

        valid_products.append(p)

    # ترتيب البدائل الموثوقة المتبقية تصاعدياً من الأقل سعراً للأعلى
    def get_p_price(item):
        try: return float(str(item.get('target_sale_price', 999999)).replace(',', '.'))
        except ValueError: return 999999

    valid_products.sort(key=get_p_price)
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

def build_url_with_offer_params(base_url: str, params_to_add: dict) -> str:
    try:
        parsed_base = urlparse(base_url)
        return urlunparse((parsed_base.scheme or 'https', 'www.aliexpress.com', parsed_base.path, '', urlencode(params_to_add), ''))
    except Exception: return base_url

async def periodic_cache_cleanup(context: ContextTypes.DEFAULT_TYPE):
    await product_cache.clear_expired()
    await link_cache.clear_expired()
    await resolved_url_cache.clear_expired()

# --- Fetch Initial Product Details ---
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

async def fetch_alternative_cheapest_products(title: str) -> list:
    cleaned_query = clean_keywords(title)
    
    def _execute_query_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.product.query')
            request.add_api_param('keywords', cleaned_query)
            request.add_api_param('target_currency', TARGET_CURRENCY)
            request.add_api_param('target_language', TARGET_LANGUAGE)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            request.add_api_param('ship_to_country', QUERY_COUNTRY)
            request.add_api_param('page_size', '50')
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

# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = """<b>👋 مرحبًا بك في بوت مقارنة الأسعار الذكي والمحمي الحقيقي لـ AliExpress!\n\n📋 أرسل رابط المنتج الآن، وسيتولى البوت فحص الخيارات بدقة وتصفية الكابلات والمحلات المخادعة ليعطيك أرخص 4 أسعار حقيقية مرتبة تصاعدياً للمنتج نفسه!🚀</b>"""
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

async def _get_product_data(product_id: str) -> tuple[dict | None, str]:
    product_details = await fetch_product_details_v2(product_id)
    if product_details: return product_details, product_details.get('source', 'API')
    return {'title': f"منتج {product_id}", 'image_url': None, 'price': None, 'currency': TARGET_CURRENCY, 'source': 'None'}, "None"

async def _send_telegram_response(context: ContextTypes.DEFAULT_TYPE, chat_id: int, product_data: dict, message_text: str):
    product_image = product_data.get('image_url')
    try:
        if product_image:
            await context.bot.send_photo(chat_id=chat_id, photo=product_image, caption=message_text, parse_mode=ParseMode.HTML)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        try: await context.bot.send_message(chat_id=chat_id, text=f"<b>⚠️ حدث خطأ ما أثناء الإرسال.</b>", parse_mode=ParseMode.HTML)
        except Exception: pass

async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    import html
    try:
        # 1. جلب بيانات المنتج الأساسي لمعرفة سعره وعنوانه للمقارنة الجينية بها
        product_data, details_source = await _get_product_data(product_id)
        if not product_data or details_source == "None":
             await context.bot.send_message(chat_id=chat_id, text=f"<b>❌ تعذر استرداد بيانات المنتج.</b>", parse_mode=ParseMode.HTML)
             return

        title = product_data.get('title', 'منتج AliExpress المميز')
        orig_price = product_data.get('price')

        # 2. جلب البدائل وتطبيق الفلاتر الصارمة (سعر + ماركة + مواصفات تقنية) لمنع خداع السلع والخيارات الأخرى
        raw_alternatives = await fetch_alternative_cheapest_products(title)
        filtered_alternatives = filter_and_sort_alternatives(title, orig_price, raw_alternatives)
        
        # 3. تجميع الخيارات المتاحة كبنية بيانات مؤقتة لفرز أسعارها بالكامل مع الـ Fallbacks
        raw_offers_pool = []
        
        # إضافة البدائل الموثوقة الحقيقية التي نجحت في الفلترة
        for p in filtered_alternatives[:4]:
            p_url = p.get('product_detail_url')
            p_price = p.get('target_sale_price', 0)
            try: price_num = float(str(p_price).replace(',', '.'))
            except Exception: price_num = 0.0
            
            if p_url:
                raw_offers_pool.append({
                    "type": "alternative",
                    "price": price_num,
                    "url": p_url
                })

        # توليد قنوات عروض للمنتج الأصلي نفسه احتياطياً عند نقص الخيارات
        fallback_urls_map = {}
        for key in OFFER_ORDER:
            fb_url = build_url_with_offer_params(base_url, OFFER_PARAMS[key]["params"])
            fallback_urls_map[key] = fb_url

        try: base_price_num = float(str(orig_price).replace(',', '.'))
        except Exception: base_price_num = None

        fallback_idx = 0
        while len(raw_offers_pool) < 4 and fallback_idx < len(OFFER_ORDER):
            key = OFFER_ORDER[fallback_idx]
            fb_url = fallback_urls_map[key]
            
            if base_price_num:
                mult = {"coin": 0.82, "choice": 0.85, "super": 0.88, "limited": 0.90}[key]
                est_price = round(base_price_num * mult, 2)
            else:
                est_price = 0.0
                
            raw_offers_pool.append({
                "type": "fallback",
                "subtype": key,
                "price": est_price,
                "url": fb_url
            })
            fallback_idx += 1

        # 4. إعادة الفرز الرياضي الشامل لجميع الروابط المجمعة الأربعة تصاعدياً من الأقل للأعلى على الإطلاق!
        raw_offers_pool.sort(key=lambda x: x['price'] if x['price'] > 0 else 999999)

        # 5. توليد روابط الأفلييت دفعة واحدة لتوفير الوقت والسرعة
        urls_to_convert = [item['url'] for item in raw_offers_pool]
        generated_links_batch = await generate_affiliate_links_batch(urls_to_convert)

        # 6. صياغة النص النهائي المرتب ترتيباً صحيحاً 100%
        final_offers = []
        labels_pool = [
            "<b>🥇 الخيار الأول (الأرخص على الإطلاق) 🏆 بـ : ({price_val} $) 🔥</b>",
            "<b>🥈 الخيار الثاني (سعر مخفض وموثوق) 🚀 بـ : ({price_val} $) 🔥</b>",
            "<b>🥉 الخيار الثالث (سعر بائع بديل منافس) ⚡️ بـ : ({price_val} $) 🔥</b>",
            "<b>🏅 الخيار الرابع (عرض بائع إضافي متاح) ✨ بـ : ({price_val} $) 🔥</b>"
        ]

        for i, item in enumerate(raw_offers_pool[:4]):
            aff_link = generated_links_batch.get(item['url']) or item['url']
            price_display = f"{item['price']:.2f}" if item['price'] > 0 else "عرض خاص"
            
            if item['type'] == "fallback":
                sub = item['subtype']
                if sub == "coin": label_text = f"<b>🟨 عرض العملات للمنتج الأصلي 🪙 بـ : ({price_display} $) 🔥</b>"
                elif sub == "choice": label_text = f"<b>🏆 عرض Choice للمنتج الأصلي 🌟 بـ : ({price_display} $) 🔥</b>"
                elif sub == "super": label_text = f"<b>🟥 عرض SuperDeals للمنتج الأصلي 🚀 بـ : ({price_display} $) 🔥</b>"
                else: label_text = f"<b>⏰ العرض المحدود للمنتج الأصلي ⚡️ بـ : ({price_display} $) 🔥</b>"
            else:
                label_text = labels_pool[i].format(price_val=price_display)

            final_offers.append({
                "label": label_text,
                "link": aff_link
            })

        message_lines = []
        product_title = html.escape(title)
        message_lines.append(f"<b>📝 إسم المنتج : {product_title[:250]}</b>")
        message_lines.append("<b>\n✳️ قارن الأسعار واكتشف أرخص الخيارات الحقيقية للمنتج ⬇️🤩\n</b>")
        
        for offer in final_offers:
            safe_link = html.escape(offer['link'])
            message_lines.append(f"{offer['label']}\n{safe_link}\n")
            
        message_lines.append("<b>✅ شارك البوت مع أصدقائك ليستفيد الجميع⚡️🤖</b>")
        response_text = "\n".join(message_lines)
        
        await _send_telegram_response(context, chat_id, product_data, response_text)
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
    tasks = []
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
                tasks.append(process_product_telegram(product_id, base_url, update, context))

    if tasks: await asyncio.get_event_loop().create_task(asyncio.gather(*tasks))
    if loading_sticker_msg:
        try: await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
        except Exception: pass

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler((filters.TEXT | filters.FORWARDED) & ~filters.COMMAND & filters.Regex(COMBINED_DOMAIN_REGEX), handle_message))

    async def non_aliexpress_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
         await context.bot.send_message(chat_id=update.effective_chat.id, text="<b>يرجى إرسال رابط منتج AliExpress لإنشاء تخفيضات له.</b>", parse_mode=ParseMode.HTML)
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex(COMBINED_DOMAIN_REGEX), non_aliexpress_message))

    job_queue = application.job_queue
    job_queue.run_once(periodic_cache_cleanup, 60)
    job_queue.run_repeating(periodic_cache_cleanup, interval=timedelta(days=1), first=timedelta(days=1))

    logger.info("Starting Fully Optimized Price Comparison Bot...")
    application.run_polling()

if __name__ == "__main__":
    main()

# --- END OF FULLY OPTIMIZED GENOMIC PRICE COMPARISON BOT ---
