# --- START OF ULTIMATE FIXED BOT WITH ANTI-SKU & ARABIC SEARCH TRAP ---

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

async def get_image_hash(image_url: str, session: aiohttp.ClientSession) -> imagehash.ImageHash or None:
    if not image_url: return None
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

# 🛠️ دالة تنظيف وتجهيز الكلمات المفتاحية المحدثة بالكامل لحل مشكلة البحث باللغة العربية
def clean_keywords(title: str) -> str:
    title_clean = re.sub(r'\[.*?\]|\(.*?\)|⚡|🔥|⭐|✨', '', title.lower())
    title_clean = re.sub(r'[^\w\s-]', ' ', title_clean)
    words = title_clean.split()
    
    # تحقق مما إذا كان العنوان يحتوي على حروف عربية
    is_arabic = bool(re.search(r'[\u0600-\u06FF]', title))
    
    if is_arabic:
        # الكلمات غير المفيدة في البيع للمستودع العربي
        arabic_fillers = {'جديد', 'جديدة', 'الموضة', 'موضة', 'عالي', 'الجودة', 'شحن', 'مجاني', 'وصول', 'من', 'في', 'على', 'مع', 'للحصول', 'تصميم'}
        filtered_words = [w for w in words if w not in arabic_fillers and len(w) > 2]
        # استخدام أول كلمتين فقط لأن محرك بحث AliExpress Affiliate يعطي نتائج ممتازة عند اختصار الكلمات العربية
        return " ".join(filtered_words[:2])
    
    # للمنتجات الإنجليزية
    stop_words = {'with', 'for', 'from', 'and', 'the', 'new', 'original', 'version', 'global', 'shipping', 'free', 'to', 'brand', 'official', 'store'}
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(filtered_words[:3])

# --- نظام الفلترة والمطابقة الفولاذي المطور والمانع للفخاخ ---
async def filter_strict_alternatives_v3(orig_title: str, orig_price_raw, orig_image_url: str, search_products: list, orig_id: str, session: aiohttp.ClientSession) -> list:
    if not search_products: return []
    
    orig_price = extract_clean_price(orig_price_raw)
    orig_title_lower = orig_title.lower()
    orig_hash = await get_image_hash(orig_image_url, session)
    
    # استخراج الأرقام المميزة للموديل أو القدرة (مثال: 240, 65, 100) لمنع خلط المنتجات
    orig_numbers = set(re.findall(r'\b\d{2,3}\b', orig_title_lower))
    orig_numbers = {n for n in orig_numbers if n not in ['2024', '2025', '2026']} # استبعاد السنوات

    valid_products = []
    seen_ids = set()
    
    strict_negative_keywords = {'bag', 'case', 'box', 'sticker', 'strap', 'holder', 'protector', 'cover', 'pouch', 'film', 'glass', 'حقيبة', 'كفر', 'ملصق', 'حامل', 'حماية', 'جراب'}

    for p in search_products:
        p_id = str(p.get('product_id') or '')
        if not p_id or p_id in seen_ids or p_id == orig_id: continue
            
        p_title_lower = p.get('product_title', '').lower()
        p_price = extract_clean_price(p.get('target_sale_price'))
        p_image_url = p.get('product_main_image_url')
        
        if p_price <= 0.1: continue

        # 1. فلتر السعر المرن قليلاً ليتناسب مع تقلبات أسعار الـ SKU داخل الموقع
        if orig_price > 0.5:
            min_allowed_price = orig_price * 0.40  
            max_allowed_price = orig_price * 1.30  
            if p_price < min_allowed_price or p_price > max_allowed_price: continue

        # 2. فلتر الكلمات المانعة الأساسية
        if any(neg in p_title_lower for neg in strict_negative_keywords):
            if not any(neg in orig_title_lower for neg in strict_negative_keywords): continue

        # 3. 🛡️ مطابقة الأرقام والمواصفات (مثل منع شاحن 65 واط من الدخول في عروض شاحن 240 واط)
        if orig_numbers:
            p_numbers = set(re.findall(r'\b\d{2,3}\b', p_title_lower))
            p_numbers = {n for n in p_numbers if n not in ['2024', '2025', '2026']}
            if not (orig_numbers & p_numbers): continue

        # 4. 👁️ حصن الأمان البصري المشدد (تطابق الصور الفعلي)
        if orig_hash and p_image_url:
            p_hash = await get_image_hash(p_image_url, session)
            if p_hash:
                if (orig_hash - p_hash) > 10: # درجة تدقيق بصرية ممتازة للصورة الرئيسية
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
                      product_info = {'title': scraped_name, 'image_url': scraped_image, 'price': None,
