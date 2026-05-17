# --- START OF DEEP-SEARCH ABSOLUTE CHEAPEST PRICE COMPARISON BOT ---

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
    """استخراج مرن وذكي يركز على لب المنتج لمنع تضييق نتائج البحث العميقة"""
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
    
    # نأخذ أول 3 كلمات جوهرية فقط لفتح المجال أمام محرك البحث ليعود بكل المتاجر المتاحة أرخصها وأغلاها
    if len(filtered_words) >= 2:
        return " ".join(filtered_words[:3])
    return " ".join(words[:3])

def filter_and_sort_alternatives(orig_title: str, orig_price_raw, search_products: list, orig_id: str) -> list:
    """غربلة وتصفية ذكية لضمان مطابقة السلعة 100% والسماح بأكبر نسبة تخفيض ممكنة"""
    if not search_products: return []
    
    try: orig_price = float(str(orig_price_raw).replace(',', '.'))
    except (ValueError, TypeError): orig_price = None

    valid_products = []
    seen_ids = set()
    orig_title_lower = orig_title.lower()
    
    # استخراج القوة والمواصفات الفنية الصارمة (مثل 65W أو 100W أو 512GB) لفرض مطابقتها
    spec_patterns = [r'\b\d+w\b', r'\b\d+a\b', r'\b\d+mah\b', r'\bgan\b', r'\bpd\b', r'\b\d+gb\b', r'\b\d+tb\b']
    orig_specs = []
    for pattern in spec_patterns:
        orig_specs.extend(re.findall(pattern, orig_title_lower))
    orig_specs = set(orig_specs)

    # تحديد الكلمات المفتاحية للفئات لتجنب تداخل المنتجات
    charger_keywords = {'charger', 'plug', 'adapter', 'gan', 'block', 'رأس', 'شاحن', 'مقبس'}
    cable_keywords = {'cable', 'cord', 'wire', 'line', 'سلك', 'كابل', 'خيط'}
    
    is_charger = any(k in orig_title_lower for k in charger_keywords)
    is_cable = any(k in orig_title_lower for k in cable_keywords)

    for p in search_products:
        p_id = str(p.get('product_id') or '')
        if not p_id or p_id in seen_ids or p_id == orig_id:
            continue
            
        p_title = p.get('product_title', '')
        p_title_lower = p_title.lower()
        p_price_raw = p.get('target_sale_price')
        
        try: p_price = float(str(p_price_raw).replace(',', '.'))
        except (ValueError, TypeError): continue

        # 1. جدار حماية السعر العميق: تم خفضه لـ 15% للسماح بالتخفيضات الضخمة وحرق الأسعار الحقيقي
        if orig_price is not None and p_price < (orig_price * 0.15):
            continue

        # 2. التحقق الصارم من الفئة لحظر الملحقات التافهة
        if is_charger and not any(k in p_title_lower for k in charger_keywords): continue
        if is_cable and not any(k in p_title_lower for k in cable_keywords): continue

        # 3. التحقق من تطابق المواصفات الفنية الحاكمة (إن وجدت) لضمان عدم جلب قوة أقل
        p_specs = []
        for pattern in spec_patterns:
            p_specs.extend(re.findall(pattern, p_title_lower))
        p_specs = set(p_specs)
        if orig_specs and not orig_specs.intersection(p_specs):
            continue

        # 4. مطابقة نسبة معجمية مرنة لاسم المنتج لضمان أنه نفس نوع السلعة
        stop_words = {'with', 'for', 'from', 'and', 'the', 'new', 'original', 'version', 'global', 'shipping', 'free', 'official', 'store', 'choice'}
        orig_words = set([w for w in re.sub(r'[^\w\s]', ' ', orig_title_lower).split() if len(w) > 2 and w not in stop_words])
        p_words = set([w for w in re.sub(r'[^\w\s]', ' ', p_title_lower).split() if len(w) > 2 and w not in stop_words])
        
        if orig_words:
            common = orig_words.intersection(p_words)
            if len(common) < 2: # يجب أن يشترك في كلمتين أساسيتين على الأقل لضمان التقارب الكامل
                continue

        valid_products.append(p)
        seen_ids.add(p_id)

    # إعادة فرز وتأكيد الترتيب التصاعدي من القرش الأقل للأعلى
    valid_products.sort(key=lambda x: float(str(x.get('target_sale_price', 999999)).replace(',', '.')))
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

# --- Fetch Product Details ---
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
    logger.info(f"Deep Deep Search query: {cleaned_query}")
    
    def _execute_query_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.product.query')
            request.add_api_param('keywords', cleaned_query)
            request.add_api_param('target_currency', TARGET_CURRENCY)
            request.add_api_param('target_language', TARGET_LANGUAGE)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            request.add_api_param('ship_to_country', QUERY_COUNTRY)
            # النقطة السحرية: إجبار السيرفر على الترتيب من الأرخص للأغلى مباشرة من المنبع
            request.add_api_param('sort', 'SALE_PRICE_ASC')
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
    welcome_message = """<b>👋 مرحبًا بك في بوت مقارنة الأسعار العميقة لـ AliExpress!\n\n📋 أرسل رابط المنتج الآن، وسيقوم البوت بمسح السيرفرات بالكامل لجلب أرخص سعر متاح للسلعة بدقة متناهية!🚀</b>"""
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
        try: await context.bot.send_message(chat_id=chat_id, text=f"<b>⚠️ حدث خطأ أثناء إرسال الرسالة.</b>", parse_mode=ParseMode.HTML)
        except Exception: pass

async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    import html
    try:
        # 1. جلب السعر الأصلي الحالي للرابط المرسل
        product_data, details_source = await _get_product_data(product_id)
        if not product_data or details_source == "None":
             await context.bot.send_message(chat_id=chat_id, text=f"<b>❌ تعذر استرداد بيانات المنتج من AliExpress.</b>", parse_mode=ParseMode.HTML)
             return

        title = product_data.get('title', 'منتج AliExpress')
        orig_price = product_data.get('price')
        try: orig_price_num = float(str(orig_price).replace(',', '.'))
        except Exception: orig_price_num = 0.0

        # 2. جلب وتصفية المنتجات البديلة المرتبة من الأرخص على الإطلاق برمجياً من السيرفر
        raw_alternatives = await fetch_alternative_cheapest_products(title)
        filtered_alternatives = filter_and_sort_alternatives(title, orig_price, raw_alternatives, product_id)
        
        final_offers_data = []
        for p in filtered_alternatives[:4]:
            p_url = p.get('product_detail_url')
            p_price = p.get('target_sale_price', 0)
            try: price_num = float(str(p_price).replace(',', '.'))
            except Exception: price_num = 0.0
            
            if p_url:
                final_offers_data.append({
                    "url": p_url,
                    "price": price_num,
                    "is_original": False
                })

        # إدراج الرابط الأصلي الذي أرسله المستخدم تلقائياً ضمن المقارنة ليتضح فارق السعر الهائل
        if not any(item['url'] == base_url for item in final_offers_data):
            final_offers_data.append({
                "url": base_url,
                "price": orig_price_num,
                "is_original": True
            })

        # الفرز التصاعدي المطلق لضمان ظهور أصغر رقم مالي في الخيار الأول رياضياً
        final_offers_data.sort(key=lambda x: x['price'] if x['price'] > 0 else 999999)

        # 4. تحويل الروابط للأفلييت
        urls_to_convert = [item['url'] for item in final_offers_data[:4]]
        generated_links_batch = await generate_affiliate_links_batch(urls_to_convert)

        # 5. صياغة التقرير النهائي للمستخدم بكل دقة وفخر بالأرخص
        labels_pool = [
            "<b>🥇 الخيار الأول (أرخص متجر على الإطلاق) 🏆 بـ : ({price_val} $) 🔥</b>",
            "<b>🥈 الخيار الثاني (بائع بديل ممتاز) 🚀 بـ : ({price_val} $) 🔥</b>",
            "<b>🥉 الخيار الثالث (متجر منافس مخفض) ⚡️ بـ : ({price_val} $) 🔥</b>",
            "<b>🏅 الخيار الرابع (عرض إضافي متاح) ✨ بـ : ({price_val} $) 🔥</b>"
        ]

        final_offers = []
        for i, item in enumerate(final_offers_data[:4]):
            aff_link = generated_links_batch.get(item['url']) or item['url']
            price_display = f"{item['price']:.2f}" if item['price'] > 0 else "عرض ممتاز"
            
            if item.get('is_original'):
                label_text = f"<b>📍 الرابط الأصلي الذي أرسلته أنت 🌟 بـ : ({price_display} $) 🔥</b>"
            else:
                label_text = labels_pool[i].format(price_val=price_display)

            final_offers.append({
                "label": label_text,
                "link": aff_link
            })

        message_lines = []
        product_title = html.escape(title)
        message_lines.append(f"<b>📝 إسم المنتج : {product_title[:250]}</b>")
        message_lines.append("<b>\n🎯 عثرنا لك على أرخص الأسعار والمتاجر البديلة في علي إكسبريس ⬇️🤩\n</b>")
        
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

    logger.info("Starting Deep-Search Absolute Cheapest Bot...")
    application.run_polling()

if __name__ == "__main__":
    main()

# --- END OF DEEP-SEARCH ABSOLUTE CHEAPEST PRICE COMPARISON BOT ---
