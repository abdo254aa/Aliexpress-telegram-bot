# --- START OF REAL PRICE COMPARISON BOT (4 CHEAPEST alternative LISTINGS) ---

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

# الفئات الاحتياطية لتوليد عروض تلقائية في حال لم يجد بائعين بدلاء للمنتج
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
    """تنظيف عنوان المنتج واستخراج الكلمات المفتاحية الأساسية للبحث عن بائعين آخرين بدقة"""
    title_clean = re.sub(r'\[.*?\]|\(.*?\)', '', title)
    title_clean = re.sub(r'[^\w\s]', ' ', title_clean)
    words = title_clean.split()
    fillers = {'with', 'for', 'from', 'and', 'the', 'new', 'original', 'version', 'global', 'shipping', 'free'}
    filtered_words = [w for w in words if len(w) > 2 and w.lower() not in fillers]
    if len(filtered_words) >= 3:
        return " ".join(filtered_words[:4])
    return " ".join(words[:4])

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

# --- البحث الفعلي عن أرخص 4 بائعين لنفس المنتج وترتيبهم تصاعدياً ---
async def fetch_alternative_cheapest_products(title: str) -> list:
    cleaned_query = clean_keywords(title)
    logger.info(f"Searching alternative items with keywords: '{cleaned_query}'")
    
    def _execute_query_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.product.query')
            request.add_api_param('keywords', cleaned_query)
            request.add_api_param('target_currency', TARGET_CURRENCY)
            request.add_api_param('target_language', TARGET_LANGUAGE)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            request.add_api_param('ship_to_country', QUERY_COUNTRY)
            request.add_api_param('page_size', '40')  # جلب عينة جيدة للفرز الدقيق
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
        
        products = result.get('result', {}).get('products', {}).get('product', [])
        if not products: return []
        
        # فرز وترتيب المنتجات برمجياً من الأقل سعراً على الإطلاق إلى الأعلى
        def get_price(p):
            try: return float(str(p.get('target_sale_price', 999999)).replace(',', '.'))
            except ValueError: return 999999
        
        products.sort(key=get_price)
        return products[:4]
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
    welcome_message = """<b>👋 مرحبًا بك في بوت مقارنة تخفيضات AliExpress!\n\n🔍 كيفية الاستخدام ⬇️:\n1️⃣ انسخ رابط منتجك من AliExpress 📋\n2️⃣ أرسل الرابط هنا 📤\n3️⃣ وسيقوم البوت بالبحث واكتشاف أرخص 4 بائعين لنفس المنتج على الإطلاق!🌟📦\n\n🚀 أرسل رابط المنتج للبدء ! 🎁</b>"""
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
        try: await context.bot.send_message(chat_id=chat_id, text=f"<b>⚠️ حدث خطأ أثناء معالجة رسالة العروض.</b>", parse_mode=ParseMode.HTML)
        except Exception: pass

async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    import html
    try:
        # 1. جلب بيانات المنتج الأساسي
        product_data, details_source = await _get_product_data(product_id)
        if not product_data or details_source == "None":
             await context.bot.send_message(chat_id=chat_id, text=f"<b>❌ تعذر استرداد بيانات المنتج من علي إكسبريس.</b>", parse_mode=ParseMode.HTML)
             return

        product_data['id'] = product_id
        title = product_data.get('title', 'منتج AliExpress مميز')

        # 2. استعلام للبحث عن أرخص 4 بائعين بدلاء لنفس السلعة
        alt_products = await fetch_alternative_cheapest_products(title)
        
        # 3. تجميع الروابط لتحويلها دفعة واحدة (Batch) لروابط تسويقية
        urls_to_convert = []
        for p in alt_products:
            p_url = p.get('product_detail_url')
            if p_url: urls_to_convert.append(p_url)

        # توليد روابط احتياطية للمنتج الأساسي لحمايته وضمان بقاء الـ 4 سطور ممتلئة دائماً
        fallback_urls_map = {}
        for key in OFFER_ORDER:
            fb_url = build_url_with_offer_params(base_url, OFFER_PARAMS[key]["params"])
            fallback_urls_map[key] = fb_url
            if len(urls_to_convert) < 4:
                urls_to_convert.append(fb_url)

        # معالجة وتوليد الروابط التسويقية عبر الـ API
        generated_links_batch = await generate_affiliate_links_batch(urls_to_convert)
        
        # 4. بناء الـ 4 خيارات بأسعارها الحقيقية الدقيقة
        final_offers = []
        labels_pool = [
            "<b>🥇 الخيار الأول (الأرخص على الإطلاق) 🏆 بـ : ({price_val} $) 🔥</b>",
            "<b>🥈 الخيار الثاني (سعر بائع بديل مخفض) 🚀 بـ : ({price_val} $) 🔥</b>",
            "<b>🥉 الخيار الثالث (سعر بائع بديل ممتاز) ⚡️ بـ : ({price_val} $) 🔥</b>",
            "<b>🏅 الخيار الرابع (سعر بائع بديل منافس) ✨ بـ : ({price_val} $) 🔥</b>"
        ]

        # إضافة المنتجات الحقيقية الأرخص التي عثر عليها محرك البحث
        for i, p in enumerate(alt_products):
            p_url = p.get('product_detail_url')
            aff_link = generated_links_batch.get(p_url) or p_url
            p_price = p.get('target_sale_price', 'عرض خاص')
            
            try: formatted_price = f"{float(str(p_price).replace(',', '.')):.2f}"
            except Exception: formatted_price = str(p_price)

            final_offers.append({
                "label": labels_pool[i].format(price_val=formatted_price),
                "link": aff_link
            })

        # خط الدفاع والاحتياط: ملء السطور المتبقية في حال عدم وجود بائعين كافيين للمنتج لضمان الـ 4 خيارات ثابتة 100%
        base_price = product_data.get('price')
        used_fallbacks = 0
        while len(final_offers) < 4 and used_fallbacks < len(OFFER_ORDER):
            key = OFFER_ORDER[used_fallbacks]
            fb_url = fallback_urls_map[key]
            aff_link = generated_links_batch.get(fb_url) or fb_url
            
            if base_price:
                mult = {"coin": 0.82, "choice": 0.85, "super": 0.88, "limited": 0.90}[key]
                est_price = f"{round(float(str(base_price).replace(',', '.')) * mult, 2):.2f}"
            else:
                est_price = "تخفيض خاص"
                
            fallback_label_text = {
                "coin": f"<b>🟨 رابط الشراء بالعملات 🥇 بـ : ({est_price} $) 🔥</b>",
                "choice": f"<b>🏆  المنتج في عرض  choice🌟 بـ : ({est_price} $) 🔥</b>",
                "super": f"<b>🟥 المنتج في SuperDeals 🚀 بـ : ({est_price} $) 🔥</b>",
                "limited": f"<b>⏰ المنتج في العرض المحدود بـ : ({est_price} $) 🔥</b>",
            }[key]
            
            final_offers.append({
                "label": fallback_label_text,
                "link": aff_link
            })
            used_fallbacks += 1

        # 5. صياغة وتجميع نص الرسالة النهائية باللغة العربية
        message_lines = []
        product_title = html.escape(title)
        message_lines.append(f"<b>📝 إسم المنتج : {product_title[:250]}</b>")
        message_lines.append("<b>\n✳️ قارن الأسعار واكتشف أرخص 4 بائعين للمنتج حالياً ⬇️🤩\n</b>")
        
        for offer in final_offers:
            safe_link = html.escape(offer['link'])
            message_lines.append(f"{offer['label']}\n{safe_link}\n")
            
        message_lines.append("<b>✅ شارك البوت مع أصدقاء ليستفيد الجميع⚡️🤖</b>")
        response_text = "\n".join(message_lines)
        
        # 6. إرسال الرسالة النهائية للمستخدم
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

    if tasks: await asyncio.gather(*tasks)
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

    logger.info("Starting Price Comparison Telegram bot polling...")
    application.run_polling()

if __name__ == "__main__":
    main()

# --- END OF REAL PRICE COMPARISON BOT (4 CHEAPEST alternative LISTINGS) ---
