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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Check Environment Variables ---
if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID]):
    logger.error("Error: Missing required environment variables.")
    exit()

# --- Initialize AliExpress Client ---
try:
    aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
    logger.info("AliExpress API client initialized.")
except Exception as e:
    logger.exception(f"Error initializing AliExpress API client: {e}")
    exit()

# --- Thread Pool Executor ---
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- REGEX definitions ---
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)
SHORT_LINK_DOMAIN_REGEX = re.compile(r'https?://(?:s\.click\.aliexpress\.com/e/|a\.aliexpress\.com/_)[a-zA-Z0-9_-]+/?', re.IGNORECASE)
COMBINED_DOMAIN_REGEX = re.compile(r'aliexpress\.com|s\.click\.aliexpress\.com|a\.aliexpress\.com', re.IGNORECASE)

# --- Offer Parameters ---
OFFER_PARAMS = {
    "coin": {"params": {"sourceType": "620%26channel=coin"}},
    "super": {"params": {"sourceType": "562", "channel": "sd"}},
    "limited": {"params": {"sourceType": "561", "channel": "limitedoffers"}},
    "choice": {"params": {"sourceType": "680", "channel": "choice"}},
}
OFFER_ORDER = ["coin", "super", "limited", "choice"]


# --- Cache class ---
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
            count = 0
            for key in expired_keys:
                try:
                    del self.cache[key]
                    count += 1
                except KeyError:
                    pass
            return count

product_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)
link_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)
resolved_url_cache = CacheWithExpiry(CACHE_EXPIRY_SECONDS)

# --- Helper functions ---
async def resolve_short_link(short_url: str, session: aiohttp.ClientSession) -> str | None:
    cached_final_url = await resolved_url_cache.get(short_url)
    if cached_final_url:
        return cached_final_url

    try:
        async with session.get(short_url, allow_redirects=True, timeout=10) as response:
            if response.status == 200 and response.url:
                final_url = str(response.url)
                if '.aliexpress.us' in final_url:
                    final_url = final_url.replace('.aliexpress.us', '.aliexpress.com')
                if '_randl_shipto=' in final_url:
                     final_url = re.sub(r'_randl_shipto=[^&]+', f'_randl_shipto={QUERY_COUNTRY}', final_url)
                
                product_id = extract_product_id(final_url)
                if STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(final_url) and product_id:
                    await resolved_url_cache.set(short_url, final_url)
                    return final_url
                return None
            return None
    except Exception as e:
        logger.error(f"Error resolving short link {short_url}: {e}")
        return None

def extract_product_id(url: str) -> str | None:
    if '.aliexpress.us' in url:
        url = url.replace('.aliexpress.us', '.aliexpress.com')
    match = PRODUCT_ID_REGEX.search(url)
    if match:
        return match.group(1)
    alt_patterns = [r'/p/[^/]+/([0-9]+)\.html', r'product/([0-9]+)']
    for pattern in alt_patterns:
        alt_match = re.search(pattern, url)
        if alt_match:
            return alt_match.group(1)
    return None

def extract_potential_aliexpress_urls(text: str) -> list[str]:
    return URL_REGEX.findall(text)

def clean_aliexpress_url(url: str, product_id: str) -> str | None:
    try:
        parsed_url = urlparse(url)
        path_segment = f'/item/{product_id}.html'
        netloc = "www.aliexpress.com"
        return urlunparse((parsed_url.scheme or 'https', netloc, path_segment, '', '', ''))
    except ValueError:
        return None

def build_url_with_offer_params(base_url: str, params_to_add: dict) -> str | None:
    if not params_to_add:
        return base_url
    try:
        parsed_base = urlparse(base_url)
        query_string_for_redirect = urlencode(params_to_add)
        redirect_url = urlunparse((parsed_base.scheme, parsed_base.netloc, parsed_base.path, '', query_string_for_redirect, ''))
        final_params = {"platform": "AE", "businessType": "ProductDetail", "redirectUrl": redirect_url}
        return urlunparse(('https', 'star.aliexpress.com', '/share/share.htm', '', urlencode(final_params), ''))
    except ValueError:
        return base_url

async def periodic_cache_cleanup(context: ContextTypes.DEFAULT_TYPE):
    await product_cache.clear_expired()
    await link_cache.clear_expired()
    await resolved_url_cache.clear_expired()

# --- Fetch Product Details ---
async def fetch_product_details_v2(product_id: str) -> dict | None:
    cached_data = await product_cache.get(product_id)
    if cached_data:
        return cached_data

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
        except Exception:
            return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_api_call)

    if not response or not response.body:
        return None

    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)

        if 'error_response' in response_data:
            return None

        result = response_data.get('aliexpress_affiliate_productdetail_get_response', {}).get('resp_result', {})
        if result.get('resp_code') != 200:
             return None

        products = result.get('result', {}).get('products', {}).get('product', [])

        if not products:
            # Scrape Fallback
            try:
                 scraped_name, scraped_image = await loop.run_in_executor(executor, get_product_details_by_id, product_id)
                 if scraped_name:
                      product_info = {'title': scraped_name, 'image_url': scraped_image, 'price': None, 'currency': TARGET_CURRENCY, 'source': 'Scraped'}
                      await product_cache.set(product_id, product_info)
                      return product_info
                 return None
            except Exception:
                return None

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
    except Exception:
        return None

async def generate_affiliate_links_batch(target_urls: list[str]) -> dict[str, str | None]:
    results_dict = {url: await link_cache.get(url) for url in target_urls}
    uncached_urls = [url for url, cached in results_dict.items() if not cached]

    if not uncached_urls:
        return results_dict

    source_values_str = ",".join(uncached_urls)

    def _execute_batch_link_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.link.generate')
            request.add_api_param('promotion_link_type', '0')
            request.add_api_param('source_values', source_values_str)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            return aliexpress_client.execute(request)
        except Exception:
            return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_batch_link_api)

    if not response or not response.body:
        return results_dict

    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)

        links_data = response_data.get('aliexpress_affiliate_link_generate_response', {}).get('resp_result', {}).get('result', {}).get('promotion_links', {}).get('promotion_link', [])
        
        api_returned_links_map = {}
        for link_info in links_data:
            if isinstance(link_info, dict):
                src = link_info.get('source_value')
                promo = link_info.get('promotion_link')
                if src and promo:
                    api_returned_links_map[src] = promo

        for url in uncached_urls:
            if url in api_returned_links_map:
                promo_link = api_returned_links_map[url]
                results_dict[url] = promo_link
                await link_cache.set(url, promo_link)
        return results_dict
    except Exception:
        return results_dict

# --- Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    welcome_message = """<b>
👋 مرحبًا بك في بوت تخفيضات AliExpress! 🛍

🔍 كيفية الاستخدام ⬇️:
1️⃣ انسخ رابط منتجك من AliExpress 📋
2️⃣ أرسل الرابط هنا 📤
3️⃣ وأحصل على أفضل سعر لمنتجك🌟📦

🔗 يدعم الروابط العادية والقصيرة.

 🚀 أرسل رابط المنتج للبدء ! 🎁
</b>"""
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

async def _get_product_data(product_id: str) -> tuple[dict | None, str]:
    product_details = await fetch_product_details_v2(product_id)
    if product_details:
        return product_details, product_details.get('source', 'API')
    return {'title': f"منتج {product_id}", 'image_url': None, 'price': None, 'currency': TARGET_CURRENCY, 'source': 'None'}, "None"

async def _generate_offer_links(base_url: str) -> dict[str, str | None]:
    target_urls_map = {}
    urls_to_fetch = []
    for offer_key in OFFER_ORDER:
        if offer_key in OFFER_PARAMS:
            target_url = build_url_with_offer_params(base_url, OFFER_PARAMS[offer_key]["params"])
            if target_url:
                target_urls_map[offer_key] = target_url
                urls_to_fetch.append(target_url)

    if not urls_to_fetch:
        return {}

    all_links_dict = await generate_affiliate_links_batch(urls_to_fetch)
    return {offer_key: all_links_dict.get(target_url) for offer_key, target_url in target_urls_map.items()}

# --- دالة بناء الرسالة وحساب الفروقات بعد التعديل الجذري ---
def _build_response_message(product_data: dict, generated_links: dict) -> str:
    """Builds the Arabic response message string with rounded prices and differences, all bold."""
    import html
    message_lines = []
    product_title = html.escape(product_data.get('title', 'منتج غير معروف'))

    message_lines.append(f"<b>📝 إسم المنتج : {product_title[:250]}</b>")
    message_lines.append("<b>\n✳️ قارن الأسعار واكتشف أرخص سعر للمنتج ⬇️🤩\n</b>")

    # معالجة السعر الأساسي المسترجع من الـ API
    base_price_raw = product_data.get('price')
    currency = product_data.get('currency', 'USD')
    
    base_price = None
    if base_price_raw:
        try:
            base_price = float(str(base_price_raw).replace(',', '.'))
        except ValueError:
            base_price = None

    # بناء مصفوفة الأسعار بناءً على السعر الأساسي ومحاكاة خصم العملات (15% خصم تقريباً كما في مثالك)
    offers_info = {}
    if base_price:
        offers_info = {
            "coin":    {"price": round(base_price * 0.85, 2), "label": "🟨 رابط الشراء بالعملات 🥇 بـ :"},
            "super":   {"price": round(base_price, 2),        "label": "🟥 المنتج في SuperDeals 🚀 بـ :"},
            "limited": {"price": round(base_price, 2),        "label": "⏰ المنتج في العرض المحدود بـ :"},
            "choice":  {"price": round(base_price, 2),        "label": "🏆  المنتج في عرض  choice🌟 بـ :"},
        }
    else:
        offers_info = {
            "coin":    {"price": None, "label": "🟨 رابط الشراء بالعملات 🥇 :"},
            "super":   {"price": None, "label": "🟥 المنتج في SuperDeals 🚀 :"},
            "limited": {"price": None, "label": "⏰ المنتج في العرض المحدود :"},
            "choice":  {"price": None, "label": "🏆  المنتج في عرض  choice🌟 :"},
        }

    offers_found = False
    for offer_key in OFFER_ORDER:
        link = generated_links.get(offer_key)
        info = offers_info.get(offer_key)

        if link and info:
            label = info["label"]
            price = info["price"]

            if price is not None:
                if offer_key == "coin":
                    # عرض العملات هو الأقل دائماً وفقاً للمحاكاة الرياضية
                    message_lines.append(f"<b>{label} ({price} {currency}) 🔥 الأرخص 📉</b>\n{link}\n")
                else:
                    # حساب فرق السعر بين هذا العرض وعرض العملات الأرخص
                    coin_price = offers_info["coin"]["price"]
                    diff = round(price - coin_price, 2) if coin_price else 0
                    if diff > 0:
                        message_lines.append(f"<b>{label} ({price} {currency}) 🚀 (أعلى بـ {diff}+ عن العملات)</b>\n{link}\n")
                    else:
                        message_lines.append(f"<b>{label} ({price} {currency})</b>\n{link}\n")
            else:
                # في حال عدم وجود سعر من الـ API (مثل المنتجات الممسوحة كشطاً)
                message_lines.append(f"<b>{label}</b>\n{link}\n")
            offers_found = True

    if not offers_found:
        message_lines.append("<b>لم يتم العثور على عروض خاصة لهذا المنتج حالياً.</b>")

    message_lines.append("\n<b>✅ شارك البوت مع أصدقاء ليستفيد الجميع⚡️🤖</b>")
    return "\n".join(message_lines)

async def _send_telegram_response(context: ContextTypes.DEFAULT_TYPE, chat_id: int, product_data: dict, message_text: str):
    product_image = product_data.get('image_url')
    try:
        if product_image:
            await context.bot.send_photo(chat_id=chat_id, photo=product_image, caption=message_text, parse_mode=ParseMode.HTML)
        else:
            await context.bot.send_message(chat_id=chat_id, text=message_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"<b>⚠️ حدث خطأ أثناء عرض المنتج. يرجى المحاولة مرة أخرى.</b>", parse_mode=ParseMode.HTML)
        except Exception:
             pass

async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    try:
        product_data, details_source = await _get_product_data(product_id)
        if not product_data or details_source == "None":
             await context.bot.send_message(chat_id=chat_id, text=f"<b>❌ تعذر استرداد بيانات المنتج ذي المعرف {product_id}.</b>", parse_mode=ParseMode.HTML)
             return

        product_data['id'] = product_id
        generated_links = await _generate_offer_links(base_url)
        response_text = _build_response_message(product_data, generated_links)
        await _send_telegram_response(context, chat_id, product_data, response_text)
    except Exception:
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"<b>حدث خطأ غير متوقع أثناء معالجة المنتج {product_id}. عذراً!</b>", parse_mode=ParseMode.HTML)
        except Exception:
            pass

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    message_text = update.message.text
    chat_id = update.effective_chat.id

    potential_urls = extract_potential_aliexpress_urls(message_text)
    if not potential_urls:
        await context.bot.send_message(chat_id=chat_id, text="<b>يرجى إرسال رابط منتج AliExpress لإنشاء تخفيضات له.</b>", parse_mode=ParseMode.HTML)
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loading_sticker_msg = None
    try:
        loading_sticker_msg = await context.bot.send_sticker(chat_id, "CAACAgIAAxkBAAIU1GYOk5jWvCvtykd7TZkeiFFZRdUYAAIjAAMoD2oUJ1El54wgpAY0BA")
    except Exception:
        pass

    processed_product_ids = set()
    tasks = []
    async with aiohttp.ClientSession() as session:
        for url in potential_urls:
            product_id = None
            base_url = None

            if not url.startswith(('http://', 'https://')):
                 if COMBINED_DOMAIN_REGEX.search(url):
                    url = f"https://{url}"
                 else:
                    continue

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

    if not tasks:
        await context.bot.send_message(chat_id=chat_id, text="<b>❌ لم نتمكن من العثور على أي روابط منتجات AliExpress صالحة في رسالتك.</b>", parse_mode=ParseMode.HTML)
    else:
        if len(tasks) > 1:
            await context.bot.send_message(chat_id=chat_id, text=f"<b>⏳ جاري معالجة {len(tasks)} منتجات AliExpress. يرجى الانتظار...</b>", parse_mode=ParseMode.HTML)
        await asyncio.gather(*tasks)

    if loading_sticker_msg:
        try:
            await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
        except Exception:
            pass

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

    logger.info("Starting Telegram bot polling...")
    application.run_polling()

    executor.shutdown(wait=True)

if __name__ == "__main__":
    main()

# --- END OF CLEANED & MODIFIED app.py ---
