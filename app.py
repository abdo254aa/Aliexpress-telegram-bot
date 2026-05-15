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

# --- إعداد المتغيرات البيئية ---
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

# --- إعداد التسجيل ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- التحقق من المتغيرات البيئية ---
if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID]):
    logger.error("Error: Missing required environment variables.")
    exit()

# --- تهيئة عميل AliExpress API ---
try:
    aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
    logger.info("AliExpress API client initialized.")
except Exception as e:
    logger.exception(f"Error initializing AliExpress API client: {e}")
    exit()

# --- مُنفذ تجمع الخيوط ---
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- تعريفات REGEX ---
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)
SHORT_LINK_DOMAIN_REGEX = re.compile(r'https?://(?:s\.click\.aliexpress\.com/e/|a\.aliexpress\.com/_)[a-zA-Z0-9_-]+/?', re.IGNORECASE)
COMBINED_DOMAIN_REGEX = re.compile(r'aliexpress\.com|s\.click\.aliexpress\.com|a\.aliexpress\.com', re.IGNORECASE)

# --- بارامترات العروض ---
OFFER_PARAMS = {
    "coin": {"params": {"sourceType": "620%26channel=coin"}},
    "super": {"params": {"sourceType": "562", "channel": "sd"}},
    "limited": {"params": {"sourceType": "561", "channel": "limitedoffers"}},
    "choice": {"params": {"sourceType": "680", "channel": "choice"}},
}
OFFER_ORDER = ["coin", "super", "limited", "choice"]


# --- فئة التخزين المؤقت مع انتهاء الصلاحية ---
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
                    logger.debug(f"Cache hit for key: {key}")
                    return item
                else:
                    logger.debug(f"Cache expired for key: {key}")
                    del self.cache[key]
            logger.debug(f"Cache miss for key: {key}")
            return None

    async def set(self, key, value):
        async with self._lock:
            self.cache[key] = (value, time.time())
            logger.debug(f"Cached value for key: {key}")

    async def clear_expired(self):
        async with self._lock:
            current_time = time.time()
            expired_keys = [k for k, (_, t) in self.cache.items()
                            if current_time - t >= self.expiry_seconds]
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

# --- دالة لحل الروابط القصيرة ---
async def resolve_short_link(short_url: str, session: aiohttp.ClientSession) -> str | None:
    cached_final_url = await resolved_url_cache.get(short_url)
    if cached_final_url:
        logger.info(f"Cache hit for resolved short link: {short_url} -> {cached_final_url}")
        return cached_final_url

    logger.info(f"Resolving short link: {short_url}")
    try:
        async with session.get(short_url, allow_redirects=True, timeout=10) as response:
            if response.status == 200 and response.url:
                final_url = str(response.url)
                logger.info(f"Resolved {short_url} to {final_url}")
                if '.aliexpress.us' in final_url:
                    final_url = final_url.replace('.aliexpress.us', '.aliexpress.com')
                    logger.info(f"Converted US domain URL: {final_url}")
                if '_randl_shipto=' in final_url:
                    final_url = re.sub(r'_randl_shipto=[^&]+', f'_randl_shipto={QUERY_COUNTRY}', final_url)
                    logger.info(f"Updated URL with query country: {final_url}")
                product_id = extract_product_id(final_url)
                if STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(final_url) and product_id:
                    await resolved_url_cache.set(short_url, final_url)
                    return final_url
                else:
                    logger.warning(f"Resolved URL {final_url} doesn't look like a valid AliExpress product page.")
                    return None
            else:
                logger.error(f"Failed to resolve short link {short_url}. Status: {response.status}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout resolving short link: {short_url}")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"HTTP ClientError resolving short link {short_url}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error resolving short link {short_url}: {e}")
        return None

# --- دالة لاستخراج معرف المنتج من الرابط ---
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
            product_id = alt_match.group(1)
            logger.info(f"Extracted product ID {product_id} using alternative pattern {pattern}")
            return product_id
    logger.warning(f"Could not extract product ID from URL: {url}")
    return None

# --- دالة لاستخراج جميع روابط AliExpress المحتملة من النص ---
def extract_potential_aliexpress_urls(text: str) -> list[str]:
    return URL_REGEX.findall(text)

# --- دالة لتنظيف رابط AliExpress الأساسي ---
def clean_aliexpress_url(url: str, product_id: str) -> str | None:
    try:
        parsed_url = urlparse(url)
        path_segment = f'/item/{product_id}.html'
        netloc = "www.aliexpress.com"
        base_url = urlunparse((
            parsed_url.scheme or 'https',
            netloc,
            path_segment,
            '', '', ''
        ))
        return base_url
    except ValueError:
        logger.warning(f"Could not parse or reconstruct URL: {url}")
        return None

# --- دالة لبناء رابط مع بارامترات العرض ---
def build_url_with_offer_params(base_url: str, params_to_add: dict) -> str | None:
    if not params_to_add:
        return base_url
    try:
        parsed_base = urlparse(base_url)
        netloc = parsed_base.netloc
        query_string_for_redirect = urlencode(params_to_add)
        redirect_url = urlunparse((
            parsed_base.scheme, netloc, parsed_base.path, '', query_string_for_redirect, ''
        ))
        final_params = {"platform": "AE", "businessType": "ProductDetail", "redirectUrl": redirect_url}
        final_query_string = urlencode(final_params)
        star_url = urlunparse(('https', 'star.aliexpress.com', '/share/share.htm', '', final_query_string, ''))
        return star_url
    except ValueError:
        logger.error(f"Error building URL with params for base: {base_url}")
        return base_url

# --- مهمة دورية لتنظيف الكاش ---
async def periodic_cache_cleanup(context: ContextTypes.DEFAULT_TYPE):
    try:
        product_expired = await product_cache.clear_expired()
        link_expired = await link_cache.clear_expired()
        resolved_expired = await resolved_url_cache.clear_expired()
        logger.info(f"Cache cleanup: Removed {product_expired} product, {link_expired} link, {resolved_expired} resolved URL items.")
        logger.info(f"Cache stats: {len(product_cache.cache)} products, {len(link_cache.cache)} links, {len(resolved_url_cache.cache)} resolved URLs in cache.")
    except Exception as e:
        logger.error(f"Error in periodic cache cleanup job: {e}")

# --- دالة لجلب تفاصيل المنتج من API مع دعم السكرابينج كبديل ---
async def fetch_product_details_v2(product_id: str) -> dict | None:
    cached_data = await product_cache.get(product_id)
    if cached_data:
        logger.info(f"Cache hit for product ID: {product_id}")
        return cached_data
    logger.info(f"Fetching product details for ID: {product_id}")
    api_language = TARGET_LANGUAGE

    def _execute_api_call():
        try:
            request = iop.IopRequest('aliexpress.affiliate.productdetail.get')
            request.add_api_param('fields', QUERY_FIELDS)
            request.add_api_param('product_ids', product_id)
            request.add_api_param('target_currency', TARGET_CURRENCY)
            request.add_api_param('target_language', api_language)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            request.add_api_param('country', QUERY_COUNTRY)
            return aliexpress_client.execute(request)
        except Exception as e:
            logger.error(f"Error in API call thread for product {product_id}: {e}")
            return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_api_call)

    if not response or not response.body:
        logger.error(f"Product detail API call failed or returned empty body for ID: {product_id}")
        return None

    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)

        if 'error_response' in response_data:
            error_details = response_data.get('error_response', {})
            logger.error(f"API Error for Product ID {product_id}: Code={error_details.get('code', 'N/A')}, Msg={error_details.get('msg', 'Unknown API error')}")
            return None

        detail_response = response_data.get('aliexpress_affiliate_productdetail_get_response')
        if not detail_response:
            logger.error(f"Missing 'aliexpress_affiliate_productdetail_get_response' key for ID {product_id}. Response: {response_data}")
            return None

        resp_result = detail_response.get('resp_result')
        if not resp_result:
            logger.error(f"Missing 'resp_result' key for ID {product_id}. Response: {detail_response}")
            return None
        resp_code = resp_result.get('resp_code')
        if resp_code != 200:
            logger.error(f"API response code not 200 for ID {product_id}. Code: {resp_code}, Msg: {resp_result.get('resp_msg', 'Unknown')}")
            return None

        result = resp_result.get('result', {})
        products = result.get('products', {}).get('product', [])

        if not products:
            logger.warning(f"No products found in API response for ID {product_id}")
            logger.info(f"Attempting scrape fallback after empty API product list for {product_id}")
            try:
                loop_inner = asyncio.get_event_loop()
                scraped_name, scraped_image = await loop_inner.run_in_executor(
                    executor, get_product_details_by_id, product_id
                )
                if scraped_name:
                    logger.info(f"Successfully scraped details after empty API response for product ID: {product_id}")
                    product_info = {'title': scraped_name, 'image_url': scraped_image, 'price': None, 'currency': None, 'source': 'Scraped'}
                    await product_cache.set(product_id, product_info)
                    return product_info
                else:
                    logger.warning(f"Scraping also failed after empty API response for product ID: {product_id}")
                    return None
            except Exception as scrape_err:
                logger.error(f"Error during scraping fallback after empty API response for {product_id}: {scrape_err}")
                return None

        product_data = products[0]
        product_info = {
            'image_url': product_data.get('product_main_image_url'),
            'price': product_data.get('target_sale_price'),
            'currency': product_data.get('target_sale_price_currency'),
            'title': product_data.get('product_title'),
            'source': 'API'
        }
        await product_cache.set(product_id, product_info)
        return product_info

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError for product ID {product_id}: {e}.  Response: {response.body if response else 'No response'}")
        return None
    except Exception as e:
        logger.exception(f"Error fetching product details for ID {product_id}: {e}")
        return None



# --- دالة للتعامل مع رسائل المستخدم ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming messages from the user."""
    chat_id = update.effective_chat.id
    user = update.effective_user
    text = update.message.text
    logger.info(f"Received message from {user.username or user.id} in chat {chat_id}: {text}")

    urls = extract_potential_aliexpress_urls(text)
    if not urls:
        await context.bot.send_message(
            chat_id=chat_id,
            text="<b>⚠️ يرجى إرسال رابط منتج AliExpress.</b>",
            parse_mode=ParseMode.HTML
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loading_sticker_msg = None
    try:
        loading_sticker_msg = await context.bot.send_sticker(chat_id, "CAACAgIAAxkBAAIU1GYOk5jWvCvtykd7TZkeiFFZRdUYAAIjAAMoD2oUJ1El54wgpAY0BA")
    except Exception as sticker_err:
        logger.warning(f"Could not send loading sticker: {sticker_err}")


    async with aiohttp.ClientSession() as session:
        for url in urls:
            resolved_url = await resolve_short_link(url, session) if SHORT_LINK_DOMAIN_REGEX.match(url) else url
            product_id = extract_product_id(resolved_url)

            if product_id:
                product_data = await fetch_product_details_v2(product_id)
                if product_data:
                    message_text = f"<b>📝 اسم المنتج:</b> {product_data['title'][:250]}\n\n"
                    message_text += f"🖼️ <a href='{product_data['image_url']}'>صورة المنتج</a>\n\n" if product_data['image_url'] else ""
                    if product_data['price'] is not None and product_data['currency']:
                        message_text += f"💰 <b>السعر الحالي:</b> <mark>{product_data['price']} {product_data['currency']}</mark>\n\n"
                    else:
                        message_text += "💰 <b>السعر:</b> غير متوفر\n\n"

                    best_offer_url = None
                    best_offer_name = None
                    for offer_name in OFFER_ORDER:
                        offer_params = OFFER_PARAMS.get(offer_name, {}).get('params')
                        if offer_params:
                            offer_url = build_url_with_offer_params(resolved_url, offer_params)
                            if offer_url:
                                await link_cache.set(f"{product_id}_{offer_name}", offer_url)
                                best_offer_url = offer_url
                                best_offer_name = offer_name
                                break

                    if best_offer_url:
                        message_text += f"🔗 أفضل عرض ({best_offer_name.capitalize()}): <a href='{best_offer_url}'>اذهب للعرض</a>\n\n"

                    for offer_name in OFFER_ORDER:
                        offer_params = OFFER_PARAMS.get(offer_name, {}).get('params')
                        if offer_params:
                            offer_url = build_url_with_offer_params(resolved_url, offer_params)
                            if offer_url:
                                await link_cache.set(f"{product_id}_{offer_name}", offer_url)
                                message_text += f"🔗 <a href='{offer_url}'>{offer_name.capitalize()}</a>\n"
                    await context.bot.send_message(chat_id=chat_id, text=message_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="<b>❌ لم يتم العثور على معلومات المنتج. يرجى التأكد من الرابط.</b>",
                        parse_mode=ParseMode.HTML
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="<b>❌ لم يتم التعرف على رابط منتج صحيح. يرجى إرسال رابط AliExpress.</b>",
                    parse_mode=ParseMode.HTML
                )
    if loading_sticker_msg:
        try:
            await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
        except Exception as delete_err:
            logger.warning(f"Could not delete loading sticker: {delete_err}")



# --- معالج أوامر جديد لعرض السعر ---
async def get_price(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """يستقبل رابط المنتج ويعرض سعره."""
    chat_id = update.effective_chat.id
    user = update.effective_user
    logger.info(f"Received /price command from {user.username or user.id} in chat {chat_id}")

    if context.args:
        url = context.args[0]
        logger.info(f"Received URL for /price command: {url}")

        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        loading_sticker_msg = None
        try:
            loading_sticker_msg = await context.bot.send_sticker(chat_id, "CAACAgIAAxkBAAIU1GYOk5jWvCvtykd7TZkeiFFZRdUYAAIjAAMoD2oUJ1El54wgpAY0BA")
        except Exception as sticker_err:
            logger.warning(f"Could not send loading sticker for /price: {sticker_err}")

        async with aiohttp.ClientSession() as session:
            resolved_url = await resolve_short_link(url, session) if SHORT_LINK_DOMAIN_REGEX.match(url) else url
            product_id = extract_product_id(resolved_url)

            if product_id:
                product_data = await fetch_product_details_v2(product_id)
                if product_data and product_data.get('price') is not None and product_data.get('currency'):
                    price = product_data['price']
                    currency = product_data['currency']
                    product_title = product_data.get('title', f'منتج برقم {product_id}')
                    response_text = f"""<b>📝 إسم المنتج : {product_title[:250]}</b>

💰 <b>السعر الحالي:</b> <mark>{price} {currency}</mark>

✅ <b>للحصول على أفضل العروض، أرسل رابط المنتج بشكل مباشر!</b> 🚀
"""
                    await context.bot.send_message(chat_id=chat_id, text=response_text, parse_mode=ParseMode.HTML)
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"<b>❌ لم يتم العثور على سعر المنتج ذي المعرف {product_id}.</b>",
                        parse_mode=ParseMode.HTML
                    )
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="<b>❌ لم يتم التعرف على رابط منتج صحيح. يرجى إرسال رابط AliExpress.</b>",
                    parse_mode=ParseMode.HTML
                )

        if loading_sticker_msg:
            try:
                await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
            except Exception as delete_err:
                logger.warning(f"Could not delete loading sticker for /price: {delete_err}")

    else:
        await context.bot.send_message(
            chat_id=chat_id,
            text="<b>⚠️ يرجى إرسال رابط المنتج بعد الأمر /سعر. مثال: /سعر https://...</b>",
            parse_mode=ParseMode.HTML
        )

# --- دالة main لبدء البوت ---
def main() -> None:
    """Starts the bot."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # إضافة معالجات الأوامر
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("price", get_price)) # إضافة معالج الأمر /price

    # إضافة معالج الرسائل
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # تشغيل مهمة تنظيف الكاش بشكل دوري
    application.job_queue.run_repeating(periodic_cache_cleanup, interval=3600, first=60)

    # بدء البوت
    logger.info("Bot started!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)



# --- دالة /start ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a greeting message when the command /start is issued."""
    user = update.effective_user
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"<b>مرحبًا {user.first_name}! 👋\n\nأنا بوت مساعد AliExpress. أرسل لي رابط منتج AliExpress وسأعرض لك تفاصيل المنتج وأفضل العروض المتاحة.</b>",
        parse_mode=ParseMode.HTML,
    )

# --- دالة /help ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="<b>للبحث عن منتج، أرسل رابط AliExpress الخاص بالمنتج.\n\nلاستخدام أمر السعر، أرسل /سعر متبوعًا برابط المنتج.\n\nمثال:\n/سعر https://s.click.aliexpress.com/e/_oB2B3RB</b>",
        parse_mode=ParseMode.HTML,
    )



# --- تشغيل البوت عند استدعاء الملف ---
if __name__ == "__main__":
    main()
# 
```
