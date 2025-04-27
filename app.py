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

# --- Keep your existing environment variable loading and initial setup ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALIEXPRESS_APP_KEY = os.getenv('ALIEXPRESS_APP_KEY')
ALIEXPRESS_APP_SECRET = os.getenv('ALIEXPRESS_APP_SECRET')
TARGET_CURRENCY = os.getenv('TARGET_CURRENCY', 'USD') # لا يزال يستخدم لل API
TARGET_LANGUAGE = os.getenv('TARGET_LANGUAGE', 'en') # لا يزال يستخدم لل API
QUERY_COUNTRY = os.getenv('QUERY_COUNTRY', 'US') # لا يزال يستخدم لل API
ALIEXPRESS_TRACKING_ID = os.getenv('ALIEXPRESS_TRACKING_ID', 'default')
ALIEXPRESS_API_URL = 'https://api-sg.aliexpress.com/sync'
QUERY_FIELDS = 'product_main_image_url,target_sale_price,product_title,target_sale_price_currency' # Price fields kept for API call, but not displayed
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

# --- Keep your existing REGEX definitions ---
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)
SHORT_LINK_DOMAIN_REGEX = re.compile(r'https?://(?:s\.click\.aliexpress\.com/e/|a\.aliexpress\.com/_)[a-zA-Z0-9_-]+/?', re.IGNORECASE)
COMBINED_DOMAIN_REGEX = re.compile(r'aliexpress\.com|s\.click\.aliexpress\.com|a\.aliexpress\.com', re.IGNORECASE)

# --- Offer Parameters (Mapping internal keys to API parameters) ---
# Note: We'll map these keys to Arabic labels in _build_response_message
OFFER_PARAMS = {
    "coin": {"params": {"sourceType": "620%26channel=coin"}},
    "super": {"params": {"sourceType": "562", "channel": "sd"}},
    "limited": {"params": {"sourceType": "561", "channel": "limitedoffers"}},
    # Mapping 'bigsave' key to the "choice" offer parameters requested visually
    # Using sourceType=680 which was Big Save, adjust if incorrect for 'Choice'
    "choice": {"params": {"sourceType": "680", "channel": "choice"}}, # Assuming Choice channel uses this
}
# Order in which offers should appear in the message
OFFER_ORDER = ["coin", "super", "limited", "choice"]


# --- Cache class remains the same ---
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

# --- Keep existing helper functions (resolve_short_link, extract_product_id, etc.) ---
# --- Ensure they don't have user-facing text ---
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

                # Standardize domain and country parameters if needed (optional but good practice)
                if '.aliexpress.us' in final_url:
                    final_url = final_url.replace('.aliexpress.us', '.aliexpress.com')
                    logger.info(f"Converted US domain URL: {final_url}")
                if '_randl_shipto=' in final_url:
                     final_url = re.sub(r'_randl_shipto=[^&]+', f'_randl_shipto={QUERY_COUNTRY}', final_url)
                     logger.info(f"Updated URL with query country: {final_url}")
                     # Optional: Re-fetch to ensure final URL after country change
                     # try:
                     #     async with session.get(final_url, allow_redirects=True, timeout=10) as country_response:
                     #         final_url = str(country_response.url)
                     #         logger.info(f"Re-fetched URL with correct country: {final_url}")
                     # except Exception as e: logger.warning(f"Error re-fetching: {e}")


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

def extract_potential_aliexpress_urls(text: str) -> list[str]:
    return URL_REGEX.findall(text)

def clean_aliexpress_url(url: str, product_id: str) -> str | None:
    try:
        parsed_url = urlparse(url)
        path_segment = f'/item/{product_id}.html'
        # Use www.aliexpress.com for consistency
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
        return base_url # Return original on error

# --- Cache cleanup remains the same ---
async def periodic_cache_cleanup(context: ContextTypes.DEFAULT_TYPE):
    try:
        product_expired = await product_cache.clear_expired()
        link_expired = await link_cache.clear_expired()
        resolved_expired = await resolved_url_cache.clear_expired()
        logger.info(f"Cache cleanup: Removed {product_expired} product, {link_expired} link, {resolved_expired} resolved URL items.")
        logger.info(f"Cache stats: {len(product_cache.cache)} products, {len(link_cache.cache)} links, {len(resolved_url_cache.cache)} resolved URLs in cache.")
    except Exception as e:
        logger.error(f"Error in periodic cache cleanup job: {e}")

# --- fetch_product_details_v2 remains the same ---
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
            'price': product_data.get('target_sale_price'), # Keep price internally if needed later
            'currency': product_data.get('target_sale_price_currency', TARGET_CURRENCY),
            'title': product_data.get('product_title', f'منتج {product_id}'), # Default Arabic title
            'source': 'API'
        }
        await product_cache.set(product_id, product_info)
        expiry_date = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
        logger.info(f"Cached product {product_id} from API until {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}")
        return product_info

    except Exception as e:
        logger.exception(f"Error parsing product details response for ID {product_id}: {e}")
        return None

# --- generate_affiliate_links_batch remains the same ---
async def generate_affiliate_links_batch(target_urls: list[str]) -> dict[str, str | None]:
    results_dict = {}
    uncached_urls = []
    for url in target_urls:
        cached_link = await link_cache.get(url)
        if cached_link:
            logger.info(f"Cache hit for affiliate link: {url}")
            results_dict[url] = cached_link
        else:
            logger.debug(f"Cache miss for affiliate link: {url}")
            results_dict[url] = None
            uncached_urls.append(url)

    if not uncached_urls:
        logger.info("All required affiliate links retrieved from cache.")
        return results_dict

    logger.info(f"Generating affiliate links for {len(uncached_urls)} uncached URLs via batch API...")
    source_values_str = ",".join(uncached_urls)

    def _execute_batch_link_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.link.generate')
            request.add_api_param('promotion_link_type', '0')
            request.add_api_param('source_values', source_values_str)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            return aliexpress_client.execute(request)
        except Exception as e:
            logger.error(f"Error in batch link API call thread: {e}")
            return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_batch_link_api)

    if not response or not response.body:
        logger.error(f"Batch link generation API call failed or returned empty body for {len(uncached_urls)} URLs.")
        return results_dict

    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)

        if 'error_response' in response_data:
            error_details = response_data.get('error_response', {})
            logger.error(f"API Error for Batch Link Generation: Code={error_details.get('code', 'N/A')}, Msg={error_details.get('msg', 'Unknown API error')}")
            return results_dict

        generate_response = response_data.get('aliexpress_affiliate_link_generate_response')
        if not generate_response:
            logger.error(f"Missing 'aliexpress_affiliate_link_generate_response' key. Response: {response_data}")
            return results_dict
        resp_result_outer = generate_response.get('resp_result')
        if not resp_result_outer:
            logger.error(f"Missing 'resp_result' key. Response: {generate_response}")
            return results_dict
        resp_code = resp_result_outer.get('resp_code')
        if resp_code != 200:
            logger.error(f"API response code not 200 for batch link generation. Code: {resp_code}, Msg: {resp_result_outer.get('resp_msg', 'Unknown')}")
            logger.error(f"Failed URLs (request): {uncached_urls}")
            return results_dict

        result = resp_result_outer.get('result', {})
        if not result:
            logger.error(f"Missing 'result' key. Response: {resp_result_outer}")
            return results_dict
        links_data = result.get('promotion_links', {}).get('promotion_link', [])
        if not links_data or not isinstance(links_data, list):
            logger.warning(f"No 'promotion_links' found or not a list in batch response. Response: {result}")
            return results_dict

        expiry_date = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
        logger.info(f"Processing {len(links_data)} links from batch API response.")
        api_returned_links_map = {}
        for link_info in links_data:
            if isinstance(link_info, dict):
                source_url_returned = link_info.get('source_value')
                promo_link = link_info.get('promotion_link')
                if source_url_returned and promo_link:
                     if source_url_returned in uncached_urls:
                          api_returned_links_map[source_url_returned] = promo_link
                     else:
                           logger.warning(f"Received link for an unexpected source_value from API: {source_url_returned}")
                else:
                    logger.warning(f"Missing 'source_value' or 'promotion_link' in batch response item: {link_info}")
            else:
                 logger.warning(f"Promotion link data item is not a dictionary: {link_info}")

        for url in uncached_urls:
            if url in api_returned_links_map:
                promo_link = api_returned_links_map[url]
                results_dict[url] = promo_link
                await link_cache.set(url, promo_link)
                logger.debug(f"Cached affiliate link for {url} until {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                 logger.warning(f"No affiliate link returned or processed from batch API for requested URL: {url}")
        return results_dict

    except Exception as e:
        logger.exception(f"Error parsing batch link generation response: {e}")
        return results_dict

# --- MODIFIED start handler ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the Arabic welcome message."""
    welcome_message = """
👋 <b>مرحبًا بك في بوت تخفيضات AliExpress!</b> 🛍

🔍 <b>كيفية الاستخدام ⬇️:</b>
1️⃣ انسخ رابط منتجك من AliExpress 📋
2️⃣ أرسل الرابط هنا 📤
3️⃣ وأحصل على أفضل سعر لمنتجك🌟📦

🔗 يدعم الروابط العادية والقصيرة.

 🚀 <b>أرسل رابط المنتج للبدء !</b> 🎁
"""
    await update.message.reply_text(welcome_message, parse_mode=ParseMode.HTML)

# --- MODIFIED _get_product_data ---
async def _get_product_data(product_id: str) -> tuple[dict | None, str]:
    """Fetches product data from API or scraping, returns data and source."""
    product_details = await fetch_product_details_v2(product_id)
    details_source = "None"
    if product_details:
        details_source = product_details.get('source', 'API')
        logger.info(f"Successfully fetched/scraped details ({details_source}) for product ID: {product_id}")
        return product_details, details_source
    else:
        logger.warning(f"API and Scrape fallback failed for product ID: {product_id}")
        # Return minimal dict indicating failure
        return {'title': f"منتج {product_id}", 'image_url': None, 'price': None, 'currency': None, 'source': 'None'}, details_source

# --- MODIFIED _generate_offer_links ---
async def _generate_offer_links(base_url: str) -> dict[str, str | None]:
    """Generates affiliate links for different offer types."""
    target_urls_map = {} # Map offer_key -> target_url_sent_to_api
    urls_to_fetch = []

    # Iterate through OFFER_ORDER to maintain sequence
    for offer_key in OFFER_ORDER:
        if offer_key in OFFER_PARAMS: # Check if the key exists in our defined offers
            offer_info = OFFER_PARAMS[offer_key]
            target_url = build_url_with_offer_params(base_url, offer_info["params"])
            if target_url:
                target_urls_map[offer_key] = target_url
                urls_to_fetch.append(target_url)
            else:
                logger.warning(f"Could not build target URL for offer {offer_key} with base {base_url}")
        else:
             logger.warning(f"Offer key '{offer_key}' from OFFER_ORDER not found in OFFER_PARAMS.")


    if not urls_to_fetch:
        logger.warning(f"No target URLs could be built for base URL: {base_url}")
        return {}

    # Generate links for all built URLs in one batch
    all_links_dict = await generate_affiliate_links_batch(urls_to_fetch)

    generated_links = {} # Map offer_key -> final_promo_link
    for offer_key, target_url in target_urls_map.items():
        promo_link = all_links_dict.get(target_url)
        generated_links[offer_key] = promo_link # Will be None if generation failed
        if not promo_link:
            logger.warning(f"Failed to get affiliate link for offer {offer_key} (target: {target_url})")

    return generated_links

# --- MODIFIED _build_response_message ---
def _build_response_message(product_data: dict, generated_links: dict) -> str:
    """Builds the Arabic response message string."""
    message_lines = []
    product_title = product_data.get('title', 'منتج غير معروف') # Default Arabic title

    # Map internal offer keys to Arabic labels and emojis
    offer_labels = {
        "coin":    "🟨 رابط الشراء بالعملات 🥇:",
        "super":   "🟥 المنتج في SuperDeals 🚀:",
        "limited": "⏰ المنتج في العرض المحدود بـ :🔥",
        "choice":  "🏆 المنتج في عرض choice 🌟:", # Mapped from 'bigsave' key
    }

    # 1. Product Name
    message_lines.append(f"📝 <b>إسم المنتج :</b> {product_title[:250]}") # Keep title reasonable length
    message_lines.append("\n✳️ <b>قارن الأسعار واكتشف أرخص سعر للمنتج ⬇️🤩</b>\n")

    # 2. Offer Links
    offers_found = False
    for offer_key in OFFER_ORDER:
        link = generated_links.get(offer_key)
        label = offer_labels.get(offer_key)

        if link and label:
            message_lines.append(f"{label}\n{link}\n") # Show label then link on new line
            offers_found = True

    # Handle case where NO offers were found
    if not offers_found:
        message_lines.append("لم يتم العثور على عروض خاصة لهذا المنتج حالياً.")

    # 3. Footer
    message_lines.append("\n✅ <b>شارك البوت مع أصدقاء ليستفيد الجميع⚡️</b>🤖")

    return "\n".join(message_lines)


# --- MODIFIED _send_telegram_response ---
async def _send_telegram_response(context: ContextTypes.DEFAULT_TYPE, chat_id: int, product_data: dict, message_text: str):
    """Sends the final response (photo or text) to Telegram without extra buttons."""
    product_image = product_data.get('image_url')
    product_id = product_data.get('id', 'N/A')

    try:
        if product_image:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=product_image,
                caption=message_text,
                parse_mode=ParseMode.HTML,
                # reply_markup=None # No extra buttons
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True, # Disable previews for text-only messages or if links are numerous
                # reply_markup=None # No extra buttons
            )
    except Exception as send_error:
        logger.error(f"Failed to send message for product {product_id} to chat {chat_id}: {send_error}")
        # Fallback message in Arabic
        try:
            fallback_text = f"⚠️ حدث خطأ أثناء عرض المنتج {product_id}. يرجى المحاولة مرة أخرى."
            await context.bot.send_message(
                chat_id=chat_id,
                text=fallback_text
            )
        except Exception as fallback_error:
             logger.error(f"فشل إرسال رسالة الخطأ البديلة للمنتج {product_id} إلى الدردشة {chat_id}: {fallback_error}")


# --- MODIFIED process_product_telegram ---
async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes a single product ID and sends the response in Arabic."""
    chat_id = update.effective_chat.id
    logger.info(f"Processing Product ID: {product_id} for chat {chat_id}")

    try:
        # 1. Get Product Data (API/Scrape)
        product_data, details_source = await _get_product_data(product_id)

        if not product_data or details_source == "None":
             logger.error(f"Failed to get any product data (API or Scraped) for {product_id}")
             # Send user-facing error in Arabic
             await context.bot.send_message(chat_id=chat_id, text=f"❌ تعذر استرداد بيانات المنتج ذي المعرف {product_id}.")
             return

        product_data['id'] = product_id # Add ID for logging/error messages

        # 2. Generate Affiliate Links
        generated_links = await _generate_offer_links(base_url)

        # 3. Build Response Message (Arabic)
        response_text = _build_response_message(product_data, generated_links)

        # 4. Send Response (No extra reply markup)
        await _send_telegram_response(context, chat_id, product_data, response_text)

    except Exception as e:
        logger.exception(f"Unhandled error processing product {product_id} in chat {chat_id}: {e}")
        try:
            # Send user-facing error in Arabic
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"حدث خطأ غير متوقع أثناء معالجة المنتج {product_id}. عذراً!"
            )
        except Exception as send_err:
            logger.error(f"Failed to send error message for product {product_id} to chat {chat_id}: {send_err}")

# --- MODIFIED handle_message ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming messages containing potential AliExpress links."""
    if not update.message or not update.message.text:
        return

    message_text = update.message.text
    user = update.effective_user
    chat_id = update.effective_chat.id
    logger.info(f"Received message from {user.username or user.id} in chat {chat_id}")

    potential_urls = extract_potential_aliexpress_urls(message_text)
    if not potential_urls:
        # Send non-link prompt in Arabic
        await context.bot.send_message(
            chat_id=chat_id,
            text="يرجى إرسال رابط منتج AliExpress لإنشاء تخفيضات له."
        )
        return

    logger.info(f"Found {len(potential_urls)} potential URLs in message from {user.username or user.id}")

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loading_sticker_msg = None
    try: # Send sticker silently
        loading_sticker_msg = await context.bot.send_sticker(chat_id, "CAACAgIAAxkBAAIU1GYOk5jWvCvtykd7TZkeiFFZRdUYAAIjAAMoD2oUJ1El54wgpAY0BA")
    except Exception as sticker_err: logger.warning(f"Could not send loading sticker: {sticker_err}")

    processed_product_ids = set()
    tasks = []
    async with aiohttp.ClientSession() as session:
        for url in potential_urls:
            original_url = url
            product_id = None
            base_url = None

            if not url.startswith(('http://', 'https://')):
                 if COMBINED_DOMAIN_REGEX.search(url):
                    logger.debug(f"Prepending https:// to potential URL: {url}")
                    url = f"https://{url}"
                 else:
                    logger.debug(f"Skipping potential URL without scheme or known AE domain: {original_url}")
                    continue

            if STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(url):
                product_id = extract_product_id(url)
                if product_id: base_url = clean_aliexpress_url(url, product_id)
                logger.debug(f"Standard URL: {url} -> ID: {product_id}, Base: {base_url}")

            elif SHORT_LINK_DOMAIN_REGEX.match(url):
                logger.debug(f"Potential short link: {url}. Resolving...")
                final_url = await resolve_short_link(url, session)
                if final_url:
                    product_id = extract_product_id(final_url)
                    if product_id: base_url = clean_aliexpress_url(final_url, product_id)
                    else: logger.warning(f"Could not extract ID from resolved URL: {final_url} (Original: {original_url})")
                else: logger.warning(f"Could not resolve short link: {original_url}")

            if product_id and base_url and product_id not in processed_product_ids:
                processed_product_ids.add(product_id)
                tasks.append(process_product_telegram(product_id, base_url, update, context))
            elif product_id and product_id in processed_product_ids:
                 logger.debug(f"Skipping duplicate product ID: {product_id}")
            elif not product_id and (STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(url) or SHORT_LINK_DOMAIN_REGEX.match(url)):
                 logger.warning(f"Could not determine Product ID for likely AE URL: {original_url}")
            elif product_id and not base_url:
                 logger.warning(f"Could not determine Base URL for Product ID {product_id} from URL: {original_url}")

    # --- Message indicating processing or no valid links ---
    if not tasks:
        logger.info(f"No processable AliExpress product links found after filtering/resolution.")
        # Send error message in Arabic
        await context.bot.send_message(
            chat_id=chat_id,
            text="❌ لم نتمكن من العثور على أي روابط منتجات AliExpress صالحة في رسالتك."
        )
    else:
        if len(tasks) > 1:
            # Send processing message in Arabic only if multiple items
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"⏳ جاري معالجة {len(tasks)} منتجات AliExpress. يرجى الانتظار..."
            )
        logger.info(f"Processing {len(tasks)} unique AliExpress products for chat {chat_id}")
        await asyncio.gather(*tasks)

    # Delete the loading sticker if it was sent
    if loading_sticker_msg:
        try: await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
        except Exception as delete_err: logger.warning(f"Could not delete loading sticker: {delete_err}")


def main() -> None:
    """Starts the bot (Arabic only)."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handler for /start
    application.add_handler(CommandHandler("start", start))

    # Message handler for AliExpress links (Text or Forwarded)
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED) & ~filters.COMMAND & filters.Regex(COMBINED_DOMAIN_REGEX),
        handle_message
    ))

    # Message handler for text that doesn't contain AE links (Arabic prompt)
    async def non_aliexpress_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
         await context.bot.send_message(
             chat_id=update.effective_chat.id,
             text="يرجى إرسال رابط منتج AliExpress لإنشاء تخفيضات له."
         )
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.Regex(COMBINED_DOMAIN_REGEX),
        non_aliexpress_message
    ))

    # Setup Job Queue for cache cleanup
    job_queue = application.job_queue
    job_queue.run_once(periodic_cache_cleanup, 60)
    job_queue.run_repeating(periodic_cache_cleanup, interval=timedelta(days=1), first=timedelta(days=1))

    # --- Log startup info ---
    logger.info("Starting Telegram bot polling (Arabic Interface)...")
    logger.info(f"Using AliExpress Key: {ALIEXPRESS_APP_KEY[:4]}...")
    logger.info(f"Using Tracking ID: {ALIEXPRESS_TRACKING_ID}")
    logger.info(f"API Settings: Currency={TARGET_CURRENCY}, Lang={TARGET_LANGUAGE}, Country={QUERY_COUNTRY}")
    logger.info(f"Cache expiry: {CACHE_EXPIRY_DAYS} days")
    offer_keys = list(OFFER_PARAMS.keys())
    logger.info(f"Offers configured: {', '.join(offer_keys)}")
    logger.info("Bot is ready and listening...")

    # Run the bot
    application.run_polling()

    # --- Shutdown ---
    logger.info("Shutting down thread pool...")
    executor.shutdown(wait=True)
    logger.info("Bot stopped.")

if __name__ == "__main__":
    main()

# --- END OF MODIFIED app.py ---
