# --- START OF app.py (Arabic Only with General Price per Link) ---

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
try:
    from aliexpress_utils import get_product_details_by_id
except ImportError:
    logger.error("Error: aliexpress_utils.py not found. Scraping fallback will not work.")
    # Define a dummy function if the import fails to avoid crashing later
    async def get_product_details_by_id(product_id: str): return None, None


load_dotenv()

# --- Constants and Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALIEXPRESS_APP_KEY = os.getenv('ALIEXPRESS_APP_KEY')
ALIEXPRESS_APP_SECRET = os.getenv('ALIEXPRESS_APP_SECRET')
TARGET_CURRENCY = os.getenv('TARGET_CURRENCY', 'USD') # العملة المستهدفة لل API
TARGET_LANGUAGE = os.getenv('TARGET_LANGUAGE', 'en') # لغة ال API
QUERY_COUNTRY = os.getenv('QUERY_COUNTRY', 'US') # بلد الاستعلام لل API
ALIEXPRESS_TRACKING_ID = os.getenv('ALIEXPRESS_TRACKING_ID', 'default')
ALIEXPRESS_API_URL = 'https://api-sg.aliexpress.com/sync'
# طلب حقول السعر من ال API
QUERY_FIELDS = 'product_main_image_url,target_sale_price,product_title,target_sale_price_currency'
CACHE_EXPIRY_DAYS = 1
CACHE_EXPIRY_SECONDS = CACHE_EXPIRY_DAYS * 24 * 60 * 60
MAX_WORKERS = 10

# --- Logging Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Environment Variable Check ---
if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID]):
    logger.error("Error: Missing required environment variables.")
    exit()

# --- AliExpress Client Initialization ---
try:
    aliexpress_client = iop.IopClient(ALIEXPRESS_API_URL, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET)
    logger.info("AliExpress API client initialized.")
except Exception as e:
    logger.exception(f"Error initializing AliExpress API client: {e}")
    exit()

# --- Thread Pool ---
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# --- REGEX Definitions ---
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

# --- Cache Implementation (remains the same) ---
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

# --- Helper Functions (resolve_short_link, extract_product_id, etc. remain the same) ---
async def resolve_short_link(short_url: str, session: aiohttp.ClientSession) -> str | None:
    cached_final_url = await resolved_url_cache.get(short_url)
    if cached_final_url: return cached_final_url
    logger.info(f"Resolving short link: {short_url}")
    try:
        async with session.get(short_url, allow_redirects=True, timeout=15) as response: # Increased timeout slightly
            response.raise_for_status() # Check for HTTP errors
            if response.url:
                final_url = str(response.url)
                logger.info(f"Resolved {short_url} to {final_url}")
                # Basic validation - should still be on aliexpress domain after redirects
                if not COMBINED_DOMAIN_REGEX.search(urlparse(final_url).netloc):
                     logger.warning(f"Resolved URL {final_url} is not an AliExpress domain.")
                     return None

                # Standardize domain and country parameters if needed
                if '.aliexpress.us' in final_url:
                    final_url = final_url.replace('.aliexpress.us', '.aliexpress.com')
                    logger.info(f"Converted US domain URL: {final_url}")
                # Example: Ensure specific country parameter if required by tracking setup
                # if '_randl_shipto=' in final_url:
                #      final_url = re.sub(r'_randl_shipto=[^&]+', f'_randl_shipto={QUERY_COUNTRY}', final_url)
                #      logger.info(f"Updated URL with query country: {final_url}")

                product_id = extract_product_id(final_url)
                if product_id: # Check if ID could be extracted
                    await resolved_url_cache.set(short_url, final_url)
                    return final_url
                else:
                    logger.warning(f"Could not extract product ID from resolved URL: {final_url}")
                    # Cache the failure? Optional.
                    # await resolved_url_cache.set(short_url, None) # Cache None to avoid retrying soon
                    return None
            else:
                logger.error(f"Failed to resolve short link {short_url}. Status: {response.status}, URL: {response.url}")
                return None
    except asyncio.TimeoutError:
        logger.error(f"Timeout resolving short link: {short_url}")
        return None
    except aiohttp.ClientResponseError as e:
         logger.error(f"HTTP Error {e.status} resolving short link {short_url}: {e.message}")
         return None
    except aiohttp.ClientError as e:
        logger.error(f"HTTP ClientError resolving short link {short_url}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error resolving short link {short_url}: {e}")
        return None


def extract_product_id(url: str) -> str | None:
    if '.aliexpress.us' in url: url = url.replace('.aliexpress.us', '.aliexpress.com')
    match = PRODUCT_ID_REGEX.search(url)
    if match: return match.group(1)
    alt_patterns = [r'/p/[^/]+/([0-9]+)\.html', r'product/([0-9]+)']
    for pattern in alt_patterns:
        alt_match = re.search(pattern, url)
        if alt_match:
            product_id = alt_match.group(1)
            logger.info(f"Extracted product ID {product_id} using alternative pattern {pattern}")
            return product_id
    # logger.warning(f"Could not extract product ID from URL: {url}") # Becomes too noisy
    return None

def extract_potential_aliexpress_urls(text: str) -> list[str]:
    return URL_REGEX.findall(text)

def clean_aliexpress_url(url: str, product_id: str) -> str | None:
    try:
        parsed_url = urlparse(url)
        path_segment = f'/item/{product_id}.html'
        netloc = "www.aliexpress.com" # Standardize domain
        # Ensure scheme is https
        scheme = 'https'
        base_url = urlunparse((scheme, netloc, path_segment, '', '', ''))
        return base_url
    except ValueError:
        logger.warning(f"Could not parse or reconstruct URL: {url}")
        return None

def build_url_with_offer_params(base_url: str, params_to_add: dict) -> str | None:
    if not params_to_add: return base_url
    try:
        parsed_base = urlparse(base_url)
        netloc = parsed_base.netloc
        query_string_for_redirect = urlencode(params_to_add)
        # Build the redirect URL first (the actual product page with offer params)
        redirect_url = urlunparse((parsed_base.scheme, netloc, parsed_base.path, '', query_string_for_redirect, ''))
        # Build the final star.aliexpress.com URL which contains the redirectUrl
        final_params = {"platform": "AE", "businessType": "ProductDetail", "redirectUrl": redirect_url}
        final_query_string = urlencode(final_params)
        star_url = urlunparse(('https', 'star.aliexpress.com', '/share/share.htm', '', final_query_string, ''))
        return star_url
    except ValueError:
        logger.error(f"Error building URL with params for base: {base_url}")
        return base_url # Return original on error

# --- Cache Cleanup Job (remains the same) ---
async def periodic_cache_cleanup(context: ContextTypes.DEFAULT_TYPE):
    try:
        product_expired = await product_cache.clear_expired()
        link_expired = await link_cache.clear_expired()
        resolved_expired = await resolved_url_cache.clear_expired()
        logger.info(f"Cache cleanup: Removed {product_expired} product, {link_expired} link, {resolved_expired} resolved URL items.")
        # Optional: Log cache sizes for monitoring
        # logger.info(f"Cache stats: {len(product_cache.cache)} products, {len(link_cache.cache)} links, {len(resolved_url_cache.cache)} resolved URLs.")
    except Exception as e:
        logger.error(f"Error in periodic cache cleanup job: {e}")


# --- fetch_product_details_v2 (Ensures price/currency are fetched) ---
async def fetch_product_details_v2(product_id: str) -> dict | None:
    """Fetches product details via API, with scraping as a fallback."""
    cached_data = await product_cache.get(product_id)
    if cached_data:
        logger.info(f"Cache hit for product ID: {product_id}")
        return cached_data

    logger.info(f"Attempting API fetch for product ID: {product_id}")
    api_language = TARGET_LANGUAGE # Use global setting for API

    product_info = None
    api_failed = False

    # --- API Call ---
    def _execute_api_call():
        # ... (API call execution code remains the same) ...
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

    # --- API Response Processing ---
    if response and response.body:
        try:
            response_data = response.body
            if isinstance(response_data, str): response_data = json.loads(response_data)

            if 'error_response' in response_data:
                error_details = response_data.get('error_response', {})
                logger.error(f"API Error for Product ID {product_id}: Code={error_details.get('code', 'N/A')}, Msg={error_details.get('msg', 'Unknown API error')}")
                api_failed = True # Mark API as failed
            else:
                detail_response = response_data.get('aliexpress_affiliate_productdetail_get_response')
                resp_result = detail_response.get('resp_result') if detail_response else None
                resp_code = resp_result.get('resp_code') if resp_result else None

                if resp_code == 200:
                    result = resp_result.get('result', {})
                    products = result.get('products', {}).get('product', [])
                    if products:
                        product_data = products[0]
                        product_info = {
                            'image_url': product_data.get('product_main_image_url'),
                            'price': product_data.get('target_sale_price'),
                            'currency': product_data.get('target_sale_price_currency', TARGET_CURRENCY),
                            'title': product_data.get('product_title', f'منتج {product_id}'),
                            'source': 'API'
                        }
                        logger.info(f"Successfully fetched product {product_id} from API.")
                    else:
                        logger.warning(f"API returned 200 but no product data for ID {product_id}.")
                        api_failed = True # Treat as failure if no product data
                else:
                    logger.error(f"API response code not 200 for ID {product_id}. Code: {resp_code}, Msg: {resp_result.get('resp_msg', 'Unknown') if resp_result else 'N/A'}")
                    api_failed = True # Mark API as failed
        except json.JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON response for product {product_id}: {json_err}. Response: {response_data[:500]}")
            api_failed = True # Mark API as failed
        except Exception as e:
            logger.exception(f"Error parsing API response for ID {product_id}: {e}")
            api_failed = True # Mark API as failed
    else:
        logger.error(f"API call failed or returned empty body for ID: {product_id}")
        api_failed = True # Mark API as failed

    # --- Scraping Fallback (if API failed and product_info is still None) ---
    if api_failed and not product_info:
        logger.info(f"Attempting scrape fallback for product ID: {product_id}")
        try:
             # Use run_in_executor for the potentially blocking scrape function
             loop_inner = asyncio.get_event_loop()
             scraped_name, scraped_image = await loop_inner.run_in_executor(
                executor, get_product_details_by_id, product_id
             )
             if scraped_name or scraped_image: # Success if we get at least one detail
                  logger.info(f"Successfully scraped details for product ID: {product_id}")
                  product_info = {
                      'title': scraped_name or f'منتج {product_id}', # Use scraped or default
                      'image_url': scraped_image,
                      'price': None, # Scraping doesn't reliably get price
                      'currency': None,
                      'source': 'Scraped'
                  }
             else:
                 logger.warning(f"Scraping fallback also failed for product ID: {product_id}")
                 # No product_info is set, function will return None later
        except Exception as scrape_err:
            logger.error(f"Error during scraping fallback for {product_id}: {scrape_err}")
            # No product_info is set

    # --- Caching and Return ---
    if product_info:
        await product_cache.set(product_id, product_info)
        expiry_date = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
        logger.debug(f"Cached product {product_id} (Source: {product_info['source']}) until {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}")
        return product_info
    else:
        # Explicitly log failure if both API and scraping failed
        logger.error(f"Failed to retrieve data via API and scraping for product ID: {product_id}")
        # Cache failure to prevent immediate retries? Optional.
        # await product_cache.set(product_id, {'source': 'None'}) # Cache a failure marker
        return None


# --- generate_affiliate_links_batch (remains the same) ---
async def generate_affiliate_links_batch(target_urls: list[str]) -> dict[str, str | None]:
    results_dict = {}; uncached_urls = []
    # 1. Check cache first
    for url in target_urls:
        cached_link = await link_cache.get(url)
        if cached_link: results_dict[url] = cached_link
        else: results_dict[url] = None; uncached_urls.append(url)
    if not uncached_urls: return results_dict

    logger.info(f"Generating affiliate links for {len(uncached_urls)} URLs via batch API...")
    source_values_str = ",".join(uncached_urls)

    # 2. Execute API call in executor
    def _execute_batch_link_api():
        # ... (API call execution same as before) ...
        try:
            request = iop.IopRequest('aliexpress.affiliate.link.generate')
            request.add_api_param('promotion_link_type', '0'); request.add_api_param('source_values', source_values_str); request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            return aliexpress_client.execute(request)
        except Exception as e: logger.error(f"Error in batch link API call thread: {e}"); return None

    loop = asyncio.get_event_loop(); response = await loop.run_in_executor(executor, _execute_batch_link_api)

    # 3. Process API response
    if not response or not response.body: logger.error(f"Batch link API call failed/empty body."); return results_dict
    try:
        response_data = response.body
        if isinstance(response_data, str): response_data = json.loads(response_data)
        # ... (Error and response structure checking same as before) ...
        if 'error_response' in response_data: logger.error(f"API Error Batch Link: {response_data.get('error_response')}"); return results_dict
        generate_response = response_data.get('aliexpress_affiliate_link_generate_response')
        resp_result_outer = generate_response.get('resp_result') if generate_response else None
        resp_code = resp_result_outer.get('resp_code') if resp_result_outer else None
        if resp_code != 200: logger.error(f"Batch link API code not 200: {resp_code}, Msg: {resp_result_outer.get('resp_msg', 'N/A') if resp_result_outer else 'N/A'}"); return results_dict

        result = resp_result_outer.get('result', {})
        links_data = result.get('promotion_links', {}).get('promotion_link', [])
        if not result or not links_data or not isinstance(links_data, list): logger.warning(f"No valid links in batch response."); return results_dict

        # 4. Map results and update cache
        api_returned_links_map = {}
        for link_info in links_data:
            if isinstance(link_info, dict):
                source_url_returned = link_info.get('source_value'); promo_link = link_info.get('promotion_link')
                if source_url_returned and promo_link and source_url_returned in uncached_urls:
                     api_returned_links_map[source_url_returned] = promo_link
        expiry_date = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
        for url in uncached_urls:
            if url in api_returned_links_map:
                promo_link = api_returned_links_map[url]
                results_dict[url] = promo_link
                await link_cache.set(url, promo_link) # Cache successful links
            else: logger.warning(f"No affiliate link returned from batch API for: {url}") # Keep as None
        return results_dict
    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON response for batch link gen: {json_err}. Response: {response_data[:500]}")
        return results_dict # Return potentially partial results
    except Exception as e: logger.exception(f"Error parsing batch link gen response: {e}"); return results_dict


# --- Command Handlers ---
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


# --- Helper Functions for Processing ---

async def _get_product_data(product_id: str) -> tuple[dict | None, str]:
    """Fetches product data from API or scraping, returns data dict and source string."""
    product_details = await fetch_product_details_v2(product_id)
    if product_details:
        details_source = product_details.get('source', 'Unknown') # API or Scraped
        logger.info(f"Retrieved product {product_id} details via {details_source}")
        return product_details, details_source
    else:
        # fetch_product_details_v2 already logged the failure
        details_source = "None" # Indicate complete failure
        # Return a minimal dict for message building consistency, marking failure
        return {'title': f"منتج {product_id}", 'image_url': None, 'price': None, 'currency': None, 'source': details_source}, details_source


async def _generate_offer_links(base_url: str) -> dict[str, str | None]:
    """Generates affiliate links for different offer types."""
    target_urls_map = {}; urls_to_fetch = []
    for offer_key in OFFER_ORDER:
        if offer_key in OFFER_PARAMS:
            offer_info = OFFER_PARAMS[offer_key]
            target_url = build_url_with_offer_params(base_url, offer_info["params"])
            if target_url: target_urls_map[offer_key] = target_url; urls_to_fetch.append(target_url)
            else: logger.warning(f"Could not build target URL for {offer_key}, base: {base_url}")
        else: logger.warning(f"Offer key '{offer_key}' not in OFFER_PARAMS.")
    if not urls_to_fetch: logger.warning(f"No target URLs built for base: {base_url}"); return {}
    all_links_dict = await generate_affiliate_links_batch(urls_to_fetch)
    # Map the generated links back to the offer keys
    generated_links = {}
    for offer_key, target_url in target_urls_map.items():
        generated_links[offer_key] = all_links_dict.get(target_url) # Will be None if failed
    return generated_links


# --- MODIFIED _build_response_message (Shows general price) ---
def _build_response_message(product_data: dict, generated_links: dict) -> str:
    """Builds the Arabic response message string including the general price per link."""
    message_lines = []
    product_title = product_data.get('title', 'منتج غير معروف')
    product_price = product_data.get('price') # General price from API
    product_currency = product_data.get('currency') # General currency from API
    source = product_data.get('source', 'Unknown') # API, Scraped, or None

    # --- Format Price String (General Price) ---
    price_string = ""
    if source == 'API' and product_price and product_currency:
        # Format as (USD 123.45)
        price_string = f"({product_currency} {product_price})"
    elif source == 'Scraped':
        price_string = "(السعر يتطلب API)" # Indicate price only available via API
    # If source is 'None' or price/currency missing from API, price_string remains ""

    # --- Offer Labels ---
    offer_labels = {
        "coin":    "🟨 رابط الشراء بالعملات 🥇",
        "super":   "🟥 المنتج في SuperDeals 🚀",
        "limited": "⏰ المنتج في العرض المحدود",
        "choice":  "🏆 المنتج في عرض choice 🌟",
    }

    # --- Build Message ---
    # 1. Product Name
    message_lines.append(f"📝 <b>إسم المنتج :</b> {product_title[:250]}") # Limit title length
    message_lines.append("\n✳️ <b>قارن الأسعار واكتشف أرخص سعر للمنتج ⬇️🤩</b>\n")

    # 2. Offer Links with General Price
    offers_found = False
    for offer_key in OFFER_ORDER:
        link = generated_links.get(offer_key)
        label = offer_labels.get(offer_key)

        if link and label:
            # Add the general price string if available
            price_part = f" بـ : {price_string}" if price_string else ""
            # Construct the line: Label + Optional Price + Emoji + Link
            message_lines.append(f"{label}{price_part} 🔥:\n{link}\n")
            offers_found = True

    # Handle case where NO offers were found / generated
    if not offers_found:
        if source != 'None': # We got product details but failed to generate links
             message_lines.append("✅ تم العثور على المنتج، لكن لم نتمكن من إنشاء روابط عروض خاصة له حالياً.")
        else: # Product data itself failed
             message_lines.append("❌ لم نتمكن من العثور على المنتج أو إنشاء روابط له.")

    # 3. Footer
    message_lines.append("\n✅ <b>شارك البوت مع أصدقاء ليستفيد الجميع⚡️</b>🤖")

    return "\n".join(message_lines)

# --- MODIFIED _send_telegram_response ---
async def _send_telegram_response(context: ContextTypes.DEFAULT_TYPE, chat_id: int, product_data: dict, message_text: str):
    """Sends the final response (photo or text) to Telegram."""
    product_image = product_data.get('image_url')
    product_id = product_data.get('id', 'N/A') # Use 'id' which is added in process_product_telegram
    source = product_data.get('source', 'Unknown')

    try:
        # Send photo only if available (from API or Scraped)
        if product_image:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=product_image,
                caption=message_text,
                parse_mode=ParseMode.HTML,
            )
        else:
            # Send as text if no image
            logger.info(f"Sending text response for product {product_id} (Source: {source}) as image was not available.")
            await context.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True, # Disable previews for text message with links
            )
    except Exception as send_error:
        logger.error(f"Failed to send message for product {product_id} to chat {chat_id}: {send_error}")
        # Fallback message in Arabic
        try:
            fallback_text = f"⚠️ حدث خطأ أثناء عرض المنتج {product_id}. يرجى المحاولة مرة أخرى."
            await context.bot.send_message(chat_id=chat_id, text=fallback_text)
        except Exception as fallback_error:
             logger.error(f"Failed to send fallback error message for {product_id} to {chat_id}: {fallback_error}")


# --- Core Processing Logic ---

async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes a single product ID and sends the response in Arabic."""
    chat_id = update.effective_chat.id
    logger.info(f"Processing Product ID: {product_id} for chat {chat_id}")

    try:
        # 1. Get Product Data (API/Scrape) - Includes price/currency if from API
        product_data, details_source = await _get_product_data(product_id)

        # Check if source is 'None', indicating complete failure
        if details_source == "None":
             # _get_product_data already logged failure, send user message
             await context.bot.send_message(chat_id=chat_id, text=f"❌ تعذر استرداد بيانات المنتج ذي المعرف {product_id}.")
             return

        # Ensure 'id' key exists in product_data for logging in _send_telegram_response
        product_data['id'] = product_id

        # 2. Generate Affiliate Links (Requires a valid base_url)
        if not base_url:
             logger.error(f"Cannot generate links for {product_id} as base_url is missing.")
             # Inform user link generation failed, even if product data exists
             await context.bot.send_message(chat_id=chat_id, text=f"✅ تم العثور على بيانات المنتج {product_id}، لكن تعذر إنشاء الروابط.")
             return # Stop processing if base_url is bad

        generated_links = await _generate_offer_links(base_url)

        # 3. Build Response Message (Uses general price if available)
        response_text = _build_response_message(product_data, generated_links)

        # 4. Send Response
        await _send_telegram_response(context, chat_id, product_data, response_text)

    except Exception as e:
        logger.exception(f"Unhandled error processing product {product_id} in chat {chat_id}: {e}")
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"حدث خطأ غير متوقع أثناء معالجة المنتج {product_id}. عذراً!"
            )
        except Exception as send_err:
            logger.error(f"Failed to send final error message for {product_id} to {chat_id}: {send_err}")


# --- Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming messages containing potential AliExpress links."""
    if not update.message or not update.message.text: return
    message_text = update.message.text; user = update.effective_user; chat_id = update.effective_chat.id
    logger.info(f"Received message from {user.username or user.id} in chat {chat_id}")

    potential_urls = extract_potential_aliexpress_urls(message_text)
    valid_links_data = [] # Store tuples of (product_id, base_url)

    if not potential_urls:
        await context.bot.send_message(chat_id=chat_id, text="يرجى إرسال رابط منتج AliExpress لإنشاء تخفيضات له.")
        return

    logger.info(f"Found {len(potential_urls)} potential URLs. Analyzing...")
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loading_sticker_msg = None
    try: loading_sticker_msg = await context.bot.send_sticker(chat_id, "CAACAgIAAxkBAAIU1GYOk5jWvCvtykd7TZkeiFFZRdUYAAIjAAMoD2oUJ1El54wgpAY0BA")
    except Exception as sticker_err: logger.warning(f"Could not send loading sticker: {sticker_err}")

    processed_product_ids = set()
    async with aiohttp.ClientSession() as session:
        for url in potential_urls:
            original_url = url; product_id = None; base_url = None; final_url_for_id = None

            # Prepend scheme if missing
            if not url.startswith(('http://', 'https://')):
                 if COMBINED_DOMAIN_REGEX.search(url): url = f"https://{url}"
                 else: logger.debug(f"Skipping non-AE URL without scheme: {original_url}"); continue

            # Resolve short links first
            if SHORT_LINK_DOMAIN_REGEX.match(url):
                resolved_url = await resolve_short_link(url, session)
                if resolved_url: final_url_for_id = resolved_url
                else: logger.warning(f"Failed to resolve short link: {original_url}"); continue # Skip if resolution failed
            elif STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(url):
                 final_url_for_id = url # Use the original URL if it's standard
            else:
                 logger.debug(f"URL does not match known AE patterns: {original_url}"); continue # Skip if not standard or short AE

            # Extract ID and create base URL from the final resolved/standard URL
            if final_url_for_id:
                product_id = extract_product_id(final_url_for_id)
                if product_id:
                    base_url = clean_aliexpress_url(final_url_for_id, product_id)
                    if not base_url:
                         logger.warning(f"Could not clean URL for ID {product_id} from: {final_url_for_id}")
                         product_id = None # Invalidate if base_url creation failed
                else:
                     logger.warning(f"Could not extract ID from final URL: {final_url_for_id} (Original: {original_url})")

            # Add to list if valid and unique
            if product_id and base_url and product_id not in processed_product_ids:
                processed_product_ids.add(product_id)
                valid_links_data.append({"id": product_id, "base": base_url})
                logger.info(f"Added valid product ID {product_id} for processing.")
            elif product_id and product_id in processed_product_ids:
                 logger.debug(f"Skipping duplicate product ID: {product_id}")

    # --- Process Valid Links ---
    if not valid_links_data:
        logger.info(f"No processable AliExpress product links found after analysis.")
        await context.bot.send_message(chat_id=chat_id, text="❌ لم نتمكن من العثور على أي روابط منتجات AliExpress صالحة في رسالتك.")
    else:
        tasks = [process_product_telegram(link_data["id"], link_data["base"], update, context) for link_data in valid_links_data]
        if len(tasks) > 1:
            await context.bot.send_message(chat_id=chat_id, text=f"⏳ جاري معالجة {len(tasks)} منتجات AliExpress. يرجى الانتظار...")
        logger.info(f"Processing {len(tasks)} unique AliExpress products for chat {chat_id}")
        await asyncio.gather(*tasks)
        logger.info(f"Finished processing batch for chat {chat_id}")

    # Delete the loading sticker
    if loading_sticker_msg:
        try: await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
        except Exception as delete_err: logger.warning(f"Could not delete loading sticker: {delete_err}")


# --- Main Function ---
def main() -> None:
    """Starts the bot (Arabic only, with general price per link)."""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # --- Add Handlers ---
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED) & ~filters.COMMAND & filters.Regex(COMBINED_DOMAIN_REGEX),
        handle_message
    ))
    # Fallback for non-link messages
    async def non_aliexpress_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
         await context.bot.send_message(chat_id=update.effective_chat.id, text="يرجى إرسال رابط منتج AliExpress لإنشاء تخفيضات له.")
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.Regex(COMBINED_DOMAIN_REGEX),
        non_aliexpress_message
    ))

    # --- Job Queue for Cache ---
    job_queue = application.job_queue
    job_queue.run_once(periodic_cache_cleanup, 60) # Run 60s after start
    job_queue.run_repeating(periodic_cache_cleanup, interval=timedelta(hours=12), first=timedelta(hours=12)) # Clean every 12 hours

    # --- Log startup info ---
    logger.info("Starting Telegram bot polling (Arabic Interface, General Price per Link)...")
    logger.info(f"Using AliExpress Key: {ALIEXPRESS_APP_KEY[:4]}...")
    logger.info(f"Using Tracking ID: {ALIEXPRESS_TRACKING_ID}")
    logger.info(f"API Settings: Currency={TARGET_CURRENCY}, Lang={TARGET_LANGUAGE}, Country={QUERY_COUNTRY}")
    logger.info(f"Cache expiry: {CACHE_EXPIRY_DAYS} days")
    offer_keys = list(OFFER_PARAMS.keys())
    logger.info(f"Offers configured: {', '.join(offer_keys)}")
    logger.info("Bot is ready and listening...")

    # --- Run the bot ---
    application.run_polling(allowed_updates=Update.ALL_TYPES) # Process all update types

    # --- Shutdown ---
    logger.info("Shutting down thread pool...")
    executor.shutdown(wait=True)
    logger.info("Bot stopped.")

if __name__ == "__main__":
    main()

# --- END OF app.py ---
