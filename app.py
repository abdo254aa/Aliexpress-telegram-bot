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
import html  # Import html module for escaping

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

load_dotenv()

# --- Environment Variables ---
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

# --- Logging ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# --- Variable Checks ---
if not all([TELEGRAM_BOT_TOKEN, ALIEXPRESS_APP_KEY, ALIEXPRESS_APP_SECRET, ALIEXPRESS_TRACKING_ID]):
    logger.error("Missing required environment variables.")
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

# --- Regex Patterns ---
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)
SHORT_LINK_DOMAIN_REGEX = re.compile(r'https?://(?:s\.click\.aliexpress\.com/e/|a\.aliexpress\.com/_)[a-zA-Z0-9_-]+/?', re.IGNORECASE)
COMBINED_DOMAIN_REGEX = re.compile(r'aliexpress\.com|s\.click\.aliexpress\.com|a\.aliexpress\.com', re.IGNORECASE)

# --- Offer Config ---
OFFER_PARAMS = {
    "coin": {"params": {"sourceType": "620%26channel=coin"}},
    "super": {"params": {"sourceType": "562", "channel": "sd"}},
    "limited": {"params": {"sourceType": "561", "channel": "limitedoffers"}},
    "choice": {"params": {"sourceType": "680", "channel": "choice"}},
}
OFFER_ORDER = ["coin", "super", "limited", "choice"]

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

# --- Helper Functions ---
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
        return base_url

# --- Cache Cleanup Job ---
async def periodic_cache_cleanup(context: ContextTypes.DEFAULT_TYPE):
    try:
        product_expired = await product_cache.clear_expired()
        link_expired = await link_cache.clear_expired()
        resolved_expired = await resolved_url_cache.clear_expired()
        logger.info(f"Cache cleanup: Removed {product_expired} product, {link_expired} link, {resolved_expired} resolved URL items.")
        logger.info(f"Cache stats: {len(product_cache.cache)} products, {len(link_cache.cache)} links, {len(resolved_url_cache.cache)} resolved URLs in cache.")
    except Exception as e:
        logger.error(f"Error in periodic cache cleanup job: {e}")

# --- Fetch Product Details ---
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
        if isinstance(response_data, str):
            response_data = json.loads(response_data)

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
            'currency': product_data.get('target_sale_price_currency', TARGET_CURRENCY),
            'title': product_data.get('product_title', f'ظ…ظ†طھط¬ {product_id}'),
            'source': 'API'
        }
        await product_cache.set(product_id, product_info)
        expiry_date = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
        logger.info(f"Cached product {product_id} from API until {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}")
        return product_info

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for product {product_id}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Error parsing product details response for ID {product_id}: {e}")
        return None

# --- Generate Affiliate Links ---
async def generate_affiliate_links(product_id: str, base_url: str) -> dict:
    links = {}
    cache_key = f"links_{product_id}"
    cached_links = await link_cache.get(cache_key)
    if cached_links:
        logger.info(f"Cache hit for affiliate links: {product_id}")
        return cached_links

    for offer_name in OFFER_ORDER:
        offer_config = OFFER_PARAMS.get(offer_name)
        if offer_config:
            params = offer_config.get("params", {})
            link = build_url_with_offer_params(base_url, params)
            if link:
                links[offer_name] = link

    if links:
        await link_cache.set(cache_key, links)
        logger.info(f"Generated and cached affiliate links for product {product_id}")

    return links

# --- Format Message ---
def format_message(product_info: dict, links: dict) -> str:
    title = html.escape(product_info.get('title', 'ظ…ظ†طھط¬ ط؛ظٹط± ظ…ط¹ط±ظˆظپ'))
    price = product_info.get('price')
    currency = product_info.get('currency', '')

    price_text = f"ًں’° <b>ط§ظ„ط³ط¹ط±:</b> {html.escape(str(price))} {html.escape(currency)}" if price else "ًں’° <b>ط§ظ„ط³ط¹ط±:</b> ط؛ظٹط± ظ…طھظˆظپط±"

    link_lines = []
    link_labels = {
        "coin": "ًںھ™ ط¹ط±ظˆط¶ ط§ظ„ظƒظˆظٹظ†",
        "super": "ًں”¥ ط³ظˆط¨ط± ط¯ظٹظ„ط²",
        "limited": "âڈ° ط¹ط±ظˆط¶ ظ…ط­ط¯ظˆط¯ط©",
        "choice": "â­گ ط§ط®طھظٹط§ط± ط¹ظ„ظٹ ط¥ظƒط³ط¨ط±ظٹط³",
    }
    for offer_name in OFFER_ORDER:
        if offer_name in links:
            label = link_labels.get(offer_name, offer_name)
            link_lines.append(f'<a href="{links[offer_name]}">{label}</a>')

    links_text = "\n".join(link_lines) if link_lines else "ظ„ط§ طھظˆط¬ط¯ ط±ظˆط§ط¨ط· ظ…طھط§ط­ط©"

    message = (
        f"ًں›چï¸ڈ <b>{title}</b>\n\n"
        f"{price_text}\n\n"
        f"ًں”— <b>ط±ظˆط§ط¨ط· ط§ظ„ط´ط±ط§ط،:</b>\n{links_text}"
    )
    return message

# --- Handle Message ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    text = update.message.text
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    potential_urls = extract_potential_aliexpress_urls(text)
    if not potential_urls:
        return

    aliexpress_urls = [url for url in potential_urls if COMBINED_DOMAIN_REGEX.search(url)]
    if not aliexpress_urls:
        return

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    async with aiohttp.ClientSession() as session:
        for url in aliexpress_urls[:1]:
            try:
                final_url = url
                if SHORT_LINK_DOMAIN_REGEX.match(url):
                    final_url = await resolve_short_link(url, session)
                    if not final_url:
                        logger.warning(f"Could not resolve short link: {url}")
                        continue

                product_id = extract_product_id(final_url)
                if not product_id:
                    logger.warning(f"Could not extract product ID from: {final_url}")
                    continue

                base_url = clean_aliexpress_url(final_url, product_id)
                if not base_url:
                    continue

                product_info = await fetch_product_details_v2(product_id)
                if not product_info:
                    logger.warning(f"Could not fetch product details for ID: {product_id}")
                    continue

                links = await generate_affiliate_links(product_id, base_url)
                message_text = format_message(product_info, links)
                image_url = product_info.get('image_url')

                if image_url:
                    try:
                        await context.bot.send_photo(
                            chat_id=chat_id,
                            photo=image_url,
                            caption=message_text,
                            parse_mode=ParseMode.HTML,
                            reply_to_message_id=message_id
                        )
                    except Exception as photo_err:
                        logger.warning(f"Failed to send photo, sending text only: {photo_err}")
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=message_text,
                            parse_mode=ParseMode.HTML,
                            reply_to_message_id=message_id,
                            disable_web_page_preview=False
                        )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=message_text,
                        parse_mode=ParseMode.HTML,
                        reply_to_message_id=message_id,
                        disable_web_page_preview=False
                    )

            except Exception as e:
                logger.exception(f"Error processing URL {url}: {e}")

# --- Start Command ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "ًں‘‹ ظ…ط±ط­ط¨ط§ظ‹! ط£ط±ط³ظ„ ظ„ظٹ ط£ظٹ ط±ط§ط¨ط· ظ…ظ†طھط¬ ظ…ظ† AliExpress ظˆط³ط£ظˆظ„ظ‘ط¯ ظ„ظƒ ط±ظˆط§ط¨ط· طھط§ط¨ط¹ط© ظ…طھط¹ط¯ط¯ط©! ًں›چï¸ڈ"
    )

# --- Main ---
def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    job_queue = application.job_queue
    job_queue.run_repeating(periodic_cache_cleanup, interval=3600, first=3600)

    logger.info("Bot started successfully!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
