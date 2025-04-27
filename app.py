# --- START OF MODIFIED app.py ---

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

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    JobQueue,
    CallbackQueryHandler, # Import CallbackQueryHandler
)
from telegram.constants import ParseMode, ChatAction

import iop
from aliexpress_utils import get_product_details_by_id

# Import translations
from translations import get_text, get_offer_name, OFFER_PARAMS_LANG, OFFER_ORDER, DEFAULT_LANG

load_dotenv()

# --- Keep your existing environment variable loading and initial setup ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ALIEXPRESS_APP_KEY = os.getenv('ALIEXPRESS_APP_KEY')
ALIEXPRESS_APP_SECRET = os.getenv('ALIEXPRESS_APP_SECRET')
TARGET_CURRENCY = os.getenv('TARGET_CURRENCY', 'USD')
TARGET_LANGUAGE = os.getenv('TARGET_LANGUAGE', 'en') # This might be less relevant now per-user lang is used
QUERY_COUNTRY = os.getenv('QUERY_COUNTRY', 'US')
ALIEXPRESS_TRACKING_ID = os.getenv('ALIEXPRESS_TRACKING_ID', 'default')
ALIEXPRESS_API_URL = 'https://api-sg.aliexpress.com/sync'
QUERY_FIELDS = 'product_main_image_url,target_sale_price,product_title,target_sale_price_currency'
CACHE_EXPIRY_DAYS = 1
CACHE_EXPIRY_SECONDS = CACHE_EXPIRY_DAYS * 24 * 60 * 60
MAX_WORKERS = 10

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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

# --- Keep your existing REGEX definitions ---
URL_REGEX = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+|\b(?:s\.click\.|a\.)?aliexpress\.(?:com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id|th|ar)(?:\.[\w-]+)?/[^\s<>"]*', re.IGNORECASE)
PRODUCT_ID_REGEX = re.compile(r'/item/(\d+)\.html')
STANDARD_ALIEXPRESS_DOMAIN_REGEX = re.compile(r'https?://(?!a\.|s\.click\.)([\w-]+\.)?aliexpress\.(com|ru|es|fr|pt|it|pl|nl|co\.kr|co\.jp|com\.br|com\.tr|com\.vn|us|id\.aliexpress\.com|th\.aliexpress\.com|ar\.aliexpress\.com)(\.([\w-]+))?(/.*)?', re.IGNORECASE)
SHORT_LINK_DOMAIN_REGEX = re.compile(r'https?://(?:s\.click\.aliexpress\.com/e/|a\.aliexpress\.com/_)[a-zA-Z0-9_-]+/?', re.IGNORECASE)
COMBINED_DOMAIN_REGEX = re.compile(r'aliexpress\.com|s\.click\.aliexpress\.com|a\.aliexpress\.com', re.IGNORECASE)

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

# --- Helper function to get user language ---
def get_user_language(context: ContextTypes.DEFAULT_TYPE) -> str:
    return context.user_data.get('language', DEFAULT_LANG)

# --- Keep existing functions (resolve_short_link, extract_product_id, etc.) ---
# --- Make sure they don't have hardcoded user-facing text ---
# --- (The existing ones seem fine in this regard) ---
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
                    logger.info(f"Updated URL with correct country: {final_url}")
                    try:
                        logger.info(f"Re-fetching URL with updated country parameter: {final_url}")
                        async with session.get(final_url, allow_redirects=True, timeout=10) as country_response:
                            if country_response.status == 200 and country_response.url:
                                final_url = str(country_response.url)
                                logger.info(f"Re-fetched URL with correct country: {final_url}")
                    except Exception as e:
                        logger.warning(f"Error re-fetching URL with updated country parameter: {e}")

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
        # Ensure the domain is www.aliexpress.com for consistency
        netloc_parts = parsed_url.netloc.split('.')
        if 'aliexpress' in netloc_parts:
            # Find the TLD (com, ru, etc.)
            tld_index = -1
            for i, part in enumerate(reversed(netloc_parts)):
                if part == 'aliexpress':
                    tld_index = len(netloc_parts) - 1 - i + 1
                    break
            if tld_index != -1 and tld_index < len(netloc_parts):
                tld = netloc_parts[tld_index]
                # Rebuild as www.aliexpress.TLD
                netloc = f"www.aliexpress.{tld}"
            else: # Fallback if structure is unexpected
                 netloc = "www.aliexpress.com"
        else: # If not an aliexpress domain somehow, default
             netloc = "www.aliexpress.com"


        base_url = urlunparse((
            parsed_url.scheme or 'https',
            netloc, # Use normalized netloc
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
        # Use the netloc directly from the cleaned base_url
        netloc = parsed_base.netloc

        # Create the redirect URL part first
        query_string_for_redirect = urlencode(params_to_add)
        redirect_url = urlunparse((
             parsed_base.scheme,
             netloc,
             parsed_base.path,
             '',
             query_string_for_redirect,
             ''
        ))

        # Now build the final star.aliexpress.com URL
        final_params = {
            "platform": "AE",
            "businessType": "ProductDetail",
            "redirectUrl": redirect_url
        }
        final_query_string = urlencode(final_params)

        star_url = urlunparse((
            'https', # Always use https for star link
            'star.aliexpress.com',
            '/share/share.htm',
            '',
            final_query_string,
            ''
        ))
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

    # Use the global TARGET_LANGUAGE for the API call, as user pref doesn't map directly
    api_language = TARGET_LANGUAGE

    def _execute_api_call():
        try:
            request = iop.IopRequest('aliexpress.affiliate.productdetail.get')
            request.add_api_param('fields', QUERY_FIELDS)
            request.add_api_param('product_ids', product_id)
            request.add_api_param('target_currency', TARGET_CURRENCY)
            request.add_api_param('target_language', api_language) # Use global setting here
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
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON response for product {product_id}: {json_err}. Response: {response_data[:500]}")
                return None

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
            # Try scraping directly if API fails to find product
            logger.info(f"Attempting scrape fallback after empty API product list for {product_id}")
            try:
                 loop_inner = asyncio.get_event_loop()
                 scraped_name, scraped_image = await loop_inner.run_in_executor(
                    executor, get_product_details_by_id, product_id
                 )
                 if scraped_name:
                      logger.info(f"Successfully scraped details after empty API response for product ID: {product_id}")
                      product_info = {'title': scraped_name, 'image_url': scraped_image, 'price': None, 'currency': None, 'source': 'Scraped'}
                      await product_cache.set(product_id, product_info) # Cache scraped result
                      return product_info
                 else:
                     logger.warning(f"Scraping also failed after empty API response for product ID: {product_id}")
                     return None # Indicate failure if both fail
            except Exception as scrape_err:
                logger.error(f"Error during scraping fallback after empty API response for {product_id}: {scrape_err}")
                return None


        product_data = products[0]
        product_info = {
            'image_url': product_data.get('product_main_image_url'),
            'price': product_data.get('target_sale_price'),
            'currency': product_data.get('target_sale_price_currency', TARGET_CURRENCY),
            'title': product_data.get('product_title', f'Product {product_id}'),
            'source': 'API' # Indicate source
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
            results_dict[url] = None # Placeholder
            uncached_urls.append(url)

    if not uncached_urls:
        logger.info("All required affiliate links retrieved from cache.")
        return results_dict

    logger.info(f"Generating affiliate links for {len(uncached_urls)} uncached URLs via batch API...")
    source_values_str = ",".join(uncached_urls) # API expects the full URL including star.aliexpress...

    def _execute_batch_link_api():
        try:
            request = iop.IopRequest('aliexpress.affiliate.link.generate')
            request.add_api_param('promotion_link_type', '0') # 0 for promotion link
            request.add_api_param('source_values', source_values_str)
            request.add_api_param('tracking_id', ALIEXPRESS_TRACKING_ID)
            return aliexpress_client.execute(request)
        except Exception as e:
            # Log the specific error from the API call thread
            logger.error(f"Error in batch link API call thread: {e}")
            return None

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, _execute_batch_link_api)

    if not response or not response.body:
        logger.error(f"Batch link generation API call failed or returned empty body for {len(uncached_urls)} URLs.")
        # Return the dictionary with cached results and Nones for failed ones
        return results_dict

    try:
        response_data = response.body
        if isinstance(response_data, str):
            try:
                response_data = json.loads(response_data)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON response for batch link generation: {json_err}. Response: {response_data[:500]}")
                return results_dict # Return current results

        if 'error_response' in response_data:
            error_details = response_data.get('error_response', {})
            logger.error(f"API Error for Batch Link Generation: Code={error_details.get('code', 'N/A')}, Msg={error_details.get('msg', 'Unknown API error')}")
            return results_dict # Return current results

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
            # Potentially log the URLs that might have caused the error if possible
            logger.error(f"Failed URLs (request): {uncached_urls}")
            return results_dict # Return current results

        result = resp_result_outer.get('result', {})
        if not result:
            logger.error(f"Missing 'result' key. Response: {resp_result_outer}")
            return results_dict

        links_data = result.get('promotion_links', {}).get('promotion_link', [])
        if not links_data or not isinstance(links_data, list):
            logger.warning(f"No 'promotion_links' found or not a list in batch response. Response: {result}")
            return results_dict # Return current results

        expiry_date = datetime.now() + timedelta(days=CACHE_EXPIRY_DAYS)
        logger.info(f"Processing {len(links_data)} links from batch API response.")
        api_returned_links_map = {}
        for link_info in links_data:
            if isinstance(link_info, dict):
                source_url_returned = link_info.get('source_value')
                promo_link = link_info.get('promotion_link')

                if source_url_returned and promo_link:
                     # The API returns the source_value exactly as sent
                     if source_url_returned in uncached_urls:
                          api_returned_links_map[source_url_returned] = promo_link
                     else:
                           logger.warning(f"Received link for an unexpected source_value from API: {source_url_returned}")
                else:
                    logger.warning(f"Missing 'source_value' or 'promotion_link' in batch response item: {link_info}")
            else:
                 logger.warning(f"Promotion link data item is not a dictionary: {link_info}")


        # Update results_dict and cache for successfully generated links
        for url in uncached_urls:
            if url in api_returned_links_map:
                promo_link = api_returned_links_map[url]
                results_dict[url] = promo_link
                await link_cache.set(url, promo_link)
                logger.debug(f"Cached affiliate link for {url} until {expiry_date.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                 # Keep results_dict[url] as None if it failed
                 logger.warning(f"No affiliate link returned or processed from batch API for requested URL: {url}")

        return results_dict

    except Exception as e:
        logger.exception(f"Error parsing batch link generation response: {e}")
        return results_dict # Return whatever was processed before the error

# --- MODIFIED start handler ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message with inline buttons to choose the language."""
    keyboard = [
        [
            InlineKeyboardButton("English 🇬🇧", callback_data='lang_en'),
            InlineKeyboardButton("العربية 🇸🇦", callback_data='lang_ar'),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # Send message asking to choose language (use default 'en' for this initial prompt)
    await update.message.reply_text(get_text('choose_language', 'en'), reply_markup=reply_markup)
    # Also send the Arabic version so the user sees both options clearly
    await update.message.reply_text(get_text('choose_language', 'ar'), reply_markup=reply_markup)


# --- NEW CallbackQueryHandler for language selection ---
async def select_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and sets the language."""
    query = update.callback_query
    await query.answer()  # Answer the callback query

    lang_code = query.data.split('_')[1] # 'lang_en' -> 'en'
    context.user_data['language'] = lang_code

    logger.info(f"User {query.from_user.id} selected language: {lang_code}")

    # Edit the message to show confirmation and the welcome message in the chosen language
    await query.edit_message_text(
        text=f"{get_text('language_set', lang_code)}\n\n{get_text('welcome', lang_code)}",
        parse_mode=ParseMode.HTML
    )

# --- MODIFIED _get_product_data ---
async def _get_product_data(product_id: str) -> tuple[dict | None, str]:
    """Fetches product data from API or scraping, returns data and source."""
    product_details = await fetch_product_details_v2(product_id)
    details_source = "None" # Default if nothing works

    if product_details:
        details_source = product_details.get('source', 'API') # Use source from fetch_product_details_v2
        logger.info(f"Successfully fetched/scraped details ({details_source}) for product ID: {product_id}")
        return product_details, details_source
    else:
        # fetch_product_details_v2 should now handle the scrape fallback internally
        logger.warning(f"API and Scrape fallback failed for product ID: {product_id}")
        # Return a minimal dict indicating failure, but still allowing message construction
        return {'title': f"Product {product_id}", 'image_url': None, 'price': None, 'currency': None, 'source': 'None'}, details_source


# --- MODIFIED _generate_offer_links ---
async def _generate_offer_links(base_url: str) -> dict[str, str | None]:
    """Generates affiliate links for different offer types."""
    target_urls_map = {} # Map offer_key -> target_url_sent_to_api
    urls_to_fetch = []

    for offer_key in OFFER_ORDER:
        offer_info = OFFER_PARAMS_LANG[offer_key] # Use the language-aware config
        # Pass the specific parameters for this offer type
        target_url = build_url_with_offer_params(base_url, offer_info["params"])
        if target_url:
            target_urls_map[offer_key] = target_url
            urls_to_fetch.append(target_url)
        else:
            logger.warning(f"Could not build target URL for offer {offer_key} with base {base_url}")

    if not urls_to_fetch:
        logger.warning(f"No target URLs could be built for base URL: {base_url}")
        return {}

    # Generate links for all built URLs in one batch
    all_links_dict = await generate_affiliate_links_batch(urls_to_fetch)

    generated_links = {} # Map offer_key -> final_promo_link
    for offer_key, target_url in target_urls_map.items():
        # Retrieve the generated link using the target_url as the key
        promo_link = all_links_dict.get(target_url)
        generated_links[offer_key] = promo_link # Will be None if generation failed
        if not promo_link:
            logger.warning(f"Failed to get affiliate link for offer {offer_key} (target: {target_url})")

    return generated_links


# --- MODIFIED _build_response_message ---
def _build_response_message(product_data: dict, generated_links: dict, details_source: str, lang: str) -> str:
    """Builds the response message string using translations."""
    message_lines = []
    product_title = product_data.get('title', get_text('unknown_product', lang)) # Add 'unknown_product' to translations if needed
    product_price = product_data.get('price')
    product_currency = product_data.get('currency', '')

    message_lines.append(f"<b>{product_title[:250]}</b>") # Keep title short

    if details_source == "API" and product_price:
        price_str = f"{product_price} {product_currency}".strip()
        message_lines.append(f"\n{get_text('price_label', lang)} {price_str}\n")
    elif details_source == "Scraped":
        message_lines.append(f"\n{get_text('price_unavailable_scraped', lang)}\n")
    else: # Source is None or failed
        message_lines.append(f"\n{get_text('product_details_unavailable', lang)}\n")

    message_lines.append(get_text('special_offers_label', lang))
    message_lines.append("──────────────\n")

    offers_available = False
    for offer_key in OFFER_ORDER:
        link = generated_links.get(offer_key)
        offer_name = get_offer_name(offer_key, lang) # Use helper to get translated name
        if link:
            message_lines.append(f'▫️ {offer_name}: <a href="{link}">{get_text("get_discount", lang)}</a>\n')
            offers_available = True
        else:
            message_lines.append(f"▫️ {offer_name}: {get_text('not_available', lang)}")

    if not offers_available and details_source != 'None':
         # If we got product details but no offers, simplify the message
         message_lines = [
             f"<b>{product_title[:250]}</b>",
             f"\n{get_text('price_label', lang)} {product_price} {product_currency}\n" if details_source == "API" and product_price else f"\n{get_text('price_unavailable_scraped', lang)}\n",
             f"\n{get_text('no_offers_found', lang)}"
         ]
    elif not offers_available and details_source == 'None':
        # If we couldn't even get product details, just show that error
         message_lines = [
             f"<b>{product_title[:250]}</b>", # Still show title if available (e.g., "Product 12345")
             f"\n{get_text('product_details_unavailable', lang)}\n",
             f"\n{get_text('no_offers_found', lang)}"
         ]

    # Add footer only if offers were available
    if offers_available:
        message_lines.append("──────────────")
        message_lines.append(f"\n{get_text('follow_us_label', lang)}")
        message_lines.append(f"{get_text('telegram_label', lang)} @Aliexpress_Deal_Dz")
        message_lines.append(f"{get_text('github_label', lang)} <a href='https://github.com/ReizoZ'>ReizoZ</a>")
        message_lines.append(f"{get_text('discord_label', lang)} {get_text('discord_cta', lang)}") # Using placeholder text for discord link
        message_lines.append(f"\n{get_text('footer_text', lang)}")

    return "\n".join(message_lines)

# --- MODIFIED _build_reply_markup ---
def _build_reply_markup(lang: str) -> InlineKeyboardMarkup:
    """Builds the inline keyboard markup using translations."""
    keyboard = [
        [
            InlineKeyboardButton(get_text('choice_day_button', lang), url="https://s.click.aliexpress.com/e/_oCPK1K1"),
            InlineKeyboardButton(get_text('best_deals_button', lang), url="https://s.click.aliexpress.com/e/_onx9vR3")
        ],
        [
            InlineKeyboardButton(get_text('github_button', lang), url="https://github.com/ReizoZ"),
            InlineKeyboardButton(get_text('discord_button', lang), url="https://discord.gg/9QzECYfmw8"),
            InlineKeyboardButton(get_text('channel_button', lang), url="https://t.me/Aliexpress_Deal_Dz")
        ],
        [
            InlineKeyboardButton(get_text('support_button', lang), url="https://ko-fi.com/reizoz")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

# --- MODIFIED _send_telegram_response ---
async def _send_telegram_response(context: ContextTypes.DEFAULT_TYPE, chat_id: int, product_data: dict, message_text: str, reply_markup: InlineKeyboardMarkup, lang: str):
    """Sends the final response (photo or text) to Telegram."""
    product_image = product_data.get('image_url')
    product_id = product_data.get('id', 'N/A')
    # Check if the message indicates no offers were found (using translated text)
    no_offer_text_en = get_text('no_offers_found', 'en') # Use key text for reliable check
    no_offer_text_ar = get_text('no_offers_found', 'ar') # Use key text for reliable check


    try:
        # Send photo only if available AND offers were found
        if product_image and no_offer_text_en not in message_text and no_offer_text_ar not in message_text:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=product_image,
                caption=message_text,
                parse_mode=ParseMode.HTML,
                reply_markup=reply_markup
            )
        else:
            # Send as text if no image or if no offers were found
            await context.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True, # Keep preview disabled for text mode
                reply_markup=reply_markup
            )
    except Exception as send_error:
        logger.error(f"Failed to send message for product {product_id} to chat {chat_id}: {send_error}")
        # Fallback message if sending fails
        try:
            fallback_text = get_text('error_displaying_product', lang, product_id=product_id)
            await context.bot.send_message(
                chat_id=chat_id,
                text=fallback_text,
                reply_markup=reply_markup # Still provide buttons if possible
            )
        except Exception as fallback_error:
             # Log using translated text if possible, otherwise default
             log_message = get_text('error_fallback_failed', lang, product_id=product_id, chat_id=chat_id)
             logger.error(f"{log_message}: {fallback_error}")


# --- MODIFIED process_product_telegram ---
async def process_product_telegram(product_id: str, base_url: str, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes a single product ID and sends the response."""
    chat_id = update.effective_chat.id
    lang = get_user_language(context) # Get user's language
    logger.info(f"Processing Product ID: {product_id} for chat {chat_id} (Lang: {lang})")

    try:
        # 1. Get Product Data (API/Scrape)
        product_data, details_source = await _get_product_data(product_id)

        if not product_data or details_source == "None":
             logger.error(f"Failed to get any product data (API or Scraped) for {product_id}")
             await context.bot.send_message(chat_id=chat_id, text=get_text('error_api_or_scrape', lang, product_id=product_id))
             return

        product_data['id'] = product_id # Add ID for logging/error messages

        # 2. Generate Affiliate Links
        generated_links = await _generate_offer_links(base_url)

        # 3. Build Response Message
        response_text = _build_response_message(product_data, generated_links, details_source, lang)

        # 4. Build Reply Markup
        reply_markup = _build_reply_markup(lang)

        # 5. Send Response
        await _send_telegram_response(context, chat_id, product_data, response_text, reply_markup, lang)

    except Exception as e:
        logger.exception(f"Unhandled error processing product {product_id} in chat {chat_id}: {e}")
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=get_text('error_processing_product', lang, product_id=product_id)
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
    lang = get_user_language(context) # Get user's language
    logger.info(f"Received message from {user.username or user.id} in chat {chat_id} (Lang: {lang})")

    potential_urls = extract_potential_aliexpress_urls(message_text)
    if not potential_urls:
        await context.bot.send_message(
            chat_id=chat_id,
            text=get_text('error_no_link', lang) # Use translated text
        )
        return

    logger.info(f"Found {len(potential_urls)} potential URLs in message from {user.username or user.id}")

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    loading_sticker_msg = None
    # Try sending sticker, but don't fail if it doesn't work
    try:
        # Consider using a language-neutral or universally understood sticker if possible
        loading_sticker_msg = await context.bot.send_sticker(chat_id, "CAACAgIAAxkBAAIU1GYOk5jWvCvtykd7TZkeiFFZRdUYAAIjAAMoD2oUJ1El54wgpAY0BA")
    except Exception as sticker_err:
        logger.warning(f"Could not send loading sticker: {sticker_err}")


    processed_product_ids = set()
    tasks = []
    async with aiohttp.ClientSession() as session:
        for url in potential_urls:
            original_url = url
            product_id = None
            base_url = None

            # Prepend scheme if missing and looks like an AE link
            if not url.startswith(('http://', 'https://')):
                 if COMBINED_DOMAIN_REGEX.search(url): # Check if it contains AE domains
                    logger.debug(f"Prepending https:// to potential URL: {url}")
                    url = f"https://{url}"
                 else:
                    logger.debug(f"Skipping potential URL without scheme or known AE domain: {original_url}")
                    continue # Skip if it doesn't look like an AE link

            # Check standard URLs first
            if STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(url):
                product_id = extract_product_id(url)
                if product_id:
                    base_url = clean_aliexpress_url(url, product_id)
                    logger.debug(f"Standard URL: {url} -> ID: {product_id}, Base: {base_url}")

            # Check short links if not a standard one or ID extraction failed
            elif SHORT_LINK_DOMAIN_REGEX.match(url):
                logger.debug(f"Potential short link: {url}. Resolving...")
                final_url = await resolve_short_link(url, session)
                if final_url:
                    product_id = extract_product_id(final_url)
                    if product_id:
                        base_url = clean_aliexpress_url(final_url, product_id)
                        logger.debug(f"Resolved short link: {url} -> {final_url} -> ID: {product_id}, Base: {base_url}")
                    else:
                         logger.warning(f"Could not extract ID from resolved URL: {final_url} (Original: {original_url})")
                else:
                     logger.warning(f"Could not resolve short link: {original_url}")
            # else: # Log URLs that didn't match either regex if needed
            #      logger.debug(f"URL did not match standard or short link patterns: {original_url}")


            # Add task if valid product ID and base URL found, and not already processed
            if product_id and base_url and product_id not in processed_product_ids:
                processed_product_ids.add(product_id)
                # Pass update and context to the task function
                tasks.append(process_product_telegram(product_id, base_url, update, context))
            elif product_id and product_id in processed_product_ids:
                 logger.debug(f"Skipping duplicate product ID: {product_id}")
            # Log cases where ID or base_url couldn't be determined
            elif not product_id and (STANDARD_ALIEXPRESS_DOMAIN_REGEX.match(url) or SHORT_LINK_DOMAIN_REGEX.match(url)):
                 logger.warning(f"Could not determine Product ID for likely AE URL: {original_url}")
            elif product_id and not base_url:
                 logger.warning(f"Could not determine Base URL for Product ID {product_id} from URL: {original_url}")


    # --- Message indicating processing or no valid links ---
    if not tasks:
        logger.info(f"No processable AliExpress product links found after filtering/resolution.")
        await context.bot.send_message(
            chat_id=chat_id,
            text=get_text('error_no_valid_links', lang) # Use translated text
        )
    else:
        if len(tasks) > 1:
            # Send processing message only if multiple items
            await context.bot.send_message(
                chat_id=chat_id,
                text=get_text('processing_multiple', lang, count=len(tasks)) # Use translated text
            )
        logger.info(f"Processing {len(tasks)} unique AliExpress products for chat {chat_id}")
        await asyncio.gather(*tasks) # Execute all processing tasks concurrently

    # Delete the loading sticker if it was sent
    if loading_sticker_msg:
        try:
            await context.bot.delete_message(chat_id, loading_sticker_msg.message_id)
        except Exception as delete_err:
            # Log deletion errors but don't crash
            logger.warning(f"Could not delete loading sticker: {delete_err}")


def main() -> None:
    """Starts the bot."""
    # Consider using persistence if you want language choice to survive restarts
    # from telegram.ext import PicklePersistence
    # persistence = PicklePersistence(filepath="bot_data.pkl")
    # application = Application.builder().token(TELEGRAM_BOT_TOKEN).persistence(persistence).build()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Command handler for /start
    application.add_handler(CommandHandler("start", start))

    # Callback query handler for language selection buttons
    application.add_handler(CallbackQueryHandler(select_language, pattern='^lang_'))

    # Message handler for AliExpress links (Text or Forwarded)
    application.add_handler(MessageHandler(
        (filters.TEXT | filters.FORWARDED) & ~filters.COMMAND & filters.Regex(COMBINED_DOMAIN_REGEX),
        handle_message
    ))

    # Message handler for text that doesn't contain AE links
    async def non_aliexpress_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
         lang = get_user_language(context)
         await context.bot.send_message(
             chat_id=update.effective_chat.id,
             text=get_text('prompt_send_link', lang) # Use translated text
         )
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND & ~filters.Regex(COMBINED_DOMAIN_REGEX),
        non_aliexpress_message
    ))

    # Setup Job Queue for cache cleanup
    job_queue = application.job_queue
    job_queue.run_once(periodic_cache_cleanup, 60) # Run once shortly after start
    job_queue.run_repeating(periodic_cache_cleanup, interval=timedelta(days=1), first=timedelta(days=1))

    logger.info("Starting Telegram bot polling...")
    logger.info(f"Using AliExpress Key: {ALIEXPRESS_APP_KEY[:4]}...")
    logger.info(f"Using Tracking ID: {ALIEXPRESS_TRACKING_ID}")
    logger.info(f"API Settings: Currency={TARGET_CURRENCY}, Lang={TARGET_LANGUAGE}, Country={QUERY_COUNTRY}")
    logger.info(f"Cache expiry: {CACHE_EXPIRY_DAYS} days")
    offer_keys = list(OFFER_PARAMS_LANG.keys())
    logger.info(f"Offers configured: {', '.join(offer_keys)}")
    logger.info("Bot is ready and listening...")

    # Run the bot
    application.run_polling()

    logger.info("Shutting down thread pool...")
    executor.shutdown(wait=True)
    logger.info("Bot stopped.")

if __name__ == "__main__":
    main()

# --- END OF MODIFIED app.py ---
