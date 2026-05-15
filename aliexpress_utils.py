import requests
from bs4 import BeautifulSoup

def get_product_details_by_id(product_id):
    url = f"https://www.aliexpress.com/item/{product_id}.html"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')
        title_element = soup.find('h1', {'class': 'product-title-text'})
        image_element = soup.find('img', {'id': 'image-0'})  # Or a more specific selector
        title = title_element.text.strip() if title_element else "Title Not Found"
        image_url = image_element.get('src') if image_element else None
        return title, image_url
    except requests.exceptions.RequestException as e:
        print(f"Error fetching product details: {e}")
        return None, None
    except Exception as e:
        print(f"Error parsing product details: {e}")
        return None, None
