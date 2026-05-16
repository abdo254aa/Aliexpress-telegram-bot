import requests
from bs4 import BeautifulSoup

def get_product_details_by_id(product_id):
    url = f"https://www.aliexpress.com/item/{product_id}.html"
    
    # 1. إضافة الهيدرز لخداع نظام الحماية وتجنب الحظر الفوري (مهم جداً)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept-Language": "ar-EG,ar;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive"
    }
    
    try:
        # إرسال الطلب مع تمرير الهيدرز ووقت انتظار كافٍ
        response = requests.get(url, headers=headers, timeout=15)
        
        # إذا واجه الموقع مشكلة حظر أو صفحة غير موجودة سيتوقف هنا ويعطي خطأ واضحاً في الـ Terminal
        response.raise_for_status()  
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 2. البحث عن اسم المنتج وصورته عبر وسوم الـ Meta الثابتة (Open Graph)
        title_meta = soup.find('meta', property='og:title') or soup.find('meta', name='twitter:title')
        image_meta = soup.find('meta', property='og:image') or soup.find('meta', name='twitter:image')
        
        title = title_meta['content'].strip() if title_meta and title_meta.get('content') else None
        image_url = image_meta['content'].strip() if image_meta and image_meta.get('content') else None
        
        # حل بديل أول: إذا لم يجد وسم الميتا الخاص بالعنوان، نأخذه من عنوان الصفحة الرئيسي <title>
        if not title and soup.title:
            title = soup.title.text.strip()
            
        # تنظيف العنوان من اللواحق الإعلانية الخاصة بـ AliExpress (مثل اسم الموقع في النهاية)
        if title:
            # إذا كان العنوان يحتوي على علامة التوقف "|" نقوم بأخذ الجزء الأول فقط ليكون نظيفاً
            if '|' in title:
                title = title.split('|')[0].strip()
            elif '-' in title:
                title = title.split('-')[0].strip()
        else:
            title = f"منتج رقم {product_id}"

        return title, image_url

    except requests.exceptions.RequestException as e:
        print(f"[Scraper Error] خطأ أثناء الاتصال بـ AliExpress للمنتج {product_id}: {e}")
        return f"منتج رقم {product_id}", None
    except Exception as e:
        print(f"[Scraper Error] خطأ أثناء تحليل بيانات كود الصفحة للمنتج {product_id}: {e}")
        return f"منتج رقم {product_id}", None
