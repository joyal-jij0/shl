import requests
from bs4 import BeautifulSoup
import sqlite3
import time
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.shl.com"
CATALOG_BASE_URL = f"{BASE_URL}/products/product-catalog/"
DB_NAME = "shl_products.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

class SHLCrawler:
    def __init__(self, db_path=DB_NAME, max_workers=10):
        self.db_path = db_path
        self.max_workers = max_workers
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    remote_testing BOOLEAN,
                    adaptive_irt BOOLEAN,
                    test_type TEXT,
                    description TEXT,
                    job_levels TEXT,
                    languages TEXT,
                    assessment_length TEXT,
                    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def save_product(self, product_data):
        # SQLite is better with one connection per write or careful locking
        # For simplicity in this parallel script, we open a connection per save
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO products (
                        name, url, remote_testing, adaptive_irt, test_type,
                        description, job_levels, languages, assessment_length
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        name=excluded.name,
                        remote_testing=excluded.remote_testing,
                        adaptive_irt=excluded.adaptive_irt,
                        test_type=excluded.test_type,
                        description=excluded.description,
                        job_levels=excluded.job_levels,
                        languages=excluded.languages,
                        assessment_length=excluded.assessment_length,
                        crawled_at=CURRENT_TIMESTAMP
                ''', (
                    product_data['name'],
                    product_data['url'],
                    product_data['remote_testing'],
                    product_data['adaptive_irt'],
                    product_data['test_type'],
                    product_data.get('description'),
                    product_data.get('job_levels'),
                    product_data.get('languages'),
                    product_data.get('assessment_length')
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving {product_data['name']} to DB: {e}")

    def get_soup(self, url):
        try:
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=HEADERS, timeout=15)
                    response.raise_for_status()
                    return BeautifulSoup(response.text, 'html.parser')
                except requests.exceptions.RequestException as e:
                    if attempt == 2: raise
                    logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}. Retrying...")
                    time.sleep(2)
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def scrape_catalog(self, start=0):
        url = f"{CATALOG_BASE_URL}?start={start}&type=1"
        logger.info(f"Scraping catalog page (start={start}): {url}")
        soup = self.get_soup(url)
        if not soup:
            return []

        tables = soup.find_all('div', class_='custom__table-responsive')
        individual_table_container = None
        
        # Search specifically for the "Individual Test Solutions" table
        for t in tables:
            heading = t.find('th', class_='custom__table-heading__title')
            if heading and "Individual Test Solutions" in heading.text:
                individual_table_container = t
                break
        
        if not individual_table_container:
            # Fallback to second table if heading search fails but there are multiple tables
            if len(tables) >= 2:
                individual_table_container = tables[1]
            else:
                logger.warning(f"Could not find 'Individual Test Solutions' table on page with start={start}.")
                return []

        individual_table = individual_table_container.find('table')
        if not individual_table:
            return []

        rows = individual_table.find_all('tr')[1:] # Skip header row
        products = []

        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 4:
                continue

            name_link = cols[0].find('a')
            if not name_link:
                continue

            name = name_link.text.strip()
            href = name_link.get('href')
            full_url = href if href.startswith('http') else f"{BASE_URL}{href}"

            remote_testing = bool(cols[1].find('span', class_='-yes'))
            adaptive_irt = bool(cols[2].find('span', class_='-yes'))
            test_types = [span.text.strip() for span in cols[3].find_all('span', class_='product-catalogue__key')]
            
            products.append({
                'name': name,
                'url': full_url,
                'remote_testing': remote_testing,
                'adaptive_irt': adaptive_irt,
                'test_type': ", ".join(test_types)
            })

        logger.info(f"Found {len(products)} products on this page.")
        return products

    def scrape_detail(self, product):
        logger.info(f"Scraping detail: {product['url']}")
        soup = self.get_soup(product['url'])
        if not soup:
            return product

        headings = soup.find_all('h4')
        for h in headings:
            label = h.text.strip().lower()
            content_node = h.find_next_sibling()
            content = content_node.text.strip() if content_node else ""

            if "description" in label:
                product['description'] = content
            elif "job levels" in label:
                product['job_levels'] = content
            elif "languages" in label:
                product['languages'] = content
            elif "assessment length" in label:
                match = re.search(r'(\d+)', content)
                product['assessment_length'] = match.group(1) if match else None

        # Save immediately after scraping detail effectively within the thread
        self.save_product(product)
        return product

    def run(self):
        start = 0
        total_discovered = 0
        
        # Parallel setup
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            while True:
                products = self.scrape_catalog(start)
                if not products:
                    break
                
                for product in products:
                    futures.append(executor.submit(self.scrape_detail, product))
                    total_discovered += 1
                
                start += 12
                if start > 500: # Safety break
                    break
            
            logger.info(f"All pages queued. Total products queued: {total_discovered}. Waiting for completion...")
            
            # Monitoring progress
            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Progress: {completed}/{total_discovered} completed.")

        logger.info(f"Crawling completed. Total products processed: {total_discovered}")

if __name__ == "__main__":
    # You can adjust max_workers to be faster (e.g., 20) but 10 is safe and significantly faster
    crawler = SHLCrawler(max_workers=10)
    crawler.run()
