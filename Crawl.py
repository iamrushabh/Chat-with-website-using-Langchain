import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def is_valid_url(url):
    """Check if the URL is well-formed."""
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_links_from_page(url, base_url, retries=3):
    """Fetch links from a single page with retry logic."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    for attempt in range(retries):
        try:
            response = requests.get(url.strip(), headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all links and resolve full URLs
            links = set()
            for link in soup.find_all('a', href=True):
                href = urljoin(base_url, link['href'])
                if is_valid_url(href) and href.startswith(base_url):
                    links.add(href)
            return links
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt == retries - 1:
                print(f"Skipping {url} after {retries} attempts.")
    return set()

def crawl_website(base_url, max_workers=20):
    """Crawl the website to get all unique URLs."""
    visited = set()
    to_visit = {base_url}
    all_links = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while to_visit:
            # Submit tasks for the current batch of URLs
            futures = {executor.submit(get_links_from_page, url, base_url): url for url in to_visit}
            to_visit = set()  # Clear the queue for the next iteration

            for future in as_completed(futures):
                url = futures[future]
                try:
                    links = future.result()
                    new_links = links - visited
                    to_visit.update(new_links)
                    all_links.update(new_links)
                    visited.add(url)
                except Exception as e:
                    print(f"Error processing {url}: {e}")

    return list(all_links)

def main():
    base_url = "https://botpenguin.com/"
    print("Starting fast crawl...")

    # Crawl the website
    all_links = crawl_website(base_url, max_workers=50)
    print(f"Found {len(all_links)} unique pages.")

    # Save to JSON
    with open("botpenguin_urls.json", "w") as json_file:
        json.dump(all_links, json_file, indent=4)

    print("Page URLs saved to 'botpenguin_urls.json'")

if __name__ == "__main__":
    main()
