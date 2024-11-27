import requests
from bs4 import BeautifulSoup
import json

def fetch_page_content(url):
    """Fetch the content of a page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url.strip(), headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract main content (modify as per website structure)
        body_text = soup.get_text(separator='\n', strip=True)
        return body_text
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch {url}: {e}")
        return None

def process_urls_and_save(input_file, output_file):
    """Read URLs from JSON, fetch their content, and save to a text file."""
    try:
        # Load URLs from JSON file
        with open(input_file, 'r') as json_file:
            urls = json.load(json_file)

        with open(output_file, 'w', encoding='utf-8') as text_file:
            for url in urls:
                print(f"Processing: {url}")
                content = fetch_page_content(url)
                if content:
                    # Write content to text file
                    text_file.write(f"URL: {url}\n")
                    text_file.write(content + "\n\n")
                else:
                    text_file.write(f"URL: {url}\nFailed to fetch content.\n\n")
    except Exception as e:
        print(f"Error processing URLs: {e}")

def main():
    input_file = "botpenguin_urls.json"  # JSON file with URLs
    output_file = "botpenguin_content.txt"  # Output text file for content

    print("Fetching content from URLs...")
    process_urls_and_save(input_file, output_file)
    print(f"Content saved to '{output_file}'")

if __name__ == "__main__":
    main()
