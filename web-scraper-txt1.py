import requests
from bs4 import BeautifulSoup
import re

start_url = "https://catalog.smu.edu/content.php?catoid=63&navoid=6032"
domain = "catalog.smu.edu"
visited_urls = set()
output_file = "scraped_data.txt"

def scrape_site(url):
    if url in visited_urls or domain not in url:
        return
    else:
        visited_urls.add(url)
        try:
            response = requests.get(url)
            if response.status_code == 200 and 'text/html' in response.headers['Content-Type']:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text()
                save_text(url, text)
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if href.startswith('/'):
                        href = 'https://' + domain + href
                    scrape_site(href)
        except Exception as e:
            print(f"Failed to process {url}: {str(e)}")

def save_text(url, text):
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(f"URL: {url}\n")
        file.write(cleaned_text)
        file.write("\n\n----------------\n\n")

scrape_site(start_url)
