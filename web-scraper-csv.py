import requests
from bs4 import BeautifulSoup
import csv

# List to keep track of URLs visited
visited_urls = []

# CSV file setup
csv_file = open('smu_data.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['URL', 'Content'])

def scrape_site(url):
    if url in visited_urls:
        return
    else:
        visited_urls.append(url)

    # Send HTTP request to the URL
    try:
        response = requests.get(url)
        # If the request was successful and the content is HTML
        if response.status_code == 200 and 'text/html' in response.headers['Content-Type']:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract all text from the page
            text = soup.get_text()

            # Write the URL and the extracted text to the CSV
            csv_writer.writerow([url, text])

            # Find all links in the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):  # relative URL
                    href = url + href
                if 'smu.edu' in href:  # keeping it within the same site
                    scrape_site(href)
    except Exception as e:
        print(f"Failed to process {url}: {str(e)}")

# Start scraping from the main page
scrape_site('https://www.smu.edu/')

# Close the CSV file
csv_file.close()
