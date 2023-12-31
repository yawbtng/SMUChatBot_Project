# Import necessary libraries
import requests  # Used to make HTTP requests
from bs4 import BeautifulSoup  # Used for parsing HTML and navigating the parse tree
import re  # Used for regular expression operations

start_url = "https://www.smu.edu/Admission" # The initial URL where scraping starts
domain = "smu.edu" # Domain to limit the scope of the scraper
visited_urls = set() # Initialize a set to keep track of visited URLs to avoid duplication
output_file = "smu_admissions_scraped_data.txt" # Specify the name of the output file where the scraped data will be saved

def scrape_site(url): # Define the main function to scrape the website
    if url in visited_urls or domain not in url or 'Admission' not in url:  # Check if the URL has already been visited or if it's outside the target domain
        return  # Skip this URL
    else:
        visited_urls.add(url) # Mark the URL as visited
        try:
            response = requests.get(url) # Send an HTTP GET request to the URL
            if response.status_code == 200 and 'text/html' in response.headers['Content-Type']: # Check if the response is successful and the content is HTML
                soup = BeautifulSoup(response.text, 'html.parser') # Parse the HTML content with BeautifulSoup
                text = soup.get_text() # Extract all text from the page
                save_text(url, text) # Save the extracted text using the save_text function
                links = soup.find_all('a', href=True) # Find all hyperlinks on the page
                for link in links: # Recursively follow each link that contains 'Admission'
                    href = link['href'] # Format relative URLs into absolute URLs
                    if 'Admission' in href and href.startswith('/'):  # Check if 'Admission' is in the href
                        href = 'https://' + domain + href
                    scrape_site(href) # Recursively call scrape_site on the new URL
        except Exception as e:
            print(f"Failed to process {url}: {str(e)}") # Print an error message if the URL couldn't be processed

# Define a function to save the scraped text to a file
def save_text(url, text):
    # Clean the text by reducing whitespace
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    # Open the output file in append mode
    with open(output_file, 'a', encoding='utf-8') as file:
        # Write the URL to the file
        file.write(f"URL: {url}\n")
        # Write the cleaned text to the file
        file.write(cleaned_text)
        for each in cleaned_text:
            file.write ('\n')
        # Write a separator to delineate different pages
        file.write("\n\n----------------\n\n")


# Start the scraping process with the specified start URL
scrape_site(start_url)
