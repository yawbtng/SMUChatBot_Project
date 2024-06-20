import os
from literalai import LiteralClient
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
literal_api_key = os.environ['LITERAL_API_KEY']
client = LiteralClient(api_key=literal_api_key)

# Function to fetch all threads with pagination
def fetch_all_threads():
    all_threads = []
    page = 1
    per_page = 100  # Adjust as needed

    while True:
        response = client.api.get_threads()
        all_threads.extend(threads)

        if len(threads) < per_page:
            break  # No more pages

        page += 1

    return all_threads

# Fetch all threads
threads = fetch_all_threads()

# Convert threads to a DataFrame
threads_data = []
for thread in threads:
    thread_info = {
        "id": thread.id,
        "name": thread.name,
        "created_at": thread.created_at,
        "duration": thread.duration,
        "token_count": thread.token_count,
        "metadata": thread.metadata
    }
    threads_data.append(thread_info)

df = pd.DataFrame(threads_data)

# Export DataFrame to Excel
df.to_excel("threads_export.xlsx", index=False)

print("Threads exported to threads_export.xlsx")