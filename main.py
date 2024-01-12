import os
from dotenv import load_dotenv
import pandas as pd
import PyPDF2
from qdrant_client import qdrant_client
from qdrant_client.http import models
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()


qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
qdrant_collection_name = os.environ['QDRANT_COLLECTION_NAME']

# Initialize Qdrant client.
client = qdrant_client.QdrantClient(
    url=qdrant_host, 
    api_key = qdrant_api_key,
)

# create collection

vectors_config = models.VectorParams(
   size=1536, #for OpenAI
   distance=models.Distance.COSINE
   )

client.recreate_collection(
   collection_name = qdrant_collection_name,
   vectors_config=vectors_config,
)

# create vector store
def get_vector_store():
    client = qdrant_client.QdrantClient(
    qdrant_host, 
    api_key = qdrant_api_key,
    )

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=qdrant_collection_name, 
        embeddings=embeddings,
    )

    return vector_store

# created vector store
vector_store = get_vector_store()

# getting text from the pdfs
def get_txt_from_pdfs(pdf_paths):
    concatenated_raw_text = ''
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                concatenated_raw_text += page.extract_text()

    return concatenated_raw_text

pdf_paths = ['C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/20232024 Undergraduate Catalog91123.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/20232024 Graduate Catalog101723.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/20232024 FinancialInformationBulletin91123.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/Students - Spring 2024.pdf']

texts = get_txt_from_pdfs(pdf_paths)


# function to split up text into chunks
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        length_function=len
    ) 

    chunks = text_splitter.split_text(text)
    return chunks
chunked_texts = get_chunks(texts)

# add pdf text chunks to vector store
vector_store.add_texts(chunked_texts)



check = False

if check == True:
    try:
        # now adding the csv files into the database

        # Function to read multiple CSV files and convert them to text.
        def read_csvs(csv_paths):
            combined_csv_text = ''
            for csv_path in csv_paths:
                df = pd.read_csv(csv_path)
                combined_csv_text += csv_to_text(df) + ' '
            return combined_csv_text

        # Function to convert CSV data to text.
        def csv_to_text(df):
            text_data = []
            for _, row in df.iterrows():
                row_text = ' '.join([f'{col}: {row[col]}' for col in df.columns])
                text_data.append(row_text)
            csv_text = ' '.join(text_data)
            return csv_text

        csv_paths = ['C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/University Calendar 2023-24 11.17.2023.csv',
                    'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/SMU360 events (last updated 1-2-2024).csv',
                    'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/List of orgs at SMU.csv']

        # Load and concatenate text from CSV files.
        combined_csv_text = read_csvs(csv_paths)

        # Split the CSV text into chunks.
        csv_chunked_texts = get_chunks(combined_csv_text)

        # Add the CSV text chunks to the vector store.
        vector_store.add_texts(csv_chunked_texts)

    except:
        print('CSV text was unable to be uploaded into the qdrant vector database')