# This is a variation of the main file to attempt to optimize the vector store data by using a different method
# from langchain shown here: https://clusteredbytes.pages.dev/posts/2023/langchain-parent-document-retriever/
import os
from dotenv import load_dotenv
import pandas as pd
import pypdf
from qdrant_client import qdrant_client
from qdrant_client.http import models
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
# added imports
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore 

load_dotenv()
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
qdrant_collection2_name = os.environ['QDRANT_COLLECTION2_NAME']

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
   collection_name = qdrant_collection2_name,
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
        collection_name=qdrant_collection2_name, 
        embeddings=embeddings,
    )

    return vector_store

# created vector store
vector_store = get_vector_store()



def load_pdf_documents(pdf_paths):
    pdf_documents = []
    for path in pdf_paths:
        try:
            # Open the PDF file in binary read mode
            with open(path, 'rb') as file:
                # Read the PDF file using PyPDF2
                pdf_reader = pypdf.PdfReader(file)
                pdf_documents.append(pdf_reader)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return pdf_documents

pdf_paths = ['C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/20232024 Undergraduate Catalog91123.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/20232024 Graduate Catalog101723.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/20232024 FinancialInformationBulletin91123.pdf',
             'C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Data/Students - Spring 2024.pdf']

pdf_documents = load_pdf_documents(pdf_paths)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, length_function=len) 

parent_splitter =RecursiveCharacterTextSplitter(chunk_size=1250, chunk_overlap=50, length_function=len)  
# storage for parent splitter
store = InMemoryStore()

# retriever
retriever = ParentDocumentRetriever(
    vectorstore=vector_store, 
    docstore=store, 
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    )


retriever.add_documents(pdf_documents)