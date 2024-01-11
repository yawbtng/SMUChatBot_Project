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


qdrantHost = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
qdrant_collection_name = os.environ['QDRANT_COLLECTION_NAME']

print(qdrantHost)