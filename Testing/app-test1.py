# The purpose of this test is to attempt contextual compression to see how we can better specificy the data/context that is retrieved
# from the vector store to answer the prompt with more accuracy and specificity if that's even a word.
# source: https://youtu.be/KQjZ68mToWo?si=OOJWGc3SK1FgqJlM&t=773
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain

from qdrant_client import qdrant_client
import chainlit as cl
import os
# added imports
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# Load environment variables.
load_dotenv()
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
qdrant_collection_name = os.environ['QDRANT_COLLECTION_NAME']
qdrant_collection2_name = os.environ['QDRANT_COLLECTION2_NAME']

# Function to get the Qdrant vector store.
def get_vector_store():
    client = qdrant_client.QdrantClient(
        qdrant_host, 
        api_key=qdrant_api_key,
    )

    embeddings = OpenAIEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=qdrant_collection_name, 
        embeddings=embeddings,
    )
    
    return vector_store

# initializing the vector store and retriever
VECTOR_STORE = get_vector_store()
retriever = VECTOR_STORE.as_retriever(cl.Message)

# retriever.get_relevant_documents()

# initializing the llm
LLM = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key, verbose=True)


# Start of the chat, set up the chat environment
@cl.on_chat_start
async def on_chat_start():
    
    # Display a welcome message and image
    elements = [cl.Image(name="smu_icon", display="inline", path="./smu_icon.png")]
    await cl.Message(content="Welcome to PerunaBot! Your guide to all things SMU", elements=elements, author="PerunaBot").send()
    await cl.Avatar(name="Chatbot", path="./smu_icon.png",).send()

    # MAJOR CHANGES 
    # https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression?ref=blog.langchain.dev

    # makes another LLM call to retrieval contextual documentsby shortening or omitting documents which reduces documents sent for final question
    # more expensive, slower but good result
    compressor = LLMChainExtractor.from_llm(LLM)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    # uses embeddings model to retrieve similar documents based on cosine similarity 
    # cheaper but documents will not be chainged (dependent on similarity threshold)
    embeddings_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.76)
    embeddings_compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

    # combines multiple compressors to a more optimized result by removing redundant documents and finding relevant documents based on cosine similarity
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    redundant_filter = EmbeddingsRedundantFilter(embeddings=OpenAIEmbeddings())
    relevant_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), similarity_threshold=0.86)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter]
    )
    combined_compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

    # will plug in all three different retrievers in chain using smu-data-1 and smu-data-2 to see which retains a better result

    memory = ConversationSummaryBufferMemory(
        llm=LLM, memory_key='chat_history', return_messages=True, max_token_limit=250, human_prefix='User')

    chain = ConversationalRetrievalChain.from_llm(
        LLM,
        chain_type="stuff",
        retriever=VECTOR_STORE.as_retriever(search_type="similarity"),
        memory=memory,
    )

    # Save the chain in the user session for later use
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    cb.answer_reached = True

    # Process the user's message using the chain and callback
    res = await chain.acall(message.content, callbacks=[cb])

    # Extract the answer from the response
    answer = res["answer"]

    await cl.Message(content=answer).send()