from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import qdrant_client
import chainlit as cl
import os

from dotenv import load_dotenv

# Load environment variables.
load_dotenv()
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
qdrant_collection_name = os.environ['QDRANT_COLLECTION_NAME']

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

# initializing the vector store
VECTOR_STORE = get_vector_store()

# initializing the llm
LLM = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key, verbose=True)


# Start of the chat, set up the chat environment
@cl.on_chat_start
async def on_chat_start():
    
    # Display a welcome message and image
    elements = [cl.Image(name="smu_icon", display="inline", path="C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Images/smu_icon.png")]
    await cl.Message(content="Welcome to PerunaBot! Your guide to all things SMU", elements=elements, author="PerunaBot").send()
    await cl.Avatar(name="Chatbot", path="C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/Images/smu_icon.png").send()


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