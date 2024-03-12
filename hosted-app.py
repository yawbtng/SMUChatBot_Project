from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import qdrant_client
from literalai import LiteralClient
import chainlit as cl
import os

from dotenv import load_dotenv
from sympy import false, true

# Load environment variables.
load_dotenv()
qdrant_host = os.environ['QDRANT_HOST']
qdrant_api_key = os.environ['QDRANT_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
qdrant_collection_name = os.environ['QDRANT_COLLECTION_NAME']
literal_api_key = os.environ['LITERAL_API_KEY']

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

client = LiteralClient(api_key=literal_api_key)

# Start of the chat, set up the chat environment
@cl.on_chat_start
async def on_chat_start():
    
    # Display a welcome message and image
    # elements = [cl.Image(name="smu_icon", display="inline", path="./Images/smu_icon.png")]
    await cl.Message(content="Welcome to PerunaBot! Your guide to all things SMU! Please go back and look at the 'README' file in the top left corner before typing any chats!", 
                     author="PerunaBot").send()
   ## await cl.Avatar(name="PerunaBot", path="./Images/smu_icon.png").send()


    memory = ConversationSummaryBufferMemory(
        llm=LLM, memory_key='chat_history', return_messages=True, max_token_limit=500, human_prefix='User')

    chain = ConversationalRetrievalChain.from_llm(
        LLM,
        chain_type="stuff",
        retriever=VECTOR_STORE.as_retriever(search_type="similarity"),
        memory=memory,
    )

    # Save the chain in the user session for later use
    cl.user_session.set("chain", chain)

count = 0

def max_number_of_questions():
    global count
    if count > 20: # checks if count is more than 20
        return False
    return True

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    cb.answer_reached = True

    # Declare 'count' as global to modify it
    global count 
    
    # Increment the count for each messag
    count += 1

    if max_number_of_questions(): # if number of questions asked is less than 5, normal response will come
        # Process the user's message using the chain and callback
        res = await chain.acall(message.content, callbacks=[cb])

        # Extract the answer from the response
        answer = res["answer"]

        await cl.Message(content=answer).send()
        
    else:
        await cl.Message(content="You have reached the maximum number of questions for this session. Thank you!").send()
