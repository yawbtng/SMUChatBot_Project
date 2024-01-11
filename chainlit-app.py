from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from qdrant_client import qdrant_client
from PIL import Image
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
LLM = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key, max_tokens=500, verbose=True)

# Define a template for system messages
system_template = """
    You are an assistant chatbot designed to support, guide, and welcome prospective and current
    students at Southern Methodist University. Your capabilities include providing accurate information
    about financial information, academic course catalog, faculty, the university academic calendar, campus events,
    and student organizations. You are programmed to be helpful and welcoming, offering precise answers 
    to relevant inquiries while maintaining a friendly and conversational tone. For queries that fall outside your 
    knowledge base or involve personal private information, you are to respond with 'I'm sorry, 
    I do not have access to that information. Please visit the SMU website or contact the relevant department 
    for more assistance'. In the face of inappropriate or harmful messages, your response should be tactful yet 
    firm, indicating the inappropriateness of the query and redirecting to relevant university-related questions. 
    In situations involving emergencies or urgent support, you're programmed to advise users to contact SMU's 
    emergency services or the appropriate university support channels. 
    Keep this in mind as this is your personality and instructions for all messages in this chat.

    {question}
"""

# Create templates for system and human messages
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
    
]

# Create a chat prompt from the defined messages
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}


# Start of the chat, set up the chat environment
@cl.on_chat_start
async def on_chat_start():
    # Display a welcome message and image
    elements = [cl.Image(name="smu_icon", display="inline", path="./smu_icon.png")]
    await cl.Message(content="Welcome to PerunaBot! Your guide to all things SMU", elements=elements, author="PerunaBot").send()

    # Initialize the retrieval chain with your Qdrant vector store

    memory = ConversationSummaryBufferMemory(
        llm=LLM, memory_key='chat_history', return_messages=True, max_token_limit=250, human_prefix='User')
    
    question_generator_chain = LLMChain(llm=LLM, prompt=prompt)

    doc_summary = StuffDocumentsChain(llm_chain=question_generator_chain,document_variable_name="", document_prompt=prompt)
    

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        LLM,
        combine_documents_chain = doc_summary,
        chain_type="stuff",
        retriever=VECTOR_STORE.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

    # Save the chain in the user session for later use
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: str):
    # Retrieve the QA chain from the user session
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain

    # Callback handler for the chain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # Process the user's message using the chain and callback
    res = await chain.acall(message, callbacks=[cb])

    # Extract the answer from the response
    answer = res["answer"]

    # Update the final stream with the answer or send as a new message
    if cb.has_streamed_final_answer:
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer).send()