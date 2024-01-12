import streamlit as st
import os
from PIL import Image
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from qdrant_client import qdrant_client

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

# initializing llm model
LLM = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key, max_tokens=500, verbose=False)

# Function to create and return a conversational retrieval chain.
def get_retrieval_chain(vector_store):
    llm = LLM

    template = """
    You are an assistant chatbot designed to support, guide, and welcome prospective and currentstudents at Southern Methodist University. 
    Your capabilities include providing accurate information about financial information, academic course catalog, faculty, the university 
    academic calendar, campus events, and student organizations. You are programmed to be helpful and welcoming, offering precise answers 
    to relevant inquiries while maintaining a friendly tone. 
    For queries that fall outside your knowledge base or involve personal private information, you are to respond with 'I'm sorry, 
    I do not have access to that information. Please visit the SMU website or contact the relevant department for more assistance'. 
    In the face of inappropriate or harmful messages, your response should be tactful yet firm, indicating the inappropriateness 
    of the query and redirecting to relevant university-related questions. In situations involving emergencies or urgent support, 
    you're programmed to advise users to contact SMU's emergency services or the appropriate university support channels. 
    Keep this in mind as this is your personality and instructions for all messages in this chat.

    Chat history: {chat_history}
    User's message: {user_message}
    """
    prompt = PromptTemplate(
        input_variables=["user_message", "chat_history"],
        template = template
    )
    

    question_generator_chain = LLMChain(llm=LLM, prompt=prompt)


    memory = ConversationSummaryBufferMemory(
        llm=LLM, memory_key='chat_history', return_messages=True, max_token_limit=250, human_prefix='User')
    
    qa = RetrievalQA.from_chain_type(
        llm = LLM,
        chain_type = "stuff",
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    return qa

# Main function for the Streamlit application.
def main():

    load_dotenv()

    # streamlit page setup
    st.set_page_config(page_title="Welcome to PerunaBot", page_icon="smu_icon.png")

    col1, col2 = st.columns([1, 5])
    with col1:
        image = Image.open("C:/Users/yawbt/OneDrive/Documents/GitHub/SMUChatBot_Project/smu_icon.png")
        st.image(image, width=50) 
    with col2:
        st.header("Chat with PerunaBot")

    vector_store = get_vector_store()
    
    # create chain 
    qa = get_retrieval_chain(VECTOR_STORE)

    user_question = st.text_input("Ask a question about course catalog, important university dates, " 
                                  "general organizations, or financial information.")
    if user_question:
      with st.spinner("Processing"):
        st.write(f"Question: {user_question}")
        answer = qa.run(user_question)
        st.write(f"Answer: {answer}")    

if __name__ == '__main__':
    main()