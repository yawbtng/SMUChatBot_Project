import streamlit as st 
from dotenv import load_dotenv
 

def get_pdf_texts(pdf_docs):

def main():
   load_dotenv()
   st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

   st.header("Chat with multiple PDFs :books:")
   st.text_input("Ask a question about your documents")

   with st.sidebar:
       st.subheader("Your documents")
       pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
       if st.button("Process"):
          with  st.spinner("Processing"):
           # get pdf text
            raw_text = get_pdf_text(pdf_docs)
           # get the text chunks

           # create vector store


if __name__ == '__main__':
    main()