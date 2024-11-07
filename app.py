import streamlit as st
from utils import *
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os


st.header('Asking about PDF file')


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
LIST_FILES = os.getenv("LIST_FILES")


pc = Pinecone(PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)



with st.sidebar:
    file_name = load_name_files(LIST_FILES)
    files_uploaded = st.file_uploader(
        'Load your file',
        type='pdf',
        accept_multiple_files=True
    )

    if st.button('Process'):
        for pdf in files_uploaded:
            if pdf is not None and pdf.name not in file_name:
                file_name.append(pdf.name)
                text_to_pinecone(pdf)


        file_name = save_name_file(LIST_FILES, file_name)

    if len(file_name) > 0:
        st.write('Files loaded')
        documents_list = st.empty()
        with documents_list.container():
            for f in file_name:
                st.write(f)
            if st.button('Cleaning files'):
                full_files = []
                clean_file(LIST_FILES)
                documents_list.empty()

if len(file_name)>0:
    user_question = st.text_input("Question: ")
    if user_question:

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        
        vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)

        results = vector_store.similarity_search(user_question, 2)

        llm = ChatOpenAI(model_name='gpt-4o-mini-2024-07-18')
        chain = load_qa_chain(llm)
        answer = chain.invoke(input={"input_documents": results, "question": user_question}, return_only_outputs=True)
        st.write(answer["output_text"])
