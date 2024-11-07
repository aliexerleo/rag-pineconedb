import os
import streamlit as st
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from uuid import uuid4
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")
LIST_FILES = os.getenv("LIST_FILES")


pc = Pinecone(PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def save_name_file(path, new_files):

    old_files = load_name_files(path)

    with open(path, 'a') as file:
        for item in new_files:
            if item not in old_files:
                file.write(item + '\n')
                old_files.append(item)

    return old_files

def load_name_files(path):

    files = []
    with open(path, 'r') as file:
        for line in file:
            files.append(line.strip())

    return files


def clean_file(path):
    with open(path, 'w') as file:
        pass
    index.delete(delete_all=True)
    return True

def text_to_pinecone(pdf):

    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, pdf.name)
    with open(temp_filepath, 'wb') as f:
        f.write(pdf.getvalue())

    loader = PyPDFLoader(temp_filepath)
    pdf_text = loader.load()

    with st.spinner(f'Creating Embeddings : {pdf.name}'):
            create_embeddings(pdf.name, pdf_text)
    
    return True

def create_embeddings(file_name, pdf_text):
    print(f'Creating embeddings of file : {file_name}')
          
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(pdf_text)

    uuids = [str(uuid4()) for _ in range(len(chunks))]

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)

    vector_store.add_documents(documents=chunks, ids=uuids)
    
    return  True



