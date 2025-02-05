import os
import tempfile
import tiktoken
import uuid
import time
import streamlit as st
import docx2txt
from dotenv import load_dotenv
from typing import Iterable
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma, AzureSearch
from langchain.chains import LLMChain, ConversationalRetrievalChain, ConversationChain, SimpleSequentialChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import openai
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
import ssl
from langchain.document_loaders import (
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    MWDumpLoader,
    UnstructuredFileLoader,
    JSONLoader
    
)
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
load_dotenv(".env.gpt4",override=True)



def count_tokens(string):
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoding.encode(string))
    return token_count


def chunkify(arr: Iterable, size: int = 16):
    for i in range(0, len(arr), size):
        yield arr[i : i + size]

def get_dataset_id():
    dataset_id = str(uuid.uuid4()) +"-"+ str(time.time()).split(".")[0]
    return dataset_id

def load_documents(uploads):
    docs = []
    texts = ""
    for file in uploads:
        ext_name = os.path.splitext(file.name)[-1]

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file.read())
            file_path = tmp_file.name
            if ext_name == ".pptx":
                loader = UnstructuredPowerPointLoader(file_path)
            elif ext_name == ".docx":
                loader = Docx2txtLoader(file_path)
            elif ext_name == ".pdf":
                loader = PyMuPDFLoader(file_path,extract_images=True)
                #loader = PyPDFLoader(file_path,extract_images=True)
            elif ext_name == ".csv":
                loader = CSVLoader(file_path=file_path)
            elif ext_name == ".xlsx|.xls":
                loader = UnstructuredExcelLoader(file_path=file_path)    
            elif ext_name == ".xml":
                loader = MWDumpLoader(file_path=file_path, encoding="utf8")
            elif ext_name == ".txt":
                loader = TextLoader(file)
            elif ext_name == ".json":
                loader = JSONLoader(file_path=file_path)
            elif ext_name == ".jpg":
                loader = UnstructuredImageLoader(file_path=file_path)

                data = loader.load()
                st.write(data)
                return data[0]
            else:
                # process .txt, .html
                loader = UnstructuredFileLoader(file_path)
        
        doc = loader.load()     
        docs.extend(doc)
    for doc in docs:
        texts += doc.page_content
    return texts

def load_excel_file(file_path):
    docs = []
    texts = ""
    loader = UnstructuredExcelLoader(file_path=file_path) 
    doc = loader.load()     
    docs.extend(doc)
    for doc in docs:
        texts += doc.page_content
    return texts

def create_docs(user_pdf_list):
    docs = []
    for filename in user_pdf_list:
        ext_name = os.path.splitext(filename.name)[-1]
        text = ""
        if ext_name == ".pdf":
            pdf_reader = PdfReader(filename)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif ext_name == ".docx":
            text = docx2txt.process(filename)
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "name": filename.name,
                    "type=": filename.type,
                    "size": filename.size,
                },
            )
        )

    return docs


def get_pdf_text(pdf_docs):
    pdf_texts = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            pdf_texts += page.extract_text()
    return pdf_texts


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=19000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_doc_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=100,
        length_function=len
    )
    text_chunks = text_splitter.create_documents([text])
    return text_chunks


def get_vectorstore_faiss(text_chunks):
    embeddings = create_embedding_model()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_vectorstore_chroma(text_chunks):
    embedding_function = create_embedding_model()
    db = Chroma(embedding_function=embedding_function)
    db.delete_collection()
    db = Chroma(embedding_function=embedding_function)
    for chunk in chunkify(text_chunks):
        db.add_texts(chunk)
    return db


def get_vectorstore_acs_from_text(text_chunks):
    dataset_id = get_dataset_id()
    embedding_function = create_embedding_model()
    acs = AzureSearch(azure_search_endpoint=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
                      azure_search_key=os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"],
                      index_name=dataset_id,
                      embedding_function=embedding_function.embed_query)
    for chunk in chunkify(text_chunks):
        acs.add_texts(chunk)
    return acs

def get_vectorstore_acs_from_text_with_index(index_name, text_chunks):
    embedding_function = create_embedding_model()
    acs = AzureSearch(azure_search_endpoint=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
                      azure_search_key=os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"],
                      index_name=index_name,
                      embedding_function=embedding_function.embed_query)
    for chunk in chunkify(text_chunks):
        acs.add_texts(chunk)
    return acs

def get_vectorstore_acs_from_doc(doc_chunks):
    dataset_id = get_dataset_id()
    embedding_function = create_embedding_model()
    acs = AzureSearch(azure_search_endpoint=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
                      azure_search_key=os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"],
                      index_name=dataset_id,
                      embedding_function=embedding_function.embed_query)
    for chunk in chunkify(doc_chunks):
        acs.add_documents(chunk)
    return acs

def get_vectorstore_acs_from_doc_with_index(index_name, doc_chunks):
    embedding_function = create_embedding_model()
    print("1")
    acs = AzureSearch(azure_search_endpoint=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
                      azure_search_key=os.environ["AZURE_COGNITIVE_SEARCH_API_KEY"],
                      index_name=index_name,
                      embedding_function=embedding_function.embed_query)
    print("2")
    for chunk in chunkify(doc_chunks):
        print("3")
        acs.add_documents(chunk)
        print("4")
    return acs

def create_embedding_model():
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['CURL_CA_BUNDLE'] = ''
    embeddings = OpenAIEmbeddings(
        deployment=os.environ["EMBEDDING_DEPLOYMENT_NAME"],
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type=os.environ["OPENAI_API_TYPE"],
        #verify_ssl=False  # Disable SSL verification
    )
    return embeddings


def create_llm_model(temperature, max_tokens, streaming=False):
    llm = AzureChatOpenAI(
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type=os.environ["OPENAI_API_TYPE"],
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
    return llm




def create_retrieval_chain(vectorstore, temperature=0.5, max_tokens=3000):
    llm = create_llm_model(temperature, max_tokens, streaming=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def create_llm_chain(prompt_template, temperature, max_tokens):
    llm = create_llm_model(temperature=temperature, max_tokens=max_tokens)
    return LLMChain(llm=llm, prompt=prompt_template)


def get_summary(current_doc, temperature, max_tokens):
    llm = create_llm_model(temperature=temperature, max_tokens=max_tokens)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(current_doc)

    return summary

def execute_sequential_chain(chains, input):
    full_chain = SimpleSequentialChain(chains=chains)
    result = full_chain.run(input)
    return result

def create_llm_with_memory_buffer(temperature, max_tokens):
    memory = ConversationBufferMemory()
    llm = create_llm_model(temperature=temperature, max_tokens=max_tokens)
    return ConversationChain(llm=llm, memory=memory, verbose=True)

def check_clear_conversation_chain():
    with st.sidebar:
        reset_button = st.button(
            "Reset",
            key="clear",
            type="primary",
            use_container_width=True,
        )
        if reset_button:
            st.session_state.clear()
            st.session_state.list_of_messages = []

    return



def create_llm_model_argent(temperature, max_tokens, streaming=False):
    llm = AzureChatOpenAI(
        openai_api_base=os.environ["OPENAI_API_BASE_ARGENT"],
        openai_api_version=os.environ["OPENAI_API_VERSION_ARGENT"],
        deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME_ARGENT"],
        openai_api_key=os.environ["OPENAI_API_KEY_ARGENT"],
        openai_api_type=os.environ["OPENAI_API_TYPE_ARGENT"],
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
    )
    return llm

def create_llm_chain_argent(prompt_template, temperature, max_tokens):
    llm = create_llm_model_argent(temperature=temperature, max_tokens=max_tokens)
    return LLMChain(llm=llm, prompt=prompt_template)

