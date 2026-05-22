from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import httpx
import os

client = httpx.Client(verify=False)

# LLM
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-LXItlj5p_LKYSHv01Xxsxg",
    http_client=client
)

# Embedding Model
embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-LXItlj5p_LKYSHv01Xxsxg",
    http_client=client
)

vectordb = None

def process_form(pdf_path):

    global vectordb

    raw_text = extract_text(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(raw_text)

    vectordb = Chroma.from_texts(
        chunks,
        embedding_model,
        persist_directory="./chroma_db"
    )

    vectordb.persist()

def ask_question(query):

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    result = qa_chain.invoke(query)

    return result["result"]


def ask_question2(query):
    """Direct LLM call for field suggestions (without vector DB retrieval)"""
    response = llm.invoke(query)
    return response.content