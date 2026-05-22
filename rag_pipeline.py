import os
import warnings
import ssl
import urllib3

# Disable SSL verification globally
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings('ignore')

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests to disable SSL verification
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Monkeypatch requests.get to disable SSL verification
original_get = requests.get
def patched_get(*args, **kwargs):
    kwargs['verify'] = False
    return original_get(*args, **kwargs)
requests.get = patched_get

# Also patch Session
session = requests.Session()
session.verify = False

from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import httpx

client = httpx.Client(verify=False)

# LLM
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-LXItlj5p_LKYSHv01Xxsxg",
    http_client=client
)

# ... rest of your code

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