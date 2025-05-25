import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import retrieval_qa
from pathlib import Path

from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader

def load_documents(path):
    documents = []
    for file in Path(path).glob("*"):
        if file.suffix == '.pdf':
            loader = PyMuPDFLoader(str(file))
        elif file.suffix == ".docx":
            loader = Docx2txtLoader(str(file))
        elif file.suffix == ".txt": 
            loader = TextLoader(str(file))
        else:
            continue
        documents.extend(loader)
    return documents
 
os.environ["OPENAI_API_KEY"] = os.environ["MISTRAL_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.mistral.ai/v1"

loader = TextLoader("documents.json")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-V6-v2"
)

db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(
    model_name="mistral-large-latest",
    openai_api_key=os.environ["MISTRAL_API_KEY"],
    openai_api_base="https://api.mistral.ai/v1",
    openai_api_type="openai"
)

qa_chain = retrieval_qa.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)