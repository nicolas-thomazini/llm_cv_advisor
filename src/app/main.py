import os
import streamlit as st
from pathlib import Path

from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import retrieval_qa

os.environ["OPENAI_API_KEY"] = os.environ["MISTRAL_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.mistral.ai/v1"

loader = TextLoader("documents.json")
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-V6-v2")
db = FAISS.from_documents(chunks, embeddings)

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

llm = ChatOpenAI(
    model_name="mistral-large-latest",
    openai_api_key=os.environ["MISTRAL_API_KEY"],
    openai_api_base="https://api.mistral.ai/v1",
    openai_api_type="openai"
)

retriever = db.as_retriever()

qa_chain = retrieval_qa.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    return_source_documents=True
)

st.set_page_config(page_title="Analisador de Currículo com LLM", layout="centered")
st.title("📄 Analisador Inteligente de Currículos com LLM")

uploaded_file = st.file_uploader("Envie seu currículo (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if uploaded_file:
    Path("uploads/").mkdir(exist_ok=True)
    file_path = Path("uploads") / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Arquivo '{uploaded_file.name}' recebido!")

    new_docs = load_documents("uploads/")
    new_chunks = splitter.split_documents(new_docs)
    db.add_documents(new_chunks)
    retriever = db.as_retriever()
    qa_chain.retriever = retriever
    st.info("Documento indexado com sucesso!")

query = st.text_input("Digite sua pergunta sobre o currículo:")

if query:
    result = qa_chain({"query": query})
    st.markdown("### Resposta:")
    st.write(result["result"])

    with st.expander("📚 Fontes utilizadas"):
        for doc in result["source_documents"]:
            st.markdown(f"**Fonte:** {doc.metadata.get('source', 'Desconhecida')}")
            st.code(doc.page_content[:500])
            