from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import shutil

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import retrieval_qa

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

os.environ["OPENAI_API_KEY"] = os.environ["MISTRAL_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://api.mistral.ai/v1"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

loader = TextLoader("documents.json")
docs = loader.load()
chunks = splitter.split_documents(docs)

if Path("faiss_index").exists():
    db = FAISS.load_local("faiss_index", embeddings)
else:
    db = FAISS.load_documents(chunks, embeddings)
    db.save_local("faiss_index")

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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filepath = UPLOAD_DIR / file.filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if filepath.suffix == ".pdf":
        loader = PyMuPDFLoader(str(filepath))
    elif file.suffix == ".docx":
            loader = Docx2txtLoader(str(filepath))
    elif file.suffix == ".txt": 
            loader = TextLoader(str(filepath))
    else:
        return JSONResponse(status_code=400, content={"error", "Formato não suportado."})
    
    new_docs = loader.load()
    new_chunks = splitter.split_documents(new_docs)
    db.add_documents(new_chunks)
    db.save_locaal("faiss_index")
    return {"message": f"Arquivo '{file.filename} indexado com sucesso."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
     result = qa_chain({"query": question})
     return {
          "answer": result["result"],
          "sources": [doc.metada.get("source", "sem nome")
                            for doc in result ["source_documents"]                                                                  
        ]
     }

history = []

@app.post("/feedback")
async def feedback(question: str = Form(...), feedback: str = Form(...)):
     history.append({"question": question, "feedback": feedback})
     return {"message": "Feedback salvo com sucesso"}

@app.get("/history")
def get_history():
     return history
