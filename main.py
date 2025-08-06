import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

import pytesseract
from PIL import Image

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTORSTORE_DIR = "../rag_db"
if os.path.exists(VECTORSTORE_DIR):
    shutil.rmtree(VECTORSTORE_DIR)


txt_loader = DirectoryLoader(
    "synthetic_company_dataset",
    glob="**/*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True
)
txt_documents = txt_loader.load()


pdf_documents = []
pdf_dir = Path("synthetic_company_dataset")
for pdf_file in pdf_dir.rglob("*.pdf"):
    loader = PyPDFLoader(str(pdf_file))
    pdf_documents.extend(loader.load())


image_documents = []
for image_file in pdf_dir.rglob("*.png"):
    text = pytesseract.image_to_string(Image.open(image_file), config='--oem 3 --psm 6')
    image_documents.append({"page_content": text, "metadata": {"source": str(image_file)}})

for image_file in pdf_dir.rglob("*.jpg"):
    text = pytesseract.image_to_string(Image.open(image_file), config='--oem 3 --psm 6')
    image_documents.append({"page_content": text, "metadata": {"source": str(image_file)}})


documents = txt_documents + pdf_documents + image_documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = Chroma.from_documents(
    chunks,
    embedding=embedding_model,
    persist_directory=VECTORSTORE_DIR
)
retriever = vectorstore.as_retriever()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(payload: Question):
    try:
        response = qa_chain.invoke({"query": payload.query})
        return {"answer": response["result"]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "RAG API running with TXT, PDF, and OCR image support (Gemini 1.5 Flash)"}
