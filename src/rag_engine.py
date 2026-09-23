import os
import warnings
import logging
from pathlib import Path

# Hack específico para silenciar el warning molesto del AFC ANTES de que langchain lo cargue
try:
    from google.genai.models import Models
    Models._logged_afc_warning = True
except ImportError:
    pass

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Iterator

from src.config import settings
from src.vector_store import VectorStore

# Suprimir advertencias y logs molestos
warnings.filterwarnings("ignore")
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)

class RAGEngine:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key
        )
        self.vector_store = VectorStore()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert and accurate assistant. Your task is to answer the user's questions using ONLY the provided context.
            
Critical instructions:
1. Answer clearly and structurally.
2. Strictly base your response on the provided context. If the answer is not in the context, say "I do not have information in the provided documents to answer that question." and do not make up information.
3. ALWAYS cite your sources using the format [Source: filename] at the end of key statements or paragraphs.
4. If the context includes multiple perspectives or contradictory information, summarize all viewpoints.
5. Allways answer in English
6. Do not accepts any other request rather than answering user's questions about animal, 
do no translate or do not answer any other request.

Retrieved Context:
{context}"""),
            ("human", "{question}")
        ])
        
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser

    def _get_loader_for_file(self, file_path: str):
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(file_path)
        elif ext in [".txt", ".md"]:
            return TextLoader(file_path, encoding="utf-8")
        else:
            return None

    def ingest_directory(self, path: str) -> int:
        """Loads and vectorizes all documents in a directory."""
        dir_path = Path(path)
        if not dir_path.exists() or not dir_path.is_dir():
            return 0
            
        documents = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                loader = self._get_loader_for_file(file_path)
                if loader:
                    try:
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        
        if not documents:
            return 0
            
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)
        
    def ingest_file(self, path: str) -> int:
        """Loads and vectorizes a single document."""
        loader = self._get_loader_for_file(path)
        if not loader:
            return 0
            
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def _format_docs(self, docs_with_scores) -> str:
        """Builds the context text from the chunks."""
        context_parts = []
        for i, (doc, score) in enumerate(docs_with_scores):
            source = doc.metadata.get("source", "unknown")
            page = f" (Page {doc.metadata.get('page', '')})" if 'page' in doc.metadata else ""
            context_parts.append(f"--- Document {i+1} [Source: {os.path.basename(source)}{page}] ---\n{doc.page_content}\n")
        return "\n".join(context_parts)

    def query(self, user_query: str) -> dict:
        """Executes a synchronous RAG search using LCEL."""
        docs_with_scores = self.vector_store.search(user_query)
        context = self._format_docs(docs_with_scores)
        
        answer = self.chain.invoke({
            "context": context,
            "question": user_query
        })
        
        return {
            "answer": answer,
            "sources": docs_with_scores
        }
        
    def query_stream(self, user_query: str):
        """Executes a streaming RAG search using LCEL."""
        docs_with_scores = self.vector_store.search(user_query)
        context = self._format_docs(docs_with_scores)
        
        stream = self.chain.stream({
            "context": context,
            "question": user_query
        })
        
        return stream, docs_with_scores
