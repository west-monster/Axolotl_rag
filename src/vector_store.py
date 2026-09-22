from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import settings, CHROMA_DB_DIR
import shutil
import os

class VectorStore:
    def __init__(self, collection_name: str = "axolotl_rag"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key
        )
        self.collection_name = collection_name
        
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )

    def add_documents(self, documents: list):
        """Adds LangChain Document objects to Chroma."""
        if not documents:
            return
        self.db.add_documents(documents)

    def search(self, query: str, top_k: int = settings.top_k):
        """Searches for the most relevant documents given a query."""
        if not query.strip():
            return []
            
        # Using similarity_search_with_score to return distance
        results = self.db.similarity_search_with_score(query, k=top_k)
        return results

    def count(self) -> int:
        """Returns the total number of documents in the collection."""
        try:
            return self.db._collection.count()
        except:
            return 0
        
    def clear(self):
        """Removes all items from the vector database."""
        try:
            self.db.delete_collection()
        except Exception:
            pass
            
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(CHROMA_DB_DIR)
        )
