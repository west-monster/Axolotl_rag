# Axolotl RAG

Axolotl RAG is a Retrieval-Augmented Generation (RAG) application that leverages **LangChain**, **Google Gemini**, and **ChromaDB** to ingest documents and accurately answer user queries with context-based precision. It features a conversational CLI with real-time streaming and source citation, showcasing an efficient local RAG pipeline designed to run using the fast `uv` package manager.

## Architecture

The project is built with a modular architecture leveraging the LangChain ecosystem:
- **Language Model & Embeddings**: Uses `langchain-google-genai` to interface with Google's Gemini models (`gemini-3.6-flash` for generation and `gemini-embedding-001` for vector embeddings).
- **Vector Database**: Uses `langchain-chroma` to persistently store and query document embeddings locally using ChromaDB.
- **Data Ingestion**: Uses `langchain-community` loaders (`PyPDFLoader`, `TextLoader`) and `RecursiveCharacterTextSplitter` to process raw files.
- **Orchestration**: The `RAGEngine` uses LangChain Expression Language (LCEL) to build a streamlined retrieval and generation pipeline.

## How the RAG Pipeline Works

1. **Ingestion**: Documents (like `.txt`, `.md`, or `.pdf`) are read from the `data/documents/` directory.
2. **Chunking**: The text is split into smaller, overlapping chunks (e.g., 1000 characters) to preserve context without exceeding LLM token limits.
3. **Embedding**: Each chunk is converted into a mathematical vector representation using the Gemini embedding model and stored in ChromaDB.
4. **Retrieval**: When a user asks a question, the query is embedded, and ChromaDB performs a similarity search to find the most relevant chunks of text.
5. **Generation**: The retrieved chunks are injected into a prompt template as "Context". The Gemini LLM generates a precise answer based *only* on that context, actively citing its sources in the response.

## Setup and Usage

### Prerequisites
1. Install the [uv](https://github.com/astral-sh/uv) package manager.
2. Rename `.env.example` to `.env` and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-3.6-flash
   EMBEDDING_MODEL=gemini-embedding-001
   ```

### Running the Application

Use `uv` to automatically handle dependencies and run the interactive CLI in an isolated environment.

**To ingest a document and start the chat immediately:**
```bash
uv run python -m src.cli --ingest data/documents/sample.md
```

**To start the chat directly (if documents are already ingested):**
```bash
uv run python -m src.cli
```

### CLI Commands
Once inside the interactive terminal, you can ask questions in natural language, or use the following special commands:
- `/ingest <path>`: Index a new file or directory on the fly.
- `/stats`: View how many document chunks are currently stored in the vector database.
- `/clear`: Wipe the vector database completely to start fresh.
- `/exit` or `/quit`: Close the application safely.
