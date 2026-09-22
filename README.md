# Axolotl RAG

Axolotl RAG is a Retrieval-Augmented Generation (RAG) application that leverages **LangChain**, **Google Gemini**, and **ChromaDB** to ingest documents and accurately answer user queries with context-based precision. It features a conversational CLI with real-time streaming and source citation, showcasing an efficient local RAG pipeline designed to run using the fast `uv` package manager.

## Architecture

The project is built with a modular architecture leveraging the LangChain ecosystem:
- **Language Model & Embeddings**: Uses `langchain-google-genai` to interface with Google's Gemini models (`gemini-3.6-flash` for generation and `gemini-embedding-001` for vector embeddings).
- **Vector Database**: Uses `langchain-chroma` to persistently store and query document embeddings locally using ChromaDB.
- **Data Ingestion**: Uses `langchain-community` loaders (`PyPDFLoader`, `TextLoader`) and `RecursiveCharacterTextSplitter` to process raw files.
- **Orchestration**: The `RAGEngine` uses LangChain Expression Language (LCEL) to build a streamlined retrieval and generation pipeline.

## How the RAG Pipeline Works (Deep Dive)

1. **Ingestion & Parsing**: Documents (like `.txt`, `.md`, or `.pdf`) are loaded from `data/documents/`. The raw files are parsed to extract purely the text content, discarding formatting (except for structured markdown/txt) using LangChain's document loaders.
2. **Chunking (Text Splitting)**: Since Large Language Models have a limited context window (and processing entire books at once is slow and expensive), the parsed text is split into smaller blocks called "chunks". 
   - We use the `RecursiveCharacterTextSplitter`.
   - It separates the text optimally by paragraphs (`\n\n`), then sentences (`\n`), and finally words.
   - Each chunk has exactly 1000 characters, with an overlap of 200 characters between consecutive chunks. This overlap ensures that a sentence split down the middle doesn't lose its context.
3. **Embeddings generation**: Once the text is chunked, each chunk is passed through the `gemini-embedding-001` model. This model acts as a translator: it reads the text and converts its semantic meaning into a dense vector (a long array of floating-point numbers, e.g., `[0.14, -0.82, 0.05, ...]`). In this multidimensional space, chunks that talk about similar topics are mathematically placed closer together.
4. **Vector Storage**: These mathematical arrays (embeddings), along with their original text and metadata (like the source file name), are stored in **ChromaDB**, which acts as our highly optimized vector database.
5. **Retrieval (Semantic Search)**: When a user asks a question (e.g., *"Where do axolotls live?"*):
   - The question is converted into an embedding using the exact same Gemini model.
   - ChromaDB calculates the geometric "distance" (cosine similarity) between the question's vector and all the stored document vectors.
   - It retrieves the top `K` (usually 4) chunks with the shortest distance, meaning they are the most semantically relevant to the question.
6. **Generation**: The retrieved chunks are injected into a strict prompt template as "Context". The Gemini LLM (`gemini-3.6-flash`) generates a precise answer based *only* on that context, actively citing its sources in the response.

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
