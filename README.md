# HALX

An intelligent research assistant that collects papers from ArXiv, analyzes them, and helps you stay at the forefront of scientific discovery.

## Project Overview

HALX is a research assistant system that helps users navigate and understand ArXiv papers.

The project implements a Retrieval-Augmented Generation (RAG) system that:
  1. Fetches recent research papers from ArXiv API
  2. Stores paper summaries and metadata in MongoDB
  3. Creates and stores embeddings using FAISS (originally planned with Chroma)
  4. Provides a query interface using Mistral-7B

### Core Components:

1. Paper Management (`paper.py`):
- Defines the core Paper model using Pydantic
- Handles paper metadata including ArXiv ID, title, abstract, summary, authors, and categories

2. ArXiv Integration (`arxiv.py`):
- Fetches papers from ArXiv based on configured categories
- Handles paper metadata extraction and initial processing
- Interfaces with the RAG service for paper summarization

3. Embeddings System (`embeddings.py`):
- Uses sentence-transformers
- Implements FAISS for efficient similarity search
- Maintains paper embeddings and metadata mapping
- Used for semantic search and finding similar papers

4. RAG Pipeline (`rag.py`):
- Uses Mistral (via Ollama) as the base LLM for:
  - Generating paper summaries
  - Answering user queries about papers
- Implements two main prompt templates:
  - Query prompt: For answering research questions using paper context
  - Summary prompt: For generating concise paper summaries

5. Storage (`storage.py`):
- MongoDB integration for persistent storage
- Handles CRUD operations for papers
- Implements text search and filtering capabilities

### RAG System Flow:

1. Paper Ingestion:
```
ArXiv Service -> RAG Service -> Embeddings Service -> MongoDB
```
- Papers are fetched from ArXiv
- Summaries are generated using Mistral
- Embeddings are created and stored in FAISS
- Full paper data is stored in MongoDB

2. Query Processing:
```
User Query -> Embeddings Search -> RAG Context Building -> Mistral Response
```
- User query is processed through embeddings to find relevant papers
- Relevant papers are formatted into context
- Mistral generates a response using the context

### Model Usage Breakdown:

1. Mistral (via Ollama):
- Used in `rag.py` for:
  - Generating paper summaries
  - Answering user queries with context
- Configured through `LLM_MODEL` environment variable

2. Sentence Transformers:
- Used in `embeddings.py` for:
  - Creating paper embeddings
  - Enabling semantic search
- "all-MiniLM-L6-v2" by default
- Configured through `EMBEDDING_MODEL` environment variable

3. FAISS:
- Storing and indexing embeddings
- Performing efficient similarity search
- Implements L2 distance-based similarity

## Tech Stack

### Core Components
- **Python 3.10+**: Primary development language
- **FastAPI**: API server framework
- **MongoDB**: Document storage for paper metadata and summaries
- **FAISS**: Vector database for embeddings (switched from Chroma for better compatibility)
- **Ollama**: Local LLM deployment
- **Mistral-7B**: Base LLM

### Key Libraries
- `arxiv`: ArXiv API client
- `pymongo`: MongoDB client
- `sentence-transformers`: Document embeddings
- `faiss-cpu`: Vector similarity search
- `langchain-community`: RAG pipeline orchestration
- `pydantic`: Data validation
- `python-dotenv`: Environment management

## Project Structure
```
research-rag/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── paper.py         # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── arxiv.py         # ArXiv fetching logic
│   │   ├── storage.py       # MongoDB operations
│   │   ├── embeddings.py    # FAISS operations
│   │   └── rag.py          # RAG pipeline
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # Utility functions (Empty for now)
├── tests/
│   └── test_arxiv.py       # ArXiv service tests
│   └── test_mongo.py       # MongoDB service tests
│   └── test_embeddings.py  # FAISS embeddings tests
│   └── test_rag.py        # RAG pipeline tests
├── .env.example
├── requirements.txt
└── README.md
```

## Implementation Notes

### Phase 1: Basic Infrastructure ✓
- [x] Project setup and dependency management
- [x] Basic ArXiv integration with paper fetching
- [x] MongoDB schema design and integration
- [x] Simple paper fetching and storage
- [x] Test suite for basic services

### Phase 2: RAG Implementation ✓
- [x] Embedding generation pipeline using sentence-transformers
- [x] FAISS integration for vector storage
  - Switched from ChromaDB to FAISS due to SQLite version compatibility issues
  - FAISS provides better performance and simpler deployment
- [x] Mistral-7B integration via Ollama
- [x] Basic RAG query implementation with async support
- [x] Test suite for RAG components

### Phase 3: API and Optimization ✓
- [x] FastAPI endpoints implemented:
  - `/papers/fetch`: Fetch and process new papers from ArXiv
  - `/papers/recent`: Get most recent papers from database
  - `/papers/{arxiv_id}`: Get specific paper by ID
  - `/papers/query`: Semantic search using RAG
  - `/papers/batch`: Process paper batches
  - `/papers/search`: Text-based search with filters
  - `/stats`: System statistics and monitoring
- [x] Batch processing support
- [x] Basic error handling and logging
  - Request/response logging middleware
  - Error handling middleware
  - Rate limiting implementation
- [?] Caching layer

### Phase 4: Future Enhancements
- [ ] Scheduler for periodic updates
- [ ] Migration to Weaviate/Qdrant
- [ ] Advanced query capabilities
- [ ] Web interface

### Technical Changes and Rationale

1. **Vector Store Migration (ChromaDB → FAISS)**
   - Originally planned to use ChromaDB but encountered SQLite version compatibility issues
   - FAISS provides several advantages:
     - No external database dependencies
     - Better performance for similarity search
     - Lighter resource footprint
     - Simpler deployment with pure Python implementation

2. **Async Implementation**
   - Implemented async support in RAG service for better performance
   - Helps handle I/O-bound operations efficiently:
     - LLM inference
     - Paper fetching
     - Embedding generation

3. **LangChain Updates**
   - Migrated to langchain-community for LLM operations
   - Following best practices for deprecated package management

## Setup Instructions

1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

2. Install Ollama and download Mistral
```bash
# Install Ollama from https://ollama.ai/
ollama pull mistral:7b-q4
```

3. Install MongoDB Community Edition
- Follow instructions at: https://docs.mongodb.com/manual/installation/

4. Configuration
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings
MONGODB_URL=mongodb://localhost:27017
ARXIV_MAX_RESULTS=50
CATEGORIES=cs.AI,cs.LG  # Paper Categories
```

## Usage

1. Start the MongoDB service
```bash
mongod --dbpath /path/to/data/db
```

2. Start Ollama
```bash
ollama serve
```

3. Run the FastAPI application
```bash
uvicorn app.main:app --reload
```


## Current Limitations

- Using base Mistral-7B model (can be resource intensive)
- Limited to ArXiv papers in specified categories
- Basic error handling in POC
- Local-only deployment
- FAISS index is in-memory only (no persistence)

## License

MIT License - feel free to use this for your portfolio/resume