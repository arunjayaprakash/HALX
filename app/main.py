from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

from .services import (
    ArxivService, 
    ArxivConfig, 
    MongoDBService, 
    EmbeddingsService,
    RAGService
)
from .models.paper import Paper

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="HALX Research Assistant",
    description="An intelligent research assistant that helps you navigate ArXiv papers",
    version="0.1.0"
)

# Initialize services
mongodb_service = MongoDBService(
    connection_url=os.getenv("MONGODB_URL", "mongodb://localhost:27017")
)

embeddings_service = EmbeddingsService(
    persist_directory=os.getenv("CHROMA_PATH", "./chroma_db")
)

rag_service = RAGService(
    embeddings_service=embeddings_service,
    model_name=os.getenv("LLM_MODEL", "mistral:7b-q4")
)

arxiv_service = ArxivService(
    config=ArxivConfig(
        max_results=int(os.getenv("ARXIV_MAX_RESULTS", "50")),
        categories=os.getenv("CATEGORIES", "cs.AI,cs.LG").split(",")
    )
)
arxiv_service.rag_service = rag_service  # Add RAG service for summary generation

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HALX API is running"}

@app.get("/papers/fetch", response_model=List[Paper])
async def fetch_papers():
    """Fetch new papers from ArXiv"""
    try:
        # Fetch papers
        papers = await arxiv_service.fetch_papers()
        
        # Process papers with RAG (generate summaries and embeddings)
        processed_papers = await rag_service.batch_process_papers(papers)
        
        # Store in MongoDB
        mongodb_service.store_papers(processed_papers)
        
        return processed_papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/recent", response_model=List[Paper])
async def get_recent_papers(
    limit: int = Query(10, gt=0, le=50)
):
    """Get most recent papers from database"""
    papers = mongodb_service.get_recent_papers(limit=limit)
    return papers

@app.get("/papers/{arxiv_id}", response_model=Paper)
async def get_paper(arxiv_id: str):
    """Get specific paper by ArXiv ID"""
    paper = mongodb_service.get_paper(arxiv_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@app.post("/papers/query")
async def query_papers(
    query: str,
    n_results: int = Query(3, gt=0, le=10),
    categories: Optional[List[str]] = Query(None)
):
    """Query papers using RAG"""
    try:
        result = await rag_service.query(
            query=query,
            n_results=n_results,
            filter_categories=categories
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/search", response_model=List[Paper])
async def search_papers(
    query: str = "",
    categories: Optional[List[str]] = Query(None),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = Query(10, gt=0, le=50)
):
    """Search papers with filters"""
    papers = mongodb_service.search_papers(
        query=query,
        categories=categories,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    return papers

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # You could add initialization logic here
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # You could add cleanup logic here
    pass