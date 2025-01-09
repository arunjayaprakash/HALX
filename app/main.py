from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional
from datetime import datetime
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


logging.info("STARTING")


from .services import (
    ArxivService, 
    ArxivConfig, 
    MongoDBService, 
    EmbeddingsService,
    RAGService
)

from .models.paper import Paper

from .config import settings

logging.info("IMPORTED MODULES")

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

# embeddings_service = EmbeddingsService(
#     persist_directory=os.getenv("CHROMA_PATH", "./chroma_db")
# )

# Update to FAISS implementation
embeddings_service = EmbeddingsService(
    model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)


rag_service = RAGService(
    embeddings_service=embeddings_service,
    model_name=os.getenv("LLM_MODEL", "mistral")
)

arxiv_service = ArxivService(
    config=ArxivConfig(
        max_results=int(os.getenv("ARXIV_MAX_RESULTS", "50")),
        # categories=os.getenv("CATEGORIES", "cs.AI,cs.LG").split(",")
        categories=settings.categories
    )
)

arxiv_service.rag_service = rag_service  # RAG service for summary gen.

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    duration = datetime.now() - start_time
    logging.info(f"{request.method} {request.url} took {duration.total_seconds():.2f}s")
    return response

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "message": "HALX API is running"}

@app.get("/papers/fetch", response_model=List[Paper])
async def fetch_papers():
    """Fetch new papers from ArXiv"""
    try:
        logger.info("Starting paper fetch process...")
        
        # Fetch papers
        papers = await arxiv_service.fetch_papers()
        logger.info(f"Fetched {len(papers)} papers from ArXiv")
        
        # Process papers with RAG
        logger.info("Starting RAG processing...")
        processed_papers = await rag_service.batch_process_papers(papers)
        logger.info("RAG processing complete")
        
        # Store in MongoDB
        logger.info("Storing papers in MongoDB...")
        await mongodb_service.store_papers(processed_papers)
        logger.info("Storage complete")
        
        return processed_papers
    except Exception as e:
        logger.error(f"Error in fetch_papers: {str(e)}")
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

@app.post("/papers/batch")
async def process_paper_batch(papers: List[Paper]):
    """Process a batch of papers"""
    try:
        processed = await rag_service.batch_process_papers(papers)
        stored = mongodb_service.store_papers(processed)
        return {"processed": len(processed), "stored": stored}
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

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "total_papers": mongodb_service.get_total_papers(),
        "categories": mongodb_service.get_unique_categories(),
        "latest_update": mongodb_service.get_latest_update_time()
    }

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