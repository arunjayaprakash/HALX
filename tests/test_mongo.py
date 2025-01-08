import sys
from pathlib import Path

# Add the project root to Python path if needed
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.arxiv import ArxivService, ArxivConfig
from app.services.storage import MongoDBService

def main():
    # Initialize services
    arxiv_config = ArxivConfig(max_results=10, categories=["cs.AI"])
    arxiv_service = ArxivService(arxiv_config)
    mongo_service = MongoDBService()  # Uses default localhost connection

    # Fetch papers from ArXiv
    print("Fetching papers from ArXiv...")
    papers = arxiv_service.fetch_papers()
    print(f"Found {len(papers)} papers")

    # Store in MongoDB
    print("\nStoring papers in MongoDB...")
    stored_count = mongo_service.store_papers(papers)
    print(f"Successfully stored {stored_count} papers")

    # Test retrieval
    print("\nRetrieving recent papers from MongoDB...")
    recent_papers = mongo_service.get_recent_papers(5)
    for paper in recent_papers:
        print(f"\nTitle: {paper.title}")
        print(f"Authors: {', '.join(paper.authors)}")
        print(f"ArXiv ID: {paper.arxiv_id}")
        print("-" * 50)

    # Test search
    print("\nTesting search functionality...")
    search_results = mongo_service.search_papers("neural networks", limit=3)
    print(f"Found {len(search_results)} papers about neural networks")
    for paper in search_results:
        print(f"\nTitle: {paper.title}")
        print("-" * 50)

if __name__ == "__main__":
    main()