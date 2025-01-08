import sys
from pathlib import Path

# Add the project root to Python path if needed
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.arxiv import ArxivService, ArxivConfig

def main():
    # Initialize the service
    config = ArxivConfig(max_results=10, categories=["cs.AI"])
    service = ArxivService(config)

    # Fetch papers
    print("Fetching papers...")
    papers = service.fetch_papers()
    
    # Print results
    for paper in papers:
        print(f"\nTitle: {paper.title}")
        print(f"Authors: {', '.join(paper.authors)}")
        print(f"ArXiv ID: {paper.arxiv_id}")
        print("-" * 50)

if __name__ == "__main__":
    main()