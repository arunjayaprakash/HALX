import sys
from pathlib import Path

# Add the project root to Python path if needed
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.arxiv import ArxivService, ArxivConfig
from app.services.embeddings import EmbeddingsService

def main():
    try:
        # Initialize services
        arxiv_config = ArxivConfig(max_results=5, categories=["cs.AI"])
        arxiv_service = ArxivService(arxiv_config)
        embeddings_service = EmbeddingsService()  # Using in-memory database

        # Fetch some papers to test with
        print("Fetching sample papers from ArXiv...")
        papers = arxiv_service.fetch_papers()
        print(f"Fetched {len(papers)} papers")

        # Test adding papers to vector store
        print("\nAdding papers to vector store...")
        success = embeddings_service.add_papers(papers)
        print(f"Adding papers {'succeeded' if success else 'failed'}")

        # Test similarity search
        print("\nTesting similarity search...")
        test_query = "recent advances in transformer architectures"
        results = embeddings_service.search_similar(
            query=test_query,
            n_results=3
        )
        
        print(f"\nTop 3 papers related to: '{test_query}'")
        for result in results:
            paper = result['paper']
            score = result['similarity_score']
            print(f"\nTitle: {paper.title}")
            print(f"Similarity Score: {score:.3f}")
            print("-" * 50)

        # Test category filtering
        print("\nTesting search with category filter...")
        filtered_results = embeddings_service.search_similar(
            query=test_query,
            n_results=3,
            filter_categories=["cs.AI"]
        )
        
        print(f"\nTop 3 AI-specific papers related to: '{test_query}'")
        for result in filtered_results:
            paper = result['paper']
            score = result['similarity_score']
            print(f"\nTitle: {paper.title}")
            print(f"Categories: {', '.join(paper.categories)}")
            print(f"Similarity Score: {score:.3f}")
            print("-" * 50)

        # Test paper deletion
        if papers:
            test_paper_id = papers[0].arxiv_id
            print(f"\nTesting paper deletion for ID: {test_paper_id}")
            deleted = embeddings_service.delete_paper(test_paper_id)
            print(f"Deletion {'succeeded' if deleted else 'failed'}")

    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    main()