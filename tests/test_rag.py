import sys
import asyncio
from pathlib import Path

# Add the project root to Python path if needed
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.arxiv import ArxivService, ArxivConfig
from app.services.embeddings import EmbeddingsService
from app.services.rag import RAGService

async def main():
    try:
        # Initialize services
        arxiv_config = ArxivConfig(max_results=5, categories=["cs.AI"])
        arxiv_service = ArxivService(arxiv_config)
        embeddings_service = EmbeddingsService()  # No need for persist_directory with FAISS
        rag_service = RAGService(embeddings_service=embeddings_service)

        # Fetch sample papers
        print("Fetching sample papers from ArXiv...")
        papers = arxiv_service.fetch_papers()
        print(f"Fetched {len(papers)} papers")

        # Test batch processing (summary generation and embedding storage)
        print("\nTesting batch processing of papers...")
        processed_papers = await rag_service.batch_process_papers(papers)
        print("Batch processing completed")

        # Print sample of generated summaries
        print("\nSample of generated summaries:")
        for paper in processed_papers[:2]:  # Show first 2 papers
            print(f"\nTitle: {paper.title}")
            print(f"Generated Summary: {paper.summary[:200]}...")  # Show first 200 chars
            print("-" * 50)

        # Test RAG querying
        test_queries = [
            # "What are the latest developments in transformer architectures?",
            # "Explain recent advances in reinforcement learning",
            "What are some recent advancements on using LLMs in languages other than English?"
        ]

        print("\nTesting RAG queries...")
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            
            result = await rag_service.query(
                query=query,
                n_results=2
            )
            
            print("\nAI Assistant's Response:")
            print(result['answer'])
            
            print("\nRelevant Papers:")
            for paper in result['papers']:
                print(f"- {paper.title}")
            
            print("-" * 50)

        # Test RAG query with category filtering
        print("\nTesting RAG query with category filter...")
        filtered_result = await rag_service.query(
            query="What are the latest developments in neural networks?",
            n_results=2,
            filter_categories=["cs.AI"]
        )
        
        print("\nAI Assistant's Response (AI-specific papers):")
        print(filtered_result['answer'])
        
        print("\nRelevant Papers (should all be in cs.AI category):")
        for paper in filtered_result['papers']:
            print(f"- {paper.title}")
            print(f"  Categories: {', '.join(paper.categories)}")

    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())