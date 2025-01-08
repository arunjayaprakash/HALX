from datetime import datetime
from typing import List, Optional
import arxiv
from pydantic import BaseModel
import logging
from ..models.paper import Paper

logger = logging.getLogger(__name__)

class ArxivConfig(BaseModel):
    """Configuration for ArXiv fetching service"""
    max_results: int = 50
    categories: List[str] = ["cs.AI", "cs.LG"]
    sort_by: str = "submittedDate"
    ascending: bool = False

class ArxivService:
    def __init__(self, config: Optional[ArxivConfig] = None):
        self.config = config or ArxivConfig()
        self.client = arxiv.Client()

    def fetch_papers(self) -> List[Paper]:
        """
        Fetch papers from ArXiv based on configured categories
        Returns list of Paper objects
        """
        try:
            # Construct category query
            category_query = " OR ".join(f"cat:{cat}" for cat in self.config.categories)
            
            # Create search object
            search = arxiv.Search(
                query=category_query,
                max_results=self.config.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )

            papers = []
            for result in self.client.results(search):
                try:
                    paper = Paper(
                        arxiv_id=result.entry_id.split('/')[-1],
                        title=result.title,
                        abstract=result.summary,
                        authors=[author.name for author in result.authors],
                        categories=[cat.lower() for cat in result.categories],
                        published_date=result.published,
                        updated_date=result.updated,
                        pdf_url=result.pdf_url,
                        summary=self._generate_summary(abstract=result.summary, title=result.title)
                    )
                    papers.append(paper)
                except Exception as e:
                    logger.error(f"Error processing paper {result.entry_id}: {str(e)}")
                    continue

            return papers

        except Exception as e:
            logger.error(f"Error fetching papers from ArXiv: {str(e)}")
            raise

    def _generate_summary(self, abstract: str, title: str) -> str:
        """
        Generate a concise summary of the abstract
        For POC, we'll just return the abstract
        TODO: Implement actual summarization using Mistral
        """
        return abstract

    def get_paper_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """Fetch a specific paper by ArXiv ID"""
        try:
            search = arxiv.Search(
                id_list=[arxiv_id],
                max_results=1
            )
            
            results = list(self.client.results(search))
            if not results:
                return None
                
            result = results[0]
            return Paper(
                arxiv_id=result.entry_id.split('/')[-1],
                title=result.title,
                abstract=result.summary,
                authors=[author.name for author in result.authors],
                categories=[cat.lower() for cat in result.categories],
                published_date=result.published,
                updated_date=result.updated,
                pdf_url=result.pdf_url,
                summary=self._generate_summary(abstract=result.summary, title=result.title)
            )
        except Exception as e:
            logger.error(f"Error fetching paper {arxiv_id}: {str(e)}")
            return None

    def search_papers(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search papers with a specific query"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = []
            for result in self.client.results(search):
                try:
                    paper = Paper(
                        arxiv_id=result.entry_id.split('/')[-1],
                        title=result.title,
                        abstract=result.summary,
                        authors=[author.name for author in result.authors],
                        categories=[cat.lower() for cat in result.categories],
                        published_date=result.published,
                        updated_date=result.updated,
                        pdf_url=result.pdf_url,
                        summary=self._generate_summary(abstract=result.summary, title=result.title)
                    )
                    papers.append(paper)
                except Exception as e:
                    logger.error(f"Error processing paper {result.entry_id}: {str(e)}")
                    continue

            return papers

        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            raise