from typing import List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from datetime import datetime
import logging
from ..models.paper import Paper

logger = logging.getLogger(__name__)

class MongoDBService:
    def __init__(self, connection_url: str = "mongodb://localhost:27017", database_name: str = "halx"):
        """Initialize MongoDB connection"""
        self.client = MongoClient(connection_url)
        self.db: Database = self.client[database_name]
        self.papers: Collection = self.db.papers
        self._setup_indexes()

    def _setup_indexes(self):
        """Setup necessary indexes"""
        # Unique index on arxiv_id
        self.papers.create_index("arxiv_id", unique=True)
        # Index for text search
        self.papers.create_index([
            ("title", "text"),
            ("abstract", "text"),
            ("summary", "text")
        ])
        # Index for date-based queries
        self.papers.create_index("published_date")

    def store_papers(self, papers: List[Paper]) -> int:
        """
        Store multiple papers, handling duplicates
        Returns number of papers successfully stored
        """
        stored_count = 0
        for paper in papers:
            try:
                # Convert to dict and handle datetime
                paper_dict = paper.model_dump()
                
                # Upsert based on arxiv_id
                result = self.papers.update_one(
                    {"arxiv_id": paper.arxiv_id},
                    {"$set": paper_dict},
                    upsert=True
                )
                
                if result.upserted_id or result.modified_count:
                    stored_count += 1
                    
            except Exception as e:
                logger.error(f"Error storing paper {paper.arxiv_id}: {str(e)}")
                continue
                
        return stored_count

    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Retrieve a specific paper by ArXiv ID"""
        try:
            paper_dict = self.papers.find_one({"arxiv_id": arxiv_id})
            if paper_dict:
                # Convert MongoDB _id to string and remove it
                paper_dict.pop('_id')
                return Paper(**paper_dict)
            return None
        except Exception as e:
            logger.error(f"Error retrieving paper {arxiv_id}: {str(e)}")
            return None

    def search_papers(self, 
                     query: str = "", 
                     categories: List[str] = None, 
                     start_date: datetime = None,
                     end_date: datetime = None,
                     limit: int = 10) -> List[Paper]:
        """Search papers with various filters"""
        try:
            # Build query
            search_query = {}
            
            # Text search
            if query:
                search_query["$text"] = {"$search": query}
                
            # Category filter
            if categories:
                search_query["categories"] = {"$in": categories}
                
            # Date range
            if start_date or end_date:
                date_query = {}
                if start_date:
                    date_query["$gte"] = start_date
                if end_date:
                    date_query["$lte"] = end_date
                if date_query:
                    search_query["published_date"] = date_query

            # Execute search
            cursor = self.papers.find(search_query).limit(limit)
            
            # Convert results to Paper objects
            papers = []
            for paper_dict in cursor:
                paper_dict.pop('_id')  # Remove MongoDB _id
                papers.append(Paper(**paper_dict))
                
            return papers
            
        except Exception as e:
            logger.error(f"Error searching papers: {str(e)}")
            return []

    def get_recent_papers(self, limit: int = 10) -> List[Paper]:
        """Get most recent papers"""
        try:
            cursor = self.papers.find().sort("published_date", -1).limit(limit)
            
            papers = []
            for paper_dict in cursor:
                paper_dict.pop('_id')
                papers.append(Paper(**paper_dict))
                
            return papers
        except Exception as e:
            logger.error(f"Error getting recent papers: {str(e)}")
            return []

    def delete_paper(self, arxiv_id: str) -> bool:
        """Delete a paper by ArXiv ID"""
        try:
            result = self.papers.delete_one({"arxiv_id": arxiv_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting paper {arxiv_id}: {str(e)}")
            return False