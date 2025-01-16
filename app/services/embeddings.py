from typing import List, Dict, Any
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import logging
from ..models.paper import Paper

logger = logging.getLogger(__name__)

class EmbeddingsService:
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 use_gpu: bool = True):
        """Initialize the embeddings service with GPU support"""
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize sentence transformer with GPU if available
        device = 'cuda' if self.use_gpu else 'cpu'
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_model.to(device)
        
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index with GPU support if available
        if self.use_gpu:
            # Create CPU index first
            self.cpu_index = faiss.IndexFlatL2(self.dimension)
            
            # Convert to GPU index
            self.res = faiss.StandardGpuResources()  # Initialize GPU resources
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
            logger.info("FAISS index initialized on GPU")
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("FAISS index initialized on CPU")
        
        # Store paper metadata
        self.papers = {}  # arxiv_id -> Paper
        self.id_to_index = {}  # arxiv_id -> index position
        self.index_to_id = {}  # index position -> arxiv_id
        self.current_index = 0
        
        if self.use_gpu:
            logger.info(f"Using GPU (CUDA) for embeddings and similarity search")
        else:
            logger.info("Using CPU for embeddings and similarity search")

    def _prepare_paper_text(self, paper: Paper) -> str:
        """Prepare paper text for embedding"""
        return f"Title: {paper.title}\nAbstract: {paper.abstract}"

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a given text"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.astype('float32')
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def add_papers(self, papers: List[Paper]) -> bool:
        """Add papers to the vector store"""
        try:
            if not papers:
                return True

            for paper in papers:
                # Generate embedding
                text = self._prepare_paper_text(paper)
                embedding = self._generate_embedding(text)

                # Add to FAISS index
                self.index.add(embedding.reshape(1, -1))
                
                # Store metadata
                self.papers[paper.arxiv_id] = paper
                self.id_to_index[paper.arxiv_id] = self.current_index
                self.index_to_id[self.current_index] = paper.arxiv_id
                self.current_index += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding papers to vector store: {str(e)}")
            return False

    def search_similar(self, 
                      query: str, 
                      n_results: int = 5, 
                      filter_categories: List[str] = None) -> List[Dict[str, Any]]:
        """Search for similar papers based on semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search in FAISS
            D, I = self.index.search(query_embedding.reshape(1, -1), n_results)
            
            # Format results
            results = []
            for i, idx in enumerate(I[0]):
                if idx != -1:  # FAISS returns -1 for empty slots
                    paper_id = self.index_to_id[idx]
                    paper = self.papers[paper_id]
                    
                    # Apply category filter if specified
                    if filter_categories and not any(cat in paper.categories for cat in filter_categories):
                        continue
                        
                    results.append({
                        'paper': paper,
                        'similarity_score': 1.0 / (1.0 + D[0][i])  # Convert distance to similarity score
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar papers: {str(e)}")
            return []

    def delete_paper(self, arxiv_id: str) -> bool:
        """Delete a paper from the vector store"""
        try:
            if arxiv_id in self.papers:
                # Note: FAISS doesn't support deletion, so we'll mark it as deleted
                # in our metadata. In a production system, we'd want to rebuild the
                # index periodically to clean up deleted entries.
                idx = self.id_to_index[arxiv_id]
                del self.papers[arxiv_id]
                del self.id_to_index[arxiv_id]
                del self.index_to_id[idx]
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting paper {arxiv_id} from vector store: {str(e)}")
            return False

    def clear_collection(self) -> bool:
        """Clear all papers from the vector store"""
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.papers.clear()
            self.id_to_index.clear()
            self.index_to_id.clear()
            self.current_index = 0
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False