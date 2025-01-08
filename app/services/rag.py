from typing import List, Dict, Any, Optional
import logging
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ..models.paper import Paper
from .embeddings import EmbeddingsService

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, 
                 embeddings_service: EmbeddingsService,
                 model_name: str = "mistral"):
        """Initialize the RAG service"""
        self.embeddings_service = embeddings_service
        self.llm = Ollama(model=model_name)
        
        # Initialize prompt templates
        self.query_prompt = PromptTemplate(
            input_variables=["context", "query"],
            template="""You are a helpful research assistant. Use the following research paper excerpts to answer the question. 
            If you cannot answer the question based solely on the provided excerpts, say so.

            Research Paper Excerpts:
            {context}

            Question: {query}

            Answer: """
        )
        
        self.summary_prompt = PromptTemplate(
            input_variables=["title", "abstract"],
            template="""Create a concise summary of the following research paper:

            Title: {title}
            Abstract: {abstract}

            Provide a clear, informative summary that captures the key points and contributions of the paper.
            
            Summary:"""
        )
        
        # Initialize chains
        self.query_chain = LLMChain(llm=self.llm, prompt=self.query_prompt)
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_prompt)

    def _format_context(self, papers: List[Dict[str, Any]]) -> str:
        """Format retrieved papers into context string"""
        context_parts = []
        for item in papers:
            paper = item['paper']
            score = item['similarity_score']
            context_parts.append(
                f"[Paper: {paper.title}]\n"
                f"Authors: {', '.join(paper.authors)}\n"
                f"Abstract: {paper.abstract}\n"
                f"Relevance Score: {score:.2f}\n"
            )
        return "\n---\n".join(context_parts)

    async def query(self, 
                   query: str, 
                   n_results: int = 3,
                   filter_categories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process a research query using RAG"""
        try:
            # Retrieve relevant papers
            similar_papers = self.embeddings_service.search_similar(
                query=query,
                n_results=n_results,
                filter_categories=filter_categories
            )
            
            if not similar_papers:
                return {
                    "answer": "I couldn't find any relevant papers to answer your question.",
                    "papers": []
                }
            
            # Format context from retrieved papers
            context = self._format_context(similar_papers)
            
            # Generate answer using LLM
            answer = await self.query_chain.arun(
                context=context,
                query=query
            )
            
            return {
                "answer": answer.strip(),
                "papers": [item['paper'] for item in similar_papers]
            }
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}")
            return {
                "answer": "An error occurred while processing your query.",
                "papers": []
            }

    async def generate_summary(self, paper: Paper) -> str:
        """Generate a concise summary of a paper"""
        try:
            summary = await self.summary_chain.arun(
                title=paper.title,
                abstract=paper.abstract
            )
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary for paper {paper.arxiv_id}: {str(e)}")
            return paper.abstract  # Fallback to abstract if summarization fails

    async def batch_process_papers(self, papers: List[Paper]) -> List[Paper]:
        """Process a batch of papers - generating summaries and storing embeddings"""
        try:
            # Generate summaries
            for paper in papers:
                if not paper.summary or paper.summary == paper.abstract:
                    paper.summary = await self.generate_summary(paper)
            
            # Store embeddings
            self.embeddings_service.add_papers(papers)
            
            return papers
            
        except Exception as e:
            logger.error(f"Error batch processing papers: {str(e)}")
            return papers  # Return original papers if processing fails