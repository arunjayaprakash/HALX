from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class Paper(BaseModel):
    """Model representing a research paper"""
    arxiv_id: str = Field(..., description="ArXiv ID of the paper")
    title: str = Field(..., description="Title of the paper")
    abstract: str = Field(..., description="Full abstract of the paper")
    summary: str = Field(..., description="Generated summary of the paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    categories: List[str] = Field(default_factory=list, description="ArXiv categories")
    published_date: datetime = Field(..., description="Original publication date")
    updated_date: Optional[datetime] = Field(None, description="Last update date")
    pdf_url: str = Field(..., description="URL to the PDF version")

    class Config:
        json_schema_extra = {
            "example": {
                "arxiv_id": "2401.00123",
                "title": "A Novel Approach to Neural Networks",
                "abstract": "This paper presents...",
                "summary": "A concise summary...",
                "authors": ["David Bowman", "Frank Poole"],
                "categories": ["cs.AI", "cs.LG"],
                "published_date": "2024-01-01T00:00:00",
                "updated_date": "2024-01-02T00:00:00",
                "pdf_url": "https://arxiv.org/pdf/2401.00123"
            }
        }