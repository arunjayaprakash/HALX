from pydantic_settings import BaseSettings
from typing import List
from pydantic import model_validator

class Settings(BaseSettings):
    mongodb_url: str = "mongodb://localhost:27017"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "mistral"
    arxiv_max_results: int = 50
    categories: List[str] = ["cs.AI", "cs.LG"]

    @model_validator(mode='before')
    @classmethod
    def parse_categories(cls, values):
        categories = values.get('categories')
        if isinstance(categories, str):
            # Remove quotes if present and split
            categories = categories.strip('"\'').split(',')
            values['categories'] = [cat.strip() for cat in categories]
        return values

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

settings = Settings()