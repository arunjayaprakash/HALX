from typing import Optional
import asyncio
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class BaselineLLM:
    def __init__(
        self,
        model_name: str = "mistral",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the baseline LLM system.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for generation (0.0 to 1.0)
            system_prompt: Optional system prompt to set context
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        
        self.system_prompt = system_prompt or (
            "You are a helpful research assistant. Provide clear, accurate, "
            "and concise responses to questions about scientific topics."
        )

    async def query(self, question: str) -> str:
        """
        Query the LLM with a question.
        
        Args:
            question: The user's question
            
        Returns:
            The LLM's response as a string
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content


# Usage example:
async def main():
    # Initialize the baseline system
    baseline = BaselineLLM()
    
    # Single question example
    question = "What are some recent studie into bias in techniques such as stable diffusion?"
    response = await baseline.query(question)
    print(f"Question: {question}\nResponse: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())