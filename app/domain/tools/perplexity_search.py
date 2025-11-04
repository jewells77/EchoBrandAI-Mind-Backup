from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Literal, List
from perplexity import Perplexity
from app.config import settings

client = Perplexity(api_key=settings.PERPLEXITY_API_KEY)


class PerplexitySearchInput(BaseModel):
    """Input for Perplexity search queries."""

    queries: List[str] = Field(description="The queries to search for")
    start_date: str = Field(description="Start date in mm/dd/yyyy format")
    end_date: str = Field(description="End date in mm/dd/yyyy format")


@tool("perplexity_search", args_schema=PerplexitySearchInput)
def perplexity_search(queries: List[str], start_date: str, end_date: str) -> str:
    """
    Perform a multi-query web search using Perplexity AI and save the results to a file.
    """
    search = client.search.create(
        query=queries,
        max_tokens=2000,
        max_results=10,
        search_after_date_filter=start_date,
        search_before_date_filter=end_date,
    )

    filtered_results = [
        {
            "title": result.title,
            "content": result.snippet,
        }
        for result in search.results
    ]
    return filtered_results
