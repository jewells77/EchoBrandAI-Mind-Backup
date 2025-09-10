from typing import List, Dict, Any
import json
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper


class CompetitorInsights(BaseModel):
    """Structured insights from competitor content analysis."""

    competitor_insights: List[Dict[str, Any]] = Field(
        description="List of insights from each competitor"
    )
    content_gaps: List[str] = Field(
        description="Content opportunities the brand could exploit"
    )
    trending_topics: List[str] = Field(
        description="Topics trending across competitor content"
    )
    content_types: List[str] = Field(
        description="Content formats being used by competitors"
    )


class CompetitorIntelligenceAgent:
    def __init__(self, llm: BaseLLMProvider, scraper=None):
        self.llm = llm
        self.scraper = scraper or PlaywrightScraper()
        self.parser = PydanticOutputParser(pydantic_object=CompetitorInsights)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a competitor intelligence analyst who examines content from competing brands.
Your task is to analyze the content provided from competitor websites and:

1. Identify key insights from each competitor
2. Discover content gaps that could be exploited
3. Recognize trending topics across competitors
4. Note the content types/formats being used

Return your analysis as a structured JSON object.
{format_instructions}
""",
                ),
                (
                    "human",
                    """Here is the content from competitor websites:

{competitor_content}

Analyze this content and provide structured insights.""",
                ),
            ]
        )

    async def summarize_competitors(
        self, competitor_links: List[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch latest content from competitors, summarize trends, and identify content gaps.

        Args:
            competitor_links: Optional list of competitor URLs to analyze (default: None)

        Returns:
            Dictionary containing competitor insights and content gap opportunities

        Note:
            If no competitor links are provided, returns a basic analysis with empty insights
        """
        # Handle empty competitor list
        competitor_links = competitor_links or []

        # If no competitors, return basic structure
        if not competitor_links:
            return {
                "competitor_insights": [],
                "content_gaps": ["No competitors provided for analysis"],
                "trending_topics": [],
                "content_types": [],
            }

        # Step 1: Scrape content from all competitor links
        scraped_results = await self.scraper.fetch_multiple(competitor_links)

        # Step 2: Check if we have any successful scrapes
        successful_scrapes = [
            r
            for r in scraped_results
            if not (
                r.get("error")
                or r.get("text_content", "").startswith("Failed to fetch")
            )
        ]

        # If all scrapes failed, return a clear error
        if not successful_scrapes and scraped_results:
            error_messages = [
                f"{r['url']}: {r.get('error', 'Unknown error')}"
                for r in scraped_results
            ]
            return {
                "competitor_insights": [
                    {
                        "url": r["url"],
                        "name": r["url"].split("//")[-1].split("/")[0],
                        "key_insights": [
                            f"Failed to access: {r.get('error', 'Access restricted')}"
                        ],
                    }
                    for r in scraped_results
                ],
                "content_gaps": [
                    "Unable to identify content gaps due to access restrictions to competitor sites"
                ],
                "trending_topics": [],
                "content_types": [],
                "scraping_error": True,
                "error_details": error_messages,
            }

        # Format content for LLM
        competitor_content = []
        for result in scraped_results:
            if "error" in result and result["error"]:
                competitor_content.append(
                    f"URL: {result['url']}\nError: {result['error']}\nNote: This competitor could not be analyzed due to access restrictions."
                )
            else:
                competitor_content.append(
                    f"URL: {result['url']}\n"
                    f"Title: {result['title']}\n"
                    f"Description: {result['meta_description']}\n"
                    f"Content: {result['text_content'][:2000]}..."  # Truncate for token limits
                )

        formatted_content = "\n\n---\n\n".join(competitor_content)

        # Step 3: Format prompt with the scraped content
        formatted_prompt = self.prompt.format_messages(
            competitor_content=formatted_content,
            format_instructions=self.parser.get_format_instructions(),
        )

        # Step 4: Get response from LLM
        response = await self.llm.generate(formatted_prompt)

        try:
            # Parse the response into our Pydantic model
            parsed_response = self.parser.parse(response.content)
            # Convert to dict for return
            return parsed_response.model_dump()
        except Exception as e:
            # Fallback in case parsing fails
            return {
                "competitor_insights": [
                    {"url": link, "summary": "Could not analyze due to parsing error"}
                    for link in competitor_links
                ],
                "content_gaps": [
                    "Could not identify content gaps due to parsing error"
                ],
                "trending_topics": [],
                "content_types": [],
                "error": str(e),
            }
