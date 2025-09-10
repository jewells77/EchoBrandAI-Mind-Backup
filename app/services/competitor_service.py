from typing import List, Dict, Any

from app.domain.agents.competitor_intelligence import CompetitorIntelligenceAgent
from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper


class CompetitorService:
    """Service for handling competitor intelligence operations."""

    def __init__(self, llm: BaseLLMProvider, scraper: PlaywrightScraper = None):
        self.competitor_agent = CompetitorIntelligenceAgent(llm, scraper)

    async def get_competitor_insights(
        self, competitor_links: List[str]
    ) -> Dict[str, Any]:
        """
        Get insights about competitors by scraping their websites.

        Args:
            competitor_links: List of competitor URLs to analyze

        Returns:
            Dictionary containing competitor insights
        """
        return await self.competitor_agent.summarize_competitors(competitor_links)
