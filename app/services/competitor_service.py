from typing import List, Dict, Any
from app.infrastructure.scraping.playwright_client import PlaywrightScraper
from app.services.embedding_service import EmbeddingService
from app.domain.agents.competitor_intelligence import CompetitorIntelligenceAgent


import re


def clean_text(text):
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
    text = re.sub(r"\n\s*\n", "\n\n", text)  # clean double newlines
    return text.strip()


class CompetitorService:
    def __init__(self, scraper: PlaywrightScraper):
        self.scraper = scraper

    async def scrape_single_competitor(self, url: str) -> Any:
        """
        Scrape a single competitor website and clean the text.

        Args:
            url: Competitor URL to scrape

        Returns:
            Dict with cleaned text or error
        """
        try:
            raw_data = await self.scraper.fetch_content(url)
            text = raw_data.get("text_content", "")
            cleaned = clean_text(text)
            return {"status": "success", "url": url, "cleaned_text": cleaned}
        except Exception as e:
            return {"error": str(e)}

    """Service for scraping competitor websites."""

    async def process_competitor_websites(
        self,
        competitor_links: List[str],
        embedding_provider_name: str = "huggingface",
        namespace: str = "",
    ) -> Dict[str, Any]:
        """
        Scrape competitor websites, clean text, then embed and upsert, returning processed data.

        Args:
            competitor_links: List of competitor URLs to process
            embedding_provider_name: Which embedding provider to use (default: huggingface)

        Returns:
            Dictionary containing processed data for each competitor
        """
        results = {}
        embedding_service = EmbeddingService(embedding_provider_name, namespace)
        for url in competitor_links:
            scrape_result = await self.scrape_single_competitor(url)
            if scrape_result.get("status") == "success":
                cleaned = scrape_result["cleaned_text"]
                embedding_result = embedding_service.process_and_upsert(cleaned, url)
                results[url] = {"status": "success", "url": url, **embedding_result}
            else:
                results[url] = scrape_result
        return results

    async def get_competitor_insights(
        self, competitor_links: List[str], llm_provider, scraper=None
    ) -> Dict[str, Any]:
        """
        Get AI-based competitor insights using CompetitorIntelligenceAgent.

        Args:
            competitor_links: List of competitor URLs to analyze
            llm_provider: LLM provider instance
            scraper: Optional scraper instance

        Returns:
            Dictionary containing competitor insights
        """
        agent = CompetitorIntelligenceAgent(llm_provider, scraper)
        return await agent.summarize_competitors(competitor_links)
