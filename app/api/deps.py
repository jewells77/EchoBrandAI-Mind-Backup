from fastapi import Depends

from app.domain.llm_providers.factory import create_llm_provider
from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper


# LLM Provider dependency
async def get_llm_provider() -> BaseLLMProvider:
    """
    Dependency for getting an LLM provider instance.
    Returns a configured LLM provider based on settings.
    """
    # This will use settings to determine which provider to create
    return create_llm_provider()


# Scraper dependency
async def get_scraper() -> PlaywrightScraper:
    """
    Dependency for getting a scraper instance.
    Returns a configured scraper for fetching web content.

    On systems where Playwright isn't fully supported, it will automatically
    fall back to using httpx for simpler web requests.
    """
    return PlaywrightScraper(headless=True, use_fallback=False)
