from typing import List, Dict, Any
from app.infrastructure.scraping.playwright_client import PlaywrightScraper
from app.services.embedding_service import EmbeddingService
from app.core.logger import get_logger
from app.config import settings
from app.api.exceptions import APIError
from app.infrastructure.vectorstores.qdrant_store import QdrantStore

import re

logger = get_logger(__name__)


# def clean_text(text):
#     text = text.replace("\xa0", " ").replace("\u200b", " ")
#     text = re.sub(r"\s+", " ", text)  # collapse multiple spaces
#     text = re.sub(r"\n\s*\n", "\n\n", text)  # clean double newlines
#     return text.strip()


class CompetitorService:
    def __init__(self, scraper: PlaywrightScraper):
        self.scraper = scraper
        self.qdrant_store = QdrantStore()

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
            text_content = raw_data.get("text_content", "")
            return {"status": "success", "url": url, "text_content": text_content}
        except Exception as e:
            logger.exception(f"Error scraping {url}")
            raise

    """Service for scraping competitor websites."""

    async def process_competitor_website(
        self,
        url: str,
        embedding_provider_name: str = "huggingface",
        user_id: str = "",
    ) -> None:
        """
        Scrape competitor website, clean text, then embed and upsert, raising errors as appropriate.
        Args:
            url: Competitor URL to process
            embedding_provider_name: Which embedding provider to use (default: huggingface)
        """
        try:
            collection_name = settings.QDRANT_WEBSITE_CONTENT_COLLECTION
            embedding_service = EmbeddingService(
                embedding_provider_name, collection_name
            )
            scrape_result = await self.scrape_single_competitor(url)

            if scrape_result.get("status") == "success":
                cleaned = scrape_result["text_content"]
                await embedding_service.process_and_upsert(
                    cleaned, metadata={"url": url, "user_id": user_id}
                )
                return  # success
            # If scrape_result is not success, treat as error
            error_message = (
                scrape_result.get("message") or "Failed to process competitor website."
            )
            raise APIError(error_message, status_code=400)
        except Exception as e:
            logger.exception(f"Error scraping {url}")
            raise APIError(f"Error scraping the URL: {str(e)}", status_code=500)


async def delete_qdrant_embeddings_service(collection_name: str, filter_dict: dict):
    qdrant_store = QdrantStore()
    return await qdrant_store.delete_points_by_filter(collection_name, filter_dict)
