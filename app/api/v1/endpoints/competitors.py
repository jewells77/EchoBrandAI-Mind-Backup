from fastapi import APIRouter, Depends, HTTPException, Body
from typing import Dict, Any
from pydantic import BaseModel

from app.api.deps import get_llm_provider, get_scraper
from app.api.v1.schemas.competitor import (
    CompetitorScrapeRequest,
    DeleteEmbeddingsRequest,
)
from app.domain.chains.content_generation import ContentRetrievalPipeline
from app.services.competitor_service import CompetitorService
from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper
from app.infrastructure.vectorstores.pinecone_store import delete_embeddings

from app.api.exceptions import APIError


router = APIRouter()

# pipeline = ContentRetrievalPipeline(namespace=user_id)
# user_query = "if we're a right fit and come up with a plan"
# # filter = {"category": {"$eq": "digestive system"}}
# # fields = ["category", "text"]
# context = pipeline.retrieve_context(
#     user_query=user_query,
#     top_k=3,
#     # metadata_filter=filter,
#     # fields=fields,
# )
# print("Retrieved context:", context)


@router.post("/insights")
async def get_competitor_insights(
    request: CompetitorScrapeRequest,
    scraper: PlaywrightScraper = Depends(get_scraper),
):
    """
    Scrape and analyze a single competitor website to extract insights.

    This endpoint can be used to:
    1. Gather competitive intelligence separate from content generation
    2. Pre-scrape competitor data to use in future content generation requests

    Returns a simple success or raises an HTTPException on error.
    """
    try:
        # Initialize the competitor service
        competitor_service = CompetitorService(scraper)

        # Get competitor insights
        await competitor_service.process_competitor_website(
            url=str(request.url), namespace=request.user_id
        )
        return {
            "status": "success",
            "message": "Competitor website processed successfully.",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing competitors: {str(e)}"
        )


@router.post("/delete-embeddings")
async def delete_embeddings_api(request: DeleteEmbeddingsRequest):
    """
    Delete embeddings from Pinecone based on metadata filter and namespace (user ID).
    """
    try:
        result = delete_embeddings(
            metadata_filter=request.metadata_filter, namespace=request.namespace
        )
        return {"status": "success", "result": str(result)}

    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting embeddings: {str(e)}"
        )
