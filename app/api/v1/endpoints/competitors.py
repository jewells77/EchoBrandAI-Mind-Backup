from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.api.deps import get_llm_provider, get_scraper
from app.api.v1.schemas.competitor import (
    CompetitorScrapeRequest,
    CompetitorInsightsResponse,
)
from app.services.competitor_service import CompetitorService
from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper


router = APIRouter()


@router.post("/insights", response_model=CompetitorInsightsResponse)
async def get_competitor_insights(
    request: CompetitorScrapeRequest,
    llm_provider: BaseLLMProvider = Depends(get_llm_provider),
    scraper: PlaywrightScraper = Depends(get_scraper),
) -> CompetitorInsightsResponse:
    """
    Scrape and analyze competitor websites to extract insights.

    This endpoint can be used to:
    1. Gather competitive intelligence separate from content generation
    2. Pre-scrape competitor data to use in future content generation requests

    Returns structured competitor insights that can be passed to content generation.
    """
    try:
        # Initialize the competitor service
        competitor_service = CompetitorService(llm_provider, scraper)

        # Get competitor insights
        insights = await competitor_service.get_competitor_insights(
            competitor_links=request.competitors
        )

        # Check for scraping errors
        if insights.get("scraping_error"):
            error_details = "\n".join(insights.get("error_details", ["Unknown error"]))
            # Return a 422 error instead of 500 since this is an expected error scenario
            # 422 is appropriate for "Unprocessable Entity" when the request is valid but cannot be processed
            raise HTTPException(
                status_code=422,
                detail=f"Failed to access competitor websites. Please verify the URLs are accessible:\n{error_details}",
            )

        # Return structured response
        return CompetitorInsightsResponse(**insights)

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing competitors: {str(e)}"
        )
