from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any

from app.api.deps import get_llm_provider, get_scraper
from app.api.v1.schemas.competitor import (
    CompetitorScrapeRequest,
    CompetitorInsightsResponse,
)
from app.domain.chains.content_generation import ContentRetrievalPipeline
from app.services.competitor_service import CompetitorService
from app.domain.llm_providers.base import BaseLLMProvider
from app.infrastructure.scraping.playwright_client import PlaywrightScraper


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


@router.post("/insights", response_model=CompetitorInsightsResponse)
async def get_competitor_insights(
    request: CompetitorScrapeRequest,
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
        competitor_service = CompetitorService(scraper)

        # Get competitor insights
        insights = await competitor_service.process_competitor_websites(
            competitor_links=request.competitors, namespace=request.user_id
        )

        # Check for scraping errors
        # if insights.get("scraping_error"):
        #     error_details = "\n".join(insights.get("error_details", ["Unknown error"]))
        #     # Return a 422 error instead of 500 since this is an expected error scenario
        #     # 422 is appropriate for "Unprocessable Entity" when the request is valid but cannot be processed
        #     raise HTTPException(
        #         status_code=422,
        #         detail=f"Failed to access competitor websites. Please verify the URLs are accessible:\n{error_details}",
        #     )

        # Return structured response
        # Dummy static data
        insights = {
            "competitor_insights": [
                {
                    "name": "Competitor A",
                    "strengths": ["SEO", "Video"],
                    "weaknesses": ["Low engagement"],
                },
                {
                    "name": "Competitor B",
                    "strengths": ["Social media"],
                    "weaknesses": ["Thin blog content"],
                },
            ],
            "content_gaps": [
                "In-depth case studies",
                "Interactive tools",
                "Localized content",
            ],
            "trending_topics": [
                "AI in marketing",
                "Short-form video",
                "Sustainability",
            ],
            "content_types": ["Blogs", "Videos", "Podcasts", "Infographics"],
            "scraping_error": False,
            "error_details": None,
        }
        return CompetitorInsightsResponse(**insights)

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing competitors: {str(e)}"
        )
